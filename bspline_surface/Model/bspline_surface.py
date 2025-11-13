import os
import json
import torch
import numpy as np
import open3d as o3d
from typing import Union

import bsp_cpp

from bspline_surface.Method.data import toNumpy
from bspline_surface.Method.check import checkShape
from bspline_surface.Method.render import renderPoints
from bspline_surface.Method.path import createFileFolder, removeFile, renameFile


class BSplineSurface(object):
    def __init__(
        self,
        degree_u: int = 3,
        degree_v: int = 3,
        size_u: int = 5,
        size_v: int = 7,
        sample_num_u: int = 20,
        sample_num_v: int = 20,
        dtype=torch.float32,
        device: str = "cpu",
    ) -> None:
        # Super Params
        self.degree_u = degree_u
        self.degree_v = degree_v
        self.size_u = size_u
        self.size_v = size_v
        self.sample_num_u = sample_num_u
        self.sample_num_v = sample_num_v
        self.dtype = dtype
        self.device = device

        # Diff Params
        self.knotvector_u = torch.zeros(
            self.size_u - self.degree_u - 1, dtype=self.dtype
        ).to(self.device)
        self.knotvector_v = torch.zeros(
            self.size_v - self.degree_v - 1, dtype=self.dtype
        ).to(self.device)
        self.ctrlpts = torch.zeros(
            [self.size_u - 1, self.size_v - 1, 3], dtype=self.dtype
        ).to(self.device)

        # Optional Params
        self.sigmoid_knotvector_u = torch.zeros(
            self.degree_u + self.size_u, dtype=self.dtype
        ).to(self.device)
        self.sigmoid_knotvector_u_loaded = False

        self.sigmoid_knotvector_v = torch.zeros(
            self.degree_v + self.size_v, dtype=self.dtype
        ).to(self.device)
        self.sigmoid_knotvector_v_loaded = False

        self.sample_uv = torch.empty(0)
        self.sample_uv_loaded = False

        # Cache for optimized computation
        self._cache = {}
        self._cache_valid = False

        self.reset()
        return

    @classmethod
    def fromParamsDict(
        cls,
        params_dict: dict,
        sample_num_u: int = 20,
        sample_num_v: int = 20,
        dtype=torch.float32,
        device: str = "cpu",
    ):
        degree_u = params_dict["degree_u"]
        degree_v = params_dict["degree_v"]
        size_u = params_dict["size_u"]
        size_v = params_dict["size_v"]

        bspline_surface = cls(
            degree_u,
            degree_v,
            size_u,
            size_v,
            sample_num_u,
            sample_num_v,
            dtype,
            device,
        )

        bspline_surface.loadParamsDict(params_dict)

        return bspline_surface

    @classmethod
    def fromParamsFile(
        cls,
        params_file_path: str,
        sample_num_u: int = 20,
        sample_num_v: int = 20,
        dtype=torch.float32,
        device: str = "cpu",
    ):
        params_dict = np.load(params_file_path, allow_pickle=True).item()

        return cls.fromParamsDict(
            params_dict,
            sample_num_u,
            sample_num_v,
            dtype,
            device,
        )

    def reset(self) -> bool:
        self.sigmoid_knotvector_u_loaded = False
        self.sigmoid_knotvector_v_loaded = False
        self.sample_uv_loaded = False
        self._cache_valid = False
        self._cache = {}
        return True

    def setGradState(self, need_grad: bool) -> bool:
        self.knotvector_u.requires_grad_(need_grad)
        self.knotvector_v.requires_grad_(need_grad)
        self.ctrlpts.requires_grad_(need_grad)
        return True

    def loadParams(
        self,
        knotvector_u: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
        knotvector_v: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
        ctrlpts: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
        sigmoid_knotvector_u: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
        sigmoid_knotvector_v: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
        sample_uv: Union[torch.Tensor, np.ndarray, list, tuple, None] = None,
    ) -> bool:
        if knotvector_u is not None:
            if isinstance(knotvector_u, list) or isinstance(knotvector_u, tuple):
                knotvector_u = np.array(knotvector_u)

            if not checkShape(knotvector_u.shape, self.knotvector_u.shape):
                print("[ERROR][BSplineSurface::loadParams]")
                print("\t checkShape failed for knotvector_u!")
                return False

            if isinstance(knotvector_u, np.ndarray):
                knotvector_u = torch.from_numpy(knotvector_u)

            self.knotvector_u.data = (
                knotvector_u.detach().clone().type(self.dtype).to(self.device)
            )

        if knotvector_v is not None:
            if isinstance(knotvector_v, list) or isinstance(knotvector_v, tuple):
                knotvector_v = np.array(knotvector_v)

            if not checkShape(knotvector_v.shape, self.knotvector_v.shape):
                print("[ERROR][BSplineSurface::loadParams]")
                print("\t checkShape failed for knotvector_v!")
                return False

            if isinstance(knotvector_v, np.ndarray):
                knotvector_v = torch.from_numpy(knotvector_v)

            self.knotvector_v.data = (
                knotvector_v.detach().clone().type(self.dtype).to(self.device)
            )

        if ctrlpts is not None:
            if isinstance(ctrlpts, list) or isinstance(ctrlpts, tuple):
                ctrlpts = np.array(ctrlpts)

            if not checkShape(ctrlpts.shape, self.ctrlpts.shape):
                print("[ERROR][BSplineSurface::loadParams]")
                print("\t checkShape failed for ctrlpts!")
                return False

            if isinstance(ctrlpts, np.ndarray):
                ctrlpts = torch.from_numpy(ctrlpts)

            self.ctrlpts.data = (
                ctrlpts.detach().clone().type(self.dtype).to(self.device)
            )

        if sigmoid_knotvector_u is not None:
            if isinstance(sigmoid_knotvector_u, list) or isinstance(
                sigmoid_knotvector_u, tuple
            ):
                sigmoid_knotvector_u = np.array(sigmoid_knotvector_u)

            if not checkShape(
                sigmoid_knotvector_u.shape, self.sigmoid_knotvector_u.shape
            ):
                print("[ERROR][BSplineSurface::loadParams]")
                print("\t checkShape failed for sigmoid_knotvector_u!")
                return False

            if isinstance(sigmoid_knotvector_u, np.ndarray):
                sigmoid_knotvector_u = torch.from_numpy(sigmoid_knotvector_u)

            self.sigmoid_knotvector_u = (
                sigmoid_knotvector_u.detach().clone().type(self.dtype).to(self.device)
            )
            self.sigmoid_knotvector_u_loaded = True

        if sigmoid_knotvector_v is not None:
            if isinstance(sigmoid_knotvector_v, list) or isinstance(
                sigmoid_knotvector_v, tuple
            ):
                sigmoid_knotvector_v = np.array(sigmoid_knotvector_v)

            if not checkShape(
                sigmoid_knotvector_v.shape, self.sigmoid_knotvector_v.shape
            ):
                print("[ERROR][BSplineSurface::loadParams]")
                print("\t checkShape failed for sigmoid_knotvector_v!")
                return False

            if isinstance(sigmoid_knotvector_v, np.ndarray):
                sigmoid_knotvector_v = torch.from_numpy(sigmoid_knotvector_v)

            self.sigmoid_knotvector_v = (
                sigmoid_knotvector_v.detach().clone().type(self.dtype).to(self.device)
            )
            self.sigmoid_knotvector_v_loaded = True

        if sample_uv is not None:
            if isinstance(sample_uv, list) or isinstance(sample_uv, tuple):
                sample_uv = np.array(sample_uv)

            if sample_uv.shape[1] != 2:
                print("[ERROR][BSplineSurface::loadParams]")
                print("\t checkShape failed for sample_uv!")
                return False

            if isinstance(sample_uv, np.ndarray):
                sample_uv = torch.from_numpy(sample_uv)

            self.sample_uv = sample_uv.detach().clone().type(self.dtype).to(self.device)
            self.sample_uv_loaded = True

        # 参数变化后需要重置缓存
        self._cache_valid = False
        return True

    def loadParamsDict(self, params_dict: dict) -> bool:
        knotvector_u = params_dict.get("knotvector_u", None)
        knotvector_v = params_dict.get("knotvector_v", None)
        ctrlpts = params_dict["ctrlpts"]
        sigmoid_knotvector_u = params_dict.get("sigmoid_knotvector_u", None)
        sigmoid_knotvector_v = params_dict.get("sigmoid_knotvector_v", None)
        sample_uv = params_dict.get("sample_uv", None)

        self.loadParams(
            knotvector_u,
            knotvector_v,
            ctrlpts,
            sigmoid_knotvector_u,
            sigmoid_knotvector_v,
            sample_uv,
        )

        return True

    def loadParamsFile(self, params_file_path: str) -> bool:
        if not os.path.exists(params_file_path):
            print("[ERROR][BSplineSurface::loadParamsFile]")
            print("\t params dict file not exist!")
            print("\t params_file_path:", params_file_path)
            return False

        params_dict = np.load(params_file_path, allow_pickle=True).item()

        if not self.loadParamsDict(params_dict):
            print("[ERROR][BSplineSurface::loadParamsFile]")
            print("\t loadParamsDict failed!")
            return False

        return True

    def toSigmoidKnotvectorU(self) -> torch.Tensor:
        if self.sigmoid_knotvector_u_loaded:
            return self.sigmoid_knotvector_u

        full_knotvector_u = torch.zeros(
            [self.degree_u + self.size_u], dtype=self.knotvector_u.dtype
        ).to(self.knotvector_u.device)

        full_knotvector_u[-self.degree_u - 1 :] = 1.0

        sigmoid_knotvector_u = torch.sigmoid(self.knotvector_u)

        sigmoid_knotvector_u_sum = torch.sum(sigmoid_knotvector_u)

        normed_sigmoid_knotvector_u = sigmoid_knotvector_u / sigmoid_knotvector_u_sum

        for i in range(normed_sigmoid_knotvector_u.shape[0] - 1):
            full_knotvector_u[self.degree_u + 1 + i] = (
                full_knotvector_u[self.degree_u + i] + normed_sigmoid_knotvector_u[i]
            )
        return full_knotvector_u

    def toSigmoidKnotvectorV(self) -> torch.Tensor:
        if self.sigmoid_knotvector_v_loaded:
            return self.sigmoid_knotvector_v

        full_knotvector_v = torch.zeros(
            [self.degree_v + self.size_v], dtype=self.knotvector_v.dtype
        ).to(self.knotvector_v.device)

        full_knotvector_v[-self.degree_v - 1 :] = 1.0

        sigmoid_knotvector_v = torch.sigmoid(self.knotvector_v)

        sigmoid_knotvector_v_sum = torch.sum(sigmoid_knotvector_v)

        normed_sigmoid_knotvector_v = sigmoid_knotvector_v / sigmoid_knotvector_v_sum

        for i in range(normed_sigmoid_knotvector_v.shape[0] - 1):
            full_knotvector_v[self.degree_v + 1 + i] = (
                full_knotvector_v[self.degree_v + i] + normed_sigmoid_knotvector_v[i]
            )

        return full_knotvector_v

    def _precomputeCache(self) -> None:
        """预计算所有可复用的中间结果，最大化缓存利用"""
        if self._cache_valid:
            return

        # 1. 预计算sigmoid knotvector (如果未加载)
        sigmoid_knotvector_u = self.toSigmoidKnotvectorU()
        sigmoid_knotvector_v = self.toSigmoidKnotvectorV()

        # 2. 预计算规则网格的knots
        knots_u = torch.linspace(0.0, 1.0, self.sample_num_u, dtype=self.dtype, device=self.device)
        knots_v = torch.linspace(0.0, 1.0, self.sample_num_v, dtype=self.dtype, device=self.device)

        # 3. 预计算索引偏移网格 (固定不变，仅依赖于degree)
        du_grid = torch.arange(0, self.degree_u + 1, dtype=torch.int64, device=self.device)
        dv_grid = torch.arange(0, self.degree_v + 1, dtype=torch.int64, device=self.device)

        # 创建外积网格 [degree_u+1, degree_v+1] -> [(degree_u+1)*(degree_v+1)]
        du_mesh, dv_mesh = torch.meshgrid(du_grid, dv_grid, indexing='ij')
        self._cache['du_grid_flat'] = du_mesh.reshape(-1)
        self._cache['dv_grid_flat'] = dv_mesh.reshape(-1)
        self._cache['elem_num'] = (self.degree_u + 1) * (self.degree_v + 1)

        # 4. 缓存sigmoid knotvector
        self._cache['sigmoid_knotvector_u'] = sigmoid_knotvector_u
        self._cache['sigmoid_knotvector_v'] = sigmoid_knotvector_v

        # 5. 缓存规则网格的knots
        self._cache['knots_u'] = knots_u
        self._cache['knots_v'] = knots_v

        # 6. 预计算规则网格的spans（仅在knotvector不变时有效）
        if not self.sample_uv_loaded:
            spans_u = self._computeSpansBatch(sigmoid_knotvector_u, knots_u, self.size_u - 1, self.degree_u)
            spans_v = self._computeSpansBatch(sigmoid_knotvector_v, knots_v, self.size_v - 1, self.degree_v)
            self._cache['spans_u'] = spans_u
            self._cache['spans_v'] = spans_v

            # 7. 预计算规则网格的基函数（仅在knotvector和knots不变时有效）
            basis_u = self._computeBasisFunctionsBatch(
                self.degree_u, sigmoid_knotvector_u, spans_u, knots_u
            )
            basis_v = self._computeBasisFunctionsBatch(
                self.degree_v, sigmoid_knotvector_v, spans_v, knots_v
            )
            self._cache['basis_u'] = basis_u
            self._cache['basis_v'] = basis_v

        self._cache_valid = True

    def toUVSamplePoints(
        self, sample_uv: Union[torch.Tensor, np.ndarray, list]
    ) -> torch.Tensor:
        if isinstance(sample_uv, list):
            sample_uv = np.array(sample_uv)
        if isinstance(sample_uv, np.ndarray):
            sample_uv = torch.from_numpy(sample_uv).to(self.device, dtype=self.dtype)

        sigmoid_knotvector_u = self.toSigmoidKnotvectorU()
        sigmoid_knotvector_v = self.toSigmoidKnotvectorV()

        sample_points = bsp_cpp.toUVTorchPoints(
            self.degree_u,
            self.degree_v,
            sigmoid_knotvector_u,
            sigmoid_knotvector_v,
            self.ctrlpts,
            sample_uv,
        )

        return sample_points

    def toSamplePoints(self) -> torch.Tensor:
        if self.sample_uv_loaded:
            return self.toUVSamplePoints(self.sample_uv)

        """高度优化的并行采样点计算，最大程度利用缓存和向量化"""
        # 确保缓存已预计算
        self._precomputeCache()

        # 从缓存获取预计算的值
        sigmoid_knotvector_u = self._cache['sigmoid_knotvector_u']
        sigmoid_knotvector_v = self._cache['sigmoid_knotvector_v']
        knots_u = self._cache['knots_u']
        knots_v = self._cache['knots_v']
        du_grid_flat = self._cache['du_grid_flat']
        dv_grid_flat = self._cache['dv_grid_flat']
        elem_num = self._cache['elem_num']

        # 控制点维度
        H, W, dim = self.ctrlpts.shape
        sample_num_u = self.sample_num_u
        sample_num_v = self.sample_num_v

        # ===== 步骤1&2: 使用缓存的spans和basis（如果可用）=====
        if 'spans_u' in self._cache and 'basis_u' in self._cache:
            # 使用预计算的结果，避免重复计算
            spans_u = self._cache['spans_u']
            spans_v = self._cache['spans_v']
            basis_u = self._cache['basis_u']
            basis_v = self._cache['basis_v']
        else:
            # 实时计算spans和basis
            spans_u = self._computeSpansBatch(sigmoid_knotvector_u, knots_u, H, self.degree_u)
            spans_v = self._computeSpansBatch(sigmoid_knotvector_v, knots_v, W, self.degree_v)

            basis_u = self._computeBasisFunctionsBatch(
                self.degree_u, sigmoid_knotvector_u, spans_u, knots_u
            )
            basis_v = self._computeBasisFunctionsBatch(
                self.degree_v, sigmoid_knotvector_v, spans_v, knots_v
            )

        # ===== 步骤3: 并行计算控制点索引和采样 =====
        # 计算基础索引 [sample_num_u] 和 [sample_num_v]
        idx_u = spans_u - self.degree_u
        idx_v = spans_v - self.degree_v

        # 扩展索引到 [sample_num_u, sample_num_v, elem_num]
        # 使用广播而不是循环
        idx_u_exp = idx_u.view(sample_num_u, 1, 1) + du_grid_flat.view(1, 1, elem_num)
        idx_v_exp = idx_v.view(1, sample_num_v, 1) + dv_grid_flat.view(1, 1, elem_num)

        # 扁平化索引 [sample_num_u * sample_num_v * elem_num]
        flat_idx = (idx_u_exp * W + idx_v_exp).reshape(-1)

        # 一次性gather所有需要的控制点
        flat_ctrlpts = self.ctrlpts.reshape(H * W, dim)
        sampled_ctrlpts = torch.index_select(flat_ctrlpts, 0, flat_idx).view(
            sample_num_u, sample_num_v, elem_num, dim
        )

        # ===== 步骤4: 并行计算权重外积 =====
        # 使用广播计算权重外积
        # basis_u: [sample_num_u, degree_u+1]
        # basis_v: [sample_num_v, degree_v+1]
        # 需要: [sample_num_u, sample_num_v, (degree_u+1)*(degree_v+1)]
        weights_u = basis_u.view(sample_num_u, 1, self.degree_u + 1, 1)
        weights_v = basis_v.view(1, sample_num_v, 1, self.degree_v + 1)
        weights = (weights_u * weights_v).reshape(sample_num_u, sample_num_v, elem_num, 1)

        # ===== 步骤5: 加权求和得到最终采样点 =====
        sample_points = (sampled_ctrlpts * weights).sum(dim=2)

        return sample_points

    def _computeSpansBatch(
        self, knotvector: torch.Tensor, knots: torch.Tensor, num_ctrlpts: int, degree: int
    ) -> torch.Tensor:
        """批量计算spans，完全并行化"""
        # 使用CPU进行searchsorted（通常更快），然后转回设备
        knotvector_cpu = knotvector.cpu().contiguous()
        knots_cpu = knots.cpu().contiguous()

        indices = torch.searchsorted(knotvector_cpu, knots_cpu, right=False)
        spans = torch.clamp(indices - 1, degree, num_ctrlpts - 1)

        return spans.to(knotvector.device)

    def _computeBasisFunctionsBatch(
        self, degree: int, knotvector: torch.Tensor, spans: torch.Tensor, knots: torch.Tensor
    ) -> torch.Tensor:
        """批量计算B样条基函数，完全向量化"""
        batch_size = spans.size(0)
        dtype = knotvector.dtype
        device = knotvector.device

        # 初始化
        left = torch.zeros(batch_size, degree + 1, dtype=dtype, device=device)
        right = torch.zeros(batch_size, degree + 1, dtype=dtype, device=device)
        N = torch.ones(batch_size, degree + 1, dtype=dtype, device=device)

        # Cox-de Boor递推公式，向量化实现
        for j in range(1, degree + 1):
            left[:, j] = knots - knotvector[spans - j + 1]
            right[:, j] = knotvector[spans + j] - knots

            saved = torch.zeros(batch_size, dtype=dtype, device=device)

            for r in range(j):
                denominator = right[:, r + 1] + left[:, j - r]
                temp = N[:, r] / denominator

                N[:, r] = saved + right[:, r + 1] * temp
                saved = left[:, j - r] * temp

            N[:, j] = saved

        return N

    def toSamplePcd(self) -> o3d.geometry.PointCloud:
        sample_points = toNumpy(self.toSamplePoints()).reshape(-1, 3).astype(np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sample_points)

        return pcd

    def renderSamplePoints(self) -> bool:
        sample_points = toNumpy(self.toSamplePoints()).reshape(-1, 3)

        renderPoints(sample_points)
        return True

    def toKNotsU(self) -> np.ndarray:
        return toNumpy(self.toSigmoidKnotvectorU())

    def toKNotsV(self) -> np.ndarray:
        return toNumpy(self.toSigmoidKnotvectorV())

    def toCtrlPts(self) -> np.ndarray:
        return toNumpy(self.ctrlpts).reshape(-1)

    def toParamsDict(self) -> dict:
        params_dict = {
            "degree_u": self.degree_u,
            "degree_v": self.degree_v,
            "size_u": self.size_u,
            "size_v": self.size_v,
            "knotvector_u": toNumpy(self.knotvector_u),
            "knotvector_v": toNumpy(self.knotvector_v),
            "ctrlpts": toNumpy(self.ctrlpts),
        }
        return params_dict

    def saveParamsFile(
        self, save_params_file_path: str, overwrite: bool = False
    ) -> bool:
        if os.path.exists(save_params_file_path):
            if not overwrite:
                return True

            removeFile(save_params_file_path)

        params_dict = self.toParamsDict()

        createFileFolder(save_params_file_path)

        tmp_save_params_file_path = save_params_file_path[:-4] + "_tmp.npy"
        removeFile(tmp_save_params_file_path)

        np.save(tmp_save_params_file_path, params_dict)
        renameFile(tmp_save_params_file_path, save_params_file_path)
        return True

    def toAMCAXParamsDict(self) -> dict:
        params_dict = self.toParamsDict()

        del params_dict["knotvector_u"]
        del params_dict["knotvector_v"]

        params_dict["Surface type"] = "BSplineSurface"
        params_dict["ctrlpts"] = params_dict["ctrlpts"].reshape(-1)

        sigmoid_knotvector_u = toNumpy(self.toSigmoidKnotvectorU())
        sigmoid_knotvector_v = toNumpy(self.toSigmoidKnotvectorV())
        knots_u, mults_u = np.unique(sigmoid_knotvector_u, return_counts=True)
        knots_v, mults_v = np.unique(sigmoid_knotvector_v, return_counts=True)
        params_dict["knots_u"] = knots_u
        params_dict["mults_u"] = mults_u
        params_dict["knots_v"] = knots_v
        params_dict["mults_v"] = mults_v

        for key, value in params_dict.items():
            if isinstance(value, np.ndarray):
                params_dict[key] = value.tolist()

        return params_dict

    def saveParamsJsonFile(
        self, save_params_json_file_path: str, overwrite: bool = False
    ) -> bool:
        if os.path.exists(save_params_json_file_path):
            if not overwrite:
                return True

            removeFile(save_params_json_file_path)

        params_dict = self.toAMCAXParamsDict()

        createFileFolder(save_params_json_file_path)

        tmp_save_params_json_file_path = save_params_json_file_path[:-4] + "_tmp.npy"
        removeFile(tmp_save_params_json_file_path)

        with open(tmp_save_params_json_file_path, "w", encoding="utf-8") as f:
            json.dump(params_dict, f, ensure_ascii=False, indent=4)

        renameFile(tmp_save_params_json_file_path, save_params_json_file_path)
        return True

    def saveAsPcdFile(
        self,
        save_pcd_file_path: str,
        overwrite: bool = False,
        print_progress: bool = False,
        uniform_color=None,
    ) -> bool:
        if os.path.exists(save_pcd_file_path):
            if overwrite:
                removeFile(save_pcd_file_path)
            else:
                print("[ERROR][BSplineSurface::saveAsPcdFile]")
                print("\t save pcd file already exist!")
                print("\t save_pcd_file_path:", save_pcd_file_path)
                return False

        createFileFolder(save_pcd_file_path)

        points = toNumpy(self.toSamplePoints()).reshape(-1, 3)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if uniform_color is not None:
            pcd.paint_uniform_color(uniform_color)

        if print_progress:
            print("[INFO][BSplineSurface::saveAsPcdFile]")
            print("\t start save as pcd file...")
        o3d.io.write_point_cloud(
            save_pcd_file_path, pcd, write_ascii=True, print_progress=print_progress
        )
        return True
