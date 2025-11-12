#pragma once

#include <torch/extension.h>

// ===== 基础函数 =====
// 计算knot spans（并行化）
const torch::Tensor toSpans(const int &degree, const torch::Tensor &knot_vector,
                            const int &num_ctrlpts, const torch::Tensor &knots);

// 计算B样条基函数（Cox-de Boor递推，向量化）
const torch::Tensor toBasisFunctions(const int &degree,
                                     const torch::Tensor &knot_vector,
                                     const torch::Tensor &spans,
                                     const torch::Tensor &knots);

// ===== 曲面求值函数 =====
// 旧版本（用于对比）
const torch::Tensor toEvalPointsOld(const torch::Tensor &spans_u,
                                    const torch::Tensor &spans_v,
                                    const torch::Tensor &basis_u,
                                    const torch::Tensor &basis_v,
                                    const torch::Tensor &ctrlpts);

// 优化版本：使用预计算的spans和basis，最大化并行
const torch::Tensor toEvalPoints(const torch::Tensor &spans_u,
                                 const torch::Tensor &spans_v,
                                 const torch::Tensor &basis_u,
                                 const torch::Tensor &basis_v,
                                 const torch::Tensor &ctrlpts);

// ===== 完整流程函数 =====
// 规则网格采样（从knots到采样点，完整流程）
const torch::Tensor toTorchPoints(const int &degree_u, const int &degree_v,
                                  const torch::Tensor &knots_u,
                                  const torch::Tensor &knots_v,
                                  const torch::Tensor &knotvector_u,
                                  const torch::Tensor &knotvector_v,
                                  const torch::Tensor &ctrlpts);

// 任意UV坐标采样（高度优化版本，使用einsum）
const torch::Tensor toUVTorchPoints(const int &degree_u, const int &degree_v,
                                    const torch::Tensor &knotvector_u,
                                    const torch::Tensor &knotvector_v,
                                    const torch::Tensor &ctrlpts,
                                    const torch::Tensor &uv);
