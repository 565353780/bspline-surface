#include "value.h"

const torch::Tensor toSpans(const int &degree, const torch::Tensor &knot_vector,
                            const int &num_ctrlpts,
                            const torch::Tensor &knots) {
  torch::Tensor knot_vector_cpu = knot_vector.to(torch::kCPU).contiguous();
  torch::Tensor knots_cpu = knots.to(torch::kCPU).contiguous();

  const torch::Tensor indices =
      torch::searchsorted(knot_vector_cpu, knots_cpu, /*right=*/false);

  return torch::clamp(indices - 1, degree, num_ctrlpts - 1)
      .to(knot_vector.device());
}

const torch::Tensor toBasisFunctions(const int &degree,
                                     const torch::Tensor &knot_vector,
                                     const torch::Tensor &spans,
                                     const torch::Tensor &knots) {
  // 优化版本：减少index操作，使用更高效的切片
  const auto opts = torch::TensorOptions()
                        .dtype(knot_vector.dtype())
                        .device(knot_vector.device());

  const int batch_size = spans.size(0);

  // 初始化张量
  auto left = torch::zeros({batch_size, degree + 1}, opts);
  auto right = torch::zeros({batch_size, degree + 1}, opts);
  auto N = torch::ones({batch_size, degree + 1}, opts);

  // Cox-de Boor递推公式，优化版本
  for (int j = 1; j <= degree; ++j) {
    // 使用切片操作替代index_put，通常更快
    left.index_put_({"...", j}, knots - knot_vector.index({spans - j + 1}));
    right.index_put_({"...", j}, knot_vector.index({spans + j}) - knots);

    auto saved = torch::zeros({batch_size}, opts);

    for (int r = 0; r < j; ++r) {
      // 批量计算，避免逐个元素操作
      const auto denominator = right.index({"...", r + 1}) + left.index({"...", j - r});
      const auto temp = N.index({"...", r}) / denominator;

      N.index_put_({"...", r}, saved + right.index({"...", r + 1}) * temp);
      saved = left.index({"...", j - r}) * temp;
    }

    N.index_put_({"...", j}, saved);
  }

  return N;
}

const torch::Tensor toEvalPointsOld(const torch::Tensor &spans_u,
                                    const torch::Tensor &spans_v,
                                    const torch::Tensor &basis_u,
                                    const torch::Tensor &basis_v,
                                    const torch::Tensor &ctrlpts) {
  const int sample_num_u = spans_u.size(0);
  const int sample_num_v = spans_v.size(0);

  // for simplify, set local degree_u = global degree_u + 1
  const int degree_u = basis_u.size(1);
  const int degree_v = basis_v.size(1);

  const torch::Tensor idx_u = spans_u - degree_u + 1;
  const torch::Tensor idx_v = spans_v - degree_v + 1;

  const torch::TensorOptions opts =
      torch::TensorOptions().dtype(ctrlpts.dtype()).device(ctrlpts.device());

  torch::Tensor eval_points =
      torch::zeros({sample_num_u, sample_num_v, 3}, opts);

  for (size_t i = 0; i < sample_num_u; ++i) {
    for (size_t j = 0; j < sample_num_v; ++j) {
      torch::Tensor spt = torch::zeros({3}, opts);

      for (int k = 0; k < degree_u; ++k) {
        torch::Tensor temp = torch::zeros({3}, opts);

        for (int l = 0; l < degree_v; ++l) {
          temp = temp + basis_v[j][l] * ctrlpts[idx_u[i] + k][idx_v[j] + l];
        }

        spt = spt + basis_u[i][k] * temp;
      }

      // eval_points[i * spans_v.size(0) + j] = spt;
      eval_points[i][j] = spt;
    }
  }

  return eval_points;
}

const torch::Tensor toEvalPoints(const torch::Tensor &spans_u,
                                 const torch::Tensor &spans_v,
                                 const torch::Tensor &basis_u,
                                 const torch::Tensor &basis_v,
                                 const torch::Tensor &ctrlpts) {
  // 优化版本：最大化并行计算，减少中间张量创建
  
  // 张量元数据
  const int sample_num_u = spans_u.size(0);
  const int sample_num_v = spans_v.size(0);
  const int degree_u = basis_u.size(1);
  const int degree_v = basis_v.size(1);
  const int H = ctrlpts.size(0);
  const int W = ctrlpts.size(1);
  const int dim = ctrlpts.size(2);
  const int elem_num = degree_u * degree_v;

  // 设备/类型配置
  const auto opts = ctrlpts.options();
  const auto idx_opts = torch::TensorOptions().dtype(torch::kInt64).device(ctrlpts.device());

  // 1. 计算基础索引 (使用更高效的方式)
  const auto idx_u = (spans_u - degree_u + 1).to(torch::kInt64);
  const auto idx_v = (spans_v - degree_v + 1).to(torch::kInt64);

  // 2. 创建局部偏移网格（可以预计算缓存）
  // 使用meshgrid更高效
  auto du_range = torch::arange(0, degree_u, idx_opts);
  auto dv_range = torch::arange(0, degree_v, idx_opts);
  auto grid = torch::meshgrid({du_range, dv_range}, "ij");
  const auto du_grid = grid[0].reshape({-1});
  const auto dv_grid = grid[1].reshape({-1});

  // 3. 使用broadcasting高效扩展索引 [sample_num_u, sample_num_v, elem_num]
  const auto idx_u_exp = idx_u.view({sample_num_u, 1, 1}) + du_grid.view({1, 1, elem_num});
  const auto idx_v_exp = idx_v.view({1, sample_num_v, 1}) + dv_grid.view({1, 1, elem_num});

  // 4. 一次性gather所有控制点（最关键的优化）
  const auto flat_idx = (idx_u_exp * W + idx_v_exp).reshape({-1});
  const auto flat_ctrlpts = ctrlpts.reshape({H * W, dim});
  const auto sampled_ctrlpts = torch::index_select(flat_ctrlpts, 0, flat_idx)
                                   .view({sample_num_u, sample_num_v, elem_num, dim});

  // 5. 高效计算权重外积（使用einsum或优化的broadcasting）
  // 方案A: 使用broadcasting (通常更快)
  const auto weights_u = basis_u.view({sample_num_u, 1, degree_u, 1});
  const auto weights_v = basis_v.view({1, sample_num_v, 1, degree_v});
  const auto weights = (weights_u * weights_v).reshape({sample_num_u, sample_num_v, elem_num, 1});

  // 6. 加权求和（使用sum而不是循环）
  return (sampled_ctrlpts * weights).sum(2);
}

const torch::Tensor toTorchPoints(const int &degree_u, const int &degree_v,
                                  const torch::Tensor &knots_u,
                                  const torch::Tensor &knots_v,
                                  const torch::Tensor &knotvector_u,
                                  const torch::Tensor &knotvector_v,
                                  const torch::Tensor &ctrlpts) {
  // 优化流程：所有中间步骤都是完全并行的
  
  // 1. 并行计算u和v方向的spans
  const auto spans_u = toSpans(degree_u, knotvector_u, ctrlpts.size(0), knots_u);
  const auto spans_v = toSpans(degree_v, knotvector_v, ctrlpts.size(1), knots_v);

  // 2. 并行计算u和v方向的基函数
  const auto basis_u = toBasisFunctions(degree_u, knotvector_u, spans_u, knots_u);
  const auto basis_v = toBasisFunctions(degree_v, knotvector_v, spans_v, knots_v);

  // 3. 并行计算所有采样点
  return toEvalPoints(spans_u, spans_v, basis_u, basis_v, ctrlpts);
}

const torch::Tensor toUVTorchPoints(const int &degree_u, const int &degree_v,
                                    const torch::Tensor &knotvector_u,
                                    const torch::Tensor &knotvector_v,
                                    const torch::Tensor &ctrlpts,
                                    const torch::Tensor &uv) {
  // 优化版本：最大化并行计算，减少不必要的tensor操作
  
  // 拆分UV坐标（使用select更高效）
  const auto u_coords = uv.select(1, 0).contiguous(); // (N,)
  const auto v_coords = uv.select(1, 1).contiguous(); // (N,)
  const int N = uv.size(0);
  const int H = ctrlpts.size(0);
  const int W = ctrlpts.size(1);
  const int dim = ctrlpts.size(2);
  const auto options = ctrlpts.options();
  const auto idx_opts = torch::TensorOptions().dtype(torch::kInt64).device(ctrlpts.device());

  // 1. 并行计算spans和basis
  const auto spans_u = toSpans(degree_u, knotvector_u, H, u_coords);
  const auto spans_v = toSpans(degree_v, knotvector_v, W, v_coords);
  const auto basis_u = toBasisFunctions(degree_u, knotvector_u, spans_u, u_coords);
  const auto basis_v = toBasisFunctions(degree_v, knotvector_v, spans_v, v_coords);

  // 2. 生成局部控制点索引偏移（可预计算缓存）
  const auto span_u_grid = torch::arange(-degree_u, 1, idx_opts);
  const auto span_v_grid = torch::arange(-degree_v, 1, idx_opts);

  // 3. 计算控制点索引（使用clamp保证边界安全）
  const auto span_u_indices = (spans_u.unsqueeze(1) + span_u_grid.unsqueeze(0))
                                  .clamp(0, H - 1);
  const auto span_v_indices = (spans_v.unsqueeze(1) + span_v_grid.unsqueeze(0))
                                  .clamp(0, W - 1);

  // 4. 使用meshgrid思想扩展索引 (N, degree_u+1, degree_v+1)
  const auto idx_u_exp = span_u_indices.unsqueeze(2).expand({N, degree_u + 1, degree_v + 1});
  const auto idx_v_exp = span_v_indices.unsqueeze(1).expand({N, degree_u + 1, degree_v + 1});

  // 5. 一次性gather所有控制点（关键优化）
  const auto flat_indices = (idx_u_exp * W + idx_v_exp).reshape({-1});
  const auto flat_ctrlpts = ctrlpts.reshape({H * W, dim});
  const auto local_ctrlpts = torch::index_select(flat_ctrlpts, 0, flat_indices)
                                 .view({N, degree_u + 1, degree_v + 1, dim});

  // 6. 使用einsum计算张量积（高效且简洁）
  // basis_u: (N, degree_u+1), basis_v: (N, degree_v+1), local_ctrlpts: (N, du+1, dv+1, dim)
  // 结果: (N, dim)
  return torch::einsum("ni,nj,nijc->nc", {basis_u, basis_v, local_ctrlpts});
}
