import torch
import numpy as np

from bspline_surface.Model.bspline_surface import BSplineSurface

def demo():
    degree_u = 3
    degree_v = 3
    size_u = 10
    size_v = 10
    sample_num_u = 10
    sample_num_v = 10
    dtype = torch.float32
    device = 'cpu'

    bspline_surface = BSplineSurface(
        degree_u,
        degree_v,
        size_u,
        size_v,
        sample_num_u,
        sample_num_v,
        dtype,
        device,
    )

    ctrlpts = np.zeros(
        [
            bspline_surface.size_u - 1,
            bspline_surface.size_v - 1,
            3,
        ],
        dtype=float,
    )

    u_values = (
        np.arange(bspline_surface.size_u - 1)
        / (bspline_surface.size_u - 2)
    ) - 0.5

    v_values = (
        np.arange(bspline_surface.size_v - 1)
        / (bspline_surface.size_v - 2)
    ) - 0.5

    surf_length = 1.0
    for i in range(bspline_surface.size_u - 1):
        ctrlpts[:, i, 0] = surf_length * u_values
    for i in range(bspline_surface.size_v - 1):
        ctrlpts[i, :, 1] = surf_length * v_values

    bspline_surface.loadParams(ctrlpts=ctrlpts)

    bspline_surface.renderSamplePoints()
    return True
