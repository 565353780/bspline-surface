import torch
import numpy as np
import open3d as o3d
from typing import Union

from bspline_surface.Method.data import toNumpy


def renderGeometries(geometry_list, window_name="Geometry List"):
    if not isinstance(geometry_list, list):
        geometry_list = [geometry_list]

    o3d.visualization.draw_geometries(geometry_list, window_name)
    return True


def renderPoints(points: Union[np.ndarray, torch.Tensor], window_name="Points"):
    if isinstance(points, torch.Tensor):
        points_array = toNumpy(points)
    else:
        points_array = points

    points_array = points_array.reshape(-1, 3).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_array)
    return renderGeometries(pcd, window_name)
