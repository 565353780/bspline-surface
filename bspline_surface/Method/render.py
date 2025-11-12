import numpy as np
import open3d as o3d


def renderGeometries(geometry_list, window_name="Geometry List"):
    if not isinstance(geometry_list, list):
        geometry_list = [geometry_list]

    o3d.visualization.draw_geometries(geometry_list, window_name)
    return True


def renderPoints(points: np.ndarray, window_name="Points"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    return renderGeometries(pcd, window_name)
