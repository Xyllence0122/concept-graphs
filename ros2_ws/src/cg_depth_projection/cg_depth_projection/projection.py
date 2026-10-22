from __future__ import annotations

import numpy as np


def project_depth_roi(
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    roi: tuple[int, int, int, int],
    *,
    stride: int = 4,
    min_depth_m: float = 0.20,
    max_depth_m: float = 4.00,
) -> np.ndarray:
    """Project valid pixels in an image ROI to XYZ camera coordinates.

    Args:
        depth_m: HxW depth image in meters.
        camera_matrix: 3x3 pinhole camera intrinsic matrix.
        roi: (u_min, v_min, u_max, v_max), with exclusive maxima.
        stride: Pixel sampling step in both image axes.
        min_depth_m: Reject measurements at or below this range.
        max_depth_m: Reject measurements at or above this range.

    Returns:
        Nx3 float32 array in the optical camera frame: +X right, +Y down,
        +Z forward.
    """
    if depth_m.ndim != 2:
        raise ValueError("depth_m must be a 2-D array")
    if camera_matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3, 3)")
    if stride < 1:
        raise ValueError("stride must be at least 1")

    height, width = depth_m.shape
    u_min, v_min, u_max, v_max = roi
    u_min = max(0, min(width, int(u_min)))
    u_max = max(0, min(width, int(u_max)))
    v_min = max(0, min(height, int(v_min)))
    v_max = max(0, min(height, int(v_max)))
    if u_min >= u_max or v_min >= v_max:
        return np.empty((0, 3), dtype=np.float32)

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    vv, uu = np.mgrid[v_min:v_max:stride, u_min:u_max:stride]
    zz = depth_m[v_min:v_max:stride, u_min:u_max:stride]
    valid = np.isfinite(zz) & (zz > min_depth_m) & (zz < max_depth_m)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    z = zz[valid].astype(np.float32, copy=False)
    u = uu[valid].astype(np.float32, copy=False)
    v = vv[valid].astype(np.float32, copy=False)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.column_stack((x, y, z)).astype(np.float32, copy=False)
