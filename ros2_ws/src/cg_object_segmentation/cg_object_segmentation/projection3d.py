from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObjectGeometry:
    points: np.ndarray
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray


def project_mask_to_3d(
    mask: np.ndarray,
    depth_m: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    stride: int = 2,
    min_depth_m: float = 0.20,
    max_depth_m: float = 4.00,
    depth_quantile_low: float = 0.02,
    depth_quantile_high: float = 0.98,
) -> ObjectGeometry | None:
    """Project one binary instance mask into robust camera-frame geometry."""

    mask_array = np.asarray(mask, dtype=bool)
    depth_array = np.asarray(depth_m, dtype=np.float32)
    matrix = np.asarray(camera_matrix, dtype=np.float64)

    if mask_array.shape != depth_array.shape:
        raise ValueError("mask and depth image dimensions differ")
    if matrix.shape != (3, 3):
        raise ValueError("camera_matrix must have shape (3, 3)")
    if stride < 1:
        raise ValueError("stride must be at least one")
    if not 0.0 <= depth_quantile_low < depth_quantile_high <= 1.0:
        raise ValueError("invalid depth quantile range")

    rows, columns = np.nonzero(mask_array)
    if rows.size == 0:
        return None
    rows = rows[::stride]
    columns = columns[::stride]
    z = depth_array[rows, columns]

    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    rows, columns, z = rows[valid], columns[valid], z[valid]
    if z.size == 0:
        return None

    if z.size >= 20:
        low, high = np.quantile(z, [depth_quantile_low, depth_quantile_high])
        inlier = (z >= low) & (z <= high)
        rows, columns, z = rows[inlier], columns[inlier], z[inlier]
    if z.size == 0:
        return None

    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    x = (columns.astype(np.float32) - cx) * z / fx
    y = (rows.astype(np.float32) - cy) * z / fy
    points = np.column_stack((x, y, z)).astype(np.float32, copy=False)

    centroid = np.median(points, axis=0).astype(np.float32)
    bbox_min = np.quantile(points, 0.02, axis=0).astype(np.float32)
    bbox_max = np.quantile(points, 0.98, axis=0).astype(np.float32)
    return ObjectGeometry(points, centroid, bbox_min, bbox_max)
