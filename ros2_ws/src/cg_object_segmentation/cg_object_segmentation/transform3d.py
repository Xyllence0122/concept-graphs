from __future__ import annotations

import itertools

import numpy as np


def quaternion_xyzw_to_rotation_matrix(
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """Convert an xyzw quaternion to a 3x3 rotation matrix."""
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError("quaternion must contain exactly four values")

    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion norm must be non-zero")
    x, y, z, w = quaternion / norm

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def transform_points(
    points: np.ndarray,
    translation_xyz: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """Apply a rigid transform to one point or an Nx3 point array."""
    values = np.asarray(points, dtype=np.float64)
    if values.shape[-1:] != (3,):
        raise ValueError("points must end with an xyz dimension")

    translation = np.asarray(translation_xyz, dtype=np.float64)
    if translation.shape != (3,):
        raise ValueError("translation must contain exactly three values")

    rotation = quaternion_xyzw_to_rotation_matrix(quaternion_xyzw)
    return values @ rotation.T + translation


def compose_transforms(
    first_translation_xyz: np.ndarray,
    first_quaternion_xyzw: np.ndarray,
    second_translation_xyz: np.ndarray,
    second_quaternion_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose A<-B and B<-C rigid transforms into A<-C."""
    first_translation = np.asarray(first_translation_xyz, dtype=np.float64)
    second_translation = np.asarray(second_translation_xyz, dtype=np.float64)
    first_quaternion = np.asarray(first_quaternion_xyzw, dtype=np.float64)
    second_quaternion = np.asarray(second_quaternion_xyzw, dtype=np.float64)
    if first_translation.shape != (3,) or second_translation.shape != (3,):
        raise ValueError("translations must contain exactly three values")
    if first_quaternion.shape != (4,) or second_quaternion.shape != (4,):
        raise ValueError("quaternions must contain exactly four values")

    first_norm = float(np.linalg.norm(first_quaternion))
    second_norm = float(np.linalg.norm(second_quaternion))
    if first_norm <= np.finfo(np.float64).eps or second_norm <= np.finfo(np.float64).eps:
        raise ValueError("quaternion norm must be non-zero")
    x1, y1, z1, w1 = first_quaternion / first_norm
    x2, y2, z2, w2 = second_quaternion / second_norm
    composed_quaternion = np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )
    composed_quaternion /= np.linalg.norm(composed_quaternion)
    composed_translation = transform_points(
        second_translation,
        first_translation,
        first_quaternion,
    )
    return composed_translation, composed_quaternion


def transform_axis_aligned_bbox(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    translation_xyz: np.ndarray,
    quaternion_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform all eight corners and return their target-frame AABB."""
    minimum = np.asarray(bbox_min, dtype=np.float64)
    maximum = np.asarray(bbox_max, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("bbox bounds must contain exactly three values")
    if np.any(minimum > maximum):
        raise ValueError("bbox_min cannot be greater than bbox_max")

    corners = np.array(
        list(
            itertools.product(
                (minimum[0], maximum[0]),
                (minimum[1], maximum[1]),
                (minimum[2], maximum[2]),
            )
        ),
        dtype=np.float64,
    )
    transformed = transform_points(
        corners, translation_xyz, quaternion_xyzw
    )
    return transformed.min(axis=0), transformed.max(axis=0)
