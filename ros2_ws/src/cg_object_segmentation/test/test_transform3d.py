import math

import numpy as np

from cg_object_segmentation.transform3d import (
    compose_transforms,
    transform_axis_aligned_bbox,
    transform_points,
)


def test_compose_transforms_preserves_application_order() -> None:
    half_angle = math.pi / 4.0
    map_to_odom_quaternion = np.array(
        [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]
    )
    translation, quaternion = compose_transforms(
        np.array([10.0, 0.0, 0.0]),
        map_to_odom_quaternion,
        np.array([2.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.0]),
    )

    np.testing.assert_allclose(translation, [10.0, 2.0, 0.0], atol=1e-7)
    transformed = transform_points(
        np.array([1.0, 0.0, 0.0]), translation, quaternion
    )
    np.testing.assert_allclose(transformed, [10.0, 3.0, 0.0], atol=1e-7)


def test_transform_point_with_rotation_and_translation() -> None:
    half_angle = math.pi / 4.0
    quaternion = np.array([0.0, 0.0, math.sin(half_angle), math.cos(half_angle)])
    result = transform_points(
        np.array([1.0, 0.0, 0.0]),
        np.array([2.0, 3.0, 4.0]),
        quaternion,
    )

    np.testing.assert_allclose(result, [2.0, 4.0, 4.0], atol=1e-7)


def test_transform_bbox_recomputes_axis_aligned_bounds() -> None:
    half_angle = math.pi / 4.0
    quaternion = np.array([0.0, 0.0, math.sin(half_angle), math.cos(half_angle)])
    minimum, maximum = transform_axis_aligned_bbox(
        np.array([0.0, 0.0, 0.0]),
        np.array([2.0, 1.0, 1.0]),
        np.zeros(3),
        quaternion,
    )

    np.testing.assert_allclose(minimum, [-1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(maximum, [0.0, 2.0, 1.0], atol=1e-7)


def test_zero_quaternion_is_rejected() -> None:
    try:
        transform_points(np.zeros(3), np.zeros(3), np.zeros(4))
    except ValueError as error:
        assert "non-zero" in str(error)
    else:
        raise AssertionError("zero quaternion should have raised ValueError")
