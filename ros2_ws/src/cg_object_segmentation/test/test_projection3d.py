import numpy as np
import pytest

from cg_object_segmentation.projection3d import project_mask_to_3d


def test_center_pixel_projects_to_optical_axis() -> None:
    depth = np.full((5, 5), 2.0, dtype=np.float32)
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True
    camera_matrix = np.array(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]]
    )

    geometry = project_mask_to_3d(mask, depth, camera_matrix)

    assert geometry is not None
    np.testing.assert_allclose(geometry.centroid, [0.0, 0.0, 2.0])


def test_invalid_depth_is_ignored() -> None:
    depth = np.zeros((4, 4), dtype=np.float32)
    mask = np.ones((4, 4), dtype=bool)
    camera_matrix = np.eye(3)

    assert project_mask_to_3d(mask, depth, camera_matrix) is None


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError):
        project_mask_to_3d(np.ones((2, 2)), np.ones((3, 3)), np.eye(3))
