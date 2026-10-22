import numpy as np

from cg_depth_projection.projection import project_depth_roi


def test_center_pixel_projects_to_optical_axis() -> None:
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[1, 1] = 2.0
    camera_matrix = np.array(
        [[100.0, 0.0, 1.0], [0.0, 100.0, 1.0], [0.0, 0.0, 1.0]]
    )

    points = project_depth_roi(depth, camera_matrix, (0, 0, 3, 3), stride=1)

    np.testing.assert_allclose(points, [[0.0, 0.0, 2.0]])


def test_range_filter_rejects_invalid_depth() -> None:
    depth = np.array([[0.0, np.nan, 0.1, 5.0, 1.0]], dtype=np.float32)
    camera_matrix = np.array(
        [[100.0, 0.0, 2.0], [0.0, 100.0, 0.0], [0.0, 0.0, 1.0]]
    )

    points = project_depth_roi(
        depth,
        camera_matrix,
        (0, 0, 5, 1),
        stride=1,
        min_depth_m=0.2,
        max_depth_m=4.0,
    )

    assert points.shape == (1, 3)
    assert points[0, 2] == 1.0
