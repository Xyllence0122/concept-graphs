import numpy as np
import pytest

from cg_object_segmentation.postprocess import compose_instance_map, xyxy_to_int_box


def test_compose_instance_map_reserves_zero_for_background() -> None:
    masks = np.zeros((2, 3, 4), dtype=bool)
    masks[0, 0, 0] = True
    masks[1, 2, 3] = True

    result = compose_instance_map(masks)

    assert result.dtype == np.uint16
    assert result[0, 0] == 1
    assert result[2, 3] == 2
    assert result[1, 1] == 0


def test_compose_instance_map_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        compose_instance_map(np.zeros((4, 5), dtype=bool))


def test_xyxy_to_int_box_clamps_to_image() -> None:
    assert xyxy_to_int_box(np.array([-3.2, 4.6, 120.1, 80.9]), 100, 60) == [
        0,
        5,
        99,
        59,
    ]
