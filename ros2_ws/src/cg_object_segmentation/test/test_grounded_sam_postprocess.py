import numpy as np
import pytest

from cg_object_segmentation.grounded_sam_segmentation_node import (
    class_id_for_label,
    non_maximum_suppression,
    parse_text_prompt,
    select_best_sam_masks,
)


def test_parse_text_prompt_normalizes_dotted_labels() -> None:
    assert parse_text_prompt(" Box . eraser.\nchair ") == [
        "box",
        "eraser",
        "chair",
    ]


def test_parse_text_prompt_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        parse_text_prompt(" . ; \n")


def test_class_id_maps_decoded_phrase_to_query() -> None:
    assert class_id_for_label("an eraser", ["box", "eraser"]) == 1
    assert class_id_for_label("unknown", ["box", "eraser"]) == -1


def test_select_best_sam_masks_uses_highest_iou_candidate() -> None:
    masks = np.zeros((2, 3, 2, 2), dtype=np.float32)
    masks[0, 1, 0, 0] = 1.0
    masks[1, 2, 1, 1] = 1.0
    scores = np.array([[0.1, 0.9, 0.2], [0.3, 0.2, 0.8]])

    selected = select_best_sam_masks(masks, scores, threshold=0.0)

    assert selected.shape == (2, 2, 2)
    assert selected[0, 0, 0]
    assert selected[1, 1, 1]


def test_nms_removes_lower_score_overlapping_box() -> None:
    boxes = np.array(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [30, 30, 40, 40]]
    )
    scores = np.array([0.9, 0.7, 0.8])

    kept = non_maximum_suppression(boxes, scores, iou_threshold=0.5)

    assert kept.tolist() == [0, 2]
