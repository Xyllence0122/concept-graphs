import pytest

from cg_object_segmentation.yolo_segmentation_node import resolve_inference_device


def test_auto_device_uses_cuda_when_available() -> None:
    assert resolve_inference_device("auto", True) == "0"


def test_auto_device_falls_back_to_cpu() -> None:
    assert resolve_inference_device("auto", False) == "cpu"


def test_explicit_device_is_preserved() -> None:
    assert resolve_inference_device("cpu", True) == "cpu"
    assert resolve_inference_device("1", True) == "1"


def test_empty_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        resolve_inference_device("  ", True)
