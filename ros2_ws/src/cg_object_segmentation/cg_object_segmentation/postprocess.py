from __future__ import annotations

import numpy as np


def compose_instance_map(masks: np.ndarray) -> np.ndarray:
    """Convert N boolean masks into a uint16 image with IDs 1..N.

    Later masks overwrite earlier masks in overlapping pixels. ID 0 is
    reserved for background.
    """

    masks_array = np.asarray(masks)
    if masks_array.ndim != 3:
        raise ValueError("masks must have shape (N, H, W)")
    if masks_array.shape[0] >= np.iinfo(np.uint16).max:
        raise ValueError("too many masks for a uint16 instance map")

    instance_map = np.zeros(masks_array.shape[1:], dtype=np.uint16)
    for index, mask in enumerate(masks_array, start=1):
        instance_map[np.asarray(mask, dtype=bool)] = index
    return instance_map


def xyxy_to_int_box(
    box: np.ndarray, image_width: int, image_height: int
) -> list[int]:
    """Round and clamp an xyxy box to valid image coordinates."""

    values = np.rint(np.asarray(box, dtype=np.float64)).astype(np.int64)
    if values.shape != (4,):
        raise ValueError("box must contain exactly four values")

    x1 = int(np.clip(values[0], 0, image_width - 1))
    y1 = int(np.clip(values[1], 0, image_height - 1))
    x2 = int(np.clip(values[2], x1, image_width - 1))
    y2 = int(np.clip(values[3], y1, image_height - 1))
    return [x1, y1, x2, y2]
