#!/usr/bin/env python3
"""Export source-frame contact sheets for ConceptGraphs object candidates."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from query_object_map import load_objects, most_common_label


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render stored masks, boxes, labels, and confidence values over "
            "the source RGB frames for selected object-map indices."
        )
    )
    parser.add_argument("--map-file", type=Path, required=True)
    parser.add_argument("--object-ids", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frames-per-object", type=int, default=6)
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--overlay-alpha", type=float, default=0.42)
    parser.add_argument(
        "--ids-are-source",
        action="store_true",
        help="Interpret --object-ids as stable source ids instead of list indices.",
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        default=0,
        help="Crop around each box and fit it to this square size; 0 keeps full frames.",
    )
    parser.add_argument(
        "--crop-padding-ratio",
        type=float,
        default=1.0,
        help="Padding around a cropped box as a fraction of its longest side.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    result = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return result or "unknown"


def selected_indices(length: int, maximum: int) -> list[int]:
    if length <= maximum:
        return list(range(length))
    return np.linspace(0, length - 1, maximum).round().astype(int).tolist()


def render_observation(
    obj: dict[str, Any],
    observation_index: int,
    object_id: int,
    alpha: float,
    crop_size: int,
    crop_padding_ratio: float,
) -> Image.Image:
    color_path = Path(obj["color_path"][observation_index])
    if not color_path.is_file():
        raise FileNotFoundError(f"source RGB frame does not exist: {color_path}")
    image = Image.open(color_path).convert("RGB")
    image_array = np.asarray(image).copy()
    mask = np.asarray(obj["mask"][observation_index], dtype=bool)
    if mask.shape != image_array.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image {image_array.shape[:2]}"
        )

    overlay_color = np.array([255, 55, 35], dtype=np.float64)
    image_array[mask] = np.clip(
        image_array[mask] * (1.0 - alpha) + overlay_color * alpha,
        0,
        255,
    ).astype(np.uint8)
    rendered = Image.fromarray(image_array)
    draw = ImageDraw.Draw(rendered)
    x1, y1, x2, y2 = [
        int(round(value)) for value in obj["xyxy"][observation_index]
    ]
    draw.rectangle((x1, y1, x2, y2), outline=(255, 230, 30), width=3)

    frame_id = int(obj["image_idx"][observation_index])
    label = str(obj["class_name"][observation_index])
    confidence = float(obj["conf"][observation_index])
    text = (
        f"object_{object_id:04d} | frame {frame_id:06d} | "
        f"{label} | conf {confidence:.3f}"
    )
    font = ImageFont.load_default()
    if crop_size > 0:
        longest_side = max(x2 - x1, y2 - y1, 1)
        padding = int(round(longest_side * crop_padding_ratio))
        crop_box = (
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(rendered.width, x2 + padding),
            min(rendered.height, y2 + padding),
        )
        cropped = rendered.crop(crop_box)
        header_height = 28
        available_height = max(1, crop_size - header_height)
        fitted = ImageOps.contain(
            cropped,
            (crop_size, available_height),
            method=Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGB", (crop_size, crop_size), "black")
        paste_x = (crop_size - fitted.width) // 2
        paste_y = header_height + (available_height - fitted.height) // 2
        canvas.paste(fitted, (paste_x, paste_y))
        crop_draw = ImageDraw.Draw(canvas)
        crop_draw.text((5, 8), text, fill=(255, 255, 255), font=font)
        return canvas

    text_box = draw.textbbox((0, 0), text, font=font)
    text_height = text_box[3] - text_box[1]
    draw.rectangle(
        (0, 0, min(rendered.width, text_box[2] + 10), text_height + 10),
        fill=(0, 0, 0),
    )
    draw.text((5, 5), text, fill=(255, 255, 255), font=font)
    return rendered


def make_contact_sheet(
    obj: dict[str, Any],
    object_id: int,
    frames_per_object: int,
    columns: int,
    alpha: float,
    crop_size: int,
    crop_padding_ratio: float,
) -> Image.Image:
    observation_indices = selected_indices(
        int(obj.get("num_detections", len(obj.get("image_idx", [])))),
        frames_per_object,
    )
    frames = [
        render_observation(
            obj,
            index,
            object_id,
            alpha,
            crop_size,
            crop_padding_ratio,
        )
        for index in observation_indices
    ]
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    rows = math.ceil(len(frames) / columns)
    header_height = 44
    sheet = Image.new("RGB", (columns * width, rows * height + header_height), "black")
    draw = ImageDraw.Draw(sheet)
    majority_label = most_common_label(obj)
    header = (
        f"object_{object_id:04d} | majority: {majority_label} | "
        f"observations: {int(obj.get('num_detections', 0))}"
    )
    draw.text((8, 14), header, fill=(255, 255, 255), font=ImageFont.load_default())
    for index, frame in enumerate(frames):
        x = (index % columns) * width
        y = header_height + (index // columns) * height
        sheet.paste(frame, (x, y))
    return sheet


def main() -> None:
    args = parse_arguments()
    if args.frames_per_object < 1 or args.columns < 1:
        raise ValueError("frame and column counts must be positive")
    if not 0.0 <= args.overlay_alpha <= 1.0:
        raise ValueError("--overlay-alpha must be in [0, 1]")
    if args.crop_size < 0:
        raise ValueError("--crop-size cannot be negative")
    if args.crop_size and args.crop_size < 128:
        raise ValueError("--crop-size must be 0 or at least 128")
    if args.crop_padding_ratio < 0:
        raise ValueError("--crop-padding-ratio cannot be negative")

    map_path = args.map_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    objects = load_objects(map_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_id_to_index = {
        int(source_id): list_index
        for list_index, obj in enumerate(objects)
        for source_id in obj.get(
            "merged_source_object_ids",
            [obj.get("source_object_id", list_index)],
        )
    }

    for requested_id in dict.fromkeys(args.object_ids):
        if args.ids_are_source:
            if requested_id not in source_id_to_index:
                raise ValueError(f"source object id {requested_id} is not in the map")
            list_index = source_id_to_index[requested_id]
            object_id = int(objects[list_index].get("source_object_id", requested_id))
        else:
            if not 0 <= requested_id < len(objects):
                raise ValueError(
                    f"object id {requested_id} must be between 0 and {len(objects) - 1}"
                )
            list_index = requested_id
            object_id = int(objects[list_index].get("source_object_id", requested_id))
        obj = objects[list_index]
        sheet = make_contact_sheet(
            obj,
            object_id,
            args.frames_per_object,
            args.columns,
            args.overlay_alpha,
            args.crop_size,
            args.crop_padding_ratio,
        )
        output_path = output_dir / (
            f"object_{object_id:04d}_{safe_name(most_common_label(obj))}.jpg"
        )
        sheet.save(output_path, quality=92, optimize=True)
        print(output_path)


if __name__ == "__main__":
    main()
