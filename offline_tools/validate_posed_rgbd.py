#!/usr/bin/env python3
"""Build a lightweight fused point-cloud preview from a converted scene."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


AXIS_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--pixel-step", type=int, default=8)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--min-depth-m", type=float, default=0.25)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    return parser.parse_args()


def write_binary_ply(
    path: Path, points: np.ndarray, colors: np.ndarray
) -> None:
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    vertex = struct.Struct("<fffBBB")
    with path.open("wb") as handle:
        handle.write(header)
        for point, color in zip(points, colors, strict=True):
            handle.write(
                vertex.pack(
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    int(color[0]),
                    int(color[1]),
                    int(color[2]),
                )
            )


def rasterize_projection(
    points: np.ndarray,
    colors: np.ndarray,
    first_axis: int,
    second_axis: int,
    path: Path,
    meters_per_pixel: float = 0.01,
) -> dict[str, object]:
    projected = points[:, [first_axis, second_axis]]
    lower = np.quantile(projected, 0.002, axis=0)
    upper = np.quantile(projected, 0.998, axis=0)
    span = np.maximum(upper - lower, meters_per_pixel)
    width, height = np.ceil(span / meters_per_pixel).astype(int) + 1
    max_dimension = 1600
    scale = max(width / max_dimension, height / max_dimension, 1.0)
    effective_resolution = meters_per_pixel * scale
    width, height = np.ceil(span / effective_resolution).astype(int) + 1

    pixels = np.floor((projected - lower) / effective_resolution).astype(int)
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    pixels = pixels[inside]
    pixel_colors = colors[inside]
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    canvas[height - 1 - pixels[:, 1], pixels[:, 0]] = pixel_colors
    Image.fromarray(canvas).save(path)
    return {
        "axis_indices": [first_axis, second_axis],
        "lower_m": lower.round(6).tolist(),
        "upper_m": upper.round(6).tolist(),
        "meters_per_pixel": round(float(effective_resolution), 6),
        "image_size": [int(width), int(height)],
    }


def main() -> None:
    args = parse_arguments()
    if args.pixel_step < 1 or args.frame_step < 1:
        raise ValueError("pixel and frame steps must be positive")
    scene = args.scene_dir.resolve()
    with (scene / "camera_config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    camera = config["camera_params"]
    fx = float(camera["fx"])
    fy = float(camera["fy"])
    cx = float(camera["cx"])
    cy = float(camera["cy"])
    depth_scale = float(camera["png_depth_scale"])

    rgb_paths = sorted((scene / "rgb").glob("*.jpg"))
    depth_paths = sorted((scene / "depth").glob("*.png"))
    pose_paths = sorted((scene / "poses").glob("*.npy"))
    if not rgb_paths or not (
        len(rgb_paths) == len(depth_paths) == len(pose_paths)
    ):
        raise ValueError("RGB, depth, and pose counts must be equal and non-zero")

    all_points = []
    all_colors = []
    for index in range(0, len(rgb_paths), args.frame_step):
        rgb = np.asarray(Image.open(rgb_paths[index]).convert("RGB"))
        depth = np.asarray(Image.open(depth_paths[index]), dtype=np.float64)
        stored_pose = np.load(pose_paths[index]).astype(np.float64)
        pose = AXIS_FLIP @ stored_pose @ AXIS_FLIP

        rows = np.arange(0, depth.shape[0], args.pixel_step)
        columns = np.arange(0, depth.shape[1], args.pixel_step)
        u, v = np.meshgrid(columns, rows)
        z = depth[v, u] / depth_scale
        valid = (z >= args.min_depth_m) & (z <= args.max_depth_m)
        z = z[valid]
        u = u[valid]
        v = v[valid]
        camera_points = np.column_stack(
            ((u - cx) * z / fx, (v - cy) * z / fy, z)
        )
        world_points = (
            pose[:3, :3] @ camera_points.T
        ).T + pose[:3, 3]
        all_points.append(world_points.astype(np.float32))
        all_colors.append(rgb[v, u].astype(np.uint8))

    points = np.concatenate(all_points)
    colors = np.concatenate(all_colors)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    if len(points) == 0:
        raise ValueError("no valid 3D points were reconstructed")

    validation_dir = scene / "validation"
    validation_dir.mkdir(exist_ok=True)
    write_binary_ply(validation_dir / "fused_preview.ply", points, colors)
    topdown = rasterize_projection(
        points, colors, 0, 1, validation_dir / "topdown_xy.png"
    )
    side = rasterize_projection(
        points, colors, 0, 2, validation_dir / "side_xz.png"
    )
    report = {
        "frame_count": len(rgb_paths),
        "sampled_frame_count": len(range(0, len(rgb_paths), args.frame_step)),
        "point_count": int(len(points)),
        "pixel_step": args.pixel_step,
        "frame_step": args.frame_step,
        "bounds_m": {
            "min": points.min(axis=0).round(6).tolist(),
            "max": points.max(axis=0).round(6).tolist(),
        },
        "topdown_projection": topdown,
        "side_projection": side,
    }
    with (validation_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))
    print(f"Validation artifacts: {validation_dir}")


if __name__ == "__main__":
    main()
