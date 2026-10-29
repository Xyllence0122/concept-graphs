#!/usr/bin/env python3
"""Convert ``rtabmap-export`` output into ConceptGraphs RealsenseDataset input.

Expected RTAB-Map export command::

    rtabmap-export --images --images_id --poses_camera --poses_format 11 \
      --output RUN_NAME --output_dir EXPORT_DIR database.db

The upstream ConceptGraphs ``RealsenseDataset`` applies ``P @ pose @ P.T``
while loading every ``poses/*.npy`` file, where ``P = diag(1, -1, -1, 1)``.
RTAB-Map already exports a camera-optical-to-map transform suitable for RGB-D
back-projection, so this converter stores the inverse-conjugated matrix.  The
upstream loader therefore reconstructs the original RTAB-Map transform.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


AXIS_FLIP = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass(frozen=True)
class PoseRecord:
    timestamp: float
    node_id: int
    translation_xyz: np.ndarray
    quaternion_xyzw: np.ndarray


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    camera_matrix: np.ndarray
    distortion: np.ndarray


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RGB, depth, calibration, and optimized camera poses "
            "exported from RTAB-Map into the directory layout used by the "
            "upstream ConceptGraphs RealsenseDataset."
        )
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        required=True,
        help="Directory passed to rtabmap-export --output_dir.",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Name passed to rtabmap-export --output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New scene directory to create. It must not already exist.",
    )
    parser.add_argument(
        "--max-depth-m",
        type=float,
        default=5.0,
        help="Depth values beyond this range are written as invalid zero.",
    )
    return parser.parse_args()


def parse_pose_file(path: Path) -> dict[int, PoseRecord]:
    records: dict[int, PoseRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 9:
                raise ValueError(
                    f"{path}:{line_number}: expected 9 fields from "
                    "--poses_format 11"
                )
            timestamp, x, y, z, qx, qy, qz, qw = map(float, fields[:8])
            node_id = int(fields[8])
            if node_id in records:
                raise ValueError(f"duplicate RTAB-Map node ID {node_id}")
            quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
            norm = float(np.linalg.norm(quaternion))
            if norm < 1.0e-12:
                raise ValueError(f"node {node_id} has a zero quaternion")
            records[node_id] = PoseRecord(
                timestamp=timestamp,
                node_id=node_id,
                translation_xyz=np.asarray([x, y, z], dtype=np.float64),
                quaternion_xyzw=quaternion / norm,
            )
    if not records:
        raise ValueError(f"no camera poses found in {path}")
    return records


def quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_to_world(record: PoseRecord) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_matrix(record.quaternion_xyzw)
    transform[:3, 3] = record.translation_xyz
    return transform


def read_calibration(path: Path) -> Calibration:
    # RTAB-Map writes OpenCV YAML 1.0 without the optional ``dt`` field.
    # OpenCV 4.6 FileStorage refuses to materialize such matrices, while the
    # rows/cols/data mapping is valid YAML after removing the legacy directive.
    yaml_text = "\n".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("%YAML:")
    )
    payload = yaml.safe_load(yaml_text)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid calibration document in {path}")

    def matrix(name: str) -> np.ndarray:
        value = payload.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"missing {name} in {path}")
        rows = int(value["rows"])
        cols = int(value["cols"])
        data = np.asarray(value["data"], dtype=np.float64)
        if data.size != rows * cols:
            raise ValueError(f"invalid {name} element count in {path}")
        return data.reshape(rows, cols)

    width = int(payload["image_width"])
    height = int(payload["image_height"])
    camera_matrix = matrix("camera_matrix")
    distortion = matrix("distortion_coefficients")
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"invalid camera matrix in {path}")
    return Calibration(
        width=width,
        height=height,
        camera_matrix=np.asarray(camera_matrix, dtype=np.float64),
        distortion=np.asarray(distortion, dtype=np.float64).reshape(-1),
    )


def numeric_ids(directory: Path, suffix: str) -> set[int]:
    result: set[int] = set()
    for path in directory.glob(f"*{suffix}"):
        try:
            result.add(int(path.stem))
        except ValueError as error:
            raise ValueError(f"non-numeric frame filename: {path.name}") from error
    return result


def require_matching_ids(
    rgb_ids: set[int],
    depth_ids: set[int],
    calibration_ids: set[int],
    pose_ids: set[int],
) -> list[int]:
    sets = {
        "rgb": rgb_ids,
        "depth": depth_ids,
        "calibration": calibration_ids,
        "pose": pose_ids,
    }
    all_ids = set().union(*sets.values())
    problems = []
    for name, ids in sets.items():
        missing = sorted(all_ids - ids)
        if missing:
            problems.append(f"{name} missing IDs {missing}")
    if problems:
        raise ValueError("frame ID mismatch: " + "; ".join(problems))
    if not all_ids:
        raise ValueError("the RTAB-Map export contains no frames")
    return sorted(all_ids)


def write_depth(source: Path, destination: Path, max_depth_mm: int) -> float:
    depth = np.asarray(Image.open(source))
    if depth.ndim != 2:
        raise ValueError(f"depth image must be single-channel: {source}")
    depth = depth.astype(np.uint16, copy=False)
    valid = (depth > 0) & (depth < 65535) & (depth <= max_depth_mm)
    cleaned = np.where(valid, depth, 0).astype(np.uint16)
    Image.fromarray(cleaned, mode="I;16").save(destination)
    return float(valid.mean())


def median_calibration(calibrations: list[Calibration]) -> Calibration:
    sizes = {(item.width, item.height) for item in calibrations}
    if len(sizes) != 1:
        raise ValueError(f"calibration image sizes changed across frames: {sizes}")
    matrices = np.stack([item.camera_matrix for item in calibrations])
    distortions = np.stack([item.distortion for item in calibrations])
    width, height = next(iter(sizes))
    return Calibration(
        width=width,
        height=height,
        camera_matrix=np.median(matrices, axis=0),
        distortion=np.median(distortions, axis=0),
    )


def main() -> None:
    args = parse_arguments()
    if args.max_depth_m <= 0.0:
        raise ValueError("--max-depth-m must be positive")
    export_dir = args.export_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a new path: {output_dir}"
        )

    rgb_source = export_dir / f"{args.run_name}_rgb"
    depth_source = export_dir / f"{args.run_name}_depth"
    calibration_source = export_dir / f"{args.run_name}_calib"
    pose_source = export_dir / f"{args.run_name}_camera_poses.txt"
    for path in (rgb_source, depth_source, calibration_source, pose_source):
        if not path.exists():
            raise FileNotFoundError(f"missing RTAB-Map export path: {path}")

    poses = parse_pose_file(pose_source)
    frame_ids = require_matching_ids(
        numeric_ids(rgb_source, ".jpg"),
        numeric_ids(depth_source, ".png"),
        numeric_ids(calibration_source, ".yaml"),
        set(poses),
    )
    calibrations = [
        read_calibration(calibration_source / f"{node_id}.yaml")
        for node_id in frame_ids
    ]
    calibration = median_calibration(calibrations)

    rgb_output = output_dir / "rgb"
    depth_output = output_dir / "depth"
    pose_output = output_dir / "poses"
    rgb_output.mkdir(parents=True)
    depth_output.mkdir()
    pose_output.mkdir()

    max_depth_mm = int(round(args.max_depth_m * 1000.0))
    manifest_frames = []
    valid_depth_ratios = []
    ros_translations = []
    for frame_index, node_id in enumerate(frame_ids):
        stem = f"{frame_index:06d}"
        shutil.copy2(rgb_source / f"{node_id}.jpg", rgb_output / f"{stem}.jpg")
        valid_ratio = write_depth(
            depth_source / f"{node_id}.png",
            depth_output / f"{stem}.png",
            max_depth_mm,
        )
        valid_depth_ratios.append(valid_ratio)

        ros_c2w = camera_to_world(poses[node_id])
        stored_pose = AXIS_FLIP @ ros_c2w @ AXIS_FLIP
        np.save(pose_output / f"{stem}.npy", stored_pose.astype(np.float32))
        ros_translations.append(ros_c2w[:3, 3])
        manifest_frames.append(
            {
                "frame_index": frame_index,
                "rtabmap_node_id": node_id,
                "timestamp": poses[node_id].timestamp,
                "rgb": f"rgb/{stem}.jpg",
                "depth": f"depth/{stem}.png",
                "pose": f"poses/{stem}.npy",
                "valid_depth_ratio": round(valid_ratio, 6),
            }
        )

    k = calibration.camera_matrix
    dataset_config = {
        "dataset_name": "realsense",
        "camera_params": {
            "image_height": calibration.height,
            "image_width": calibration.width,
            "fx": float(k[0, 0]),
            "fy": float(k[1, 1]),
            "cx": float(k[0, 2]),
            "cy": float(k[1, 2]),
            "png_depth_scale": 1000.0,
            "crop_edge": 0,
            "distortion": [float(value) for value in calibration.distortion],
        },
    }
    with (output_dir / "camera_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_config, handle, sort_keys=False)

    translations = np.stack(ros_translations)
    manifest = {
        "schema_version": 1,
        "source": (
            "rtabmap-export --images --images_id --poses_camera "
            "--poses_format 11"
        ),
        "run_name": args.run_name,
        "frame_count": len(frame_ids),
        "max_depth_m": args.max_depth_m,
        "depth_unit": "millimeter",
        "pose_input_convention": "RTAB-Map map_T_camera_optical",
        "pose_storage_convention": (
            "P @ map_T_camera_optical @ P.T, P=diag(1,-1,-1,1); "
            "upstream RealsenseDataset applies P @ pose @ P.T while loading"
        ),
        "median_valid_depth_ratio": round(
            float(np.median(valid_depth_ratios)), 6
        ),
        "translation_bounds_m": {
            "min": translations.min(axis=0).round(6).tolist(),
            "max": translations.max(axis=0).round(6).tolist(),
        },
        "frames": manifest_frames,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    print(f"Created ConceptGraphs scene: {output_dir}")
    print(f"Frames: {len(frame_ids)}")
    print(
        "Intrinsics: "
        f"fx={k[0, 0]:.6f}, fy={k[1, 1]:.6f}, "
        f"cx={k[0, 2]:.6f}, cy={k[1, 2]:.6f}"
    )
    print(
        "Median valid depth ratio after clipping: "
        f"{np.median(valid_depth_ratios):.4f}"
    )
    print(
        "Camera translation bounds (m): "
        f"min={translations.min(axis=0).round(4).tolist()} "
        f"max={translations.max(axis=0).round(4).tolist()}"
    )


if __name__ == "__main__":
    main()
