#!/usr/bin/env python3
"""Query a ConceptGraphs object map and emit a candidate navigation pose."""

from __future__ import annotations

import argparse
from collections import Counter
from difflib import SequenceMatcher
import gzip
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank objects in a ConceptGraphs map by label or CLIP similarity "
            "and compute a geometric candidate pose facing the best match."
        )
    )
    parser.add_argument("--map-file", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument(
        "--mode",
        choices=("clip", "label"),
        default="clip",
        help="CLIP is open-vocabulary; label mode is fast and model-free.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--robot-x", type=float, default=0.0)
    parser.add_argument("--robot-y", type=float, default=0.0)
    parser.add_argument("--standoff-m", type=float, default=0.30)
    parser.add_argument(
        "--clip-model",
        default="ViT-H-14",
    )
    parser.add_argument(
        "--clip-pretrained",
        default="laion2b_s32b_b79k",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def load_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"object map does not exist: {path}")
    try:
        with gzip.open(path, "rb") as handle:
            result = pickle.load(handle)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "the official map contains Hydra/OmegaConf objects; activate the "
            "conceptgraphs-official environment before running this tool"
        ) from error
    objects = result.get("objects") if isinstance(result, dict) else result
    if not isinstance(objects, list) or not objects:
        raise ValueError("object map contains no objects")
    return objects


def most_common_label(obj: dict[str, Any]) -> str:
    reviewed_label = str(obj.get("reviewed_label", "")).strip()
    if reviewed_label:
        return reviewed_label
    labels = [str(label) for label in obj.get("class_name", [])]
    if not labels:
        return "unknown"
    return Counter(labels).most_common(1)[0][0]


def source_object_ids(obj: dict[str, Any], list_index: int) -> list[int]:
    values = obj.get("merged_source_object_ids")
    if values is None:
        return [int(obj.get("source_object_id", list_index))]
    normalized = sorted({int(value) for value in values})
    return normalized or [int(obj.get("source_object_id", list_index))]


def label_similarity(query: str, label: str) -> float:
    query_normalized = " ".join(query.lower().split())
    label_normalized = " ".join(label.lower().split())
    if query_normalized == label_normalized:
        return 1.0
    query_tokens = set(query_normalized.split())
    label_tokens = set(label_normalized.split())
    union = query_tokens | label_tokens
    jaccard = len(query_tokens & label_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, query_normalized, label_normalized).ratio()
    containment = (
        0.95
        if query_normalized in label_normalized
        or label_normalized in query_normalized
        else 0.0
    )
    return float(max(jaccard, sequence, containment))


def clip_similarities(
    query: str,
    objects: list[dict[str, Any]],
    device: str,
    model_name: str,
    pretrained: str,
) -> list[float]:
    try:
        import open_clip
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError(
            "CLIP mode requires torch and open_clip_torch; activate the "
            "conceptgraphs-official environment"
        ) from error

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    model, _, _ = open_clip.create_model_and_transforms(
        model_name,
        pretrained,
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(model_name)

    object_features = np.stack(
        [np.asarray(obj["clip_ft"], dtype=np.float32).reshape(-1) for obj in objects]
    )
    object_features_tensor = torch.from_numpy(object_features).to(device)

    with torch.inference_mode():
        tokens = tokenizer([query]).to(device)
        query_feature = model.encode_text(tokens)
        query_feature = functional.normalize(query_feature.float(), dim=-1)
        object_features_tensor = functional.normalize(
            object_features_tensor.float(), dim=-1
        )
        similarities = object_features_tensor @ query_feature.squeeze(0)

    return similarities.detach().cpu().numpy().astype(float).tolist()


def object_geometry(obj: dict[str, Any]) -> dict[str, list[float] | int]:
    points = np.asarray(obj.get("pcd_np"), dtype=np.float64)
    bbox_points = np.asarray(obj.get("bbox_np"), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("object point cloud is missing or invalid")
    if bbox_points.ndim != 2 or bbox_points.shape[1] != 3:
        raise ValueError("object bounding box is missing or invalid")

    centroid = points.mean(axis=0)
    median = np.median(points, axis=0)
    bbox_center = bbox_points.mean(axis=0)
    bbox_extent = bbox_points.max(axis=0) - bbox_points.min(axis=0)
    return {
        "point_count": int(len(points)),
        "centroid_m": centroid.round(6).tolist(),
        "median_m": median.round(6).tolist(),
        "bbox_center_m": bbox_center.round(6).tolist(),
        "bbox_axis_aligned_extent_m": bbox_extent.round(6).tolist(),
    }


def candidate_pose(
    target_xyz: list[float],
    robot_x: float,
    robot_y: float,
    standoff_m: float,
    frame_id: str,
) -> dict[str, Any]:
    target_x, target_y = target_xyz[:2]
    delta_x = target_x - robot_x
    delta_y = target_y - robot_y
    distance = math.hypot(delta_x, delta_y)
    if distance > 1e-9:
        direction_x = delta_x / distance
        direction_y = delta_y / distance
        applied_standoff = min(standoff_m, distance)
        goal_x = target_x - applied_standoff * direction_x
        goal_y = target_y - applied_standoff * direction_y
        yaw = math.atan2(target_y - goal_y, target_x - goal_x)
    else:
        applied_standoff = 0.0
        goal_x = robot_x
        goal_y = robot_y
        yaw = 0.0

    return {
        "candidate_only": True,
        "requires_nav2_costmap_validation": True,
        "reference_robot_xy_m": [round(robot_x, 6), round(robot_y, 6)],
        "requested_standoff_m": round(standoff_m, 6),
        "applied_standoff_m": round(applied_standoff, 6),
        "target_distance_from_reference_m": round(distance, 6),
        "yaw_rad": round(yaw, 6),
        "pose_stamped": {
            "header": {"frame_id": frame_id},
            "pose": {
                "position": {
                    "x": round(goal_x, 6),
                    "y": round(goal_y, 6),
                    "z": 0.0,
                },
                "orientation": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": round(math.sin(yaw / 2.0), 6),
                    "w": round(math.cos(yaw / 2.0), 6),
                },
            },
        },
    }


def main() -> None:
    args = parse_arguments()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")
    if args.standoff_m < 0:
        raise ValueError("--standoff-m cannot be negative")

    map_path = args.map_file.expanduser().resolve()
    objects = load_objects(map_path)
    labels = [most_common_label(obj) for obj in objects]

    if args.mode == "clip":
        similarities = clip_similarities(
            args.query,
            objects,
            args.device,
            args.clip_model,
            args.clip_pretrained,
        )
    else:
        similarities = [label_similarity(args.query, label) for label in labels]

    ranked_indices = sorted(
        range(len(objects)),
        key=lambda index: similarities[index],
        reverse=True,
    )[: min(args.top_k, len(objects))]

    matches = []
    for rank, index in enumerate(ranked_indices, start=1):
        obj = objects[index]
        geometry = object_geometry(obj)
        confidences = np.asarray(obj.get("conf", []), dtype=np.float64)
        match = {
            "rank": rank,
            "object_id": int(obj.get("source_object_id", index)),
            "merged_source_object_ids": source_object_ids(obj, index),
            "map_list_index": index,
            "label": labels[index],
            "similarity": round(float(similarities[index]), 6),
            "observations": int(obj.get("num_detections", 0)),
            "mean_detection_confidence": (
                round(float(confidences.mean()), 6)
                if len(confidences)
                else None
            ),
            **geometry,
        }
        matches.append(match)

    best_match = matches[0]
    result = {
        "schema_version": 1,
        "query": args.query,
        "query_mode": args.mode,
        "map_file": str(map_path),
        "coordinate_frame": args.frame_id,
        "object_count": len(objects),
        "matches": matches,
        "best_match": best_match,
        "candidate_goal": candidate_pose(
            best_match["bbox_center_m"],
            args.robot_x,
            args.robot_y,
            args.standoff_m,
            args.frame_id,
        ),
    }

    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    print(serialized)
    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
