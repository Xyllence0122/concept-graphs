#!/usr/bin/env python3
"""Find possible duplicate ConceptGraphs nodes without modifying the map."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from query_object_map import load_objects


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Report pairs that are similar in CLIP space and overlap in 3D. "
            "The report is for source-frame review; it does not merge nodes."
        )
    )
    parser.add_argument("--map-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--min-clip-similarity", type=float, default=0.92)
    parser.add_argument("--min-aabb-containment", type=float, default=0.30)
    parser.add_argument("--point-distance", type=float, default=0.05)
    parser.add_argument("--min-symmetric-overlap", type=float, default=0.50)
    return parser.parse_args()


def majority_label(obj: dict[str, Any]) -> str:
    labels = [str(value) for value in obj.get("class_name", [])]
    return Counter(labels).most_common(1)[0][0] if labels else "unknown"


def normalized_clip_feature(obj: dict[str, Any]) -> np.ndarray:
    feature = np.asarray(obj["clip_ft"], dtype=np.float64).reshape(-1)
    norm = np.linalg.norm(feature)
    if norm == 0:
        raise ValueError("encountered an object with a zero CLIP feature")
    return feature / norm


def source_id(obj: dict[str, Any], list_index: int) -> int:
    return int(obj.get("source_object_id", list_index))


def rounded(value: float) -> float:
    return round(float(value), 6)


def main() -> None:
    args = parse_arguments()
    if not 0.0 <= args.min_clip_similarity <= 1.0:
        raise ValueError("--min-clip-similarity must be between 0 and 1")
    if not 0.0 <= args.min_aabb_containment <= 1.0:
        raise ValueError("--min-aabb-containment must be between 0 and 1")
    if args.point_distance <= 0:
        raise ValueError("--point-distance must be positive")
    if not 0.0 <= args.min_symmetric_overlap <= 1.0:
        raise ValueError("--min-symmetric-overlap must be between 0 and 1")

    map_path = args.map_file.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    objects = load_objects(map_path)
    points = [np.asarray(obj["pcd_np"], dtype=np.float64) for obj in objects]
    if any(array.ndim != 2 or array.shape[1] != 3 or not len(array) for array in points):
        raise ValueError("every object must contain a non-empty Nx3 pcd_np array")

    boxes = [(array.min(axis=0), array.max(axis=0)) for array in points]
    features = [normalized_clip_feature(obj) for obj in objects]
    candidates = []
    prefiltered_pair_count = 0

    for first_index in range(len(objects)):
        first_min, first_max = boxes[first_index]
        first_volume = float(np.prod(np.maximum(first_max - first_min, 0.0)))
        for second_index in range(first_index + 1, len(objects)):
            clip_similarity = float(features[first_index] @ features[second_index])
            if clip_similarity < args.min_clip_similarity:
                continue

            second_min, second_max = boxes[second_index]
            second_volume = float(
                np.prod(np.maximum(second_max - second_min, 0.0))
            )
            intersection_extent = np.maximum(
                0.0,
                np.minimum(first_max, second_max)
                - np.maximum(first_min, second_min),
            )
            intersection_volume = float(np.prod(intersection_extent))
            smaller_volume = min(first_volume, second_volume)
            aabb_containment = (
                intersection_volume / smaller_volume if smaller_volume > 0 else 0.0
            )
            if aabb_containment < args.min_aabb_containment:
                continue
            prefiltered_pair_count += 1

            first_close_fraction = float(
                np.mean(
                    cKDTree(points[second_index]).query(
                        points[first_index], k=1, workers=-1
                    )[0]
                    <= args.point_distance
                )
            )
            second_close_fraction = float(
                np.mean(
                    cKDTree(points[first_index]).query(
                        points[second_index], k=1, workers=-1
                    )[0]
                    <= args.point_distance
                )
            )
            symmetric_overlap = (
                2.0
                * first_close_fraction
                * second_close_fraction
                / (first_close_fraction + second_close_fraction)
                if first_close_fraction + second_close_fraction > 0
                else 0.0
            )
            if symmetric_overlap < args.min_symmetric_overlap:
                continue

            candidates.append(
                {
                    "first_object_id": source_id(
                        objects[first_index], first_index
                    ),
                    "first_map_list_index": first_index,
                    "first_label": majority_label(objects[first_index]),
                    "first_observations": int(
                        objects[first_index].get("num_detections", 0)
                    ),
                    "second_object_id": source_id(
                        objects[second_index], second_index
                    ),
                    "second_map_list_index": second_index,
                    "second_label": majority_label(objects[second_index]),
                    "second_observations": int(
                        objects[second_index].get("num_detections", 0)
                    ),
                    "clip_similarity": rounded(clip_similarity),
                    "aabb_containment": rounded(aabb_containment),
                    "first_close_fraction": rounded(first_close_fraction),
                    "second_close_fraction": rounded(second_close_fraction),
                    "symmetric_point_overlap": rounded(symmetric_overlap),
                    "review_status": "unreviewed",
                }
            )

    candidates.sort(
        key=lambda item: (
            item["symmetric_point_overlap"], item["clip_similarity"]
        ),
        reverse=True,
    )
    result = {
        "schema_version": 1,
        "map_file": str(map_path),
        "object_count": len(objects),
        "thresholds": {
            "min_clip_similarity": args.min_clip_similarity,
            "min_aabb_containment": args.min_aabb_containment,
            "point_distance_m": args.point_distance,
            "min_symmetric_overlap": args.min_symmetric_overlap,
        },
        "aabb_and_clip_prefilter_pair_count": prefiltered_pair_count,
        "candidate_pair_count": len(candidates),
        "candidates": candidates,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "object_count": len(objects),
                "aabb_and_clip_prefilter_pair_count": prefiltered_pair_count,
                "candidate_pair_count": len(candidates),
            },
            indent=2,
        )
    )
    print(f"Saved duplicate-candidate report to {output_path}")


if __name__ == "__main__":
    main()
