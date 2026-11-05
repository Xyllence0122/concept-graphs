#!/usr/bin/env python3
"""Infer geometric ``subject --on--> support`` relations in an object map.

The support surface is estimated from the dominant plane of the candidate
support object's point cloud.  A subject is considered supported when its
lower points are close to that plane, most of it lies on the positive side of
the plane, and its projection overlaps the support surface footprint.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer an on/support relation from a ConceptGraphs map."
    )
    parser.add_argument("--map-file", type=Path, required=True)
    parser.add_argument("--subject-label", required=True)
    parser.add_argument("--support-label", required=True)
    parser.add_argument(
        "--subject-id",
        type=int,
        help=(
            "Use one verified object index as the subject instead of selecting "
            "all objects whose majority label matches --subject-label."
        ),
    )
    parser.add_argument(
        "--support-id",
        type=int,
        help=(
            "Use one verified object index as the support instead of selecting "
            "all objects whose majority label matches --support-label."
        ),
    )
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--plane-distance-threshold-m", type=float, default=0.02)
    parser.add_argument("--ransac-iterations", type=int, default=3000)
    parser.add_argument("--support-trim-percent", type=float, default=0.5)
    parser.add_argument("--min-plane-inlier-ratio", type=float, default=0.30)
    parser.add_argument("--min-footprint-overlap", type=float, default=0.50)
    parser.add_argument("--max-contact-gap-m", type=float, default=0.04)
    parser.add_argument("--max-penetration-m", type=float, default=0.03)
    parser.add_argument("--min-median-height-m", type=float, default=0.01)
    parser.add_argument("--max-median-height-m", type=float, default=0.50)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--require-match",
        action="store_true",
        help="Exit with status 2 if no relation satisfies all thresholds.",
    )
    return parser.parse_args()


def normalized_label(value: str) -> str:
    return " ".join(value.lower().split())


def most_common_label(obj: dict[str, Any]) -> str:
    labels = [str(label) for label in obj.get("class_name", [])]
    return Counter(labels).most_common(1)[0][0] if labels else "unknown"


def load_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"object map does not exist: {path}")
    with gzip.open(path, "rb") as handle:
        result = pickle.load(handle)
    objects = result.get("objects") if isinstance(result, dict) else result
    if not isinstance(objects, list) or not objects:
        raise ValueError("object map contains no objects")
    return objects


def object_points(obj: dict[str, Any], label: str) -> np.ndarray:
    points = np.asarray(obj.get("pcd_np"), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3:
        raise ValueError(f"{label!r} has an invalid point cloud")
    if not np.isfinite(points).all():
        points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        raise ValueError(f"{label!r} has fewer than three finite points")
    return points


def dominant_plane(
    points: np.ndarray,
    distance_threshold: float,
    iterations: int,
    random_seed: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "support inference requires Open3D; use the "
            "conceptgraphs-official environment"
        ) from error

    try:
        o3d.utility.random.seed(random_seed)
    except AttributeError:
        pass

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    model, _ = cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=iterations,
    )
    normal = np.asarray(model[:3], dtype=np.float64)
    scale = float(np.linalg.norm(normal))
    if scale <= 1e-12:
        raise ValueError("RANSAC returned a degenerate support plane")
    normal /= scale
    offset = float(model[3]) / scale
    inlier_mask = np.abs(points @ normal + offset) <= distance_threshold
    return normal, offset, inlier_mask


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(reference @ normal)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    axis_u = reference - normal * float(reference @ normal)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)
    return axis_u, axis_v


def footprint_overlap(
    support_points: np.ndarray,
    subject_points: np.ndarray,
    normal: np.ndarray,
    trim_percent: float,
) -> tuple[float, int]:
    try:
        from scipy.spatial import ConvexHull, Delaunay, QhullError
    except ImportError as error:
        raise RuntimeError(
            "support inference requires SciPy; use the "
            "conceptgraphs-official environment"
        ) from error

    origin = support_points.mean(axis=0)
    axis_u, axis_v = plane_basis(normal)
    support_2d = np.column_stack(
        ((support_points - origin) @ axis_u, (support_points - origin) @ axis_v)
    )
    subject_2d = np.column_stack(
        ((subject_points - origin) @ axis_u, (subject_points - origin) @ axis_v)
    )

    lower = np.percentile(support_2d, trim_percent, axis=0)
    upper = np.percentile(support_2d, 100.0 - trim_percent, axis=0)
    keep = np.all((support_2d >= lower) & (support_2d <= upper), axis=1)
    trimmed_support = support_2d[keep]
    if len(trimmed_support) < 3:
        raise ValueError("support footprint has fewer than three usable points")

    try:
        hull = ConvexHull(trimmed_support)
        triangulation = Delaunay(trimmed_support[hull.vertices])
    except QhullError as error:
        raise ValueError("support footprint is degenerate") from error

    inside = triangulation.find_simplex(subject_2d) >= 0
    return float(inside.mean()), int(inside.sum())


def evaluate_pair(
    subject_id: int,
    subject: dict[str, Any],
    support_id: int,
    support: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    subject_label = most_common_label(subject)
    support_label = most_common_label(support)
    subject_points = object_points(subject, subject_label)
    support_points = object_points(support, support_label)

    normal, offset, inlier_mask = dominant_plane(
        support_points,
        args.plane_distance_threshold_m,
        args.ransac_iterations,
        args.random_seed,
    )
    subject_distances = subject_points @ normal + offset
    if float(np.median(subject_distances)) < 0:
        normal = -normal
        offset = -offset
        subject_distances = -subject_distances

    support_distances = support_points @ normal + offset
    inlier_mask = np.abs(support_distances) <= args.plane_distance_threshold_m
    support_plane_points = support_points[inlier_mask]
    plane_inlier_ratio = float(inlier_mask.mean())
    overlap, inside_count = footprint_overlap(
        support_plane_points,
        subject_points,
        normal,
        args.support_trim_percent,
    )

    quantiles = np.percentile(subject_distances, [1, 5, 50, 95, 99])
    contact_gap = float(quantiles[1])
    median_height = float(quantiles[2])
    checks = {
        "support_plane": plane_inlier_ratio >= args.min_plane_inlier_ratio,
        "contact_gap": (
            -args.max_penetration_m <= contact_gap <= args.max_contact_gap_m
        ),
        "subject_above_plane": (
            args.min_median_height_m
            <= median_height
            <= args.max_median_height_m
        ),
        "footprint_overlap": overlap >= args.min_footprint_overlap,
    }
    accepted = all(checks.values())

    return {
        "source_id": f"object_{subject_id:04d}",
        "target_id": f"object_{support_id:04d}",
        "source_label": subject_label,
        "target_label": support_label,
        "relation": "on",
        "directed": True,
        "accepted": accepted,
        "checks": checks,
        "evidence": {
            "support_plane_normal": normal.round(6).tolist(),
            "support_plane_offset": round(offset, 6),
            "support_plane_inlier_ratio": round(plane_inlier_ratio, 6),
            "support_plane_point_count": int(len(support_plane_points)),
            "subject_point_count": int(len(subject_points)),
            "subject_points_inside_support": inside_count,
            "footprint_overlap_ratio": round(overlap, 6),
            "subject_plane_distance_m": {
                "q01": round(float(quantiles[0]), 6),
                "q05_contact_gap": round(contact_gap, 6),
                "q50_median_height": round(median_height, 6),
                "q95": round(float(quantiles[3]), 6),
                "q99": round(float(quantiles[4]), 6),
            },
            "thresholds": {
                "plane_distance_m": args.plane_distance_threshold_m,
                "min_plane_inlier_ratio": args.min_plane_inlier_ratio,
                "min_footprint_overlap": args.min_footprint_overlap,
                "max_contact_gap_m": args.max_contact_gap_m,
                "max_penetration_m": args.max_penetration_m,
                "min_median_height_m": args.min_median_height_m,
                "max_median_height_m": args.max_median_height_m,
            },
        },
    }


def main() -> None:
    args = parse_arguments()
    if args.plane_distance_threshold_m <= 0:
        raise ValueError("--plane-distance-threshold-m must be positive")
    if args.ransac_iterations < 1:
        raise ValueError("--ransac-iterations must be positive")
    if not 0 <= args.support_trim_percent < 50:
        raise ValueError("--support-trim-percent must be in [0, 50)")

    map_path = args.map_file.expanduser().resolve()
    objects = load_objects(map_path)
    labels = [most_common_label(obj) for obj in objects]
    subject_query = normalized_label(args.subject_label)
    support_query = normalized_label(args.support_label)
    if args.subject_id is not None:
        if not 0 <= args.subject_id < len(objects):
            raise ValueError(
                f"--subject-id must be between 0 and {len(objects) - 1}"
            )
        subject_ids = [args.subject_id]
    else:
        subject_ids = [
            index
            for index, label in enumerate(labels)
            if normalized_label(label) == subject_query
        ]

    if args.support_id is not None:
        if not 0 <= args.support_id < len(objects):
            raise ValueError(
                f"--support-id must be between 0 and {len(objects) - 1}"
            )
        support_ids = [args.support_id]
    else:
        support_ids = [
            index
            for index, label in enumerate(labels)
            if normalized_label(label) == support_query
        ]
    if not subject_ids:
        raise ValueError(f"no object has subject label {args.subject_label!r}")
    if not support_ids:
        raise ValueError(f"no object has support label {args.support_label!r}")

    candidates = [
        evaluate_pair(subject_id, objects[subject_id], support_id, objects[support_id], args)
        for subject_id in subject_ids
        for support_id in support_ids
        if subject_id != support_id
    ]
    for candidate in candidates:
        # Preserve the automatic map labels as provenance while emitting the
        # user-verified semantic labels requested on the command line.  This is
        # important when an otherwise valid support surface has a noisy
        # majority label in the automatically fused map.
        candidate["source_map_label"] = candidate["source_label"]
        candidate["target_map_label"] = candidate["target_label"]
        candidate["source_label"] = args.subject_label
        candidate["target_label"] = args.support_label

    relations = [candidate for candidate in candidates if candidate["accepted"]]
    result = {
        "schema_version": 1,
        "coordinate_frame": args.frame_id,
        "map_file": str(map_path),
        "query": {
            "subject_label": args.subject_label,
            "subject_id": args.subject_id,
            "relation": "on",
            "support_label": args.support_label,
            "support_id": args.support_id,
        },
        "candidate_count": len(candidates),
        "relation_count": len(relations),
        "relations": relations,
        "candidates": candidates,
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    print(serialized)

    if args.output_json is not None:
        output_path = args.output_json.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialized + "\n", encoding="utf-8")

    if args.require_match and not relations:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
