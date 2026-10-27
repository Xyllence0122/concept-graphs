"""Build deterministic spatial-relation edges from map-frame 3D tracks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np

from .tracker import ObjectTrack


@dataclass(frozen=True)
class SpatialRelationConfig:
    """Thresholds for deterministic map-frame spatial relations."""

    near_distance_m: float = 0.75
    max_pair_distance_m: float = 3.0
    horizontal_margin_m: float = 0.05
    vertical_margin_m: float = 0.05
    min_xy_overlap_ratio: float = 0.10

    def __post_init__(self) -> None:
        if self.near_distance_m < 0.0:
            raise ValueError("near_distance_m must be non-negative")
        if self.max_pair_distance_m <= 0.0:
            raise ValueError("max_pair_distance_m must be positive")
        if self.horizontal_margin_m < 0.0:
            raise ValueError("horizontal_margin_m must be non-negative")
        if self.vertical_margin_m < 0.0:
            raise ValueError("vertical_margin_m must be non-negative")
        if not 0.0 <= self.min_xy_overlap_ratio <= 1.0:
            raise ValueError("min_xy_overlap_ratio must be in [0, 1]")


def _node_id(track: ObjectTrack) -> str:
    return f"object_{track.node_id:04d}"


def _rounded(value: float) -> float:
    return round(float(value), 5)


def _aabb_gap(a: ObjectTrack, b: ObjectTrack) -> float:
    gap = np.maximum(
        np.maximum(a.bbox_min - b.bbox_max, b.bbox_min - a.bbox_max),
        0.0,
    )
    return float(np.linalg.norm(gap))


def _interval_overlap(
    first_min: float,
    first_max: float,
    second_min: float,
    second_max: float,
) -> float:
    return max(0.0, min(first_max, second_max) - max(first_min, second_min))


def _xy_overlap_ratio(a: ObjectTrack, b: ObjectTrack) -> float:
    overlap_x = _interval_overlap(
        float(a.bbox_min[0]),
        float(a.bbox_max[0]),
        float(b.bbox_min[0]),
        float(b.bbox_max[0]),
    )
    overlap_y = _interval_overlap(
        float(a.bbox_min[1]),
        float(a.bbox_max[1]),
        float(b.bbox_min[1]),
        float(b.bbox_max[1]),
    )
    overlap_area = overlap_x * overlap_y
    area_a = max(0.0, float(a.bbox_max[0] - a.bbox_min[0])) * max(
        0.0, float(a.bbox_max[1] - a.bbox_min[1])
    )
    area_b = max(0.0, float(b.bbox_max[0] - b.bbox_min[0])) * max(
        0.0, float(b.bbox_max[1] - b.bbox_min[1])
    )
    smaller_area = min(area_a, area_b)
    return overlap_area / smaller_area if smaller_area > 0.0 else 0.0


def _edge(
    source: ObjectTrack,
    target: ObjectTrack,
    relation: str,
    *,
    directed: bool,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "source_id": _node_id(source),
        "target_id": _node_id(target),
        "relation": relation,
        "directed": directed,
        "evidence": evidence,
    }


def build_spatial_edges(
    tracks: list[ObjectTrack],
    config: SpatialRelationConfig,
) -> list[dict[str, Any]]:
    """Build deterministic edges using AABBs expressed in one map frame.

    ``left_of`` and ``right_of`` follow the frame X axis. ``above`` and
    ``below`` follow the frame Z axis and additionally require overlap in the
    XY projection. ``near`` is undirected and uses the shortest AABB surface
    distance rather than centroid distance.
    """

    edges: list[dict[str, Any]] = []
    ordered_tracks = sorted(tracks, key=lambda track: track.node_id)
    for first, second in combinations(ordered_tracks, 2):
        centroid_distance = float(
            np.linalg.norm(first.centroid - second.centroid)
        )
        if centroid_distance > config.max_pair_distance_m:
            continue

        gap = _aabb_gap(first, second)
        common_evidence = {"centroid_distance_m": _rounded(centroid_distance)}
        if gap <= config.near_distance_m:
            edges.append(
                _edge(
                    first,
                    second,
                    "near",
                    directed=False,
                    evidence={
                        **common_evidence,
                        "aabb_gap_m": _rounded(gap),
                        "threshold_m": _rounded(config.near_distance_m),
                    },
                )
            )

        first_before_second = (
            first.bbox_max[0] + config.horizontal_margin_m
            <= second.bbox_min[0]
        )
        second_before_first = (
            second.bbox_max[0] + config.horizontal_margin_m
            <= first.bbox_min[0]
        )
        if first_before_second:
            left, right = first, second
        elif second_before_first:
            left, right = second, first
        else:
            left = right = None
        if left is not None and right is not None:
            axis_evidence = {
                **common_evidence,
                "axis": "x",
                "frame_convention": "increasing_x",
            }
            edges.append(
                _edge(
                    left,
                    right,
                    "left_of",
                    directed=True,
                    evidence=axis_evidence,
                )
            )
            edges.append(
                _edge(
                    right,
                    left,
                    "right_of",
                    directed=True,
                    evidence=axis_evidence,
                )
            )

        xy_overlap = _xy_overlap_ratio(first, second)
        if xy_overlap < config.min_xy_overlap_ratio:
            continue
        if first.bbox_min[2] >= second.bbox_max[2] + config.vertical_margin_m:
            upper, lower = first, second
        elif (
            second.bbox_min[2]
            >= first.bbox_max[2] + config.vertical_margin_m
        ):
            upper, lower = second, first
        else:
            upper = lower = None
        if upper is not None and lower is not None:
            vertical_evidence = {
                **common_evidence,
                "axis": "z",
                "frame_convention": "increasing_z",
                "xy_overlap_ratio": _rounded(xy_overlap),
            }
            edges.append(
                _edge(
                    upper,
                    lower,
                    "above",
                    directed=True,
                    evidence=vertical_evidence,
                )
            )
            edges.append(
                _edge(
                    lower,
                    upper,
                    "below",
                    directed=True,
                    evidence=vertical_evidence,
                )
            )

    return edges
