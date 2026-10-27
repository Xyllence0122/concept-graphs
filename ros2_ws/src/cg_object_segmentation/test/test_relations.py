import numpy as np
import pytest

from cg_object_segmentation.relations import (
    SpatialRelationConfig,
    build_spatial_edges,
)
from cg_object_segmentation.tracker import ObjectTrack


def track(
    node_id: int,
    center: tuple[float, float, float],
    half_size: tuple[float, float, float] = (0.1, 0.1, 0.1),
) -> ObjectTrack:
    centroid = np.asarray(center, dtype=np.float64)
    extent = np.asarray(half_size, dtype=np.float64)
    return ObjectTrack(
        node_id=node_id,
        label="object",
        centroid=centroid,
        bbox_min=centroid - extent,
        bbox_max=centroid + extent,
        confidence_mean=0.9,
        observations=5,
        missed_frames=0,
        source_instance_id=node_id,
    )


def relation_pairs(edges: list[dict]) -> set[tuple[str, str, str]]:
    return {
        (edge["source_id"], edge["relation"], edge["target_id"])
        for edge in edges
    }


def test_near_uses_aabb_surface_distance() -> None:
    config = SpatialRelationConfig(near_distance_m=0.5)
    edges = build_spatial_edges(
        [track(1, (0.0, 0.0, 0.0)), track(2, (0.6, 0.0, 0.0))],
        config,
    )

    near = next(edge for edge in edges if edge["relation"] == "near")
    assert near["directed"] is False
    assert near["evidence"]["aabb_gap_m"] == pytest.approx(0.4)


def test_x_separation_creates_inverse_directional_edges() -> None:
    config = SpatialRelationConfig(near_distance_m=0.1)
    edges = build_spatial_edges(
        [track(1, (0.0, 0.0, 0.0)), track(2, (1.0, 0.0, 0.0))],
        config,
    )

    pairs = relation_pairs(edges)
    assert ("object_0001", "left_of", "object_0002") in pairs
    assert ("object_0002", "right_of", "object_0001") in pairs


def test_vertical_separation_requires_xy_overlap() -> None:
    config = SpatialRelationConfig(near_distance_m=0.1)
    aligned = build_spatial_edges(
        [track(1, (0.0, 0.0, 1.0)), track(2, (0.0, 0.0, 0.0))],
        config,
    )
    separated = build_spatial_edges(
        [track(1, (1.0, 0.0, 1.0)), track(2, (0.0, 0.0, 0.0))],
        config,
    )

    aligned_pairs = relation_pairs(aligned)
    assert ("object_0001", "above", "object_0002") in aligned_pairs
    assert ("object_0002", "below", "object_0001") in aligned_pairs
    assert not {"above", "below"} & {
        edge["relation"] for edge in separated
    }


def test_max_pair_distance_limits_graph_density() -> None:
    config = SpatialRelationConfig(max_pair_distance_m=1.0)
    edges = build_spatial_edges(
        [track(1, (0.0, 0.0, 0.0)), track(2, (2.0, 0.0, 0.0))],
        config,
    )

    assert edges == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"near_distance_m": -0.1},
        {"max_pair_distance_m": 0.0},
        {"horizontal_margin_m": -0.1},
        {"vertical_margin_m": -0.1},
        {"min_xy_overlap_ratio": 1.1},
    ],
)
def test_invalid_thresholds_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        SpatialRelationConfig(**kwargs)
