#!/usr/bin/env python3
"""Export a curated ConceptGraphs object map as a portable semantic graph.

The official pipeline stores object geometry and embeddings in a compressed
pickle.  That representation is useful for retrieval, but it is not a stable
interface for ROS 2 or another computer.  This tool writes a JSON graph whose
node ids are based on ``source_object_id`` and whose edges come only from
accepted, evidence-backed relation files.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np


OBJECT_ID_PATTERN = re.compile(r"^object_(\d+)$")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a ConceptGraphs object map as a semantic graph JSON."
    )
    parser.add_argument("--map-file", type=Path, required=True)
    parser.add_argument(
        "--relations-file",
        type=Path,
        action="append",
        default=[],
        help="Accepted relation JSON. Repeat this option to combine files.",
    )
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--scene-id")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--allow-unresolved-relations",
        action="store_true",
        help="Skip an edge whose stored object id is absent from the map.",
    )
    return parser.parse_args()


def load_map(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"object map does not exist: {path}")
    try:
        with gzip.open(path, "rb") as handle:
            data = pickle.load(handle)
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "the official map contains Hydra/OmegaConf objects; activate the "
            "conceptgraphs-official environment before running this tool"
        ) from error

    objects = data.get("objects") if isinstance(data, dict) else data
    if not isinstance(objects, list) or not objects:
        raise ValueError("object map contains no foreground objects")
    metadata = data if isinstance(data, dict) else {}
    return metadata, objects


def source_ids(obj: dict[str, Any], list_index: int) -> list[int]:
    values = obj.get("merged_source_object_ids")
    if values is None:
        return [int(obj.get("source_object_id", list_index))]
    normalized = sorted({int(value) for value in values})
    return normalized or [int(obj.get("source_object_id", list_index))]


def label_summary(obj: dict[str, Any]) -> tuple[str, str, list[dict[str, Any]]]:
    labels = [str(value).strip() for value in obj.get("class_name", [])]
    labels = [value for value in labels if value]
    counts = Counter(labels)
    votes = [
        {"label": label, "count": count}
        for label, count in counts.most_common(5)
    ]
    reviewed = str(obj.get("reviewed_label", "")).strip()
    if reviewed:
        return reviewed, "manual_review", votes
    if votes:
        return str(votes[0]["label"]), "automatic_majority_vote", votes
    return "unknown", "missing", votes


def finite_points(value: Any, field_name: str, object_id: int) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            f"object_{object_id:04d} has invalid {field_name}; expected Nx3"
        )
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) == 0:
        raise ValueError(f"object_{object_id:04d} has no finite {field_name}")
    return points


def rounded_xyz(values: np.ndarray) -> dict[str, float]:
    return {
        "x": round(float(values[0]), 6),
        "y": round(float(values[1]), 6),
        "z": round(float(values[2]), 6),
    }


def make_node(obj: dict[str, Any], list_index: int) -> dict[str, Any]:
    represented_ids = source_ids(obj, list_index)
    primary_id = int(obj.get("source_object_id", min(represented_ids)))
    if primary_id not in represented_ids:
        represented_ids = sorted({primary_id, *represented_ids})

    points = finite_points(obj.get("pcd_np"), "point cloud", primary_id)
    bbox_value = obj.get("bbox_np")
    if bbox_value is None:
        bbox_points = points
    else:
        bbox_points = finite_points(bbox_value, "bounding box", primary_id)

    bbox_min = bbox_points.min(axis=0)
    bbox_max = bbox_points.max(axis=0)
    label, label_source, label_votes = label_summary(obj)
    confidences = np.asarray(obj.get("conf", []), dtype=np.float64).reshape(-1)
    confidences = confidences[np.isfinite(confidences)]

    node: dict[str, Any] = {
        "node_id": f"object_{primary_id:04d}",
        "source_object_id": primary_id,
        "merged_source_object_ids": represented_ids,
        "map_list_index": list_index,
        "label": label,
        "label_source": label_source,
        "automatic_label_votes": label_votes,
        "centroid_m": rounded_xyz(points.mean(axis=0)),
        "median_m": rounded_xyz(np.median(points, axis=0)),
        "bbox_3d_min_m": rounded_xyz(bbox_min),
        "bbox_3d_max_m": rounded_xyz(bbox_max),
        "bbox_3d_extent_m": rounded_xyz(bbox_max - bbox_min),
        "point_count": int(len(points)),
        "observations": int(obj.get("num_detections", 0)),
        "mean_detection_confidence": (
            round(float(confidences.mean()), 6) if len(confidences) else None
        ),
        "needs_relabel": bool(obj.get("needs_relabel", False)),
    }
    if label_source == "manual_review":
        node["review"] = {
            "confidence": obj.get("reviewed_label_confidence"),
            "notes": obj.get("reviewed_label_notes"),
        }
    return node


def relation_object_number(value: Any) -> int:
    if isinstance(value, int):
        return value
    match = OBJECT_ID_PATTERN.fullmatch(str(value))
    if match is None:
        raise ValueError(f"invalid relation object id: {value!r}")
    return int(match.group(1))


def load_relation_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"relations file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("relations", data) if isinstance(data, dict) else data
    if not isinstance(records, list):
        raise ValueError(f"relations file has no relation list: {path}")
    return [record for record in records if record.get("accepted", True)]


def make_edges(
    relation_paths: list[Path],
    alias_to_node: dict[int, str],
    allow_unresolved: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    edges: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, bool]] = set()

    for path in relation_paths:
        for record in load_relation_records(path):
            source_number = relation_object_number(record.get("source_id"))
            target_number = relation_object_number(record.get("target_id"))
            source_id = alias_to_node.get(source_number)
            target_id = alias_to_node.get(target_number)
            if source_id is None or target_id is None:
                detail = {
                    "relations_file": str(path),
                    "source_id": record.get("source_id"),
                    "target_id": record.get("target_id"),
                    "reason": "relation endpoint is absent from the curated map",
                }
                if allow_unresolved:
                    skipped.append(detail)
                    continue
                raise ValueError(json.dumps(detail, ensure_ascii=False))

            relation = str(record.get("relation", "unknown"))
            directed = bool(record.get("directed", True))
            key = (source_id, target_id, relation, directed)
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "edge_id": f"edge_{len(edges) + 1:04d}",
                    "source_id": source_id,
                    "target_id": target_id,
                    "relation": relation,
                    "directed": directed,
                    "source_label": record.get("source_label"),
                    "target_label": record.get("target_label"),
                    "evidence": record.get("evidence", {}),
                    "provenance": {
                        "relations_file": str(path),
                        "stored_source_id": record.get("source_id"),
                        "stored_target_id": record.get("target_id"),
                    },
                }
            )
    return edges, skipped


def main() -> None:
    args = parse_arguments()
    map_path = args.map_file.expanduser().resolve()
    relation_paths = [path.expanduser().resolve() for path in args.relations_file]
    output_path = args.output_json.expanduser().resolve()

    metadata, objects = load_map(map_path)
    nodes = [make_node(obj, index) for index, obj in enumerate(objects)]
    node_ids = [node["node_id"] for node in nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("map contains duplicate primary source_object_id values")

    alias_to_node: dict[int, str] = {}
    for node in nodes:
        for source_id in node["merged_source_object_ids"]:
            existing = alias_to_node.setdefault(int(source_id), node["node_id"])
            if existing != node["node_id"]:
                raise ValueError(
                    f"source object id {source_id} belongs to multiple nodes"
                )

    edges, skipped = make_edges(
        relation_paths,
        alias_to_node,
        args.allow_unresolved_relations,
    )
    label_counts = Counter(node["label"] for node in nodes)
    reviewed_count = sum(node["label_source"] == "manual_review" for node in nodes)
    result = {
        "schema_version": 2,
        "graph_type": "static_open_vocabulary_semantic_map",
        "coordinate_frame": args.frame_id,
        "scene_id": args.scene_id or map_path.parent.parent.name,
        "source_map_file": str(map_path),
        "relation_files": [str(path) for path in relation_paths],
        "summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "unique_label_count": len(label_counts),
            "manually_reviewed_node_count": reviewed_count,
            "nodes_needing_relabel": sum(node["needs_relabel"] for node in nodes),
            "skipped_relation_count": len(skipped),
            "top_labels": [
                {"label": label, "node_count": count}
                for label, count in label_counts.most_common(20)
            ],
        },
        "map_provenance": {
            "has_curation_metadata": "curation" in metadata,
            "has_reviewed_merge_metadata": "reviewed_merge" in metadata,
            "has_reviewed_label_metadata": "reviewed_labels" in metadata,
        },
        "nodes": nodes,
        "edges": edges,
        "skipped_relations": skipped,
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    print(f"semantic graph: {output_path}")


if __name__ == "__main__":
    main()
