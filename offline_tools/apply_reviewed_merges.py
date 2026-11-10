#!/usr/bin/env python3
"""Apply only source-frame-reviewed object merges to a separate map file."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.utils import merge_obj2_into_obj1

from curate_object_map import atomic_pickle_dump


METADATA_KEYS = {
    "source_object_id",
    "merged_source_object_ids",
    "needs_relabel",
    "merge_review_notes",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply explicit merge decisions after source-frame review. "
            "Rejected pairs are recorded but never modified."
        )
    )
    parser.add_argument("--input-map", type=Path, required=True)
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return pickle.load(handle)


def original_ids(obj: dict[str, Any], list_index: int) -> set[int]:
    values = obj.get("merged_source_object_ids")
    if values is None:
        return {int(obj.get("source_object_id", list_index))}
    return {int(value) for value in values}


def load_decisions(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    decisions = data.get("decisions") if isinstance(data, dict) else None
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("review JSON must contain a non-empty decisions list")
    normalized = []
    seen_pairs = set()
    for index, item in enumerate(decisions):
        if not isinstance(item, dict):
            raise ValueError(f"decision {index} is not a JSON object")
        first = int(item["first_object_id"])
        second = int(item["second_object_id"])
        if first == second:
            raise ValueError(f"decision {index} refers to the same object twice")
        pair = tuple(sorted((first, second)))
        if pair in seen_pairs:
            raise ValueError(f"duplicate review pair: {pair}")
        seen_pairs.add(pair)
        action = str(item.get("action", "")).strip().lower()
        if action not in {"merge", "reject"}:
            raise ValueError(f"decision {index} has invalid action: {action!r}")
        normalized.append(
            {
                "first_object_id": first,
                "second_object_id": second,
                "action": action,
                "needs_relabel": bool(item.get("needs_relabel", False)),
                "notes": str(item.get("notes", "")),
            }
        )
    return normalized


def union_groups(ids: set[int], merge_decisions: list[dict[str, Any]]) -> list[set[int]]:
    parent = {value: value for value in ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for decision in merge_decisions:
        union(decision["first_object_id"], decision["second_object_id"])

    groups: dict[int, set[int]] = {}
    for value in ids:
        groups.setdefault(find(value), set()).add(value)
    return [group for group in groups.values() if len(group) > 1]


def main() -> None:
    args = parse_arguments()
    input_path = args.input_map.expanduser().resolve()
    review_path = args.review_json.expanduser().resolve()
    output_path = args.output_map.expanduser().resolve()
    report_path = args.report_json.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input map does not exist: {input_path}")
    if not review_path.is_file():
        raise FileNotFoundError(f"review JSON does not exist: {review_path}")
    if input_path == output_path:
        raise ValueError("--output-map must differ from --input-map")

    data = load_pickle(input_path)
    if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
        raise ValueError("expected a map dictionary containing an objects list")
    decisions = load_decisions(review_path)

    id_to_list_index: dict[int, int] = {}
    object_id_sets = []
    for list_index, obj in enumerate(data["objects"]):
        ids = original_ids(obj, list_index)
        object_id_sets.append(ids)
        for source_id in ids:
            if source_id in id_to_list_index:
                raise ValueError(f"source id {source_id} occurs in multiple nodes")
            id_to_list_index[source_id] = list_index

    referenced_ids = {
        decision[key]
        for decision in decisions
        for key in ("first_object_id", "second_object_id")
    }
    missing_ids = sorted(referenced_ids - set(id_to_list_index))
    if missing_ids:
        raise ValueError(f"review decisions reference missing ids: {missing_ids}")

    merge_decisions = [item for item in decisions if item["action"] == "merge"]
    groups = union_groups(referenced_ids, merge_decisions)
    for group in groups:
        indexes = {id_to_list_index[source_id] for source_id in group}
        if len(indexes) != len(group):
            raise ValueError(
                "a reviewed merge joins ids already represented by one node; "
                "use the pre-merge curated map as input"
            )

    serializable_objects = []
    for obj in data["objects"]:
        clean = {key: value for key, value in obj.items() if key not in METADATA_KEYS}
        serializable_objects.append(clean)
    map_objects = MapObjectList()
    map_objects.load_serializable(serializable_objects)

    cfg = copy.deepcopy(data["cfg"])
    cfg.device = args.device
    grouped_indexes: dict[int, set[int]] = {}
    group_metadata: dict[int, dict[str, Any]] = {}
    for group in groups:
        indexes = sorted(id_to_list_index[source_id] for source_id in group)
        representative_index = indexes[0]
        grouped_indexes[representative_index] = set(indexes)
        relevant = [
            item
            for item in merge_decisions
            if item["first_object_id"] in group
            and item["second_object_id"] in group
        ]
        group_metadata[representative_index] = {
            "source_ids": sorted(group),
            "needs_relabel": any(item["needs_relabel"] for item in relevant),
            "notes": [item["notes"] for item in relevant if item["notes"]],
        }

    removed_indexes = {
        index
        for representative, indexes in grouped_indexes.items()
        for index in indexes
        if index != representative
    }
    output_runtime_objects = MapObjectList()
    output_metadata = []
    for list_index, runtime_obj in enumerate(map_objects):
        if list_index in removed_indexes:
            continue
        if list_index in grouped_indexes:
            merged = runtime_obj
            for second_index in sorted(grouped_indexes[list_index] - {list_index}):
                merged = merge_obj2_into_obj1(
                    cfg, merged, map_objects[second_index], run_dbscan=False
                )
            output_runtime_objects.append(merged)
            output_metadata.append(group_metadata[list_index])
        else:
            output_runtime_objects.append(runtime_obj)
            ids = sorted(object_id_sets[list_index])
            output_metadata.append(
                {
                    "source_ids": ids,
                    "needs_relabel": bool(
                        data["objects"][list_index].get("needs_relabel", False)
                    ),
                    "notes": list(
                        data["objects"][list_index].get("merge_review_notes", [])
                    ),
                }
            )

    output_objects = output_runtime_objects.to_serializable()
    for obj, metadata in zip(output_objects, output_metadata):
        obj["source_object_id"] = min(metadata["source_ids"])
        obj["merged_source_object_ids"] = metadata["source_ids"]
        obj["needs_relabel"] = metadata["needs_relabel"]
        obj["merge_review_notes"] = metadata["notes"]

    output_data = copy.copy(data)
    output_data["objects"] = output_objects
    output_data["reviewed_merge"] = {
        "schema_version": 1,
        "source_map": str(input_path),
        "review_file": str(review_path),
        "input_object_count": len(data["objects"]),
        "output_object_count": len(output_objects),
        "merged_group_count": len(groups),
        "merged_node_reduction": len(data["objects"]) - len(output_objects),
        "rejected_pair_count": sum(
            item["action"] == "reject" for item in decisions
        ),
    }
    atomic_pickle_dump(output_data, output_path)

    report = {
        "schema_version": 1,
        "input_map": str(input_path),
        "output_map": str(output_path),
        **output_data["reviewed_merge"],
        "merged_groups": [
            {
                "source_object_ids": group_metadata[index]["source_ids"],
                "representative_object_id": min(
                    group_metadata[index]["source_ids"]
                ),
                "needs_relabel": group_metadata[index]["needs_relabel"],
                "notes": group_metadata[index]["notes"],
            }
            for index in sorted(group_metadata)
        ],
        "rejected_pairs": [
            item for item in decisions if item["action"] == "reject"
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_data["reviewed_merge"], indent=2))
    print(f"Saved reviewed-merge map to {output_path}")
    print(f"Saved reviewed-merge report to {report_path}")


if __name__ == "__main__":
    main()
