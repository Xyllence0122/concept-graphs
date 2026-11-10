#!/usr/bin/env python3
"""Attach source-frame-reviewed semantic labels to a separate object map."""

from __future__ import annotations

from collections import Counter
import argparse
import copy
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

from curate_object_map import atomic_pickle_dump


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach reviewed labels while preserving the automatic label "
            "votes and CLIP features. The input map is never overwritten."
        )
    )
    parser.add_argument("--input-map", type=Path, required=True)
    parser.add_argument("--labels-json", type=Path, required=True)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return pickle.load(handle)


def load_labels(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels = data.get("labels") if isinstance(data, dict) else None
    if not isinstance(labels, list) or not labels:
        raise ValueError("labels JSON must contain a non-empty labels list")
    normalized = []
    seen_ids = set()
    allowed_confidences = {"low", "medium", "high"}
    for index, item in enumerate(labels):
        if not isinstance(item, dict):
            raise ValueError(f"label entry {index} is not a JSON object")
        source_id = int(item["source_object_id"])
        reviewed_label = str(item.get("reviewed_label", "")).strip()
        confidence = str(item.get("confidence", "")).strip().lower()
        if source_id in seen_ids:
            raise ValueError(f"duplicate reviewed source id: {source_id}")
        if not reviewed_label:
            raise ValueError(f"label entry {index} has an empty reviewed_label")
        if confidence not in allowed_confidences:
            raise ValueError(
                f"label entry {index} confidence must be low, medium, or high"
            )
        seen_ids.add(source_id)
        normalized.append(
            {
                "source_object_id": source_id,
                "reviewed_label": reviewed_label,
                "confidence": confidence,
                "notes": str(item.get("notes", "")),
            }
        )
    return normalized


def represented_ids(obj: dict[str, Any], list_index: int) -> set[int]:
    values = obj.get("merged_source_object_ids")
    if values is None:
        return {int(obj.get("source_object_id", list_index))}
    return {int(value) for value in values}


def main() -> None:
    args = parse_arguments()
    input_path = args.input_map.expanduser().resolve()
    labels_path = args.labels_json.expanduser().resolve()
    output_path = args.output_map.expanduser().resolve()
    report_path = args.report_json.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input map does not exist: {input_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels JSON does not exist: {labels_path}")
    if input_path == output_path:
        raise ValueError("--output-map must differ from --input-map")

    data = load_pickle(input_path)
    if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
        raise ValueError("expected a map dictionary containing an objects list")
    reviewed_labels = load_labels(labels_path)

    id_to_index: dict[int, int] = {}
    for list_index, obj in enumerate(data["objects"]):
        for source_id in represented_ids(obj, list_index):
            if source_id in id_to_index:
                raise ValueError(f"source id {source_id} occurs in multiple nodes")
            id_to_index[source_id] = list_index

    missing_ids = sorted(
        item["source_object_id"]
        for item in reviewed_labels
        if item["source_object_id"] not in id_to_index
    )
    if missing_ids:
        raise ValueError(f"reviewed labels reference missing ids: {missing_ids}")

    output_data = copy.copy(data)
    output_data["objects"] = list(data["objects"])
    applied = []
    for item in reviewed_labels:
        list_index = id_to_index[item["source_object_id"]]
        original = data["objects"][list_index]
        updated = original.copy()
        automatic_votes = Counter(
            str(value) for value in original.get("class_name", [])
        )
        updated["reviewed_label"] = item["reviewed_label"]
        updated["reviewed_label_confidence"] = item["confidence"]
        updated["reviewed_label_notes"] = item["notes"]
        updated["automatic_label_votes"] = dict(automatic_votes.most_common())
        updated["needs_relabel"] = False
        output_data["objects"][list_index] = updated
        applied.append(
            {
                **item,
                "map_list_index": list_index,
                "representative_object_id": int(
                    updated.get("source_object_id", list_index)
                ),
                "represented_source_object_ids": sorted(
                    represented_ids(updated, list_index)
                ),
                "automatic_label_votes": dict(automatic_votes.most_common()),
            }
        )

    output_data["reviewed_labels"] = {
        "schema_version": 1,
        "source_map": str(input_path),
        "labels_file": str(labels_path),
        "applied_label_count": len(applied),
    }
    atomic_pickle_dump(output_data, output_path)

    report = {
        "schema_version": 1,
        "input_map": str(input_path),
        "output_map": str(output_path),
        **output_data["reviewed_labels"],
        "applied_labels": applied,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_data["reviewed_labels"], indent=2))
    print(f"Saved reviewed-label map to {output_path}")
    print(f"Saved reviewed-label report to {report_path}")


if __name__ == "__main__":
    main()
