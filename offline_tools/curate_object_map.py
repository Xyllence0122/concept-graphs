#!/usr/bin/env python3
"""Non-destructively remove low-evidence fragments from a ConceptGraphs map."""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import json
import os
import pickle
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write a separate curated ConceptGraphs map while preserving the "
            "source object ids and a machine-readable removal report."
        )
    )
    parser.add_argument("--input-map", type=Path, required=True)
    parser.add_argument("--output-map", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    parser.add_argument("--min-observations", type=int, default=3)
    parser.add_argument("--min-points", type=int, default=50)
    parser.add_argument(
        "--min-label-purity",
        type=float,
        default=0.0,
        help=(
            "Optional hard threshold. Leave at 0 for broad open-vocabulary "
            "maps because correct objects can accumulate inconsistent labels."
        ),
    )
    return parser.parse_args()


def open_pickle(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb")


def label_summary(obj: dict[str, Any]) -> tuple[str, float]:
    labels = [str(value) for value in obj.get("class_name", [])]
    if not labels:
        return "unknown", 0.0
    label, count = Counter(labels).most_common(1)[0]
    return label, count / len(labels)


def object_summary(obj: dict[str, Any], source_id: int) -> dict[str, Any]:
    label, purity = label_summary(obj)
    points = np.asarray(obj.get("pcd_np", []))
    return {
        "source_object_id": source_id,
        "majority_label": label,
        "observations": int(obj.get("num_detections", 0)),
        "point_count": int(len(points)),
        "label_purity": round(float(purity), 6),
    }


def atomic_pickle_dump(data: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".pkl.gz" if output_path.suffix == ".gz" else ".pkl"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=suffix, dir=output_path.parent
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        if output_path.suffix == ".gz":
            with gzip.open(temporary_path, "wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with temporary_path.open("wb") as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(output_path)
        output_path.chmod(0o644)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def main() -> None:
    args = parse_arguments()
    if args.min_observations < 1:
        raise ValueError("--min-observations must be at least 1")
    if args.min_points < 1:
        raise ValueError("--min-points must be at least 1")
    if not 0.0 <= args.min_label_purity <= 1.0:
        raise ValueError("--min-label-purity must be between 0 and 1")

    input_path = args.input_map.expanduser().resolve()
    output_path = args.output_map.expanduser().resolve()
    report_path = args.report_json.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input map does not exist: {input_path}")
    if input_path == output_path:
        raise ValueError("--output-map must differ from --input-map")

    with open_pickle(input_path) as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
        raise ValueError("expected a map dictionary containing an objects list")

    retained = []
    retained_records = []
    rejected_records = []
    for list_index, original in enumerate(data["objects"]):
        if not isinstance(original, dict):
            raise ValueError(f"object at list index {list_index} is not a dictionary")
        source_id = int(original.get("source_object_id", list_index))
        summary = object_summary(original, source_id)
        reasons = []
        if summary["observations"] < args.min_observations:
            reasons.append("too_few_observations")
        if summary["point_count"] < args.min_points:
            reasons.append("too_few_points")
        if summary["label_purity"] < args.min_label_purity:
            reasons.append("low_label_purity")

        if reasons:
            rejected_records.append({**summary, "reasons": reasons})
            continue

        curated = original.copy()
        curated["source_object_id"] = source_id
        retained.append(curated)
        retained_records.append(summary)

    curated_data = copy.copy(data)
    curated_data["objects"] = retained
    curated_data["curation"] = {
        "schema_version": 1,
        "source_map": str(input_path),
        "thresholds": {
            "min_observations": args.min_observations,
            "min_points": args.min_points,
            "min_label_purity": args.min_label_purity,
        },
        "source_object_count": len(data["objects"]),
        "retained_object_count": len(retained),
        "rejected_object_count": len(rejected_records),
    }
    atomic_pickle_dump(curated_data, output_path)

    report = {
        "schema_version": 1,
        "input_map": str(input_path),
        "output_map": str(output_path),
        **curated_data["curation"],
        "retained_source_object_ids": [
            item["source_object_id"] for item in retained_records
        ],
        "rejected_objects": rejected_records,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(curated_data["curation"], indent=2, ensure_ascii=False))
    print(f"Saved curated map to {output_path}")
    print(f"Saved curation report to {report_path}")


if __name__ == "__main__":
    main()
