#!/usr/bin/env python3
"""Batch CLIP retrieval benchmark for a ConceptGraphs object map."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np

from query_object_map import load_objects, most_common_label, object_geometry


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load CLIP once, rank every object for multiple text queries, and "
            "report Top-1, Top-k, and mean reciprocal rank when ground-truth "
            "object ids are supplied."
        )
    )
    parser.add_argument("--map-file", type=Path, required=True)
    parser.add_argument("--queries-file", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-model", default="ViT-H-14")
    parser.add_argument("--clip-pretrained", default="laion2b_s32b_b79k")
    return parser.parse_args()


def load_query_specs(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"queries file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data.get("queries") if isinstance(data, dict) else data
    if not isinstance(queries, list) or not queries:
        raise ValueError("queries file must contain a non-empty query list")

    normalized = []
    seen_ids = set()
    for index, item in enumerate(queries):
        if not isinstance(item, dict):
            raise ValueError(f"query {index} must be a JSON object")
        query_id = str(item.get("query_id", "")).strip()
        text = str(item.get("text", "")).strip()
        if not query_id or not text:
            raise ValueError(f"query {index} requires query_id and text")
        if query_id in seen_ids:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen_ids.add(query_id)

        expected_ids = item.get("expected_object_ids", [])
        if not isinstance(expected_ids, list):
            raise ValueError(
                f"expected_object_ids for {query_id!r} must be a list"
            )
        normalized.append(
            {
                "query_id": query_id,
                "text": text,
                "expected_object_ids": [int(value) for value in expected_ids],
                "notes": str(item.get("notes", "")),
            }
        )
    return normalized


def label_purity(obj: dict[str, Any]) -> float:
    labels = [str(label) for label in obj.get("class_name", [])]
    if not labels:
        return 0.0
    return Counter(labels).most_common(1)[0][1] / len(labels)


def source_object_id(obj: dict[str, Any], list_index: int) -> int:
    """Return the stable pre-curation id when the map provides one."""
    return int(obj.get("source_object_id", list_index))


def source_object_ids(obj: dict[str, Any], list_index: int) -> list[int]:
    """Return every original id represented by a possibly merged node."""
    values = obj.get("merged_source_object_ids")
    if values is None:
        return [source_object_id(obj, list_index)]
    normalized = sorted({int(value) for value in values})
    return normalized or [source_object_id(obj, list_index)]


def encode_similarities(
    texts: list[str],
    objects: list[dict[str, Any]],
    device: str,
    model_name: str,
    pretrained: str,
) -> np.ndarray:
    try:
        import open_clip
        import torch
        import torch.nn.functional as functional
    except ImportError as error:
        raise RuntimeError(
            "benchmarking requires torch and open_clip_torch; activate the "
            "conceptgraphs-official environment"
        ) from error

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained)
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(model_name)

    object_features = np.stack(
        [np.asarray(obj["clip_ft"], dtype=np.float32).reshape(-1) for obj in objects]
    )
    object_tensor = torch.from_numpy(object_features).to(device)

    with torch.inference_mode():
        query_tokens = tokenizer(texts).to(device)
        query_tensor = model.encode_text(query_tokens).float()
        query_tensor = functional.normalize(query_tensor, dim=-1)
        object_tensor = functional.normalize(object_tensor.float(), dim=-1)
        similarities = query_tensor @ object_tensor.T

    return similarities.detach().cpu().numpy().astype(float)


def match_record(
    rank: int,
    list_index: int,
    similarity: float,
    objects: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, Any]:
    obj = objects[list_index]
    confidences = np.asarray(obj.get("conf", []), dtype=np.float64)
    return {
        "rank": rank,
        "object_id": source_object_id(obj, list_index),
        "merged_source_object_ids": source_object_ids(obj, list_index),
        "map_list_index": list_index,
        "label": labels[list_index],
        "similarity": round(float(similarity), 6),
        "observations": int(obj.get("num_detections", 0)),
        "label_purity": round(label_purity(obj), 6),
        "mean_detection_confidence": (
            round(float(confidences.mean()), 6) if len(confidences) else None
        ),
        **object_geometry(obj),
    }


def main() -> None:
    args = parse_arguments()
    if args.top_k < 1:
        raise ValueError("--top-k must be positive")

    map_path = args.map_file.expanduser().resolve()
    queries_path = args.queries_file.expanduser().resolve()
    output_path = args.output_json.expanduser().resolve()
    objects = load_objects(map_path)
    labels = [most_common_label(obj) for obj in objects]
    source_ids = [source_object_id(obj, index) for index, obj in enumerate(objects)]
    all_source_ids = [
        source_id
        for index, obj in enumerate(objects)
        for source_id in source_object_ids(obj, index)
    ]
    if len(all_source_ids) != len(set(all_source_ids)):
        raise ValueError("object map contains duplicate source_object_id values")
    source_id_to_index = {
        source_id: index
        for index, obj in enumerate(objects)
        for source_id in source_object_ids(obj, index)
    }
    query_specs = load_query_specs(queries_path)

    similarities = encode_similarities(
        [spec["text"] for spec in query_specs],
        objects,
        args.device,
        args.clip_model,
        args.clip_pretrained,
    )

    query_results = []
    evaluated_results = []
    for query_index, spec in enumerate(query_specs):
        scores = similarities[query_index]
        ranked_indices = np.argsort(scores)[::-1].astype(int).tolist()
        list_index_to_rank = {
            list_index: rank
            for rank, list_index in enumerate(ranked_indices, start=1)
        }
        matches = [
            match_record(rank, list_index, scores[list_index], objects, labels)
            for rank, list_index in enumerate(
                ranked_indices[: args.top_k], start=1
            )
        ]

        expected_ids = spec["expected_object_ids"]
        expected_ranks = [
            {
                "object_id": object_id,
                "present_in_map": object_id in source_id_to_index,
                "rank": (
                    list_index_to_rank[source_id_to_index[object_id]]
                    if object_id in source_id_to_index
                    else None
                ),
            }
            for object_id in expected_ids
        ]
        present_expected_ranks = [
            item["rank"] for item in expected_ranks if item["rank"] is not None
        ]
        best_expected_rank = (
            min(present_expected_ranks)
            if present_expected_ranks
            else None
        )
        evaluation = None
        if expected_ids:
            evaluation = {
                "best_expected_rank": best_expected_rank,
                "missing_expected_object_ids": [
                    item["object_id"]
                    for item in expected_ranks
                    if not item["present_in_map"]
                ],
                "hit_at_1": best_expected_rank == 1,
                f"hit_at_{args.top_k}": (
                    best_expected_rank is not None
                    and best_expected_rank <= args.top_k
                ),
                "reciprocal_rank": (
                    round(1.0 / best_expected_rank, 6)
                    if best_expected_rank is not None
                    else 0.0
                ),
            }
            evaluated_results.append(evaluation)

        query_results.append(
            {
                **spec,
                "expected_ranks": expected_ranks,
                "evaluation": evaluation,
                "matches": matches,
            }
        )

    evaluated_count = len(evaluated_results)
    summary = {
        "query_count": len(query_results),
        "evaluated_query_count": evaluated_count,
        "top_k": args.top_k,
        "hit_at_1": (
            round(
                sum(result["hit_at_1"] for result in evaluated_results)
                / evaluated_count,
                6,
            )
            if evaluated_count
            else None
        ),
        f"hit_at_{args.top_k}": (
            round(
                sum(result[f"hit_at_{args.top_k}"] for result in evaluated_results)
                / evaluated_count,
                6,
            )
            if evaluated_count
            else None
        ),
        "mean_reciprocal_rank": (
            round(
                sum(result["reciprocal_rank"] for result in evaluated_results)
                / evaluated_count,
                6,
            )
            if evaluated_count
            else None
        ),
    }

    result = {
        "schema_version": 1,
        "map_file": str(map_path),
        "queries_file": str(queries_path),
        "object_count": len(objects),
        "clip_model": args.clip_model,
        "clip_pretrained": args.clip_pretrained,
        "quality_filter_applied": any(
            "source_object_id" in obj for obj in objects
        ),
        "summary": summary,
        "queries": query_results,
    }
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved benchmark to {output_path}")


if __name__ == "__main__":
    main()
