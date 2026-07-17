from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_prediction(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"domain", "indices", "labels", "probabilities", "world_size"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Prediction file {path} is missing: {', '.join(missing)}")
    lengths = {
        len(payload["indices"]),
        len(payload["labels"]),
        len(payload["probabilities"]),
    }
    if len(lengths) != 1:
        raise ValueError(f"Prediction arrays have different lengths in {path}")
    return payload


def _by_index(payload: dict[str, Any]) -> dict[int, tuple[int, float]]:
    labels: dict[int, int] = {}
    probabilities: dict[int, list[float]] = defaultdict(list)
    for raw_index, raw_label, raw_probability in zip(
        payload["indices"], payload["labels"], payload["probabilities"]
    ):
        index = int(raw_index)
        label = int(raw_label)
        if index in labels and labels[index] != label:
            raise ValueError(f"Sampler index {index} has inconsistent labels")
        labels[index] = label
        probabilities[index].append(float(raw_probability))
    return {
        index: (labels[index], float(np.mean(values)))
        for index, values in probabilities.items()
    }


def compare_predictions(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    sample_ids: list[str] | None = None,
) -> dict[str, Any]:
    if left["domain"] != right["domain"]:
        raise ValueError(
            f"Domain mismatch: {left['domain']!r} versus {right['domain']!r}"
        )
    left_by_index = _by_index(left)
    right_by_index = _by_index(right)
    common = sorted(left_by_index.keys() & right_by_index.keys())
    if not common:
        raise ValueError(f"No common sampler indices for domain {left['domain']}")

    left_probabilities = np.asarray(
        [left_by_index[index][1] for index in common], dtype=np.float64
    )
    right_probabilities = np.asarray(
        [right_by_index[index][1] for index in common], dtype=np.float64
    )
    left_labels = np.asarray(
        [left_by_index[index][0] for index in common], dtype=np.int64
    )
    right_labels = np.asarray(
        [right_by_index[index][0] for index in common], dtype=np.int64
    )
    if not np.array_equal(left_labels, right_labels):
        raise ValueError(f"Labels differ for common indices in {left['domain']}")

    absolute = np.abs(left_probabilities - right_probabilities)
    correlation = (
        float(np.corrcoef(left_probabilities, right_probabilities)[0, 1])
        if len(common) > 1
        and np.std(left_probabilities) > 0
        and np.std(right_probabilities) > 0
        else None
    )
    largest_positions = np.argsort(absolute)[-10:][::-1]
    largest = []
    for position in largest_positions:
        index = common[int(position)]
        row = {
            "index": index,
            "label": int(left_labels[position]),
            "left_probability": float(left_probabilities[position]),
            "right_probability": float(right_probabilities[position]),
            "absolute_delta": float(absolute[position]),
        }
        if sample_ids is not None and 0 <= index < len(sample_ids):
            row["sample_id"] = sample_ids[index]
        largest.append(row)

    return {
        "domain": left["domain"],
        "left_world_size": int(left["world_size"]),
        "right_world_size": int(right["world_size"]),
        "left_samples_with_padding": len(left["indices"]),
        "right_samples_with_padding": len(right["indices"]),
        "left_duplicate_indices": len(left["indices"]) - len(left_by_index),
        "right_duplicate_indices": len(right["indices"]) - len(right_by_index),
        "common_unique_indices": len(common),
        "index_sequence_equal": left["indices"] == right["indices"],
        "label_sequence_equal": left["labels"] == right["labels"],
        "mean_absolute_probability_delta": float(np.mean(absolute)),
        "max_absolute_probability_delta": float(np.max(absolute)),
        "probability_correlation": correlation,
        "threshold_disagreements": int(
            np.sum((left_probabilities > 0.5) != (right_probabilities > 0.5))
        ),
        "largest_probability_deltas": largest,
    }


def _manifest_ids(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        domain: [record["sample_id"] for record in item["records"]]
        for domain, item in payload["domains"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifests = _manifest_ids(args.manifest)
    left_paths = {path.stem: path for path in args.left.glob("*.json")}
    right_paths = {path.stem: path for path in args.right.glob("*.json")}
    domains = sorted(left_paths.keys() & right_paths.keys())
    if not domains:
        raise RuntimeError("Prediction directories have no common domains")
    report = {
        "left": str(args.left),
        "right": str(args.right),
        "domains": {
            domain: compare_predictions(
                load_prediction(left_paths[domain]),
                load_prediction(right_paths[domain]),
                sample_ids=manifests.get(domain),
            )
            for domain in domains
        },
    }
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
