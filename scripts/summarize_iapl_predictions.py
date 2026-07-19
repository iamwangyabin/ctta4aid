#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        raise ValueError("Average precision requires at least one positive label")

    rows = sorted(zip(scores, labels), key=lambda row: row[0], reverse=True)
    true_positives = 0
    false_positives = 0
    result = 0.0
    start = 0
    while start < len(rows):
        score = rows[start][0]
        end = start
        group_positives = 0
        while end < len(rows) and rows[end][0] == score:
            group_positives += rows[end][1]
            end += 1
        true_positives += group_positives
        false_positives += (end - start) - group_positives
        result += (group_positives / positives) * (
            true_positives / (true_positives + false_positives)
        )
        start = end
    return result


def prediction_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = [int(label) for label in payload["labels"]]
    scores = [float(score) for score in payload["probabilities"]]
    indices = [int(index) for index in payload["indices"]]
    if not (len(labels) == len(scores) == len(indices)):
        raise ValueError(f"Prediction arrays have different lengths in {path}")
    if not labels or set(labels) != {0, 1}:
        raise ValueError(f"Prediction labels must contain both binary classes in {path}")

    predicted = [score > 0.5 for score in scores]
    correct = [prediction == bool(label) for prediction, label in zip(predicted, labels)]
    real_correct = [value for value, label in zip(correct, labels) if label == 0]
    fake_correct = [value for value, label in zip(correct, labels) if label == 1]
    return {
        "domain": str(payload["domain"]),
        "dataset_size": int(payload["dataset_size"]),
        "samples_with_distributed_padding": len(labels),
        "unique_sampler_indices": len(set(indices)),
        "duplicate_sampler_indices": len(indices) - len(set(indices)),
        "accuracy": fmean(correct),
        "average_precision": average_precision(labels, scores),
        "real_accuracy": fmean(real_correct),
        "fake_accuracy": fmean(fake_correct),
        "world_size": int(payload["world_size"]),
    }


def summarize(directory: Path) -> dict[str, Any]:
    by_domain = {
        path.stem: prediction_metrics(path)
        for path in sorted(directory.glob("*.json"))
    }
    if not by_domain:
        raise ValueError(f"No prediction JSON files found in {directory}")
    return {
        "by_domain": by_domain,
        "mean": {
            metric: fmean(item[metric] for item in by_domain.values())
            for metric in (
                "accuracy",
                "average_precision",
                "real_accuracy",
                "fake_accuracy",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = summarize(args.prediction_dir)
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
