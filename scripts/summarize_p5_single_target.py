#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


HOST_BY_SEED = {0: "A6000", 1: "3090", 2: "4090-2"}
METRICS = ("auc", "accuracy", "balanced_accuracy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def distribution(values: list[float]) -> dict[str, Any]:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("Cannot summarize empty or non-finite values")
    return {
        "mean": mean(values),
        "population_std": pstdev(values),
        "values_by_seed": values,
    }


def efficiency_summary(method_summary: dict[str, Any], targets: list[str]) -> dict[str, Any]:
    efficiency = [method_summary[target]["efficiency"] for target in targets]
    total_batches = sum(int(item["batches"]) for item in efficiency)
    trainable_parameters = {int(item["trainable_parameters"]) for item in efficiency}
    if len(trainable_parameters) != 1:
        raise ValueError("Trainable parameter count changed between target domains")

    def weighted(field: str) -> float:
        return sum(float(item[field]) * int(item["batches"]) for item in efficiency) / total_batches

    return {
        "trainable_parameters": trainable_parameters.pop(),
        "batches": total_batches,
        "weighted_mean_predict_ms_per_batch": weighted("mean_predict_ms_per_batch"),
        "weighted_mean_adapt_ms_per_batch": weighted("mean_adapt_ms_per_batch"),
        "weighted_mean_total_ms_per_batch": weighted("mean_total_ms_per_batch"),
        "maximum_peak_memory_mb": max(float(item["peak_memory_mb"]) for item in efficiency),
    }


def summarize(
    summary_paths: list[Path],
    audit_paths: list[Path],
) -> dict[str, Any]:
    if len(summary_paths) != 3 or len(audit_paths) != 3:
        raise ValueError("P5 requires exactly three summaries and three audits")

    summaries_by_seed: dict[int, dict[str, Any]] = {}
    audits_by_seed: dict[int, dict[str, Any]] = {}
    summary_hashes: dict[int, str] = {}
    audit_hashes: dict[int, str] = {}
    for audit_path in audit_paths:
        audit = read_json(audit_path)
        seed = int(audit["seed"])
        if audit.get("status") != "completed_and_audited" or audit.get("errors"):
            raise ValueError(f"Seed {seed} has not passed its single-target audit")
        audits_by_seed[seed] = audit
        audit_hashes[seed] = sha256(audit_path)
    for summary_path in summary_paths:
        candidate_hash = sha256(summary_path)
        matching_seeds = [
            seed
            for seed, audit in audits_by_seed.items()
            if audit["artifact_hashes"]["single_target_summary_json"] == candidate_hash
        ]
        if len(matching_seeds) != 1:
            raise ValueError(f"Cannot match summary to exactly one audited seed: {summary_path}")
        seed = matching_seeds[0]
        summaries_by_seed[seed] = read_json(summary_path)
        summary_hashes[seed] = candidate_hash
    if set(summaries_by_seed) != {0, 1, 2} or set(audits_by_seed) != {0, 1, 2}:
        raise ValueError("The audited P5 seed set must be exactly 0, 1, and 2")

    methods = list(audits_by_seed[0]["methods"])
    targets = list(audits_by_seed[0]["targets"])
    checkpoint_hashes = {
        audit["checkpoint_sha256"] for audit in audits_by_seed.values()
    }
    if len(checkpoint_hashes) != 1:
        raise ValueError("Single-target seeds used different source checkpoints")
    for seed in (0, 1, 2):
        audit = audits_by_seed[seed]
        if audit["methods"] != methods or audit["targets"] != targets:
            raise ValueError(f"Seed {seed} method/target contract differs")
        if set(summaries_by_seed[seed]) != set(methods):
            raise ValueError(f"Seed {seed} summary method set differs")

    methods_report: dict[str, Any] = {}
    for method in methods:
        per_domain: dict[str, Any] = {}
        macro_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
        for target in targets:
            target_report = {}
            for metric in METRICS:
                values = [
                    float(summaries_by_seed[seed][method][target]["overall"][metric])
                    for seed in (0, 1, 2)
                ]
                target_report[metric] = distribution(values)
            per_domain[target] = target_report

        per_seed_macro: dict[str, dict[str, float]] = {}
        for seed in (0, 1, 2):
            seed_macro = {}
            for metric in METRICS:
                value = mean(
                    float(summaries_by_seed[seed][method][target]["overall"][metric])
                    for target in targets
                )
                seed_macro[metric] = value
                macro_values[metric].append(value)
            per_seed_macro[str(seed)] = seed_macro

        methods_report[method] = {
            "cross_seed_macro": {
                metric: distribution(macro_values[metric]) for metric in METRICS
            },
            "per_seed_macro": per_seed_macro,
            "per_domain": per_domain,
            "efficiency_by_hardware": {
                str(seed): {
                    "host": HOST_BY_SEED[seed],
                    **efficiency_summary(summaries_by_seed[seed][method], targets),
                }
                for seed in (0, 1, 2)
            },
        }

    rankings = {
        metric: sorted(
            methods,
            key=lambda method: methods_report[method]["cross_seed_macro"][metric]["mean"],
            reverse=True,
        )
        for metric in METRICS
    }
    return {
        "stage": "P5",
        "experiment": "single_target_7_methods_x_3_seeds",
        "status": "completed_and_audited",
        "seeds": [0, 1, 2],
        "methods": methods,
        "targets": targets,
        "source_checkpoint_sha256": checkpoint_hashes.pop(),
        "aggregation": {
            "primary": "unweighted domain macro, followed by mean across seeds",
            "spread": "population standard deviation across seeds 0, 1, and 2",
            "latency": "never averaged across unlike GPUs",
            "controlled_efficiency": "seed 0 on A6000",
        },
        "methods_report": methods_report,
        "rankings": rankings,
        "input_hashes": {
            "summary_by_seed": {str(seed): summary_hashes[seed] for seed in (0, 1, 2)},
            "audit_by_seed": {str(seed): audit_hashes[seed] for seed in (0, 1, 2)},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize audited P5 single-target seeds")
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.summary, args.audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
