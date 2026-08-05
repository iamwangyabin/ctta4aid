#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from scripts.summarize_p5_single_target import (
        HOST_BY_SEED,
        distribution,
        read_json,
        sha256,
    )
except ModuleNotFoundError:
    from summarize_p5_single_target import (  # type: ignore[no-redef]
        HOST_BY_SEED,
        distribution,
        read_json,
        sha256,
    )


SUMMARY_METRICS = (
    "online_pooled_auc",
    "online_pooled_accuracy",
    "online_average_domain_auc",
    "final_pooled_auc",
    "final_pooled_accuracy",
    "final_average_domain_auc",
    "average_forgetting",
)


def efficiency(summary: dict[str, Any]) -> dict[str, Any]:
    value = summary["efficiency"]
    return {
        "trainable_parameters": int(value["trainable_parameters"]),
        "batches": int(value["batches"]),
        "mean_predict_ms_per_batch": float(value["mean_predict_ms_per_batch"]),
        "mean_adapt_ms_per_batch": float(value["mean_adapt_ms_per_batch"]),
        "mean_total_ms_per_batch": float(value["mean_total_ms_per_batch"]),
        "peak_memory_mb": float(value["peak_memory_mb"]),
    }


def summarize(
    summary_paths: list[Path],
    audit_paths: list[Path],
) -> dict[str, Any]:
    if len(summary_paths) != 3 or len(audit_paths) != 3:
        raise ValueError("P5 requires exactly three summaries and three audits")
    audits_by_seed: dict[int, dict[str, Any]] = {}
    summaries_by_seed: dict[int, dict[str, Any]] = {}
    audit_hashes: dict[int, str] = {}
    summary_hashes: dict[int, str] = {}
    for path in audit_paths:
        audit = read_json(path)
        seed = int(audit["seed"])
        if audit.get("status") != "completed_and_audited" or audit.get("errors"):
            raise ValueError(f"Seed {seed} has not passed its continual audit")
        audits_by_seed[seed] = audit
        audit_hashes[seed] = sha256(path)
    for path in summary_paths:
        candidate_hash = sha256(path)
        matching_seeds = [
            seed
            for seed, audit in audits_by_seed.items()
            if audit["artifact_hashes"]["continual_summary_json"] == candidate_hash
        ]
        if len(matching_seeds) != 1:
            raise ValueError(f"Cannot match summary to exactly one audited seed: {path}")
        seed = matching_seeds[0]
        summaries_by_seed[seed] = read_json(path)
        summary_hashes[seed] = candidate_hash
    if set(audits_by_seed) != {0, 1, 2} or set(summaries_by_seed) != {0, 1, 2}:
        raise ValueError("The audited P5 seed set must be exactly 0, 1, and 2")

    methods = list(audits_by_seed[0]["methods"])
    stream = list(audits_by_seed[0]["stream"])
    checkpoint_hashes = {audit["checkpoint_sha256"] for audit in audits_by_seed.values()}
    if len(checkpoint_hashes) != 1:
        raise ValueError("Continual seeds used different source checkpoints")
    for seed in (0, 1, 2):
        if audits_by_seed[seed]["methods"] != methods:
            raise ValueError(f"Seed {seed} method order differs")
        if audits_by_seed[seed]["stream"] != stream:
            raise ValueError(f"Seed {seed} stream order differs")
        if set(summaries_by_seed[seed]) != set(methods):
            raise ValueError(f"Seed {seed} summary method set differs")

    methods_report: dict[str, Any] = {}
    for method in methods:
        scalar_values: dict[str, list[float]] = {metric: [] for metric in SUMMARY_METRICS}
        per_seed: dict[str, Any] = {}
        per_domain: dict[str, Any] = {}
        for seed in (0, 1, 2):
            summary = summaries_by_seed[seed][method]
            seed_values = {
                "online_pooled_auc": float(summary["overall"]["auc"]),
                "online_pooled_accuracy": float(summary["overall"]["accuracy"]),
                "online_average_domain_auc": float(summary["online_average_domain_auc"]),
                "final_pooled_auc": float(summary["final_pooled_auc"]),
                "final_pooled_accuracy": float(
                    audits_by_seed[seed]["method_metrics"][method]["final_pooled_accuracy"]
                ),
                "final_average_domain_auc": float(summary["final_average_auc"]),
                "average_forgetting": float(summary["average_forgetting"]),
            }
            per_seed[str(seed)] = seed_values
            for metric, value in seed_values.items():
                scalar_values[metric].append(value)

        for domain in stream:
            per_domain[domain] = {
                "online": {
                    metric: distribution(
                        [
                            float(summaries_by_seed[seed][method]["by_domain"][domain][metric])
                            for seed in (0, 1, 2)
                        ]
                    )
                    for metric in ("auc", "accuracy", "balanced_accuracy")
                },
                "final": {
                    metric: distribution(
                        [
                            float(
                                audits_by_seed[seed]["method_metrics"][method][
                                    "final_by_domain"
                                ][domain][metric]
                            )
                            for seed in (0, 1, 2)
                        ]
                    )
                    for metric in ("auc", "accuracy", "balanced_accuracy")
                },
                "forgetting": distribution(
                    [
                        float(
                            summaries_by_seed[seed][method]["forgetting_by_domain"][domain]
                        )
                        for seed in (0, 1, 2)
                    ]
                ),
            }
        methods_report[method] = {
            "cross_seed": {
                metric: distribution(values) for metric, values in scalar_values.items()
            },
            "per_seed": per_seed,
            "per_domain": per_domain,
            "efficiency_by_hardware": {
                str(seed): {
                    "host": HOST_BY_SEED[seed],
                    **efficiency(summaries_by_seed[seed][method]),
                }
                for seed in (0, 1, 2)
            },
        }

    rankings = {
        metric: sorted(
            methods,
            key=lambda method: methods_report[method]["cross_seed"][metric]["mean"],
            reverse=metric != "average_forgetting",
        )
        for metric in SUMMARY_METRICS
    }
    return {
        "stage": "P5",
        "experiment": "continual_stream_7_methods_x_3_seeds",
        "status": "completed_and_audited",
        "seeds": [0, 1, 2],
        "methods": methods,
        "stream": stream,
        "source_checkpoint_sha256": checkpoint_hashes.pop(),
        "aggregation": {
            "performance": "mean and population standard deviation across seeds 0, 1, and 2",
            "forgetting": "fixed-holdout best AUC minus final AUC; last stream domain excluded from average",
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
    parser = argparse.ArgumentParser(description="Summarize audited P5 continual seeds")
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
