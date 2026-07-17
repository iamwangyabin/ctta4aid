from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from online_aig_tta.config import load_config
from run_iapl_official import validate_official_metrics


METRICS = ("acc", "ap", "racc", "facc")


def merge_shards(shards: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_domain: dict[str, dict[str, float]] = {}
    commits = {shard.get("official_commit") for shard in shards}
    commits.discard(None)
    if len(commits) > 1:
        raise RuntimeError(f"IAPL shard commit mismatch: {sorted(commits)}")
    for shard in shards:
        for domain, metrics in shard["by_domain"].items():
            if domain in by_domain:
                raise RuntimeError(f"Duplicate IAPL domain across shards: {domain}")
            by_domain[domain] = {metric: float(metrics[metric]) for metric in METRICS}

    expected = set(map(str, config["test_selected_subsets"]))
    actual = set(by_domain)
    if actual != expected:
        raise RuntimeError(
            f"IAPL shard domain mismatch: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    mean = {
        metric: statistics.fmean(values[metric] for values in by_domain.values())
        for metric in METRICS
    }
    parsed = {"by_domain": by_domain, "mean": mean}
    reference_check = validate_official_metrics(parsed, config)
    return {
        "method": "IAPL",
        "implementation": "official_sharded_by_domain",
        "official_commit": next(iter(commits), None),
        "protocol": "per_image_reset_adapt_then_predict",
        "dataset": config["dataset"],
        "reported_metrics": [
            "accuracy",
            "average_precision",
            "real_accuracy",
            "fake_accuracy",
        ],
        "by_domain": by_domain,
        "mean": mean,
        "reference_check": reference_check,
        "shard_assets": [shard.get("assets", {}) for shard in shards],
        "numerical_validation": (
            "reference_gate_passed"
            if reference_check.get("passed")
            else "reference_gate_failed"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge independent official IAPL domains")
    parser.add_argument("--config", required=True)
    parser.add_argument("--metrics", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.metrics]
    merged = merge_shards(shards, config)
    merged["shard_metrics"] = [str(path.resolve()) for path in args.metrics]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(args.output)
    if config.get("require_reference_match", True) and not merged["reference_check"].get(
        "passed"
    ):
        raise RuntimeError(
            "Merged IAPL metrics miss the configured paper-reference tolerance: "
            f"{merged['reference_check']}"
        )


if __name__ == "__main__":
    main()
