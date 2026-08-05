#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from scripts.audit_p5_single_target import (
        CHECKPOINT_SHA256,
        MANIFEST_FIELDS,
        METHODS,
        check_rate,
        read_json,
        sha256,
        tree_sha256,
    )
except ModuleNotFoundError:
    from audit_p5_single_target import (  # type: ignore[no-redef]
        CHECKPOINT_SHA256,
        MANIFEST_FIELDS,
        METHODS,
        check_rate,
        read_json,
        sha256,
        tree_sha256,
    )


STREAM = (
    "biggan",
    "cyclegan",
    "stylegan",
    "gaugan",
    "guided",
    "glide_100_27",
    "ldm_200",
    "dalle",
)
ONLINE_SAMPLES_PER_DOMAIN = 1500
HOLDOUT_SAMPLES_PER_DOMAIN = 500


def close(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def read_csv(path: Path, errors: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or ()), list(reader)
    except OSError as error:
        errors.append(f"cannot read CSV {path}: {error}")
        return [], []


def validate_manifest(
    path: Path,
    *,
    stream: tuple[str, ...],
    samples_per_domain: int,
    label: str,
    errors: list[str],
) -> list[dict[str, str]]:
    fields, rows = read_csv(path, errors)
    if set(fields) != MANIFEST_FIELDS:
        errors.append(f"{label} manifest columns are invalid")
    expected_total = len(stream) * samples_per_domain
    if len(rows) != expected_total:
        errors.append(f"{label} manifest has {len(rows)} rows, expected {expected_total}")
    positions: list[int] = []
    for row in rows:
        try:
            positions.append(int(row["position"]))
        except (KeyError, TypeError, ValueError):
            positions.append(-1)
    if positions != list(range(len(rows))):
        errors.append(f"{label} manifest positions are not contiguous")
    sample_ids = [row.get("sample_id") for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        errors.append(f"{label} manifest has an empty sample ID")
    if len(sample_ids) != len(set(sample_ids)):
        errors.append(f"{label} manifest has duplicate sample IDs")
    observed_domains = [row.get("domain") for row in rows]
    expected_domains = [
        domain for domain in stream for _ in range(samples_per_domain)
    ]
    if observed_domains != expected_domains:
        errors.append(f"{label} manifest domain blocks do not match the fixed stream")
    return rows


def audit_continual(
    root: Path,
    *,
    seed: int,
    methods: tuple[str, ...] = METHODS,
    stream: tuple[str, ...] = STREAM,
    checkpoint_sha256: str = CHECKPOINT_SHA256,
    log_path: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    required_files = (
        "metrics.json",
        "online_curve.csv",
        "batch_stats.csv",
        "sample_manifest.csv",
        "holdout_matrix.csv",
        "final_holdout_manifest.csv",
    )
    effective_config_path = root / "effective_config.json"
    aggregate_path = root / "continual_summary.json"
    effective_config = read_json(effective_config_path, errors)
    aggregate = read_json(aggregate_path, errors)
    if effective_config.get("seed") != seed:
        errors.append(
            f"effective seed {effective_config.get('seed')!r} does not match {seed}"
        )
    if tuple(effective_config.get("methods", [])) != methods:
        errors.append("effective method order does not match the frozen P5 method order")
    if tuple(effective_config.get("data", {}).get("stream", [])) != stream:
        errors.append("effective stream order does not match the frozen P5 stream")
    if set(aggregate) != set(methods):
        errors.append("aggregate method keys are incomplete or unexpected")

    metric_hashes: list[tuple[str, str]] = []
    online_manifest_hashes: dict[str, str] = {}
    holdout_manifest_hashes: dict[str, str] = {}
    checkpoint_hashes: set[str] = set()
    method_metrics: dict[str, Any] = {}
    expected_online_total = len(stream) * ONLINE_SAMPLES_PER_DOMAIN
    expected_holdout_total = len(stream) * HOLDOUT_SAMPLES_PER_DOMAIN
    expected_batches = len(stream) * math.ceil(ONLINE_SAMPLES_PER_DOMAIN / 16)
    expected_matrix_pairs = [
        (checkpoint, stream[checkpoint], domain)
        for checkpoint in range(len(stream))
        for domain in stream[: checkpoint + 1]
    ]

    for method in methods:
        destination = root / method
        missing = [name for name in required_files if not (destination / name).is_file()]
        if missing:
            errors.append(f"{method} missing files: {', '.join(missing)}")
            continue
        empty = [name for name in required_files if (destination / name).stat().st_size == 0]
        if empty:
            errors.append(f"{method} has empty files: {', '.join(empty)}")
            continue

        metrics_path = destination / "metrics.json"
        metric_hashes.append((metrics_path.relative_to(root).as_posix(), sha256(metrics_path)))
        online_manifest_path = destination / "sample_manifest.csv"
        holdout_manifest_path = destination / "final_holdout_manifest.csv"
        online_manifest_hashes[method] = sha256(online_manifest_path)
        holdout_manifest_hashes[method] = sha256(holdout_manifest_path)
        validate_manifest(
            online_manifest_path,
            stream=stream,
            samples_per_domain=ONLINE_SAMPLES_PER_DOMAIN,
            label=f"{method} online",
            errors=errors,
        )
        validate_manifest(
            holdout_manifest_path,
            stream=stream,
            samples_per_domain=HOLDOUT_SAMPLES_PER_DOMAIN,
            label=f"{method} final holdout",
            errors=errors,
        )

        metrics = read_json(metrics_path, errors)
        if metrics.get("protocol") != "predict_then_adapt":
            errors.append(f"{method} has the wrong protocol")
        if metrics.get("method") != method:
            errors.append(f"{method} identity field does not match its path")
        if tuple(metrics.get("stream_order", [])) != stream:
            errors.append(f"{method} recorded the wrong stream order")
        reproduction = metrics.get("reproduction", {})
        recorded_checkpoint = reproduction.get("source_checkpoint", {}).get("sha256")
        if isinstance(recorded_checkpoint, str):
            checkpoint_hashes.add(recorded_checkpoint)
        if recorded_checkpoint != checkpoint_sha256:
            errors.append(f"{method} has the wrong source checkpoint")
        forgetting_protocol = metrics.get("forgetting_protocol", {})
        expected_forgetting_protocol = {
            "definition": "fixed_holdout_best_auc_minus_final_auc",
            "evaluate_after_each_domain": True,
            "average_excludes_last_domain": True,
            "holdout_shuffle": "seeded_global",
            "loader_seed_offset": 1_000_000,
            "evaluation_seed": seed + 2_000_000,
            "random_state_restored_after_evaluation": True,
        }
        if forgetting_protocol != expected_forgetting_protocol:
            errors.append(f"{method} forgetting protocol metadata differs")

        summary = metrics.get("summary", {})
        overall = summary.get("overall", {})
        final_holdout = metrics.get("final_holdout", {})
        final_overall = final_holdout.get("overall", {})
        for label, values, expected_total in (
            ("online", overall, expected_online_total),
            ("final holdout", final_overall, expected_holdout_total),
        ):
            for metric in ("auc", "accuracy", "balanced_accuracy"):
                check_rate(values.get(metric), f"{method} {label} {metric}", errors)
            if values.get("samples") != expected_total:
                errors.append(f"{method} {label} sample total is inconsistent")

        online_domains = summary.get("by_domain", {})
        final_domains = final_holdout.get("by_domain", {})
        if tuple(online_domains) != stream or tuple(final_domains) != stream:
            errors.append(f"{method} online/final domain order differs")
        for domain in stream:
            for label, values, expected_count in (
                ("online", online_domains.get(domain, {}), ONLINE_SAMPLES_PER_DOMAIN),
                ("final", final_domains.get(domain, {}), HOLDOUT_SAMPLES_PER_DOMAIN),
            ):
                for metric in ("auc", "accuracy", "balanced_accuracy"):
                    check_rate(values.get(metric), f"{method} {domain} {label} {metric}", errors)
                if values.get("samples") != expected_count:
                    errors.append(f"{method} {domain} {label} sample count is inconsistent")

        if not close(
            summary.get("online_average_domain_auc"),
            mean(float(online_domains[domain]["auc"]) for domain in stream),
        ):
            errors.append(f"{method} online average domain AUC is inconsistent")
        if not close(
            summary.get("final_average_auc"),
            mean(float(final_domains[domain]["auc"]) for domain in stream),
        ):
            errors.append(f"{method} final average domain AUC is inconsistent")
        if not close(summary.get("final_pooled_auc"), final_overall.get("auc")):
            errors.append(f"{method} final pooled AUC is inconsistent")

        batch_fields, batch_rows = read_csv(destination / "batch_stats.csv", errors)
        if not {"batch", "domain", "samples"}.issubset(batch_fields):
            errors.append(f"{method} batch stats columns are incomplete")
        efficiency = summary.get("efficiency", {})
        if len(batch_rows) != expected_batches or efficiency.get("batches") != expected_batches:
            errors.append(f"{method} online batch count is inconsistent")
        try:
            batch_samples = sum(int(row["samples"]) for row in batch_rows)
        except (KeyError, TypeError, ValueError):
            batch_samples = -1
        if batch_samples != expected_online_total:
            errors.append(f"{method} online batch sample total is inconsistent")
        for field in (
            "mean_predict_ms_per_batch",
            "mean_adapt_ms_per_batch",
            "mean_total_ms_per_batch",
            "peak_memory_mb",
        ):
            try:
                value = float(efficiency[field])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{method} efficiency {field} is missing")
                continue
            if not math.isfinite(value) or value < 0:
                errors.append(f"{method} efficiency {field} is invalid")

        _, curve_rows = read_csv(destination / "online_curve.csv", errors)
        if len(curve_rows) != expected_batches:
            errors.append(f"{method} online curve batch count is inconsistent")
        matrix_fields, matrix_rows = read_csv(destination / "holdout_matrix.csv", errors)
        expected_matrix_fields = {
            "checkpoint",
            "after_domain",
            "eval_domain",
            "auc",
            "accuracy",
            "balanced_accuracy",
            "samples",
        }
        if set(matrix_fields) != expected_matrix_fields:
            errors.append(f"{method} holdout matrix columns are invalid")
        observed_pairs: list[tuple[int, str, str]] = []
        histories: dict[str, list[float]] = {domain: [] for domain in stream}
        for row in matrix_rows:
            try:
                checkpoint = int(row["checkpoint"])
                pair = (checkpoint, row["after_domain"], row["eval_domain"])
                samples = int(row["samples"])
                auc = float(row["auc"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{method} has a malformed holdout matrix row")
                continue
            observed_pairs.append(pair)
            histories[row["eval_domain"]].append(auc)
            if samples != HOLDOUT_SAMPLES_PER_DOMAIN:
                errors.append(f"{method} holdout matrix sample count is inconsistent")
            for metric in ("auc", "accuracy", "balanced_accuracy"):
                check_rate(row.get(metric), f"{method} matrix {pair} {metric}", errors)
        if observed_pairs != expected_matrix_pairs:
            errors.append(f"{method} holdout matrix is not the expected triangle")
        for domain in stream:
            history = histories[domain]
            expected_forgetting = max(history) - float(final_domains[domain]["auc"])
            if not close(
                summary.get("forgetting_by_domain", {}).get(domain), expected_forgetting
            ):
                errors.append(f"{method} forgetting is inconsistent for {domain}")
        expected_average_forgetting = mean(
            float(summary["forgetting_by_domain"][domain]) for domain in stream[:-1]
        )
        if not close(summary.get("average_forgetting"), expected_average_forgetting):
            errors.append(f"{method} average forgetting is inconsistent")

        if aggregate.get(method) != summary:
            errors.append(f"{method} does not match aggregate summary")
        method_metrics[method] = {
            "online_pooled_auc": float(overall.get("auc", math.nan)),
            "online_pooled_accuracy": float(overall.get("accuracy", math.nan)),
            "online_average_domain_auc": float(
                summary.get("online_average_domain_auc", math.nan)
            ),
            "final_pooled_auc": float(final_overall.get("auc", math.nan)),
            "final_pooled_accuracy": float(final_overall.get("accuracy", math.nan)),
            "final_average_domain_auc": float(summary.get("final_average_auc", math.nan)),
            "average_forgetting": float(summary.get("average_forgetting", math.nan)),
            "final_by_domain": final_domains,
            "efficiency": efficiency,
        }

    online_equal = set(online_manifest_hashes) == set(methods) and len(
        set(online_manifest_hashes.values())
    ) == 1
    holdout_equal = set(holdout_manifest_hashes) == set(methods) and len(
        set(holdout_manifest_hashes.values())
    ) == 1
    if not online_equal:
        errors.append("ordered online manifests differ across methods")
    if not holdout_equal:
        errors.append("ordered final-holdout manifests differ across methods")

    exit_status = None
    log_sha256 = None
    if log_path is not None:
        if not log_path.is_file():
            errors.append(f"run log does not exist: {log_path}")
        else:
            log_sha256 = sha256(log_path)
            matches = re.findall(
                r"^\s*Exit status:\s*(\d+)\s*$",
                log_path.read_text(encoding="utf-8", errors="replace"),
                flags=re.MULTILINE,
            )
            if matches:
                exit_status = int(matches[-1])
            if exit_status != 0:
                errors.append(f"run log exit status is not zero: {exit_status!r}")

    return {
        "stage": "P5",
        "experiment": "continual_stream_7_methods",
        "seed": seed,
        "status": "completed_and_audited" if not errors else "audit_failed",
        "root": str(root),
        "methods": list(methods),
        "stream": list(stream),
        "method_results": len(metric_hashes),
        "expected_method_results": len(methods),
        "online_samples_per_method": expected_online_total,
        "final_holdout_samples_per_method": expected_holdout_total,
        "holdout_matrix_rows_per_method": len(expected_matrix_pairs),
        "checkpoint_sha256": checkpoint_sha256,
        "recorded_checkpoint_hashes": sorted(checkpoint_hashes),
        "exact_online_manifest_equality": online_equal,
        "exact_final_holdout_manifest_equality": holdout_equal,
        "method_metrics": method_metrics,
        "artifact_hashes": {
            "effective_config_json": (
                sha256(effective_config_path) if effective_config_path.is_file() else None
            ),
            "continual_summary_json": (
                sha256(aggregate_path) if aggregate_path.is_file() else None
            ),
            "metrics_tree": tree_sha256(metric_hashes),
            "online_manifest": next(iter(online_manifest_hashes.values()), None),
            "final_holdout_manifest": next(iter(holdout_manifest_hashes.values()), None),
            "run_log": log_sha256,
        },
        "process_exit_status": exit_status,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one completed P5 continual seed")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_continual(
        args.root.expanduser().resolve(),
        seed=args.seed,
        log_path=args.log.expanduser().resolve() if args.log else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "completed_and_audited":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
