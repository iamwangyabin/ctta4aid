#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


METHODS = ("source", "tent", "eata", "cotta", "rotta", "lame", "t2a")
TARGET_SAMPLE_COUNTS = {
    "cyclegan": 2000,
    "biggan": 2000,
    "stylegan": 2000,
    "stargan": 2000,
    "gaugan": 2000,
    "crn": 2000,
    "dalle": 2000,
    "deepfake": 2000,
    "glide_50_27": 2000,
    "glide_100_10": 2000,
    "glide_100_27": 2000,
    "guided": 2000,
    "imle": 2000,
    "ldm_100": 2000,
    "ldm_200": 2000,
    "ldm_200_cfg": 2000,
    "san": 438,
    "seeingdark": 360,
}
MANIFEST_FIELDS = {"batch", "domain", "position", "sample_id"}
CHECKPOINT_SHA256 = (
    "57d3e1ea43b914226449ecb5d4267d86324002f5b0210bad5a7667673acd3840"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(items: list[tuple[str, str]]) -> str:
    payload = "".join(f"{file_hash}  {name}\n" for name, file_hash in sorted(items))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"cannot read JSON {path}: {error}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"expected JSON object in {path}")
        return {}
    return value


def check_rate(value: Any, name: str, errors: list[str]) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} is not numeric: {value!r}")
        return None
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        errors.append(f"{name} is outside [0, 1]: {number!r}")
        return None
    return number


def audit_single_target(
    root: Path,
    *,
    seed: int,
    methods: tuple[str, ...] = METHODS,
    expected_samples: dict[str, int] = TARGET_SAMPLE_COUNTS,
    checkpoint_sha256: str = CHECKPOINT_SHA256,
    log_path: Path | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    required_files = (
        "metrics.json",
        "online_curve.csv",
        "batch_stats.csv",
        "sample_manifest.csv",
    )
    effective_config_path = root / "effective_config.json"
    aggregate_path = root / "single_target_summary.json"
    effective_config = read_json(effective_config_path, errors)
    aggregate = read_json(aggregate_path, errors)

    if effective_config.get("seed") != seed:
        errors.append(
            f"effective seed {effective_config.get('seed')!r} does not match {seed}"
        )
    if tuple(effective_config.get("methods", [])) != methods:
        errors.append("effective method order does not match the frozen P5 method order")
    targets = tuple(expected_samples)
    configured_targets = tuple(effective_config.get("data", {}).get("targets", []))
    if configured_targets != targets:
        errors.append("effective target order does not match the frozen P5 target order")
    if set(aggregate) != set(methods):
        errors.append("aggregate method keys are incomplete or unexpected")

    metric_hashes: list[tuple[str, str]] = []
    manifest_hashes: list[tuple[str, str]] = []
    manifest_by_target: dict[str, dict[str, str]] = {target: {} for target in targets}
    checkpoint_hashes: set[str] = set()
    macro_metrics: dict[str, dict[str, float]] = {}
    total_samples = 0

    for method in methods:
        method_auc: list[float] = []
        method_accuracy: list[float] = []
        method_balanced_accuracy: list[float] = []
        method_summary = aggregate.get(method, {})
        if not isinstance(method_summary, dict) or set(method_summary) != set(targets):
            errors.append(f"aggregate targets are incomplete or unexpected for {method}")

        for target, expected_count in expected_samples.items():
            destination = root / method / target
            missing = [name for name in required_files if not (destination / name).is_file()]
            if missing:
                errors.append(f"{method}/{target} missing files: {', '.join(missing)}")
                continue
            empty = [name for name in required_files if (destination / name).stat().st_size == 0]
            if empty:
                errors.append(f"{method}/{target} has empty files: {', '.join(empty)}")
                continue

            metrics_path = destination / "metrics.json"
            manifest_path = destination / "sample_manifest.csv"
            relative_metrics = metrics_path.relative_to(root).as_posix()
            relative_manifest = manifest_path.relative_to(root).as_posix()
            metric_hashes.append((relative_metrics, sha256(metrics_path)))
            manifest_digest = sha256(manifest_path)
            manifest_hashes.append((relative_manifest, manifest_digest))
            manifest_by_target[target][method] = manifest_digest

            metrics = read_json(metrics_path, errors)
            if metrics.get("protocol") != "predict_then_adapt":
                errors.append(f"{method}/{target} has the wrong protocol")
            if metrics.get("method") != method or metrics.get("target") != target:
                errors.append(f"{method}/{target} identity fields do not match its path")

            reproduction = metrics.get("reproduction", {})
            recorded_checkpoint = reproduction.get("source_checkpoint", {}).get("sha256")
            if isinstance(recorded_checkpoint, str):
                checkpoint_hashes.add(recorded_checkpoint)
            if recorded_checkpoint != checkpoint_sha256:
                errors.append(f"{method}/{target} has the wrong source checkpoint")

            overall = metrics.get("summary", {}).get("overall", {})
            auc = check_rate(overall.get("auc"), f"{method}/{target} AUC", errors)
            accuracy = check_rate(
                overall.get("accuracy"), f"{method}/{target} accuracy", errors
            )
            balanced_accuracy = check_rate(
                overall.get("balanced_accuracy"),
                f"{method}/{target} balanced accuracy",
                errors,
            )
            if auc is not None:
                method_auc.append(auc)
            if accuracy is not None:
                method_accuracy.append(accuracy)
            if balanced_accuracy is not None:
                method_balanced_accuracy.append(balanced_accuracy)
            if overall.get("samples") != expected_count:
                errors.append(
                    f"{method}/{target} reports {overall.get('samples')!r} samples, "
                    f"expected {expected_count}"
                )

            by_domain = metrics.get("summary", {}).get("by_domain", {})
            if set(by_domain) != {target} or by_domain.get(target) != overall:
                errors.append(f"{method}/{target} by-domain summary is inconsistent")
            if method_summary.get(target) != metrics.get("summary"):
                errors.append(f"{method}/{target} does not match aggregate summary")

            with manifest_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if set(reader.fieldnames or ()) != MANIFEST_FIELDS:
                    errors.append(f"{method}/{target} manifest columns are invalid")
                manifest_rows = list(reader)
            if len(manifest_rows) != expected_count:
                errors.append(
                    f"{method}/{target} manifest has {len(manifest_rows)} rows, "
                    f"expected {expected_count}"
                )
            positions: list[int] = []
            for row in manifest_rows:
                try:
                    positions.append(int(row["position"]))
                except (KeyError, TypeError, ValueError):
                    positions.append(-1)
            if positions != list(range(len(manifest_rows))):
                errors.append(f"{method}/{target} manifest positions are not contiguous")
            if any(row.get("domain") != target for row in manifest_rows):
                errors.append(f"{method}/{target} manifest contains another domain")
            sample_ids = [row.get("sample_id") for row in manifest_rows]
            if any(not sample_id for sample_id in sample_ids):
                errors.append(f"{method}/{target} manifest has an empty sample ID")
            if len(sample_ids) != len(set(sample_ids)):
                errors.append(f"{method}/{target} manifest has duplicate sample IDs")

            with (destination / "batch_stats.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                batch_rows = list(csv.DictReader(handle))
            efficiency = metrics.get("summary", {}).get("efficiency", {})
            if efficiency.get("batches") != len(batch_rows):
                errors.append(f"{method}/{target} efficiency batch count is inconsistent")
            try:
                batch_samples = sum(int(row["samples"]) for row in batch_rows)
            except (KeyError, TypeError, ValueError):
                batch_samples = -1
            if batch_samples != expected_count:
                errors.append(f"{method}/{target} batch sample total is inconsistent")
            for field in (
                "mean_predict_ms_per_batch",
                "mean_adapt_ms_per_batch",
                "mean_total_ms_per_batch",
                "peak_memory_mb",
            ):
                try:
                    value = float(efficiency[field])
                except (KeyError, TypeError, ValueError):
                    errors.append(f"{method}/{target} efficiency {field} is missing")
                    continue
                if not math.isfinite(value) or value < 0:
                    errors.append(f"{method}/{target} efficiency {field} is invalid")

            if method == methods[0]:
                total_samples += expected_count

        if len(method_auc) == len(targets):
            macro_metrics[method] = {
                "mean_auc": mean(method_auc),
                "mean_accuracy": mean(method_accuracy),
                "mean_balanced_accuracy": mean(method_balanced_accuracy),
            }

    manifest_equal_by_target: dict[str, bool] = {}
    for target, hashes in manifest_by_target.items():
        equal = set(hashes) == set(methods) and len(set(hashes.values())) == 1
        manifest_equal_by_target[target] = equal
        if not equal:
            errors.append(f"ordered sample manifests differ across methods for {target}")

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

    report = {
        "stage": "P5",
        "experiment": "single_target_7_methods",
        "seed": seed,
        "status": "completed_and_audited" if not errors else "audit_failed",
        "root": str(root),
        "methods": list(methods),
        "targets": list(targets),
        "method_target_results": len(metric_hashes),
        "expected_method_target_results": len(methods) * len(targets),
        "samples_per_method": total_samples,
        "checkpoint_sha256": checkpoint_sha256,
        "recorded_checkpoint_hashes": sorted(checkpoint_hashes),
        "exact_manifest_equality_by_target": manifest_equal_by_target,
        "all_exact_manifest_equality": all(manifest_equal_by_target.values()),
        "macro_metrics": macro_metrics,
        "artifact_hashes": {
            "effective_config_json": (
                sha256(effective_config_path) if effective_config_path.is_file() else None
            ),
            "single_target_summary_json": (
                sha256(aggregate_path) if aggregate_path.is_file() else None
            ),
            "metrics_tree": tree_sha256(metric_hashes),
            "sample_manifest_tree": tree_sha256(manifest_hashes),
            "sample_manifest_by_target": {
                target: next(iter(hashes.values()), None)
                for target, hashes in manifest_by_target.items()
            },
            "run_log": log_sha256,
        },
        "process_exit_status": exit_status,
        "errors": errors,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit one completed P5 single-target seed")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = audit_single_target(
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
