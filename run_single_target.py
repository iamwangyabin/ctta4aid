from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from statistics import fmean

from src.cli.common import (
    build_fresh_method,
    data_config_for_method,
    resolve_device,
    seed_everything,
    write_json,
)
from src.config import load_config, require, validate_experiment_identity
from src.data.arrow import validate_arrow_data_profile
from src.data.streams import (
    as_stream,
    build_domain_loader,
    load_online_manifest_lock,
    validate_locked_sample_order,
)
from src.evaluation import OnlineEvaluator, evaluate_without_adaptation, save_evaluation


CROSS_HOLDOUT_LOADER_SEED_OFFSET = 3_000_000
CROSS_HOLDOUT_EVALUATION_SEED_OFFSET = 4_000_000


def cross_generator_holdout_stream(
    config: dict,
    domains: list[str],
    seed: int,
    *,
    transform=None,
    data_config: dict | None = None,
):
    data_config = config["data"] if data_config is None else data_config
    offset = int(data_config.get("max_samples_per_class", 0))
    limit = int(data_config.get("cross_eval_max_samples_per_class", 250))
    for domain_index, domain in enumerate(domains):
        loader = build_domain_loader(
            data_config,
            domain,
            seed=seed + domain_index,
            sample_seed=seed + domain_index,
            loader_seed=seed + CROSS_HOLDOUT_LOADER_SEED_OFFSET + domain_index,
            max_samples_per_class=limit,
            sample_offset_per_class=offset,
            shuffle=True,
            transform=transform,
        )
        yield from as_stream(loader, domain)


def pairwise_transfer_rows(
    *,
    method: str,
    seed: int,
    adapted_on: str,
    initial: dict,
    adapted: dict,
) -> list[dict]:
    rows = []
    initial_by_domain = initial.get("by_domain", {})
    adapted_by_domain = adapted.get("by_domain", {})
    for evaluated_on, initial_metrics in initial_by_domain.items():
        adapted_metrics = adapted_by_domain[evaluated_on]
        row = {
            "method": method,
            "seed": seed,
            "adapted_on": adapted_on,
            "evaluated_on": evaluated_on,
            "relation": "current" if adapted_on == evaluated_on else "cross_generator",
        }
        for metric in ("auc", "accuracy", "balanced_accuracy"):
            initial_value = float(initial_metrics[metric])
            adapted_value = float(adapted_metrics[metric])
            row[f"initial_{metric}"] = initial_value
            row[f"adapted_{metric}"] = adapted_value
            row[f"delta_{metric}"] = adapted_value - initial_value
        row["samples"] = int(adapted_metrics["samples"])
        rows.append(row)
    return rows


def summarize_pairwise_transfer(rows: list[dict]) -> dict:
    current = [
        float(row["delta_auc"])
        for row in rows
        if row["relation"] == "current" and math.isfinite(float(row["delta_auc"]))
    ]
    cross = [
        float(row["delta_auc"])
        for row in rows
        if row["relation"] == "cross_generator"
        and math.isfinite(float(row["delta_auc"]))
    ]
    return {
        "definition": "post_adaptation_auc_minus_method_initial_auc_on_fixed_holdouts",
        "current_pairs": len(current),
        "cross_generator_pairs": len(cross),
        "mean_current_auc_delta": fmean(current) if current else math.nan,
        "mean_cross_generator_auc_delta": fmean(cross) if cross else math.nan,
        "cross_generator_negative_transfer_rate": (
            fmean(delta < 0.0 for delta in cross) if cross else math.nan
        ),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def release_method_resources() -> None:
    """Release a finished target model before constructing the next fresh model."""

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent online TTA run per target generator")
    parser.add_argument(
        "--config", default="configs/experiments/controlled_ctta/single_target_seed0.yaml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    require(config, "data", "methods", "output_dir")
    dedicated_models = {"iapl", "ost"}
    normalized_methods = {
        str(name).lower().replace("_", "").replace("-", "")
        for name in config["methods"]
    }
    if any(name not in dedicated_models for name in normalized_methods):
        require(config, "model")
    require(config["data"], "format", "root", "targets")
    experiment_identity = validate_experiment_identity(config)
    validate_arrow_data_profile(
        config["data"]["root"], config["data"].get("bias_control_profile")
    )

    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    evaluation_config = config.get("evaluation", {})
    evaluate_cross_generators = bool(
        evaluation_config.get("evaluate_cross_generators", False)
    )
    evaluation_domains = list(
        evaluation_config.get("cross_generator_targets", config["data"]["targets"])
    )
    evaluator = OnlineEvaluator(
        threshold=float(evaluation_config.get("threshold", 0.5)),
        curve_window_batches=int(evaluation_config.get("curve_window_batches", 20)),
    )
    output_root = Path(config["output_dir"])
    identity_path = output_root / "experiment_identity.json"
    if identity_path.is_file():
        with identity_path.open(encoding="utf-8") as handle:
            existing_identity = json.load(handle)
        if existing_identity != experiment_identity:
            raise ValueError(
                f"Refusing to mix experiment identities below {output_root}"
            )
    write_json(identity_path, experiment_identity)
    write_json(output_root / "effective_config.json", config)
    aggregate = {}
    all_pairwise_rows: list[dict] = []
    online_manifest_lock = load_online_manifest_lock(
        config, list(config["data"]["targets"])
    )

    for method_name in config["methods"]:
        method_data_config = data_config_for_method(config, method_name)
        aggregate[method_name] = {"targets": {}} if evaluate_cross_generators else {}
        initial_holdout = None
        if evaluate_cross_generators:
            seed_everything(seed)
            initial_method, _ = build_fresh_method(config, method_name, device)
            if getattr(
                initial_method, "protocol_name", "predict_then_adapt"
            ) != "predict_then_adapt":
                raise ValueError(
                    "Cross-generator transfer requires a state-neutral "
                    "Predict-Then-Adapt prediction path"
                )
            initial_holdout = evaluate_without_adaptation(
                initial_method,
                cross_generator_holdout_stream(
                    config,
                    evaluation_domains,
                    seed,
                    transform=getattr(initial_method, "input_transform", None),
                    data_config=method_data_config,
                ),
                threshold=float(evaluation_config.get("threshold", 0.5)),
                evaluation_seed=seed + CROSS_HOLDOUT_EVALUATION_SEED_OFFSET,
            )
            aggregate[method_name]["initial_holdout"] = initial_holdout
            del initial_method
            release_method_resources()

        for target_index, target in enumerate(config["data"]["targets"]):
            seed_everything(seed)
            method, _ = build_fresh_method(config, method_name, device)
            loader = build_domain_loader(
                method_data_config,
                target,
                seed=seed + target_index,
                transform=getattr(method, "input_transform", None),
                locked_sample_ids=(
                    None
                    if online_manifest_lock is None
                    else online_manifest_lock["sample_ids_by_domain"][target]
                ),
            )
            result = evaluator.run(method, as_stream(loader, target))
            if online_manifest_lock is not None:
                validate_locked_sample_order(
                    [str(row["sample_id"]) for row in result["sample_manifest"]],
                    online_manifest_lock["sample_ids_by_domain"][target],
                )
                result["sample_lock"] = online_manifest_lock["config"]
            result["protocol"] = getattr(method, "protocol_name", "predict_then_adapt")
            result["method"] = method_name
            result["target"] = target
            result["experiment_identity"] = experiment_identity
            if initial_holdout is not None:
                cross_holdout = evaluate_without_adaptation(
                    method,
                cross_generator_holdout_stream(
                    config,
                    evaluation_domains,
                    seed,
                    transform=getattr(method, "input_transform", None),
                    data_config=method_data_config,
                    ),
                    threshold=float(evaluation_config.get("threshold", 0.5)),
                    evaluation_seed=seed + CROSS_HOLDOUT_EVALUATION_SEED_OFFSET,
                )
                target_rows = pairwise_transfer_rows(
                    method=method_name,
                    seed=seed,
                    adapted_on=target,
                    initial=initial_holdout,
                    adapted=cross_holdout,
                )
                result["initial_holdout"] = initial_holdout
                result["cross_generator_holdout"] = cross_holdout
                result["holdout_matrix"] = target_rows
                all_pairwise_rows.extend(target_rows)
            destination = output_root / method_name / target.replace("/", "_")
            save_evaluation(result, destination)
            if initial_holdout is not None:
                aggregate[method_name]["targets"][target] = {
                    "online": result["summary"],
                    "post_adaptation": result["cross_generator_holdout"],
                }
            else:
                aggregate[method_name][target] = result["summary"]
            print(
                f"method={method_name:>8s} target={target:<28s} "
                f"auc={result['summary']['overall']['auc']:.5f}"
            )
            # Loading the next target constructs a fresh ViT-L/14 method. Do
            # not retain the finished method, source Fisher tensors, or CUDA
            # allocator blocks while that new model is being created.
            del result
            del loader
            del method
            release_method_resources()

        if evaluate_cross_generators:
            method_rows = [
                row for row in all_pairwise_rows if row["method"] == method_name
            ]
            aggregate[method_name]["pairwise_transfer"] = summarize_pairwise_transfer(
                method_rows
            )

    if evaluate_cross_generators:
        write_csv(output_root / "pairwise_transfer_matrix.csv", all_pairwise_rows)
    write_json(output_root / "single_target_summary.json", aggregate)


if __name__ == "__main__":
    main()
