from __future__ import annotations

import argparse
from pathlib import Path

from src.cli.common import (
    build_fresh_method,
    resolve_device,
    seed_everything,
    write_json,
)
from src.config import load_config, require
from src.data import build_domain_loader, concatenate_domain_streams
from src.data.streams import as_stream
from src.evaluation import (
    OnlineEvaluator,
    continual_forgetting,
    evaluate_without_adaptation,
    save_evaluation,
    temporal_generalization,
)

HOLDOUT_LOADER_SEED_OFFSET = 1_000_000
HOLDOUT_EVALUATION_SEED_OFFSET = 2_000_000


def final_holdout_stream(
    config: dict, domains: list[str], seed: int, *, transform=None
):
    data_config = config["data"]
    offset = int(data_config.get("max_samples_per_class", 0))
    limit = int(data_config.get("final_eval_max_samples_per_class", 250))
    for domain_index, domain in enumerate(domains):
        loader = build_domain_loader(
            data_config,
            domain,
            seed=seed + domain_index,
            sample_seed=seed + domain_index,
            loader_seed=seed + HOLDOUT_LOADER_SEED_OFFSET + domain_index,
            max_samples_per_class=limit,
            sample_offset_per_class=offset,
            shuffle=True,
            transform=transform,
        )
        yield from as_stream(loader, domain)


def holdout_matrix_rows(
    checkpoints: list[dict],
    domains: list[str],
    *,
    initial_evaluation: dict | None = None,
) -> list[dict]:
    rows = []
    domain_indices = {domain: index for index, domain in enumerate(domains)}
    if initial_evaluation is not None:
        for eval_domain, metrics in initial_evaluation["by_domain"].items():
            rows.append(
                {
                    "checkpoint": -1,
                    "after_domain": "method_initialization",
                    "eval_domain": eval_domain,
                    "temporal_relation": "initial",
                    **metrics,
                }
            )
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        for eval_domain, metrics in checkpoint["evaluation"]["by_domain"].items():
            eval_index = domain_indices[eval_domain]
            if eval_index < checkpoint_index:
                relation = "past"
            elif eval_index == checkpoint_index:
                relation = "current"
            else:
                relation = "future"
            rows.append(
                {
                    "checkpoint": checkpoint_index,
                    "after_domain": checkpoint["after_domain"],
                    "eval_domain": eval_domain,
                    "temporal_relation": relation,
                    **metrics,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Continual multi-generator online TTA stream")
    parser.add_argument(
        "--config", default="configs/experiments/controlled_ctta/continual_seed0.yaml"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    require(config, "data", "methods", "output_dir")
    if any(str(name).lower() != "iapl" for name in config["methods"]):
        require(config, "model")
    require(config["data"], "format", "root", "stream")

    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    domains = list(config["data"]["stream"])
    evaluation_config = config.get("evaluation", {})
    threshold = float(evaluation_config.get("threshold", 0.5))
    evaluator = OnlineEvaluator(
        threshold=threshold,
        curve_window_batches=int(evaluation_config.get("curve_window_batches", 20)),
    )
    evaluate_future_domains = bool(
        evaluation_config.get("evaluate_future_generators", False)
    )
    output_root = Path(config["output_dir"])
    write_json(output_root / "effective_config.json", config)
    aggregate = {}

    for method_name in config["methods"]:
        seed_everything(seed)
        method, _ = build_fresh_method(config, method_name, device)
        if evaluate_future_domains and getattr(
            method, "protocol_name", "predict_then_adapt"
        ) != "predict_then_adapt":
            raise ValueError(
                "Future-generator holdout evaluation requires a state-neutral "
                "Predict-Then-Adapt prediction path"
            )
        seen_domains: list[str] = []
        final_holdout_manifest: list[dict] = []
        initial_holdout = None
        if evaluate_future_domains:
            initial_holdout = evaluate_without_adaptation(
                method,
                final_holdout_stream(
                    config,
                    domains,
                    seed,
                    transform=getattr(method, "input_transform", None),
                ),
                threshold=threshold,
                evaluation_seed=seed + HOLDOUT_EVALUATION_SEED_OFFSET,
            )

        def evaluate_after_domain(current_method, completed_domain: str):
            expected_domain = domains[len(seen_domains)]
            if completed_domain != expected_domain:
                raise RuntimeError(
                    f"Unexpected stream domain boundary: {completed_domain} != {expected_domain}"
                )
            seen_domains.append(completed_domain)
            evaluation_domains = domains if evaluate_future_domains else seen_domains
            holdout = evaluate_without_adaptation(
                current_method,
                final_holdout_stream(
                    config,
                    evaluation_domains,
                    seed,
                    transform=getattr(current_method, "input_transform", None),
                ),
                threshold=threshold,
                include_manifest=True,
                evaluation_seed=seed + HOLDOUT_EVALUATION_SEED_OFFSET,
            )
            final_holdout_manifest.clear()
            final_holdout_manifest.extend(holdout.pop("sample_manifest"))
            return holdout

        result = evaluator.run(
            method,
            concatenate_domain_streams(
                config["data"],
                domains,
                seed=seed,
                transform=getattr(method, "input_transform", None),
            ),
            on_domain_end=evaluate_after_domain,
        )
        checkpoints = result.pop("domain_end_evaluations")
        if len(checkpoints) != len(domains):
            raise RuntimeError(
                f"Expected {len(domains)} domain checkpoints, got {len(checkpoints)}"
            )
        checkpoint_metrics = [checkpoint["evaluation"] for checkpoint in checkpoints]
        final = checkpoint_metrics[-1]
        final_auc = {
            domain: float(metrics["auc"]) for domain, metrics in final["by_domain"].items()
        }
        online_domain_auc = {
            domain: float(metrics["auc"])
            for domain, metrics in result["summary"]["by_domain"].items()
        }
        result["final_holdout"] = final
        result["final_holdout_manifest"] = final_holdout_manifest
        result["holdout_matrix"] = holdout_matrix_rows(
            checkpoints,
            domains,
            initial_evaluation=initial_holdout,
        )
        if initial_holdout is not None:
            result["initial_holdout"] = initial_holdout
            result["summary"]["temporal_generalization"] = temporal_generalization(
                initial_holdout,
                checkpoint_metrics,
                domains,
            )
        result["forgetting_protocol"] = {
            "definition": "fixed_holdout_best_auc_minus_final_auc",
            "evaluate_after_each_domain": True,
            "average_excludes_last_domain": True,
            "holdout_shuffle": "seeded_global",
            "loader_seed_offset": HOLDOUT_LOADER_SEED_OFFSET,
            "evaluation_seed": seed + HOLDOUT_EVALUATION_SEED_OFFSET,
            "random_state_restored_after_evaluation": True,
            "evaluate_future_generators": evaluate_future_domains,
            "future_holdouts_used_for_adaptation": False,
        }
        result["summary"]["online_average_domain_auc"] = sum(
            online_domain_auc.values()
        ) / len(online_domain_auc)
        result["summary"]["final_average_auc"] = sum(final_auc.values()) / len(final_auc)
        result["summary"]["final_pooled_auc"] = float(final["overall"]["auc"])
        forgetting = continual_forgetting(checkpoint_metrics, domains)
        result["summary"]["forgetting_by_domain"] = {
            domain: metrics["forgetting"]
            for domain, metrics in forgetting["by_domain"].items()
        }
        result["summary"]["average_forgetting"] = forgetting["average"]
        result["protocol"] = getattr(method, "protocol_name", "predict_then_adapt")
        result["stream_order"] = domains
        result["method"] = method_name
        save_evaluation(result, output_root / method_name)
        aggregate[method_name] = result["summary"]
        print(
            f"method={method_name:>8s} online_auc={result['summary']['overall']['auc']:.5f} "
            f"final_auc={final['overall']['auc']:.5f} "
            f"forgetting={result['summary']['average_forgetting']:.5f}"
        )

    write_json(output_root / "continual_summary.json", aggregate)


if __name__ == "__main__":
    main()
