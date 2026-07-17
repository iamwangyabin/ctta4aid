from __future__ import annotations

import argparse
from pathlib import Path

from online_aig_tta.cli.common import (
    build_fresh_method,
    resolve_device,
    seed_everything,
    write_json,
)
from online_aig_tta.config import load_config, require
from online_aig_tta.data.streams import as_stream, build_domain_loader
from online_aig_tta.evaluation import OnlineEvaluator, save_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent online TTA run per target generator")
    parser.add_argument("--config", default="configs/single_target.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    require(config, "model", "data", "methods", "output_dir")
    require(config["data"], "format", "root", "targets")

    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    evaluation_config = config.get("evaluation", {})
    evaluator = OnlineEvaluator(
        threshold=float(evaluation_config.get("threshold", 0.5)),
        curve_window_batches=int(evaluation_config.get("curve_window_batches", 20)),
    )
    output_root = Path(config["output_dir"])
    write_json(output_root / "effective_config.json", config)
    aggregate = {}

    for method_name in config["methods"]:
        aggregate[method_name] = {}
        for target_index, target in enumerate(config["data"]["targets"]):
            seed_everything(seed)
            method, _ = build_fresh_method(config, method_name, device)
            loader = build_domain_loader(
                config["data"], target, seed=seed + target_index
            )
            result = evaluator.run(method, as_stream(loader, target))
            result["protocol"] = "predict_then_adapt"
            result["method"] = method_name
            result["target"] = target
            destination = output_root / method_name / target.replace("/", "_")
            save_evaluation(result, destination)
            aggregate[method_name][target] = result["summary"]
            print(
                f"method={method_name:>8s} target={target:<28s} "
                f"auc={result['summary']['overall']['auc']:.5f}"
            )

    write_json(output_root / "single_target_summary.json", aggregate)


if __name__ == "__main__":
    main()
