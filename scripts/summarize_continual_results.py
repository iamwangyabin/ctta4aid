from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


METRICS = ("auc", "accuracy")


def _float(row: dict[str, str], name: str) -> float:
    try:
        return float(row[name])
    except KeyError as error:
        raise ValueError(f"Missing {name!r} in holdout matrix") from error


def _holdout_metrics(
    matrix_path: Path, domains: list[str], metric: str
) -> tuple[float, float]:
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty holdout matrix: {matrix_path}")

    checkpoints = [int(row["checkpoint"]) for row in rows if int(row["checkpoint"]) >= 0]
    if not checkpoints:
        raise ValueError(f"No post-domain checkpoints in {matrix_path}")
    final_checkpoint = max(checkpoints)
    final_by_domain = {
        row["eval_domain"]: _float(row, metric)
        for row in rows
        if int(row["checkpoint"]) == final_checkpoint
    }
    missing = [domain for domain in domains if domain not in final_by_domain]
    if missing:
        raise ValueError(f"Final holdout is missing domains {missing} in {matrix_path}")

    forgetting = []
    for domain_index, domain in enumerate(domains[:-1]):
        history = [
            _float(row, metric)
            for row in rows
            if row["eval_domain"] == domain and int(row["checkpoint"]) >= domain_index
        ]
        if not history:
            raise ValueError(f"Missing holdout history for {domain} in {matrix_path}")
        forgetting.append(max(history) - final_by_domain[domain])
    return mean(final_by_domain[domain] for domain in domains), mean(forgetting)


def summarize_seed(seed_root: Path, metric: str) -> dict[str, Any]:
    summary_path = seed_root / "continual_summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid continual summary: {summary_path}")

    seed_summary: dict[str, Any] = {}
    for method, method_summary in raw.items():
        online = method_summary.get("by_domain")
        if not isinstance(online, dict) or not online:
            raise ValueError(f"Missing online domain metrics for {method} in {summary_path}")
        domains = list(online)
        online_by_domain = {domain: float(online[domain][metric]) for domain in domains}
        final_holdout, average_forgetting = _holdout_metrics(
            seed_root / method / "holdout_matrix.csv", domains, metric
        )
        seed_summary[method] = {
            "online_by_domain": online_by_domain,
            "online_mean": mean(online_by_domain.values()),
            "final_holdout": final_holdout,
            "average_forgetting": average_forgetting,
        }
    return seed_summary


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": stdev(values) if len(values) > 1 else 0.0}


def aggregate_results(results_root: Path, metric: str) -> dict[str, Any]:
    if metric not in METRICS:
        raise ValueError(f"Unsupported metric {metric!r}; choose from {METRICS}")
    seed_paths = sorted(results_root.glob("seed*/continual_summary.json"))
    if not seed_paths:
        raise FileNotFoundError(f"No seed*/continual_summary.json below {results_root}")

    per_seed = {
        summary_path.parent.name: summarize_seed(summary_path.parent, metric)
        for summary_path in seed_paths
    }
    methods = list(next(iter(per_seed.values())))
    domains = list(next(iter(per_seed.values()))[methods[0]]["online_by_domain"])
    for seed, values in per_seed.items():
        if list(values) != methods:
            raise ValueError(f"Method set differs in {seed}")
        for method in methods:
            if list(values[method]["online_by_domain"]) != domains:
                raise ValueError(f"Domain order differs for {method} in {seed}")

    aggregate: dict[str, Any] = {}
    for method in methods:
        aggregate[method] = {
            "online_by_domain": {
                domain: _mean_std(
                    [
                        per_seed[seed][method]["online_by_domain"][domain]
                        for seed in per_seed
                    ]
                )
                for domain in domains
            },
            "online_mean": _mean_std(
                [per_seed[seed][method]["online_mean"] for seed in per_seed]
            ),
            "final_holdout": _mean_std(
                [per_seed[seed][method]["final_holdout"] for seed in per_seed]
            ),
            "average_forgetting": _mean_std(
                [per_seed[seed][method]["average_forgetting"] for seed in per_seed]
            ),
        }
    return {
        "metric": metric,
        "seeds": list(per_seed),
        "domains": domains,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def write_summary(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric = str(summary["metric"])
    (output_dir / f"continual_{metric}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output_dir / f"continual_{metric}_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "method",
            *summary["domains"],
            "online_mean",
            "final_holdout",
            "average_forgetting",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, values in summary["aggregate"].items():
            row = {"method": method}
            row.update(
                {
                    domain: _format_mean_std(values["online_by_domain"][domain])
                    for domain in summary["domains"]
                }
            )
            row.update(
                {
                    name: _format_mean_std(values[name])
                    for name in ("online_mean", "final_holdout", "average_forgetting")
                }
            )
            writer.writerow(row)


def _format_mean_std(value: dict[str, float]) -> str:
    return f"{value['mean']:.6f} +/- {value['std']:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate three-seed continual CTTA results into paper-ready summaries"
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    args = parser.parse_args()
    for metric in args.metrics:
        write_summary(
            aggregate_results(args.results_root.expanduser().resolve(), metric),
            args.output_dir.expanduser().resolve(),
        )


if __name__ == "__main__":
    main()
