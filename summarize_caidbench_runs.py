from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


SEED_DIRECTORIES = {0: "all_methods", 1: "seed1", 2: "seed2"}
TRACKS = ("genimage", "progan")


def mean_std(values: list[float]) -> dict[str, float | int]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "count": len(values),
    }


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_single(track_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_seed = {
        seed: read_json(track_root / directory / "single_target_summary.json")
        for seed, directory in SEED_DIRECTORIES.items()
    }
    methods = list(by_seed[0])
    domains = list(by_seed[0][methods[0]])
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for method in methods:
        summary[method] = {"by_domain": {}}
        seed_macro = []
        for seed in SEED_DIRECTORIES:
            seed_macro.append(
                statistics.fmean(
                    float(by_seed[seed][method][domain]["overall"]["auc"])
                    for domain in domains
                )
            )
        summary[method]["macro_average_auc"] = mean_std(seed_macro)
        for domain in domains:
            values = [
                float(by_seed[seed][method][domain]["overall"]["auc"])
                for seed in SEED_DIRECTORIES
            ]
            aggregate = mean_std(values)
            aggregate["seeds"] = {
                str(seed): value for seed, value in zip(SEED_DIRECTORIES, values)
            }
            summary[method]["by_domain"][domain] = aggregate
            rows.append({"method": method, "domain": domain, **aggregate})
        rows.append(
            {
                "method": method,
                "domain": "macro_average",
                **summary[method]["macro_average_auc"],
            }
        )
    return summary, rows


def collect_continual(
    track_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_seed = {
        seed: read_json(track_root / directory / "continual_summary.json")
        for seed, directory in SEED_DIRECTORIES.items()
    }
    metrics = (
        ("online_pooled_auc", lambda item: item["overall"]["auc"]),
        ("online_average_domain_auc", lambda item: item["online_average_domain_auc"]),
        ("final_average_auc", lambda item: item["final_average_auc"]),
        ("final_pooled_auc", lambda item: item["final_pooled_auc"]),
        ("average_forgetting", lambda item: item["average_forgetting"]),
    )
    summary: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for method in by_seed[0]:
        summary[method] = {}
        for metric, getter in metrics:
            values = [float(getter(by_seed[seed][method])) for seed in SEED_DIRECTORIES]
            aggregate = mean_std(values)
            aggregate["seeds"] = {
                str(seed): value for seed, value in zip(SEED_DIRECTORIES, values)
            }
            summary[method][metric] = aggregate
            rows.append({"method": method, "metric": metric, **aggregate})
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = [
        {key: value for key, value in row.items() if key != "seeds"} for row in rows
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate three-seed CAIDBench runs")
    parser.add_argument("--root", type=Path, default=Path("outputs/caidbench"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/caidbench/three_seed_summary")
    )
    parser.add_argument("--tracks", nargs="+", choices=TRACKS, default=TRACKS)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    payload = {
        "seeds": list(SEED_DIRECTORIES),
        "std_definition": "sample_standard_deviation",
        "tracks": {},
    }
    for track in args.tracks:
        root = args.root.expanduser().resolve() / track
        single, single_rows = collect_single(root / "single_target")
        continual, continual_rows = collect_continual(root / "continual")
        payload["tracks"][track] = {
            "single_target": single,
            "continual": continual,
        }
        write_csv(output / f"{track}_single_target.csv", single_rows)
        write_csv(output / f"{track}_continual.csv", continual_rows)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(output / "summary.json")


if __name__ == "__main__":
    main()
