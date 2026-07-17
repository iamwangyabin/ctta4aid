from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

PAPER_UFD = {
    "progan": {"acc": 1.0000, "ap": 1.0000},
    "cyclegan": {"acc": 0.9860, "ap": 0.9999},
    "biggan": {"acc": 0.9865, "ap": 0.9995},
    "stylegan": {"acc": 0.9489, "ap": 0.9975},
    "gaugan": {"acc": 0.9939, "ap": 1.0000},
    "stargan": {"acc": 0.9670, "ap": 1.0000},
    "deepfake": {"acc": 0.9589, "ap": 0.9759},
    "seeingdark": {"acc": 0.9083, "ap": 0.9727},
    "san": {"acc": 0.9384, "ap": 0.9812},
    "crn": {"acc": 0.9247, "ap": 0.9997},
    "imle": {"acc": 0.9272, "ap": 1.0000},
    "guided": {"acc": 0.7275, "ap": 0.9625},
    "ldm_200": {"acc": 0.9950, "ap": 0.9986},
    "ldm_200_cfg": {"acc": 0.9770, "ap": 0.9960},
    "ldm_100": {"acc": 0.9915, "ap": 0.9974},
    "glide_100_27": {"acc": 0.9795, "ap": 0.9940},
    "glide_50_27": {"acc": 0.9830, "ap": 0.9973},
    "glide_100_10": {"acc": 0.9835, "ap": 0.9986},
    "dalle": {"acc": 0.9890, "ap": 0.9995},
}


def parse_official_summary(text: str) -> dict[str, Any]:
    pattern = re.compile(
        r"^\((?P<index>\d+)\s+(?P<name>.+?)\)\s+acc:\s*(?P<acc>[\d.]+);\s*"
        r"ap:\s*(?P<ap>[\d.]+);\s*racc:\s*(?P<racc>[\d.]+);\s*"
        r"facc:\s*(?P<facc>[\d.]+);",
        flags=re.MULTILINE,
    )
    rows = {}
    for match in pattern.finditer(text):
        name = match.group("name").strip()
        rows[name] = {
            key: float(match.group(key)) / 100.0
            for key in ("acc", "ap", "racc", "facc")
        }
    return {"by_domain": {key: value for key, value in rows.items() if key != "mean"}}


def _prediction_metrics(path: Path) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, average_precision_score

    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = np.asarray(payload["labels"], dtype=np.int64)
    probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
    predictions = probabilities > 0.5
    real = labels == 0
    fake = labels == 1
    indices = [int(index) for index in payload["indices"]]
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "ap": float(average_precision_score(labels, probabilities)),
        "racc": float(accuracy_score(labels[real], predictions[real])),
        "facc": float(accuracy_score(labels[fake], predictions[fake])),
        "samples": int(labels.size),
        "duplicate_sampler_indices": len(indices) - len(set(indices)),
        "world_size": int(payload["world_size"]),
    }


def load_run(path: Path) -> dict[str, dict[str, Any]]:
    if path.is_dir():
        return {
            item.stem: _prediction_metrics(item)
            for item in sorted(path.glob("*.json"))
        }
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["by_domain"]
    return parse_official_summary(path.read_text(encoding="utf-8"))["by_domain"]


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be LABEL=PATH")
    label, path = value.split("=", 1)
    return label, Path(path)


def aggregate_run(run: dict[str, dict[str, Any]]) -> dict[str, Any]:
    domains = [domain for domain in PAPER_UFD if domain in run]
    metrics = {
        metric: float(np.mean([run[domain][metric] for domain in domains]))
        for metric in ("acc", "ap", "racc", "facc")
        if domains and all(metric in run[domain] for domain in domains)
    }
    paper = {
        metric: float(np.mean([PAPER_UFD[domain][metric] for domain in domains]))
        for metric in ("acc", "ap")
        if domains
    }
    return {
        "domains": domains,
        "domain_count": len(domains),
        "metrics": metrics,
        "paper_same_domains": paper,
        "delta_to_paper": {
            metric: metrics[metric] - paper[metric]
            for metric in ("acc", "ap")
            if metric in metrics and metric in paper
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=parse_run)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = {label: load_run(path) for label, path in args.run}
    domains = [domain for domain in PAPER_UFD if any(domain in run for run in runs.values())]
    report: dict[str, Any] = {
        "paper": PAPER_UFD,
        "runs": runs,
        "aggregates": {label: aggregate_run(run) for label, run in runs.items()},
        "deltas": {},
    }
    for label, run in runs.items():
        report["deltas"][label] = {
            domain: {
                metric: float(run[domain][metric]) - PAPER_UFD[domain][metric]
                for metric in ("acc", "ap")
            }
            for domain in domains
            if domain in run
        }

    header = ["domain", "paper_acc", "paper_ap"]
    for label in runs:
        header.extend([f"{label}_acc", f"{label}_ap"])
    print("\t".join(header))
    for domain in domains:
        row = [domain, f"{PAPER_UFD[domain]['acc']:.4f}", f"{PAPER_UFD[domain]['ap']:.4f}"]
        for run in runs.values():
            metrics = run.get(domain)
            row.extend(
                [f"{metrics['acc']:.4f}", f"{metrics['ap']:.4f}"]
                if metrics
                else ["", ""]
            )
        print("\t".join(row))

    print("\naggregate")
    print("run\tdomains\tacc\tap\tdelta_acc\tdelta_ap")
    for label, aggregate in report["aggregates"].items():
        metrics = aggregate["metrics"]
        deltas = aggregate["delta_to_paper"]
        print(
            "\t".join(
                [
                    label,
                    str(aggregate["domain_count"]),
                    f"{metrics.get('acc', float('nan')):.4f}",
                    f"{metrics.get('ap', float('nan')):.4f}",
                    f"{deltas.get('acc', float('nan')):+.4f}",
                    f"{deltas.get('ap', float('nan')):+.4f}",
                ]
            )
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
