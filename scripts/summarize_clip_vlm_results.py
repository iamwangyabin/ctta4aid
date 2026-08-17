"""Aggregate CLIP ViT-L/14 single-target runs into the paper's main table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping


DATASET_ORDER = (
    "genimage",
    "aigc_detection_benchmark",
    "aigi_holmes_p3",
    "opensdid_global",
)
DATASET_LABELS = {
    "genimage": "GenImage (7)",
    "aigc_detection_benchmark": "AIGC DB (17)",
    "aigi_holmes_p3": "AIGI-Holmes P3 (10)",
    "opensdid_global": "OpenSDID (5)",
}
METRICS = ("auc", "average_precision", "balanced_accuracy")
TABLE_ROWS = (
    ("source", "Frozen CLIP", "CLIP", "P only", True),
    ("tent_ln", "Tent-LN$^{\\ddagger}$", "CLIP", "P$\\rightarrow$A", True),
    ("sar", "SAR", "CLIP", "P$\\rightarrow$A", True),
    ("lame", "LAME", "CLIP", "A$\\rightarrow$P", True),
    ("tda", "TDA", "CLIP", "A$\\rightarrow$P", True),
    ("dynaprompt", "DynaPrompt", "CLIP", "A$\\rightarrow$P", True),
    ("cliptta", "CLIPTTA", "CLIP", "A$\\rightarrow$P", True),
    ("batclip", "BATCLIP", "CLIP", "P$\\rightarrow$A", True),
    ("iapl", "IAPL$^{\\dagger}$", "IAPL", "A$\\rightarrow$P", False),
    ("ours", "Ours", "CLIP", "TBD", True),
)


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": fmean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
    }


def _finite_metric(value: Any, *, context: str) -> float:
    try:
        metric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid metric in {context}") from error
    if not math.isfinite(metric):
        raise ValueError(f"Non-finite metric in {context}")
    return metric


def _seed_summary(path: Path) -> tuple[list[str], dict[str, dict[str, float]]]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid single-target summary: {path}")

    methods: dict[str, dict[str, float]] = {}
    expected_targets: list[str] | None = None
    for method, by_target in raw.items():
        if not isinstance(by_target, dict) or not by_target:
            raise ValueError(f"Missing targets for {method} in {path}")
        targets = list(by_target)
        if expected_targets is None:
            expected_targets = targets
        elif targets != expected_targets:
            raise ValueError(f"Target order differs for {method} in {path}")
        method_metrics: dict[str, float] = {}
        for metric in METRICS:
            values = []
            for target, result in by_target.items():
                try:
                    values.append(
                        _finite_metric(
                            result["overall"][metric],
                            context=f"{path}:{method}:{target}:{metric}",
                        )
                    )
                except (KeyError, TypeError) as error:
                    raise ValueError(
                        f"Missing {metric!r} for {method}/{target} in {path}"
                    ) from error
            method_metrics[metric] = fmean(values)
        methods[str(method)] = method_metrics
    assert expected_targets is not None
    return expected_targets, methods


def aggregate_dataset(results_root: Path) -> dict[str, Any]:
    """Macro-average each target within a dataset, then aggregate three seeds."""

    summaries = sorted(results_root.glob("seed*/single_target_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No seed*/single_target_summary.json below {results_root}")

    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    expected_targets: list[str] | None = None
    expected_methods: list[str] | None = None
    for summary_path in summaries:
        targets, methods = _seed_summary(summary_path)
        if expected_targets is None:
            expected_targets = targets
        elif targets != expected_targets:
            raise ValueError(f"Target order differs in {summary_path}")
        method_names = list(methods)
        if expected_methods is None:
            expected_methods = method_names
        elif method_names != expected_methods:
            raise ValueError(f"Method order differs in {summary_path}")
        per_seed[summary_path.parent.name] = methods

    assert expected_targets is not None
    assert expected_methods is not None
    aggregate = {
        method: {
            metric: _mean_std([per_seed[seed][method][metric] for seed in per_seed])
            for metric in METRICS
        }
        for method in expected_methods
    }
    return {
        "seeds": list(per_seed),
        "targets": expected_targets,
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def aggregate_results(dataset_roots: Mapping[str, Path]) -> dict[str, Any]:
    missing = [name for name in DATASET_ORDER if name not in dataset_roots]
    unexpected = [name for name in dataset_roots if name not in DATASET_ORDER]
    if missing or unexpected:
        messages = []
        if missing:
            messages.append("missing=" + ", ".join(missing))
        if unexpected:
            messages.append("unexpected=" + ", ".join(unexpected))
        raise ValueError("Dataset roots must match the CLIP main-table set: " + "; ".join(messages))
    return {
        "backbone": "OpenAI CLIP ViT-L/14",
        "metric": "macro_target_auc",
        "datasets": {
            name: aggregate_dataset(Path(dataset_roots[name]).expanduser().resolve())
            for name in DATASET_ORDER
        },
    }


def _method_value(
    summary: Mapping[str, Any] | None, method: str, dataset: str
) -> dict[str, float] | None:
    if summary is None:
        return None
    value = summary["datasets"][dataset]["aggregate"].get(method, {}).get("auc")
    return value if isinstance(value, dict) else None


def _best_methods(summary: Mapping[str, Any] | None, dataset: str) -> set[str]:
    if summary is None:
        return set()
    candidates = []
    for method, _label, _source, _protocol, same_source in TABLE_ROWS:
        if not same_source:
            continue
        value = _method_value(summary, method, dataset)
        if value is not None:
            candidates.append((method, float(value["mean"])))
    if not candidates:
        return set()
    best = max(value for _method, value in candidates)
    return {method for method, value in candidates if math.isclose(value, best)}


def _format_auc(value: Mapping[str, float] | None, *, bold: bool = False) -> str:
    if value is None:
        return "--"
    mean = float(value["mean"])
    std = float(value["std"])
    rendered = f"{100.0 * mean:.2f} $\\pm$ {100.0 * std:.2f}"
    return f"\\textbf{{{rendered}}}" if bold else rendered


def render_latex_table(summary: Mapping[str, Any] | None = None) -> str:
    """Render the blank template or a filled paper-ready main table."""

    best_by_dataset = {
        dataset: _best_methods(summary, dataset) for dataset in DATASET_ORDER
    }
    latex_newline = r"\\"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Method-native online AUC (\\%, mean $\\pm$ std over three "
        "locked target-stream seeds) with OpenAI CLIP ViT-L/14. Each dataset "
        "score first macro-averages its target generators.}",
        "\\label{tab:clip-vitl14-main}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lccrrrrr}",
        "\\toprule",
        "Method & Source & Online & GenImage & AIGC DB & AIGI-Holmes P3 & OpenSDID & Mean "
        + latex_newline,
        "\\midrule",
    ]
    for method, label, source, protocol, same_source in TABLE_ROWS:
        values = [
            _method_value(summary, method, dataset) for dataset in DATASET_ORDER
        ]
        rendered = [
            _format_auc(
                value,
                bold=same_source and method in best_by_dataset[dataset],
            )
            for dataset, value in zip(DATASET_ORDER, values, strict=True)
        ]
        if all(value is not None for value in values):
            mean_value = _mean_std([float(value["mean"]) for value in values if value])
            mean_cell = _format_auc(mean_value)
        else:
            mean_cell = "--"
        row = " & ".join([label, source, protocol, *rendered, mean_cell])
        lines.append(row + " " + latex_newline)
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{2pt}",
            "\\parbox{\\textwidth}{\\footnotesize "
            "All non-IAPL rows start from the same frozen OpenAI CLIP checkpoint "
            "and real/fake class labels; DynaPrompt retains its native context "
            "updates. P$\\rightarrow$A reports the "
            "pre-update prediction; A$\\rightarrow$P uses the current input during "
            "the method's native adaptation before its reported prediction. "
            "Native batch/view contracts are retained. $^{\\dagger}$IAPL uses its "
            "authors' task checkpoint and is therefore not a same-source ranking "
            "row. $^{\\ddagger}$Tent-LN uses the LayerNorm-capable Tent path in the "
            "SAR release rather than claiming a BatchNorm-only Tent reproduction.}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def write_summary(summary: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clip_vitl14_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (output_dir / "clip_vitl14_auc_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", *DATASET_ORDER, "mean"],
        )
        writer.writeheader()
        for method, _label, _source, _protocol, _same_source in TABLE_ROWS:
            row = {"method": method}
            values = []
            for dataset in DATASET_ORDER:
                value = _method_value(summary, method, dataset)
                row[dataset] = "" if value is None else _format_auc(value)
                if value is not None:
                    values.append(float(value["mean"]))
            row["mean"] = "" if len(values) != len(DATASET_ORDER) else f"{fmean(values):.6f}"
            writer.writerow(row)
    (output_dir / "clip_vitl14_main_table.tex").write_text(
        render_latex_table(summary), encoding="utf-8"
    )


def _parse_dataset_arguments(values: list[str]) -> dict[str, Path]:
    dataset_roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--dataset values must be NAME=RESULTS_ROOT")
        name, raw_path = value.split("=", maxsplit=1)
        if not name or not raw_path or name in dataset_roots:
            raise ValueError("--dataset values must have unique non-empty NAME=RESULTS_ROOT")
        dataset_roots[name] = Path(raw_path)
    return dataset_roots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate CLIP ViT-L/14 target-stream results into the paper table"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=RESULTS_ROOT",
        help="One root containing seed*/single_target_summary.json; repeat four times",
    )
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Write the blank LaTeX table without requiring completed runs",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if args.template_only:
        if args.dataset:
            parser.error("--template-only cannot be combined with --dataset")
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "clip_vitl14_main_table.tex").write_text(
            render_latex_table(), encoding="utf-8"
        )
        return
    try:
        summary = aggregate_results(_parse_dataset_arguments(args.dataset))
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    write_summary(summary, output_dir)


if __name__ == "__main__":
    main()
