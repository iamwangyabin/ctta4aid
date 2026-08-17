"""Aggregate CLIP ViT-L/14 single-target runs into the paper's main table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean, stdev
from typing import Any, Mapping, NamedTuple


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


class TableRow(NamedTuple):
    method: str
    label: str
    rank_group: str | None
    available: bool = True


TABLE_GROUPS = (
    (
        "source_trained",
        "Source-trained CLIP detector (shared source checkpoint)",
        (
            TableRow("source_ft", "Source", "source_trained"),
            TableRow("tent_ln", "Tent$^{\\dagger}$", "source_trained"),
            TableRow("eata", "EATA$^{\\dagger}$", "source_trained"),
            TableRow("sar", "SAR", "source_trained"),
            TableRow("cotta", "CoTTA$^{\\dagger}$", "source_trained"),
            TableRow("rotta", "RoTTA$^{\\ddagger}$", None, available=False),
            TableRow("lame", "LAME", "source_trained"),
            TableRow("t2a", "T$^2$A$^{\\dagger}$", "source_trained"),
        ),
    ),
    (
        "clip_native",
        "CLIP-native (method-native text classifier)",
        (
            TableRow("source", "Frozen CLIP", "clip_native"),
            TableRow("tda", "TDA", "clip_native"),
            TableRow("dynaprompt", "DynaPrompt", "clip_native"),
            TableRow("cliptta", "CLIPTTA", "clip_native"),
            TableRow("batclip", "BATCLIP", "clip_native"),
        ),
    ),
    (
        "method_specific",
        "Method-specific source training",
        (
            TableRow("iapl", "IAPL", None),
            TableRow("ttc", "TTC$^{\\S}$", None, available=False),
            TableRow("ours", "Ours", None),
        ),
    ),
)
TABLE_ROWS = tuple(row for _key, _label, rows in TABLE_GROUPS for row in rows)


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


def _best_methods(
    summary: Mapping[str, Any] | None, dataset: str, rank_group: str
) -> set[str]:
    if summary is None:
        return set()
    candidates = []
    for row in TABLE_ROWS:
        if not row.available or row.rank_group != rank_group:
            continue
        value = _method_value(summary, row.method, dataset)
        if value is not None:
            candidates.append((row.method, float(value["mean"])))
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

    best_by_group_dataset = {
        (rank_group, dataset): _best_methods(summary, dataset, rank_group)
        for rank_group in ("source_trained", "clip_native")
        for dataset in DATASET_ORDER
    }
    latex_newline = r"\\"
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Main comparison on four AI-generated image detection "
        "benchmarks using OpenAI CLIP ViT-L/14 as the sole pretrained model. "
        "Dataset cells report generator-macro AUROC (\\%, mean $\\pm$ "
        "standard deviation over three locked seeds); Mean averages the four "
        "dataset means.}",
        "\\label{tab:clip-vitl14-main}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & GenImage & AIGC DB & AIGI-Holmes P3 & OpenSDID & Mean "
        + latex_newline,
        "\\midrule",
    ]
    for group_index, (_group_key, group_label, rows) in enumerate(TABLE_GROUPS):
        if group_index:
            lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{6}}{{l}}{{\\textit{{{group_label}}}}} " + latex_newline
        )
        for row in rows:
            if not row.available:
                rendered = ["N/A"] * len(DATASET_ORDER)
                mean_cell = "N/A"
            else:
                values = [
                    _method_value(summary, row.method, dataset)
                    for dataset in DATASET_ORDER
                ]
                rendered = [
                    _format_auc(
                        value,
                        bold=(
                            row.rank_group is not None
                            and row.method
                            in best_by_group_dataset[(row.rank_group, dataset)]
                        ),
                    )
                    for dataset, value in zip(DATASET_ORDER, values, strict=True)
                ]
                if all(value is not None for value in values):
                    dataset_mean = fmean(
                        float(value["mean"]) for value in values if value
                    )
                    mean_cell = f"{100.0 * dataset_mean:.2f}"
                else:
                    mean_cell = "--"
            lines.append(
                " & ".join([row.label, *rendered, mean_cell])
                + " "
                + latex_newline
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{2pt}",
            "\\parbox{\\textwidth}{\\footnotesize "
            "Every method uses the same OpenAI CLIP ViT-L/14 pretrained "
            "initialization. Source-trained methods share one binary source "
            "checkpoint; CLIP-native methods retain their original text-classifier "
            "or prompt construction; method-specific rows retain their native "
            "source training. Batch/view contracts, state transitions, and "
            "prediction/adaptation order follow each public method. Bold marks the "
            "best result only within a shared-source block. $^{\\dagger}$The "
            "method's BatchNorm parameter selection is minimally mapped to CLIP "
            "LayerNorm affine parameters without changing its objective or online "
            "logic. $^{\\ddagger}$RoTTA is not run because robust BatchNorm is a "
            "core component and replacing it would redesign the method. "
            "$^{\\S}$TTC remains N/A until an authors' implementation can be "
            "pinned. Target labels are never used for prompt or hyperparameter "
            "selection.}",
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
        for table_row in TABLE_ROWS:
            row = {"method": table_row.method}
            if not table_row.available:
                row.update({dataset: "N/A" for dataset in DATASET_ORDER})
                row["mean"] = "N/A"
                writer.writerow(row)
                continue
            values = []
            for dataset in DATASET_ORDER:
                value = _method_value(summary, table_row.method, dataset)
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
