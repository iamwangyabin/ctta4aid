"""Aggregate CLIP ViT-L/14 single-target runs into paper result tables."""

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
DATASET_TITLES = {
    "genimage": "GenImage",
    "aigc_detection_benchmark": "AIGCDetectionBenchmark",
    "aigi_holmes_p3": "AIGI-Holmes P3",
    "opensdid_global": "OpenSDID Global",
}
DATASET_TARGETS = {
    "genimage": (
        ("BigGAN", "BigGAN"),
        ("ADM", "ADM"),
        ("glide", "GLIDE"),
        ("stable_diffusion_v_1_5", "SD v1.5"),
        ("VQDM", "VQDM"),
        ("wukong", "Wukong"),
        ("Midjourney", "Midjourney"),
    ),
    "aigc_detection_benchmark": (
        ("ProGAN", "ProGAN"),
        ("StyleGAN", "StyleGAN"),
        ("BigGAN", "BigGAN"),
        ("CycleGAN", "CycleGAN"),
        ("StarGAN", "StarGAN"),
        ("GauGAN", "GauGAN"),
        ("StyleGAN2", "StyleGAN2"),
        ("WFIR", "WFIR"),
        ("ADM", "ADM"),
        ("GLIDE", "GLIDE"),
        ("Midjourney", "Midjourney"),
        ("SD v1.4", "SD v1.4"),
        ("SD v1.5", "SD v1.5"),
        ("VQDM", "VQDM"),
        ("Wukong", "Wukong"),
        ("DALL-E2", "DALL-E2"),
        ("SDXL", "SDXL"),
    ),
    "aigi_holmes_p3": (
        ("Janus", "Janus"),
        ("Janus-Pro-1B", "Janus-Pro-1B"),
        ("Janus-Pro-7B", "Janus-Pro-7B"),
        ("Show-o", "Show-o"),
        ("LlamaGen", "LlamaGen"),
        ("Infinity", "Infinity"),
        ("VAR", "VAR"),
        ("PixArt-XL", "PixArt-XL"),
        ("SD3.5-L", "SD3.5-L"),
        ("FLUX", "FLUX"),
    ),
    "opensdid_global": (
        ("SD1.5", "SD1.5"),
        ("SD2.1", "SD2.1"),
        ("SDXL", "SDXL"),
        ("SD3", "SD3"),
        ("Flux.1", "Flux.1"),
    ),
}
METRICS = ("auc", "average_precision", "accuracy", "balanced_accuracy")
TABLE_METRICS = ("auc", "accuracy")
METRIC_LABELS = {"auc": "AUC", "accuracy": "Accuracy"}
LOCKED_SEED_DIRS = ("seed0", "seed2", "seed3")


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
            TableRow("tent", "Tent$^{\\dagger}$", "source_trained"),
            TableRow("eata", "EATA$^{\\dagger}$", "source_trained"),
            TableRow("sar", "SAR", "source_trained"),
            TableRow("cotta", "CoTTA", "source_trained"),
            TableRow("rotta", "RoTTA-LN$^{\\ddagger}$", "source_trained"),
            TableRow("lame", "LAME", "source_trained"),
            TableRow("t2a", "T$^2$A$^{\\dagger}$", "source_trained"),
        ),
    ),
    (
        "clip_native",
        "CLIP-native (method-native text classifier)",
        (
            TableRow("frozen_clip", "Frozen CLIP", "clip_native"),
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
            TableRow("ours_static", "Ours-Static", None),
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


def _seed_summary(
    path: Path,
) -> tuple[list[str], dict[str, dict[str, dict[str, float]]]]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Invalid single-target summary: {path}")

    methods: dict[str, dict[str, dict[str, float]]] = {}
    expected_targets: list[str] | None = None
    for method, by_target in raw.items():
        if not isinstance(by_target, dict) or not by_target:
            raise ValueError(f"Missing targets for {method} in {path}")
        targets = list(by_target)
        if expected_targets is None:
            expected_targets = targets
        elif targets != expected_targets:
            raise ValueError(f"Target order differs for {method} in {path}")
        target_metrics: dict[str, dict[str, float]] = {}
        for target, result in by_target.items():
            metrics: dict[str, float] = {}
            for metric in METRICS:
                try:
                    metrics[metric] = _finite_metric(
                        result["overall"][metric],
                        context=f"{path}:{method}:{target}:{metric}",
                    )
                except (KeyError, TypeError) as error:
                    raise ValueError(
                        f"Missing {metric!r} for {method}/{target} in {path}"
                    ) from error
            target_metrics[str(target)] = metrics
        methods[str(method)] = target_metrics
    assert expected_targets is not None
    return expected_targets, methods


def _seed_experiment_identity(summary_path: Path) -> dict[str, Any] | None:
    identity_path = summary_path.parent / "experiment_identity.json"
    if identity_path.is_file():
        with identity_path.open(encoding="utf-8") as handle:
            identity = json.load(handle)
        if not isinstance(identity, dict):
            raise ValueError(f"Invalid experiment identity: {identity_path}")
        return identity

    effective_config_path = summary_path.parent / "effective_config.json"
    if not effective_config_path.is_file():
        return None
    with effective_config_path.open(encoding="utf-8") as handle:
        effective_config = json.load(handle)
    data_profile = effective_config.get("data", {}).get("bias_control_profile")
    if data_profile is not None:
        raise ValueError(
            "Bias-controlled result is missing experiment_identity.json: "
            f"{summary_path.parent}"
        )
    return {"campaign": "raw", "data_profile": "raw"}


def aggregate_dataset(
    results_root: Path, *, expected_targets: tuple[str, ...] | None = None
) -> dict[str, Any]:
    """Macro-average each target within a dataset, then aggregate three seeds."""

    summaries = sorted(results_root.glob("seed*/single_target_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No seed*/single_target_summary.json below {results_root}")
    observed_seed_dirs = tuple(path.parent.name for path in summaries)
    if observed_seed_dirs != LOCKED_SEED_DIRS:
        raise ValueError(
            f"Expected complete seeds {LOCKED_SEED_DIRS} below {results_root}; "
            f"found {observed_seed_dirs}"
        )

    per_seed: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    identities: list[dict[str, Any] | None] = []
    observed_targets: list[str] | None = None
    expected_methods: list[str] | None = None
    for summary_path in summaries:
        identities.append(_seed_experiment_identity(summary_path))
        targets, methods = _seed_summary(summary_path)
        if observed_targets is None:
            observed_targets = targets
        elif targets != observed_targets:
            raise ValueError(f"Target order differs in {summary_path}")
        method_names = list(methods)
        if expected_methods is None:
            expected_methods = method_names
        elif method_names != expected_methods:
            raise ValueError(f"Method order differs in {summary_path}")
        per_seed[summary_path.parent.name] = methods

    assert observed_targets is not None
    assert expected_methods is not None
    present_identities = [identity for identity in identities if identity is not None]
    if present_identities and len(present_identities) != len(identities):
        raise ValueError(
            f"Experiment identity is missing for some seeds below {results_root}"
        )
    experiment_identity = present_identities[0] if present_identities else None
    if any(identity != experiment_identity for identity in present_identities[1:]):
        raise ValueError(f"Experiment identities differ below {results_root}")
    if expected_targets is not None and observed_targets != list(expected_targets):
        raise ValueError(
            f"Target order below {results_root} does not match the locked dataset order"
        )

    per_target_aggregate = {
        method: {
            target: {
                metric: _mean_std(
                    [
                        per_seed[seed][method][target][metric]
                        for seed in per_seed
                    ]
                )
                for metric in METRICS
            }
            for target in observed_targets
        }
        for method in expected_methods
    }
    aggregate = {
        method: {
            metric: _mean_std(
                [
                    fmean(
                        per_seed[seed][method][target][metric]
                        for target in observed_targets
                    )
                    for seed in per_seed
                ]
            )
            for metric in METRICS
        }
        for method in expected_methods
    }
    return {
        "seeds": list(per_seed),
        "targets": observed_targets,
        "per_seed": per_seed,
        "per_target_aggregate": per_target_aggregate,
        "aggregate": aggregate,
        "experiment_identity": experiment_identity,
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
    datasets = {
        name: aggregate_dataset(
            Path(dataset_roots[name]).expanduser().resolve(),
            expected_targets=tuple(target for target, _label in DATASET_TARGETS[name]),
        )
        for name in DATASET_ORDER
    }
    identities = [datasets[name]["experiment_identity"] for name in DATASET_ORDER]
    present_identities = [identity for identity in identities if identity is not None]
    if present_identities and len(present_identities) != len(identities):
        raise ValueError("Cannot mix identified and unidentified experiment campaigns")
    experiment_identity = present_identities[0] if present_identities else None
    if any(identity != experiment_identity for identity in present_identities[1:]):
        raise ValueError("Cannot mix raw data or different bias-control profiles")
    return {
        "backbone": "OpenAI CLIP ViT-L/14",
        "metric": "macro_target_auc",
        "table_metrics": list(TABLE_METRICS),
        "experiment_identity": experiment_identity,
        "datasets": datasets,
    }


def _method_value(
    summary: Mapping[str, Any] | None,
    method: str,
    dataset: str,
    metric: str = "auc",
) -> dict[str, float] | None:
    if summary is None:
        return None
    value = summary["datasets"][dataset]["aggregate"].get(method, {}).get(metric)
    return value if isinstance(value, dict) else None


def _target_value(
    summary: Mapping[str, Any] | None,
    method: str,
    dataset: str,
    target: str,
    metric: str,
) -> dict[str, float] | None:
    if summary is None:
        return None
    value = (
        summary["datasets"][dataset]["per_target_aggregate"]
        .get(method, {})
        .get(target, {})
        .get(metric)
    )
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


def _format_mean_std_percentage(
    value: Mapping[str, float] | None, *, bold: bool = False
) -> str:
    if value is None:
        return "--"
    mean = float(value["mean"])
    std = float(value["std"])
    rendered = f"{100.0 * mean:.2f} $\\pm$ {100.0 * std:.2f}"
    return f"\\textbf{{{rendered}}}" if bold else rendered


def _format_target_metric(
    value: Mapping[str, float] | None, *, bold: bool = False
) -> str:
    if value is None:
        return "--"
    rendered = f"{100.0 * float(value['mean']):.2f}"
    return f"\\textbf{{{rendered}}}" if bold else rendered


def _best_detailed_methods(
    summary: Mapping[str, Any] | None,
    dataset: str,
    metric: str,
    rank_group: str,
    target: str | None,
) -> set[str]:
    if summary is None:
        return set()
    candidates = []
    for row in TABLE_ROWS:
        if not row.available or row.rank_group != rank_group:
            continue
        value = (
            _method_value(summary, row.method, dataset, metric)
            if target is None
            else _target_value(summary, row.method, dataset, target, metric)
        )
        if value is not None:
            candidates.append((row.method, float(value["mean"])))
    if not candidates:
        return set()
    best = max(value for _method, value in candidates)
    return {method for method, value in candidates if math.isclose(value, best)}


def detailed_table_filename(dataset: str, metric: str) -> str:
    if dataset not in DATASET_ORDER or metric not in TABLE_METRICS:
        raise ValueError(f"Unsupported detailed table: {dataset}/{metric}")
    return f"clip_vitl14_{dataset}_{metric}_table.tex"


def render_dataset_table(
    dataset: str,
    metric: str,
    summary: Mapping[str, Any] | None = None,
) -> str:
    """Render one target-wise AUC or Accuracy table for a dataset."""

    if dataset not in DATASET_ORDER:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if metric not in TABLE_METRICS:
        raise ValueError(f"Unsupported table metric: {metric}")

    targets = DATASET_TARGETS[dataset]
    metric_label = METRIC_LABELS[metric]
    total_columns = len(targets) + 2
    target_columns = "".join("c" for _target in targets)
    accuracy_note = (
        ", using a fixed 0.5 decision threshold" if metric == "accuracy" else ""
    )
    wide_table = len(targets) > 5
    font_size = "\\scriptsize" if wide_table else "\\small"
    tabcolsep = "1.5pt" if wide_table else "4pt"
    resize_width = "\\textwidth" if wide_table else "0.82\\textwidth"
    status_note = (
        " Result cells remain blank until the complete campaign is validated."
        if summary is None
        else ""
    )
    best_by_group_column = {
        (rank_group, target): _best_detailed_methods(
            summary, dataset, metric, rank_group, target
        )
        for rank_group in ("source_trained", "clip_native")
        for target in (None, *(target for target, _label in targets))
    }
    latex_newline = r"\\"
    label_dataset = dataset.replace("_", "-")
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{Target-wise {metric_label} (\\%) on {DATASET_TITLES[dataset]} "
        f"with OpenAI CLIP ViT-L/14{accuracy_note}. Target columns report "
        "means over three locked seeds; Mean reports the target-macro mean "
        f"$\\pm$ standard deviation across seeds.{status_note}}}",
        f"\\label{{tab:clip-vitl14-{label_dataset}-{metric}}}",
        font_size,
        f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}",
        "\\renewcommand{\\arraystretch}{1.05}",
        f"\\resizebox{{{resize_width}}}{{!}}{{%",
        f"\\begin{{tabular}}{{l{target_columns}c}}",
        "\\toprule",
        f"& \\multicolumn{{{len(targets)}}}{{c}}{{{metric_label} (\\%) by target}} "
        f"& \\multicolumn{{1}}{{c}}{{Summary}} {latex_newline}",
        f"\\cmidrule(lr){{2-{len(targets) + 1}}}",
        f"\\cmidrule(lr){{{len(targets) + 2}-{len(targets) + 2}}}",
        "Method & "
        + " & ".join(label for _target, label in targets)
        + " & Mean "
        + latex_newline,
        "\\midrule",
    ]
    for group_index, (_group_key, group_label, rows) in enumerate(TABLE_GROUPS):
        if group_index:
            lines.append("\\midrule")
        lines.append(
            f"\\multicolumn{{{total_columns}}}{{l}}{{\\textit{{{group_label}}}}} "
            + latex_newline
        )
        for row in rows:
            if not row.available:
                rendered_targets = ["--"] * len(targets)
                rendered_mean = "--"
            else:
                rendered_targets = []
                for target, _label in targets:
                    value = _target_value(
                        summary, row.method, dataset, target, metric
                    )
                    rendered_targets.append(
                        _format_target_metric(
                            value,
                            bold=(
                                row.rank_group is not None
                                and row.method
                                in best_by_group_column[(row.rank_group, target)]
                            ),
                        )
                    )
                mean_value = _method_value(summary, row.method, dataset, metric)
                rendered_mean = _format_mean_std_percentage(
                    mean_value,
                    bold=(
                        row.rank_group is not None
                        and row.method
                        in best_by_group_column[(row.rank_group, None)]
                    ),
                )
            lines.append(
                " & ".join([row.label, *rendered_targets, rendered_mean])
                + " "
                + latex_newline
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\vspace{2pt}",
            "\\parbox{\\textwidth}{\\footnotesize Blocks distinguish source "
            "setup; bold is restricted to the two blocks with shared starting "
            "states. $^{\\dagger}$BatchNorm parameter selection is minimally "
            "mapped to LayerNorm affine parameters. $^{\\ddagger}$RoTTA-LN "
            "explicitly replaces RobustBN with CLIP visual LayerNorm affine "
            "adaptation while retaining CSTU, teacher/student EMA, entropy loss, "
            "and the 64-instance online update schedule. It uses stream/update "
            "microbatch 2 on 24 GB GPUs, accumulating the full weighted-mean loss "
            "before one optimizer/EMA update, and is a "
            "disclosed ViT transfer, not the original RobustBN method. "
            "Target labels are used only by the evaluator.}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_latex_table(summary: Mapping[str, Any] | None = None) -> str:
    """Render the optional four-dataset AUC overview table."""

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
                rendered = ["--"] * len(DATASET_ORDER)
                mean_cell = "--"
            else:
                values = [
                    _method_value(summary, row.method, dataset)
                    for dataset in DATASET_ORDER
                ]
                rendered = [
                    _format_mean_std_percentage(
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
            "logic. $^{\\ddagger}$RoTTA-LN explicitly replaces RobustBN with CLIP "
            "visual LayerNorm affine adaptation while retaining CSTU, "
            "teacher/student EMA, entropy loss, and the online update schedule; "
            "its stream/update microbatch is 2 on 24 GB GPUs, while the full "
            "64-sample weighted-mean loss still receives one optimizer/EMA update. "
            "It is a disclosed ViT transfer rather "
            "than the original RobustBN method. Target labels are never used for "
            "prompt or hyperparameter "
            "selection.}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def write_latex_tables(
    output_dir: Path, summary: Mapping[str, Any] | None = None
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in DATASET_ORDER:
        for metric in TABLE_METRICS:
            (output_dir / detailed_table_filename(dataset, metric)).write_text(
                render_dataset_table(dataset, metric, summary), encoding="utf-8"
            )


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
                row.update({dataset: "" for dataset in DATASET_ORDER})
                row["mean"] = ""
                writer.writerow(row)
                continue
            values = []
            for dataset in DATASET_ORDER:
                value = _method_value(summary, table_row.method, dataset)
                row[dataset] = (
                    "" if value is None else _format_mean_std_percentage(value)
                )
                if value is not None:
                    values.append(float(value["mean"]))
            row["mean"] = "" if len(values) != len(DATASET_ORDER) else f"{fmean(values):.6f}"
            writer.writerow(row)
    write_latex_tables(output_dir, summary)


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
        description="Aggregate CLIP ViT-L/14 target-stream results into paper tables"
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
        write_latex_tables(output_dir)
        return
    try:
        summary = aggregate_results(_parse_dataset_arguments(args.dataset))
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    write_summary(summary, output_dir)


if __name__ == "__main__":
    main()
