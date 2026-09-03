"""Aggregate CLIP ViT-L/14 single-target runs into paper result tables."""

from __future__ import annotations

import argparse
import copy
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
AIGI_DET_CALIB_METHOD = "aigi_det_calib"
AIGI_DET_CALIB_OFFICIAL_COMMIT = (
    "66d4bc606f7cf325d9bd4e67ca34b0c59d6a9d53"
)
AIGI_DET_CALIB_PROJECT_COMMIT = "ab059b650d46b3c1df92d479f1bb936ef46b6924"
AIGI_DET_CALIB_CLIP_SHA256 = (
    "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
)
AIGI_DET_CALIB_SOURCE_SHA256 = (
    "ef4b3cb3b2f83b6ffa227bb4a53af8f64151681e4d7fe3010d1c5ad9d926ae59"
)
AIGI_DET_CALIB_OURS_SHA256 = (
    "f7a351a0649b48ecb91d1121d724bf4640bde27dc4586d9c2c0d138cbf72d03f"
)
AIGI_DET_CALIB_FILE_SHA256 = {
    "run.py": "72bacc0c4a77c81c040b60b105a205fdc18cca655f83e4f4b324b495b167bb7d",
    "src/calibration.py": (
        "c179d5aee8dd577606f5382d9fdb403ed5dc1cb28c79dff39cc82d9c3613d927"
    ),
    "src/metric.py": (
        "d60a6ee7797a50491cf9157f0f000628c87c22b1136f3edd06421542c6bcdd1f"
    ),
}
AIGI_DET_CALIB_ALL_METRICS = (
    *METRICS,
    "brier_score",
    "nll",
    "ece",
)


class TableRow(NamedTuple):
    method: str
    label: str
    rank_group: str | None
    available: bool = True


RANK_GROUPS = ("baselines", "ours")

TABLE_SECTIONS = (
    (
        "baselines",
        (
            TableRow("source_ft", "Source", "baselines"),
            TableRow(
                AIGI_DET_CALIB_METHOD,
                "AIGI-Det-Calib$^{\\S}$",
                "baselines",
            ),
            TableRow("tent", "Tent$^{\\dagger}$", "baselines"),
            TableRow("eata", "EATA$^{\\dagger}$", "baselines"),
            TableRow("sar", "SAR", "baselines"),
            TableRow("cotta", "CoTTA", "baselines"),
            TableRow("rotta", "RoTTA-LN$^{\\ddagger}$", "baselines"),
            TableRow("lame", "LAME", "baselines"),
            TableRow("t2a", "T$^2$A$^{\\dagger}$", "baselines"),
            # The frozen zero-shot prompt probe remains in validated summaries
            # as a supplementary diagnostic. It has neither task-specific
            # detector training nor a test-time adaptation mechanism, so it is
            # intentionally excluded from the main paper tables.
            TableRow("tda", "TDA", "baselines"),
            TableRow("dynaprompt", "DynaPrompt", "baselines"),
            TableRow("cliptta", "CLIPTTA", "baselines"),
            TableRow("batclip", "BATCLIP", "baselines"),
            TableRow("iapl", "IAPL", "baselines"),
        ),
    ),
    (
        "ours",
        (
            TableRow("ours_static", "Ours-Static", "ours"),
            TableRow("ours", "Ours", "ours"),
        ),
    ),
)
TABLE_ROWS = tuple(row for _key, rows in TABLE_SECTIONS for row in rows)


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


def _expect_equal(actual: Any, expected: Any, *, context: str) -> None:
    if actual != expected:
        raise ValueError(f"{context}: expected {expected!r}, found {actual!r}")


def load_aigi_det_calib_campaign(results_root: Path) -> dict[str, Any]:
    """Load and strictly validate the completed external calibration campaign.

    The independent table row is the official AIGI-Det-Calib scalar correction
    applied to ``source_ft``.  The companion ``ours_static`` branch is loaded
    only to verify paired sample identities; it is never imported as a method.
    """

    results_root = Path(results_root).expanduser().resolve()
    seed_values = {"seed0": 0, "seed2": 2, "seed3": 3}
    expected_paths = {
        results_root / method / dataset / f"{seed_dir}.json"
        for method in ("source_ft", "ours_static")
        for dataset in DATASET_ORDER
        for seed_dir in LOCKED_SEED_DIRS
    }
    observed_paths = set(results_root.rglob("*.json"))
    missing = sorted(str(path) for path in expected_paths - observed_paths)
    unexpected = sorted(str(path) for path in observed_paths - expected_paths)
    if missing or unexpected:
        messages = []
        if missing:
            messages.append("missing=" + ", ".join(missing))
        if unexpected:
            messages.append("unexpected=" + ", ".join(unexpected))
        raise ValueError(
            "AIGI-Det-Calib campaign must contain exactly 24 final JSON files: "
            + "; ".join(messages)
        )

    documents: dict[str, dict[str, dict[str, Any]]] = {
        "source_ft": {},
        "ours_static": {},
    }
    target_units = 0
    max_auc_invariance_error = 0.0
    for method in documents:
        expected_checkpoint = (
            AIGI_DET_CALIB_SOURCE_SHA256
            if method == "source_ft"
            else AIGI_DET_CALIB_OURS_SHA256
        )
        for dataset in DATASET_ORDER:
            documents[method][dataset] = {}
            expected_targets = [
                target for target, _label in DATASET_TARGETS[dataset]
            ]
            for seed_dir, seed in seed_values.items():
                path = results_root / method / dataset / f"{seed_dir}.json"
                with path.open(encoding="utf-8") as handle:
                    document = json.load(handle)
                context = str(path)
                _expect_equal(
                    document.get("format"),
                    "aigi_det_calib_strict_causal_v1",
                    context=f"{context}:format",
                )
                _expect_equal(
                    document.get("status"), "complete", context=f"{context}:status"
                )
                _expect_equal(document.get("method"), method, context=f"{context}:method")
                _expect_equal(
                    document.get("dataset"), dataset, context=f"{context}:dataset"
                )
                _expect_equal(document.get("seed"), seed, context=f"{context}:seed")

                protocol = document.get("protocol", {})
                protocol_contract = {
                    "data_profile": "matched_jpeg",
                    "locked_sample_order": True,
                    "warmup_samples": 100,
                    "calibration_subset": "first_n_samples_in_locked_target_stream",
                    "calibration_labels": (
                        "constant_dummy_only; hidden labels never passed"
                    ),
                    "real_ratio": 0.5,
                    "causal_rule": (
                        "source predictions for warmup, fixed offset thereafter"
                    ),
                    "secondary_metric": "full_causal_prequential",
                    "retroactive_full": "diagnostic_only",
                }
                for key, expected in protocol_contract.items():
                    _expect_equal(
                        protocol.get(key), expected, context=f"{context}:protocol:{key}"
                    )

                identity = document.get("identity", {})
                identity_contract = {
                    "project_commit": AIGI_DET_CALIB_PROJECT_COMMIT,
                    "official_commit": AIGI_DET_CALIB_OFFICIAL_COMMIT,
                    "official_file_sha256": AIGI_DET_CALIB_FILE_SHA256,
                    "clip_checkpoint_sha256": AIGI_DET_CALIB_CLIP_SHA256,
                    "source_checkpoint_sha256": expected_checkpoint,
                }
                for key, expected in identity_contract.items():
                    _expect_equal(
                        identity.get(key), expected, context=f"{context}:identity:{key}"
                    )

                targets = document.get("targets")
                if not isinstance(targets, dict):
                    raise ValueError(f"{context}: targets must be an object")
                _expect_equal(
                    list(targets), expected_targets, context=f"{context}:target order"
                )
                for target, result in targets.items():
                    target_context = f"{context}:{target}"
                    _expect_equal(
                        result.get("samples"), 1500, context=f"{target_context}:samples"
                    )
                    _expect_equal(
                        result.get("warmup_samples"),
                        100,
                        context=f"{target_context}:warmup_samples",
                    )
                    _expect_equal(
                        result.get("heldout_samples"),
                        1400,
                        context=f"{target_context}:heldout_samples",
                    )
                    sample_digest = result.get("sample_ids_sha256")
                    if not isinstance(sample_digest, str) or len(sample_digest) != 64:
                        raise ValueError(f"{target_context}: invalid sample identity digest")
                    for slice_name, expected_samples in (
                        ("base_full", 1500),
                        ("base_heldout", 1400),
                        ("calibrated_heldout", 1400),
                        ("causal_prequential_full", 1500),
                    ):
                        slice_metrics = result.get(slice_name)
                        if not isinstance(slice_metrics, dict):
                            raise ValueError(
                                f"{target_context}: missing {slice_name}"
                            )
                        _expect_equal(
                            slice_metrics.get("samples"),
                            expected_samples,
                            context=f"{target_context}:{slice_name}:samples",
                        )
                        for metric in AIGI_DET_CALIB_ALL_METRICS:
                            _finite_metric(
                                slice_metrics.get(metric),
                                context=f"{target_context}:{slice_name}:{metric}",
                            )
                    invariance_error = _finite_metric(
                        result.get("heldout_logit_auc_invariance_error"),
                        context=f"{target_context}:heldout AUC invariance",
                    )
                    if invariance_error > 1e-12:
                        raise ValueError(
                            f"{target_context}: heldout AUC invariance error "
                            f"{invariance_error} exceeds 1e-12"
                        )
                    max_auc_invariance_error = max(
                        max_auc_invariance_error, invariance_error
                    )
                    target_units += 1
                documents[method][dataset][seed_dir] = document

    paired_units = 0
    for dataset in DATASET_ORDER:
        for seed_dir in LOCKED_SEED_DIRS:
            source_targets = documents["source_ft"][dataset][seed_dir]["targets"]
            ours_targets = documents["ours_static"][dataset][seed_dir]["targets"]
            for target, _label in DATASET_TARGETS[dataset]:
                _expect_equal(
                    source_targets[target]["sample_ids_sha256"],
                    ours_targets[target]["sample_ids_sha256"],
                    context=f"paired sample identity:{dataset}:{seed_dir}:{target}",
                )
                paired_units += 1

    return {
        "documents": documents,
        "audit": {
            "format": "aigi_det_calib_table_import_audit_v1",
            "reported_method": AIGI_DET_CALIB_METHOD,
            "reported_source": "source_ft",
            "reported_slice": "causal_prequential_full",
            "diagnostic_only": "ours_static",
            "json_files": len(observed_paths),
            "target_units": target_units,
            "paired_sample_identity_units": paired_units,
            "max_heldout_logit_auc_invariance_error": max_auc_invariance_error,
            "formal_seeds": [0, 2, 3],
            "datasets": list(DATASET_ORDER),
            "official_commit": AIGI_DET_CALIB_OFFICIAL_COMMIT,
            "project_commit": AIGI_DET_CALIB_PROJECT_COMMIT,
            "clip_checkpoint_sha256": AIGI_DET_CALIB_CLIP_SHA256,
            "source_checkpoint_sha256": AIGI_DET_CALIB_SOURCE_SHA256,
        },
    }


def add_aigi_det_calib_results(
    summary: Mapping[str, Any], campaign: Mapping[str, Any]
) -> dict[str, Any]:
    """Add the strict causal Source + AIGI-Det-Calib row to a main summary."""

    augmented = copy.deepcopy(summary)
    documents = campaign["documents"]["source_ft"]
    max_base_auc_difference = 0.0
    max_base_ap_difference = 0.0
    for dataset in DATASET_ORDER:
        dataset_summary = augmented["datasets"][dataset]
        method_per_seed: dict[str, dict[str, dict[str, float]]] = {}
        for seed_dir in LOCKED_SEED_DIRS:
            if AIGI_DET_CALIB_METHOD in dataset_summary["per_seed"][seed_dir]:
                raise ValueError(
                    f"{AIGI_DET_CALIB_METHOD} already exists in {dataset}/{seed_dir}"
                )
            document = documents[dataset][seed_dir]
            target_metrics: dict[str, dict[str, float]] = {}
            for target in dataset_summary["targets"]:
                result = document["targets"][target]
                base = result["base_full"]
                expected_base = dataset_summary["per_seed"][seed_dir]["source_ft"][
                    target
                ]
                auc_difference = abs(float(base["auc"]) - expected_base["auc"])
                ap_difference = abs(
                    float(base["average_precision"])
                    - expected_base["average_precision"]
                )
                max_base_auc_difference = max(
                    max_base_auc_difference, auc_difference
                )
                max_base_ap_difference = max(max_base_ap_difference, ap_difference)
                if auc_difference > 1e-5 or ap_difference > 1e-5:
                    raise ValueError(
                        f"AIGI-Det-Calib base scores do not match Source for "
                        f"{dataset}/{seed_dir}/{target}"
                    )
                for metric in ("accuracy", "balanced_accuracy"):
                    if not math.isclose(
                        float(base[metric]), expected_base[metric], abs_tol=1e-12
                    ):
                        raise ValueError(
                            f"AIGI-Det-Calib base {metric} does not match Source for "
                            f"{dataset}/{seed_dir}/{target}"
                        )
                causal = result["causal_prequential_full"]
                target_metrics[target] = {
                    metric: _finite_metric(
                        causal[metric],
                        context=f"AIGI-Det-Calib:{dataset}:{seed_dir}:{target}:{metric}",
                    )
                    for metric in METRICS
                }
            method_per_seed[seed_dir] = target_metrics
            dataset_summary["per_seed"][seed_dir][AIGI_DET_CALIB_METHOD] = (
                target_metrics
            )

        dataset_summary["per_target_aggregate"][AIGI_DET_CALIB_METHOD] = {
            target: {
                metric: _mean_std(
                    [method_per_seed[seed][target][metric] for seed in LOCKED_SEED_DIRS]
                )
                for metric in METRICS
            }
            for target in dataset_summary["targets"]
        }
        dataset_summary["aggregate"][AIGI_DET_CALIB_METHOD] = {
            metric: _mean_std(
                [
                    fmean(
                        method_per_seed[seed][target][metric]
                        for target in dataset_summary["targets"]
                    )
                    for seed in LOCKED_SEED_DIRS
                ]
            )
            for metric in METRICS
        }

    audit = copy.deepcopy(campaign["audit"])
    audit["max_base_full_auc_difference_from_main_source"] = (
        max_base_auc_difference
    )
    audit["max_base_full_average_precision_difference_from_main_source"] = (
        max_base_ap_difference
    )
    auxiliary_methods = augmented.setdefault("auxiliary_methods", {})
    if AIGI_DET_CALIB_METHOD in auxiliary_methods:
        raise ValueError(
            f"{AIGI_DET_CALIB_METHOD} already exists in auxiliary method metadata"
        )
    auxiliary_methods[AIGI_DET_CALIB_METHOD] = {
        "display_name": "AIGI-Det-Calib",
        "paper": "arXiv:2602.01973; AAAI 2026",
        "role": "independent threshold-calibration baseline",
        "source_method": "source_ft",
        "reported_slice": "causal_prequential_full",
        "protocol": {
            "warmup_samples": 100,
            "warmup_prediction": "unchanged source prediction",
            "heldout_samples": 1400,
            "heldout_prediction": "fixed official unsupervised scalar offset",
            "target_labels": "evaluator_only",
            "real_ratio": 0.5,
        },
        "audit": audit,
    }
    return augmented


def _aigi_target_macro(
    document: Mapping[str, Any], slice_name: str, metrics: tuple[str, ...]
) -> dict[str, float]:
    targets = document["targets"]
    return {
        metric: fmean(
            _finite_metric(
                result[slice_name][metric],
                context=(
                    f"AIGI-Det-Calib:{document['method']}:{document['dataset']}:"
                    f"seed{document['seed']}:{target}:{slice_name}:{metric}"
                ),
            )
            for target, result in targets.items()
        )
        for metric in metrics
    }


def add_aigi_det_calib_per_seed_summary(
    per_seed_summary: Mapping[str, Any], campaign: Mapping[str, Any]
) -> dict[str, Any]:
    """Add dataset-level causal metrics to the release per-seed summary."""

    augmented = copy.deepcopy(per_seed_summary)
    documents = campaign["documents"]["source_ft"]
    for dataset in DATASET_ORDER:
        for seed_dir in LOCKED_SEED_DIRS:
            methods = augmented[dataset][seed_dir]
            if AIGI_DET_CALIB_METHOD in methods:
                raise ValueError(
                    f"{AIGI_DET_CALIB_METHOD} already exists in per-seed summary "
                    f"for {dataset}/{seed_dir}"
                )
            methods[AIGI_DET_CALIB_METHOD] = _aigi_target_macro(
                documents[dataset][seed_dir],
                "causal_prequential_full",
                AIGI_DET_CALIB_ALL_METRICS,
            )
    return augmented


def add_aigi_det_calib_calibration_summary(
    calibration_summary: Mapping[str, Any], campaign: Mapping[str, Any]
) -> dict[str, Any]:
    """Add causal Brier/NLL/ECE aggregates to the release calibration file."""

    augmented = copy.deepcopy(calibration_summary)
    documents = campaign["documents"]["source_ft"]
    metrics = ("brier_score", "nll", "ece")
    for dataset in DATASET_ORDER:
        dataset_summary = augmented["datasets"][dataset]
        per_seed_values = {}
        for seed_dir in LOCKED_SEED_DIRS:
            methods = dataset_summary["per_seed"][seed_dir]
            if AIGI_DET_CALIB_METHOD in methods:
                raise ValueError(
                    f"{AIGI_DET_CALIB_METHOD} already exists in calibration "
                    f"summary for {dataset}/{seed_dir}"
                )
            values = _aigi_target_macro(
                documents[dataset][seed_dir],
                "causal_prequential_full",
                metrics,
            )
            methods[AIGI_DET_CALIB_METHOD] = values
            per_seed_values[seed_dir] = values

        dataset_summary["aggregate"][AIGI_DET_CALIB_METHOD] = {
            metric: _mean_std(
                [per_seed_values[seed][metric] for seed in LOCKED_SEED_DIRS]
            )
            for metric in metrics
        }
        dataset_summary["per_target_aggregate"][AIGI_DET_CALIB_METHOD] = {
            target: {
                metric: _mean_std(
                    [
                        _finite_metric(
                            documents[dataset][seed]["targets"][target]
                            ["causal_prequential_full"][metric],
                            context=(
                                f"AIGI-Det-Calib:{dataset}:{seed}:{target}:{metric}"
                            ),
                        )
                        for seed in LOCKED_SEED_DIRS
                    ]
                )
                for metric in metrics
            }
            for target, _label in DATASET_TARGETS[dataset]
        }
    return augmented


def summarize_aigi_det_calib_campaign(
    campaign: Mapping[str, Any]
) -> dict[str, Any]:
    """Build a compact final report for both the table row and diagnostic branch."""

    slices = (
        "base_full",
        "base_heldout",
        "calibrated_heldout",
        "causal_prequential_full",
    )
    metrics = AIGI_DET_CALIB_ALL_METRICS
    methods = {}
    for raw_method, role in (
        ("source_ft", "independent_table_baseline"),
        ("ours_static", "diagnostic_only_not_a_table_method"),
    ):
        documents = campaign["documents"][raw_method]
        datasets = {}
        for dataset in DATASET_ORDER:
            per_seed = {
                seed: {
                    slice_name: _aigi_target_macro(
                        documents[dataset][seed], slice_name, metrics
                    )
                    for slice_name in slices
                }
                for seed in LOCKED_SEED_DIRS
            }
            aggregate = {
                slice_name: {
                    metric: _mean_std(
                        [
                            per_seed[seed][slice_name][metric]
                            for seed in LOCKED_SEED_DIRS
                        ]
                    )
                    for metric in metrics
                }
                for slice_name in slices
            }
            datasets[dataset] = {"per_seed": per_seed, "aggregate": aggregate}

        overall = {}
        for slice_name in slices:
            per_seed_overall = {}
            for seed in LOCKED_SEED_DIRS:
                seed_documents = [documents[dataset][seed] for dataset in DATASET_ORDER]
                per_seed_overall[seed] = {
                    metric: fmean(
                        _finite_metric(
                            result[slice_name][metric],
                            context=(
                                f"AIGI-Det-Calib:{raw_method}:{document['dataset']}:"
                                f"{seed}:{target}:{slice_name}:{metric}"
                            ),
                        )
                        for document in seed_documents
                        for target, result in document["targets"].items()
                    )
                    for metric in metrics
                }
            overall[slice_name] = {
                metric: _mean_std(
                    [
                        per_seed_overall[seed][metric]
                        for seed in LOCKED_SEED_DIRS
                    ]
                )
                for metric in metrics
            }
        methods[raw_method] = {
            "role": role,
            "datasets": datasets,
            "overall_39_target_macro": overall,
        }

    return {
        "format": "aigi_det_calib_strict_causal_summary_v1",
        "display_name": "AIGI-Det-Calib",
        "paper": {
            "title": (
                "Your AI-Generated Image Detector Can Secretly Achieve SOTA "
                "Accuracy, If Calibrated"
            ),
            "arxiv": "2602.01973",
            "venue": "AAAI 2026",
        },
        "reported_table_method": {
            "name": AIGI_DET_CALIB_METHOD,
            "input": "source_ft",
            "slice": "causal_prequential_full",
        },
        "protocol": {
            "data_profile": "matched_jpeg",
            "samples_per_target": 1500,
            "warmup_samples": 100,
            "heldout_samples": 1400,
            "warmup_prediction": "unchanged source prediction",
            "heldout_prediction": "fixed official unsupervised scalar offset",
            "target_labels": "evaluator_only",
            "real_ratio": 0.5,
        },
        "audit": copy.deepcopy(campaign["audit"]),
        "methods": methods,
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
        for rank_group in RANK_GROUPS
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
    for section_index, (_section_key, rows) in enumerate(TABLE_SECTIONS):
        if section_index:
            lines.append("\\midrule")
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
            "\\parbox{\\textwidth}{\\footnotesize The single rule separates "
            "prior methods from our paired static and adaptive models. Bold "
            "marks the best result on each side of the rule. "
            "$^{\\dagger}$BN-to-LN parameter mapping; $^{\\ddagger}$RoTTA-LN "
            "ViT transfer; $^{\\S}$strictly causal AIGI-Det-Calib. Full source "
            "setups and transfer details appear in "
            "Section~\\ref{sec:experiments}.}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_latex_table(summary: Mapping[str, Any] | None = None) -> str:
    """Render the optional four-dataset AUC overview table."""

    best_by_group_dataset = {
        (rank_group, dataset): _best_methods(summary, dataset, rank_group)
        for rank_group in RANK_GROUPS
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
    for section_index, (_section_key, rows) in enumerate(TABLE_SECTIONS):
        if section_index:
            lines.append("\\midrule")
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
            "The single rule separates prior methods from our paired static and "
            "adaptive models. Bold marks the best result on each side of the "
            "rule. $^{\\dagger}$BN-to-LN parameter mapping; "
            "$^{\\ddagger}$RoTTA-LN ViT transfer; $^{\\S}$strictly causal "
            "AIGI-Det-Calib. Full source setups and transfer details appear in "
            "Section~\\ref{sec:experiments}.}",
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
            lineterminator="\n",
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
        "--base-summary",
        type=Path,
        help=(
            "Existing validated clip_vitl14_summary.json to augment instead of "
            "re-aggregating dataset run directories"
        ),
    )
    parser.add_argument(
        "--aigi-det-calib-results",
        type=Path,
        help=(
            "Root containing the validated source_ft and ours_static strict-causal "
            "AIGI-Det-Calib JSON files"
        ),
    )
    parser.add_argument(
        "--template-only",
        action="store_true",
        help="Write the blank LaTeX table without requiring completed runs",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if args.template_only:
        if args.dataset or args.base_summary or args.aigi_det_calib_results:
            parser.error(
                "--template-only cannot be combined with result input arguments"
            )
        write_latex_tables(output_dir)
        return
    try:
        if args.base_summary:
            if args.dataset:
                raise ValueError("--base-summary cannot be combined with --dataset")
            base_summary = args.base_summary.expanduser().resolve()
            if base_summary.parent == output_dir:
                raise ValueError(
                    "Refusing to overwrite the validated release containing "
                    "--base-summary; choose a new --output-dir"
                )
            with base_summary.open(encoding="utf-8") as handle:
                summary = json.load(handle)
        else:
            summary = aggregate_results(_parse_dataset_arguments(args.dataset))
        audit = None
        if args.aigi_det_calib_results:
            campaign = load_aigi_det_calib_campaign(args.aigi_det_calib_results)
            summary = add_aigi_det_calib_results(summary, campaign)
            audit = summary["auxiliary_methods"][AIGI_DET_CALIB_METHOD]["audit"]
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    write_summary(summary, output_dir)
    if audit is not None:
        (output_dir / "aigi_det_calib_audit.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
