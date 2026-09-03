from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "summarize_clip_vlm_results.py"
SPEC = importlib.util.spec_from_file_location("clip_vlm_summary_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SUMMARY_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY_SCRIPT)


class ClipVlmSummaryTests(unittest.TestCase):
    def test_aggregates_target_metrics_and_renders_detailed_tables(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_roots = {}
            for dataset_index, dataset in enumerate(SUMMARY_SCRIPT.DATASET_ORDER):
                dataset_root = root / dataset
                dataset_roots[dataset] = dataset_root
                for seed_dir, offset in zip(
                    SUMMARY_SCRIPT.LOCKED_SEED_DIRS,
                    (0.0, 0.01, 0.02),
                    strict=True,
                ):
                    _write_seed(
                        dataset_root,
                        seed_dir,
                        dataset=dataset,
                        source_ft_auc=0.60 + dataset_index * 0.01 + offset,
                        sar_auc=0.70 + dataset_index * 0.01 + offset,
                        frozen_clip_auc=0.80 + dataset_index * 0.01 + offset,
                        tda_auc=0.75 + dataset_index * 0.01 + offset,
                        iapl_auc=0.80 + dataset_index * 0.01 + offset,
                        rotta_auc=0.68 + dataset_index * 0.01 + offset,
                    )

            summary = SUMMARY_SCRIPT.aggregate_results(dataset_roots)

            genimage_source = summary["datasets"]["genimage"]["aggregate"]["source_ft"]
            self.assertAlmostEqual(genimage_source["auc"]["mean"], 0.613)
            self.assertGreater(genimage_source["auc"]["std"], 0.0)
            self.assertIn(
                "frozen_clip", summary["datasets"]["genimage"]["aggregate"]
            )
            biggan_source = summary["datasets"]["genimage"][
                "per_target_aggregate"
            ]["source_ft"]["BigGAN"]
            self.assertAlmostEqual(biggan_source["auc"]["mean"], 0.61)
            self.assertAlmostEqual(biggan_source["accuracy"]["mean"], 0.51)

            table = SUMMARY_SCRIPT.render_latex_table(summary)
            self.assertIn("sole pretrained model", table)
            self.assertNotIn("Source-trained CLIP detector", table)
            self.assertNotIn("CLIP-native", table)
            self.assertNotIn("\\textbf{71.30 $\\pm$ 1.00}", table)
            self.assertIn("\\underline{76.30 $\\pm$ 1.00}", table)
            self.assertIn("\\textbf{81.30 $\\pm$ 1.00}", table)
            self.assertIn(
                "bold and underline mark the best and second-best result", table
            )
            self.assertEqual(table.count("\\midrule"), 2)
            self.assertNotIn("Frozen CLIP", table)
            self.assertIn(
                "RoTTA-LN$^{\\ddagger}$ & 69.30 $\\pm$ 1.00", table
            )
            self.assertNotIn("TTC", table)
            self.assertIn("AIGI-Det-Calib$^{\\S}$ & -- & -- & -- & -- & --", table)
            self.assertIn("Ours-Static & -- & -- & -- & -- & --", table)
            self.assertIn("Ours & -- & -- & -- & -- & --", table)

            auc_table = SUMMARY_SCRIPT.render_dataset_table(
                "genimage", "auc", summary
            )
            self.assertIn("AUC (\\%) by target", auc_table)
            self.assertIn("BigGAN & ADM & GLIDE & SD v1.5", auc_table)
            self.assertIn("Source & 61.00 & 61.10", auc_table)
            self.assertIn("61.30 $\\pm$ 1.00", auc_table)
            self.assertNotIn("\\textbf{71.00}", auc_table)
            self.assertIn("\\underline{76.00}", auc_table)
            self.assertIn("\\textbf{81.00}", auc_table)
            self.assertEqual(auc_table.count("\\midrule"), 2)
            _header, baseline_rows, ours_rows = auc_table.split("\\midrule")
            self.assertIn("IAPL", baseline_rows)
            self.assertNotIn("Ours-Static", baseline_rows)
            self.assertIn("Ours-Static", ours_rows)
            self.assertNotIn("Source-trained CLIP detector", auc_table)
            self.assertNotIn("Method-specific source training", auc_table)
            self.assertNotIn("Result cells remain blank", auc_table)

            accuracy_table = SUMMARY_SCRIPT.render_dataset_table(
                "genimage", "accuracy", summary
            )
            self.assertIn("fixed 0.5 decision threshold", accuracy_table)
            self.assertIn("Source & 51.00 & 51.10", accuracy_table)
            self.assertIn("51.30 $\\pm$ 1.00", accuracy_table)

            output = root / "paper"
            SUMMARY_SCRIPT.write_summary(summary, output)
            self.assertTrue((output / "clip_vitl14_summary.json").is_file())
            self.assertTrue((output / "clip_vitl14_auc_table.csv").is_file())
            for dataset in SUMMARY_SCRIPT.DATASET_ORDER:
                for metric in SUMMARY_SCRIPT.TABLE_METRICS:
                    self.assertTrue(
                        (
                            output
                            / SUMMARY_SCRIPT.detailed_table_filename(dataset, metric)
                        ).is_file()
                    )
            with (output / "clip_vitl14_auc_table.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = {row["method"]: row for row in csv.DictReader(handle)}
            self.assertEqual(rows["ours"]["genimage"], "")
            self.assertEqual(rows["ours_static"]["genimage"], "")
            self.assertEqual(rows["aigi_det_calib"]["genimage"], "")
            self.assertIn("61.30", rows["source_ft"]["genimage"])
            self.assertIn("69.30", rows["rotta"]["genimage"])
            self.assertNotIn("frozen_clip", rows)
            self.assertNotIn("ttc", rows)

    def test_blank_detailed_templates_keep_every_result_cell_empty(self) -> None:
        genimage_auc = SUMMARY_SCRIPT.render_dataset_table("genimage", "auc")
        genimage_accuracy = SUMMARY_SCRIPT.render_dataset_table(
            "genimage", "accuracy"
        )
        aigc_auc = SUMMARY_SCRIPT.render_dataset_table(
            "aigc_detection_benchmark", "auc"
        )

        self.assertIn(
            "Source & -- & -- & -- & -- & -- & -- & -- & --", genimage_auc
        )
        self.assertNotIn("Frozen CLIP", genimage_auc)
        self.assertIn("AIGI-Det-Calib$^{\\S}$ & -- & --", genimage_auc)
        self.assertIn("Tent$^{\\dagger}$", genimage_auc)
        self.assertIn("RoTTA-LN$^{\\ddagger}$", genimage_auc)
        self.assertIn("IAPL & -- & --", genimage_auc)
        self.assertNotIn("TTC", genimage_auc)
        self.assertIn("Result cells remain blank", genimage_auc)
        self.assertIn("fixed 0.5 decision threshold", genimage_accuracy)
        self.assertIn("DALL-E2 & SDXL & Mean", aigc_auc)
        self.assertEqual(aigc_auc.count(" & --"), len(SUMMARY_SCRIPT.TABLE_ROWS) * 18)
        for table in (genimage_auc, genimage_accuracy, aigc_auc):
            self.assertNotIn("a real photograph", table)
            self.assertNotIn("N/A", table)
            self.assertNotIn("\\textbf", table)
            self.assertNotIn("\\underline", table)

    def test_table_ranking_uses_reported_precision_and_distinct_ranks(self) -> None:
        best, second = SUMMARY_SCRIPT._top_two_methods(
            [("best_a", 0.80001), ("best_b", 0.79999), ("second", 0.75)]
        )

        self.assertEqual(best, {"best_a", "best_b"})
        self.assertEqual(second, {"second"})

    def test_rejects_target_order_that_differs_from_dataset_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_roots = {}
            for dataset in SUMMARY_SCRIPT.DATASET_ORDER:
                dataset_root = root / dataset
                dataset_roots[dataset] = dataset_root
                for seed_dir in SUMMARY_SCRIPT.LOCKED_SEED_DIRS:
                    _write_seed(
                        dataset_root,
                        seed_dir,
                        dataset=dataset,
                        source_ft_auc=0.6,
                        sar_auc=0.7,
                        frozen_clip_auc=0.8,
                        tda_auc=0.75,
                        iapl_auc=0.8,
                        reverse_targets=(dataset == "genimage"),
                    )

            with self.assertRaisesRegex(ValueError, "locked dataset order"):
                SUMMARY_SCRIPT.aggregate_results(dataset_roots)

    def test_rejects_incomplete_seed_campaign(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_seed(
                root,
                "seed0",
                dataset="genimage",
                source_ft_auc=0.6,
                sar_auc=0.7,
                frozen_clip_auc=0.8,
                tda_auc=0.75,
                iapl_auc=0.8,
            )

            with self.assertRaisesRegex(ValueError, "Expected complete seeds"):
                SUMMARY_SCRIPT.aggregate_dataset(root)

    def test_rejects_mixed_bias_control_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_roots = {}
            for dataset in SUMMARY_SCRIPT.DATASET_ORDER:
                dataset_root = root / dataset
                dataset_roots[dataset] = dataset_root
                profile = (
                    "matched_jpeg"
                    if dataset == "opensdid_global"
                    else "all_jpeg_q90"
                )
                for seed_dir in SUMMARY_SCRIPT.LOCKED_SEED_DIRS:
                    _write_seed(
                        dataset_root,
                        seed_dir,
                        dataset=dataset,
                        source_ft_auc=0.6,
                        sar_auc=0.7,
                        frozen_clip_auc=0.8,
                        tda_auc=0.75,
                        iapl_auc=0.8,
                        experiment_identity={
                            "campaign": "clip_vlm_bias_controlled",
                            "data_profile": profile,
                            "profile_spec_sha256": profile,
                        },
                    )

            with self.assertRaisesRegex(ValueError, "different bias-control profiles"):
                SUMMARY_SCRIPT.aggregate_results(dataset_roots)

    def test_imports_strict_causal_aigi_det_calib_as_source_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _write_complete_main_summary(root / "main")
            campaign_root = root / "aigi"
            _write_aigi_campaign(campaign_root, summary)

            campaign = SUMMARY_SCRIPT.load_aigi_det_calib_campaign(campaign_root)
            augmented = SUMMARY_SCRIPT.add_aigi_det_calib_results(
                summary, campaign
            )

            audit = augmented["auxiliary_methods"]["aigi_det_calib"]["audit"]
            self.assertEqual(audit["json_files"], 24)
            self.assertEqual(audit["target_units"], 234)
            self.assertEqual(audit["paired_sample_identity_units"], 117)
            self.assertEqual(audit["max_heldout_logit_auc_invariance_error"], 0.0)
            self.assertIn(
                "aigi_det_calib",
                augmented["datasets"]["genimage"]["aggregate"],
            )
            self.assertNotIn(
                "ours_static_aigi_det_calib",
                augmented["datasets"]["genimage"]["aggregate"],
            )
            self.assertAlmostEqual(
                augmented["datasets"]["genimage"]["aggregate"]
                ["aigi_det_calib"]["accuracy"]["mean"],
                0.563,
            )
            table = SUMMARY_SCRIPT.render_dataset_table(
                "genimage", "accuracy", augmented
            )
            self.assertIn("AIGI-Det-Calib$^{\\S}$ & 56.00 & 56.10", table)
            self.assertIn("strictly causal AIGI-Det-Calib", table)

            per_seed_base = {
                dataset: {
                    seed: {"source_ft": {}}
                    for seed in SUMMARY_SCRIPT.LOCKED_SEED_DIRS
                }
                for dataset in SUMMARY_SCRIPT.DATASET_ORDER
            }
            per_seed = SUMMARY_SCRIPT.add_aigi_det_calib_per_seed_summary(
                per_seed_base, campaign
            )
            self.assertAlmostEqual(
                per_seed["genimage"]["seed0"]["aigi_det_calib"]["accuracy"],
                0.553,
            )

            calibration_base = {
                "seeds": [0, 2, 3],
                "datasets": {
                    dataset: {
                        "per_seed": {
                            seed: {"source_ft": {}}
                            for seed in SUMMARY_SCRIPT.LOCKED_SEED_DIRS
                        },
                        "aggregate": {"source_ft": {}},
                        "per_target_aggregate": {"source_ft": {}},
                    }
                    for dataset in SUMMARY_SCRIPT.DATASET_ORDER
                },
            }
            calibration = (
                SUMMARY_SCRIPT.add_aigi_det_calib_calibration_summary(
                    calibration_base, campaign
                )
            )
            self.assertAlmostEqual(
                calibration["datasets"]["genimage"]["aggregate"]
                ["aigi_det_calib"]["ece"]["mean"],
                0.05,
            )
            report = SUMMARY_SCRIPT.summarize_aigi_det_calib_campaign(campaign)
            self.assertEqual(
                report["methods"]["ours_static"]["role"],
                "diagnostic_only_not_a_table_method",
            )

    def test_rejects_aigi_det_calib_campaign_that_uses_target_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            summary = _write_complete_main_summary(root / "main")
            campaign_root = root / "aigi"
            _write_aigi_campaign(campaign_root, summary)
            path = campaign_root / "source_ft" / "genimage" / "seed0.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            document["protocol"]["calibration_labels"] = "target labels"
            path.write_text(json.dumps(document), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "calibration_labels"):
                SUMMARY_SCRIPT.load_aigi_det_calib_campaign(campaign_root)


def _write_seed(
    dataset_root: Path,
    seed: str,
    *,
    dataset: str,
    source_ft_auc: float,
    sar_auc: float,
    frozen_clip_auc: float,
    tda_auc: float,
    iapl_auc: float,
    rotta_auc: float | None = None,
    reverse_targets: bool = False,
    experiment_identity: dict[str, str] | None = None,
) -> None:
    summary = {}
    targets = [
        target for target, _label in SUMMARY_SCRIPT.DATASET_TARGETS[dataset]
    ]
    if reverse_targets:
        targets.reverse()
    method_values = [
        ("source_ft", source_ft_auc),
        ("sar", sar_auc),
        ("rotta", rotta_auc),
        ("frozen_clip", frozen_clip_auc),
        ("tda", tda_auc),
        ("iapl", iapl_auc),
    ]
    for method, auc in method_values:
        if auc is None:
            continue
        summary[method] = {}
        for target_offset, target in enumerate(targets):
            value = auc + target_offset * 0.001
            summary[method][target] = {
                "overall": {
                    "auc": value,
                    "average_precision": value - 0.01,
                    "accuracy": value - 0.10,
                    "balanced_accuracy": value - 0.02,
                }
            }
    seed_root = dataset_root / seed
    seed_root.mkdir(parents=True, exist_ok=True)
    (seed_root / "single_target_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    if experiment_identity is not None:
        (seed_root / "experiment_identity.json").write_text(
            json.dumps(experiment_identity), encoding="utf-8"
        )


def _write_complete_main_summary(root: Path) -> dict:
    dataset_roots = {}
    for dataset_index, dataset in enumerate(SUMMARY_SCRIPT.DATASET_ORDER):
        dataset_root = root / dataset
        dataset_roots[dataset] = dataset_root
        for seed_dir, offset in zip(
            SUMMARY_SCRIPT.LOCKED_SEED_DIRS,
            (0.0, 0.01, 0.02),
            strict=True,
        ):
            _write_seed(
                dataset_root,
                seed_dir,
                dataset=dataset,
                source_ft_auc=0.60 + dataset_index * 0.01 + offset,
                sar_auc=0.70 + dataset_index * 0.01 + offset,
                frozen_clip_auc=0.80 + dataset_index * 0.01 + offset,
                tda_auc=0.75 + dataset_index * 0.01 + offset,
                iapl_auc=0.80 + dataset_index * 0.01 + offset,
            )
    return SUMMARY_SCRIPT.aggregate_results(dataset_roots)


def _write_aigi_campaign(root: Path, summary: dict) -> None:
    for method in ("source_ft", "ours_static"):
        checkpoint_sha256 = (
            SUMMARY_SCRIPT.AIGI_DET_CALIB_SOURCE_SHA256
            if method == "source_ft"
            else SUMMARY_SCRIPT.AIGI_DET_CALIB_OURS_SHA256
        )
        for dataset_index, dataset in enumerate(SUMMARY_SCRIPT.DATASET_ORDER):
            targets = [
                target for target, _label in SUMMARY_SCRIPT.DATASET_TARGETS[dataset]
            ]
            for seed_index, (seed_dir, seed) in enumerate(
                zip(SUMMARY_SCRIPT.LOCKED_SEED_DIRS, (0, 2, 3), strict=True)
            ):
                target_results = {}
                for target_index, target in enumerate(targets):
                    if method == "source_ft":
                        base = dict(
                            summary["datasets"][dataset]["per_seed"][seed_dir]
                            ["source_ft"][target]
                        )
                    else:
                        value = 0.7 + dataset_index * 0.01 + seed_index * 0.01
                        base = {
                            "auc": value,
                            "average_precision": value - 0.01,
                            "accuracy": value - 0.10,
                            "balanced_accuracy": value - 0.02,
                        }
                    base.update(
                        {
                            "brier_score": 0.40,
                            "nll": 1.20,
                            "ece": 0.30,
                            "samples": 1500,
                        }
                    )
                    base_heldout = dict(base)
                    base_heldout["samples"] = 1400
                    calibrated_heldout = dict(base_heldout)
                    calibrated_heldout.update(
                        {"brier_score": 0.24, "nll": 0.68, "ece": 0.04}
                    )
                    causal = {
                        "auc": base["auc"] - 0.01,
                        "average_precision": base["average_precision"] - 0.01,
                        "accuracy": base["accuracy"] + 0.05,
                        "balanced_accuracy": base["balanced_accuracy"] + 0.04,
                        "brier_score": 0.25,
                        "nll": 0.70,
                        "ece": 0.05,
                        "samples": 1500,
                    }
                    digest_value = (
                        dataset_index * 1000 + seed_index * 100 + target_index + 1
                    )
                    target_results[target] = {
                        "samples": 1500,
                        "warmup_samples": 100,
                        "heldout_samples": 1400,
                        "sample_ids_sha256": f"{digest_value:064x}",
                        "base_full": base,
                        "base_heldout": base_heldout,
                        "calibrated_heldout": calibrated_heldout,
                        "causal_prequential_full": causal,
                        "heldout_logit_auc_invariance_error": 0.0,
                    }
                document = {
                    "format": "aigi_det_calib_strict_causal_v1",
                    "status": "complete",
                    "method": method,
                    "dataset": dataset,
                    "seed": seed,
                    "protocol": {
                        "data_profile": "matched_jpeg",
                        "locked_sample_order": True,
                        "warmup_samples": 100,
                        "calibration_subset": (
                            "first_n_samples_in_locked_target_stream"
                        ),
                        "calibration_labels": (
                            "constant_dummy_only; hidden labels never passed"
                        ),
                        "real_ratio": 0.5,
                        "causal_rule": (
                            "source predictions for warmup, fixed offset thereafter"
                        ),
                        "secondary_metric": "full_causal_prequential",
                        "retroactive_full": "diagnostic_only",
                    },
                    "identity": {
                        "project_commit": (
                            SUMMARY_SCRIPT.AIGI_DET_CALIB_PROJECT_COMMIT
                        ),
                        "official_commit": (
                            SUMMARY_SCRIPT.AIGI_DET_CALIB_OFFICIAL_COMMIT
                        ),
                        "official_file_sha256": (
                            SUMMARY_SCRIPT.AIGI_DET_CALIB_FILE_SHA256
                        ),
                        "clip_checkpoint_sha256": (
                            SUMMARY_SCRIPT.AIGI_DET_CALIB_CLIP_SHA256
                        ),
                        "source_checkpoint_sha256": checkpoint_sha256,
                    },
                    "targets": target_results,
                }
                path = root / method / dataset / f"{seed_dir}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(document), encoding="utf-8")
