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
                for seed, offset in enumerate((0.0, 0.01, 0.02)):
                    _write_seed(
                        dataset_root,
                        f"seed{seed}",
                        dataset=dataset,
                        source_ft_auc=0.60 + dataset_index * 0.01 + offset,
                        sar_auc=0.70 + dataset_index * 0.01 + offset,
                        frozen_clip_auc=0.80 + dataset_index * 0.01 + offset,
                        tda_auc=0.75 + dataset_index * 0.01 + offset,
                        iapl_auc=0.80 + dataset_index * 0.01 + offset,
                    )

            summary = SUMMARY_SCRIPT.aggregate_results(dataset_roots)

            genimage_source = summary["datasets"]["genimage"]["aggregate"]["source_ft"]
            self.assertAlmostEqual(genimage_source["auc"]["mean"], 0.613)
            self.assertGreater(genimage_source["auc"]["std"], 0.0)
            biggan_source = summary["datasets"]["genimage"][
                "per_target_aggregate"
            ]["source_ft"]["BigGAN"]
            self.assertAlmostEqual(biggan_source["auc"]["mean"], 0.61)
            self.assertAlmostEqual(biggan_source["accuracy"]["mean"], 0.51)

            table = SUMMARY_SCRIPT.render_latex_table(summary)
            self.assertIn("sole pretrained model", table)
            self.assertIn("Source-trained CLIP detector", table)
            self.assertIn("CLIP-native", table)
            self.assertIn("\\textbf{71.30 $\\pm$ 1.00}", table)
            self.assertIn("\\textbf{81.30 $\\pm$ 1.00}", table)
            self.assertIn("RoTTA$^{\\ddagger}$ & -- & -- & -- & -- & --", table)
            self.assertIn("TTC$^{\\S}$ & -- & -- & -- & -- & --", table)
            self.assertIn("Ours & -- & -- & -- & -- & --", table)

            auc_table = SUMMARY_SCRIPT.render_dataset_table(
                "genimage", "auc", summary
            )
            self.assertIn("AUC (\\%) by target", auc_table)
            self.assertIn("BigGAN & ADM & GLIDE & SD v1.5", auc_table)
            self.assertIn("Source & 61.00 & 61.10", auc_table)
            self.assertIn("61.30 $\\pm$ 1.00", auc_table)
            self.assertIn("\\textbf{71.00}", auc_table)
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
            self.assertIn("61.30", rows["source_ft"]["genimage"])
            self.assertEqual(rows["rotta"]["genimage"], "")
            self.assertEqual(rows["ttc"]["mean"], "")

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
        self.assertIn("Frozen CLIP & -- & --", genimage_auc)
        self.assertIn("Tent$^{\\dagger}$", genimage_auc)
        self.assertIn("IAPL & -- & --", genimage_auc)
        self.assertIn("Result cells remain blank", genimage_auc)
        self.assertIn("fixed 0.5 decision threshold", genimage_accuracy)
        self.assertIn("DALL-E2 & SDXL & Mean", aigc_auc)
        self.assertEqual(aigc_auc.count(" & --"), len(SUMMARY_SCRIPT.TABLE_ROWS) * 18)
        for table in (genimage_auc, genimage_accuracy, aigc_auc):
            self.assertNotIn("a real photograph", table)
            self.assertNotIn("N/A", table)
            self.assertNotIn("\\textbf", table)

    def test_rejects_target_order_that_differs_from_dataset_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_roots = {}
            for dataset in SUMMARY_SCRIPT.DATASET_ORDER:
                dataset_root = root / dataset
                dataset_roots[dataset] = dataset_root
                for seed in range(3):
                    _write_seed(
                        dataset_root,
                        f"seed{seed}",
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
    reverse_targets: bool = False,
) -> None:
    summary = {}
    targets = [
        target for target, _label in SUMMARY_SCRIPT.DATASET_TARGETS[dataset]
    ]
    if reverse_targets:
        targets.reverse()
    for method, auc in (
        ("source_ft", source_ft_auc),
        ("sar", sar_auc),
        ("frozen_clip", frozen_clip_auc),
        ("tda", tda_auc),
        ("iapl", iapl_auc),
    ):
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
