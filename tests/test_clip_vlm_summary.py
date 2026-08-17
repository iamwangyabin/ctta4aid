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
    def test_aggregates_target_macro_auc_and_renders_main_table(self) -> None:
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
                        source_ft_auc=0.60 + dataset_index * 0.01 + offset,
                        sar_auc=0.70 + dataset_index * 0.01 + offset,
                        frozen_clip_auc=0.80 + dataset_index * 0.01 + offset,
                        tda_auc=0.75 + dataset_index * 0.01 + offset,
                        iapl_auc=0.80 + dataset_index * 0.01 + offset,
                    )

            summary = SUMMARY_SCRIPT.aggregate_results(dataset_roots)

            genimage_source = summary["datasets"]["genimage"]["aggregate"]["source_ft"]
            self.assertAlmostEqual(genimage_source["auc"]["mean"], 0.62)
            self.assertGreater(genimage_source["auc"]["std"], 0.0)
            table = SUMMARY_SCRIPT.render_latex_table(summary)
            self.assertIn("sole pretrained model", table)
            self.assertIn("Source-trained CLIP detector", table)
            self.assertIn("CLIP-native", table)
            self.assertIn("\\textbf{72.00 $\\pm$ 1.00}", table)
            self.assertIn("\\textbf{82.00 $\\pm$ 1.00}", table)
            self.assertIn("RoTTA$^{\\ddagger}$ & -- & -- & -- & -- & --", table)
            self.assertIn("TTC$^{\\S}$ & -- & -- & -- & -- & --", table)
            self.assertIn("Ours & -- & -- & -- & -- & --", table)

            output = root / "paper"
            SUMMARY_SCRIPT.write_summary(summary, output)
            self.assertTrue((output / "clip_vitl14_summary.json").is_file())
            self.assertTrue((output / "clip_vitl14_auc_table.csv").is_file())
            self.assertTrue((output / "clip_vitl14_main_table.tex").is_file())
            with (output / "clip_vitl14_auc_table.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = {row["method"]: row for row in csv.DictReader(handle)}
            self.assertEqual(rows["ours"]["genimage"], "")
            self.assertIn("62.00", rows["source_ft"]["genimage"])
            self.assertEqual(rows["rotta"]["genimage"], "")
            self.assertEqual(rows["ttc"]["mean"], "")

    def test_blank_template_keeps_every_result_cell_empty(self) -> None:
        table = SUMMARY_SCRIPT.render_latex_table()

        self.assertIn("Source & -- & -- & -- & -- & --", table)
        self.assertIn("Frozen CLIP & -- & -- & -- & -- & --", table)
        self.assertIn("Tent$^{\\dagger}$", table)
        self.assertIn("IAPL & -- & -- & -- & -- & --", table)
        self.assertNotIn("a real photograph", table)
        self.assertNotIn("N/A", table)
        self.assertNotIn("\\textbf", table)


def _write_seed(
    dataset_root: Path,
    seed: str,
    *,
    source_ft_auc: float,
    sar_auc: float,
    frozen_clip_auc: float,
    tda_auc: float,
    iapl_auc: float,
) -> None:
    summary = {}
    for method, auc in (
        ("source_ft", source_ft_auc),
        ("sar", sar_auc),
        ("source", frozen_clip_auc),
        ("tda", tda_auc),
        ("iapl", iapl_auc),
    ):
        summary[method] = {}
        for target_offset, target in enumerate(("A", "B")):
            value = auc + target_offset * 0.02
            summary[method][target] = {
                "overall": {
                    "auc": value,
                    "average_precision": value - 0.01,
                    "balanced_accuracy": value - 0.02,
                }
            }
    seed_root = dataset_root / seed
    seed_root.mkdir(parents=True, exist_ok=True)
    (seed_root / "single_target_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
