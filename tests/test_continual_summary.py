from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "summarize_continual_results.py"
SPEC = importlib.util.spec_from_file_location("continual_summary_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SUMMARY_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY_SCRIPT)


class ContinualSummaryTests(unittest.TestCase):
    def test_aggregates_online_final_and_forgetting_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_seed(root, "seed0", 0.0)
            _write_seed(root, "seed1", 0.1)
            _write_seed(root, "seed2", 0.2)

            summary = SUMMARY_SCRIPT.aggregate_results(root, "auc")

            self.assertEqual(summary["seeds"], ["seed0", "seed1", "seed2"])
            source = summary["aggregate"]["source"]
            self.assertAlmostEqual(source["online_by_domain"]["A"]["mean"], 0.7)
            self.assertAlmostEqual(source["online_mean"]["mean"], 0.8)
            self.assertAlmostEqual(source["final_holdout"]["mean"], 0.775)
            self.assertAlmostEqual(source["average_forgetting"]["mean"], 0.1)
            self.assertGreater(source["online_mean"]["std"], 0.0)

    def test_writes_metric_specific_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "runs"
            _write_seed(root, "seed0", 0.0)
            output = Path(temporary) / "summary"

            summary = SUMMARY_SCRIPT.aggregate_results(root, "accuracy")
            SUMMARY_SCRIPT.write_summary(summary, output)

            self.assertTrue((output / "continual_accuracy_summary.json").is_file())
            with (output / "continual_accuracy_table.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["method"] for row in rows], ["source", "tent"])
            self.assertIn("+/-", rows[0]["final_holdout"])


def _write_seed(root: Path, name: str, offset: float) -> None:
    seed_root = root / name
    methods = {
        "source": (0.60 + offset, 0.80 + offset, 0.50 + offset, 0.85 + offset),
        "tent": (0.55 + offset, 0.75 + offset, 0.45 + offset, 0.80 + offset),
    }
    summary = {}
    for method, (online_a, online_b, final_a, final_b) in methods.items():
        summary[method] = {
            "by_domain": {
                "A": {"auc": online_a, "accuracy": online_a},
                "B": {"auc": online_b, "accuracy": online_b},
            }
        }
        method_root = seed_root / method
        method_root.mkdir(parents=True, exist_ok=True)
        with (method_root / "holdout_matrix.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["checkpoint", "after_domain", "eval_domain", "auc", "accuracy"],
            )
            writer.writeheader()
            writer.writerows(
                [
                    {
                        "checkpoint": 0,
                        "after_domain": "A",
                        "eval_domain": "A",
                        "auc": online_a,
                        "accuracy": online_a,
                    },
                    {
                        "checkpoint": 1,
                        "after_domain": "B",
                        "eval_domain": "A",
                        "auc": final_a,
                        "accuracy": final_a,
                    },
                    {
                        "checkpoint": 1,
                        "after_domain": "B",
                        "eval_domain": "B",
                        "auc": final_b,
                        "accuracy": final_b,
                    },
                ]
            )
    seed_root.mkdir(parents=True, exist_ok=True)
    (seed_root / "continual_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )


if __name__ == "__main__":
    unittest.main()
