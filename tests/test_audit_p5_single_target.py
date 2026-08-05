from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_p5_single_target import audit_single_target


class AuditP5SingleTargetTest(unittest.TestCase):
    def _write_run(self, root: Path, *, mismatched_manifest: bool = False) -> None:
        methods = ["source", "tent"]
        targets = ["cyclegan"]
        (root / "effective_config.json").write_text(
            json.dumps(
                {"seed": 2, "methods": methods, "data": {"targets": targets}}
            ),
            encoding="utf-8",
        )
        aggregate = {}
        for method in methods:
            destination = root / method / "cyclegan"
            destination.mkdir(parents=True)
            summary = {
                "overall": {
                    "auc": 0.75,
                    "accuracy": 0.5,
                    "balanced_accuracy": 0.5,
                    "samples": 2,
                },
                "by_domain": {
                    "cyclegan": {
                        "auc": 0.75,
                        "accuracy": 0.5,
                        "balanced_accuracy": 0.5,
                        "samples": 2,
                    }
                },
                "efficiency": {
                    "batches": 1,
                    "mean_predict_ms_per_batch": 1.0,
                    "mean_adapt_ms_per_batch": 2.0,
                    "mean_total_ms_per_batch": 3.0,
                    "peak_memory_mb": 4.0,
                },
            }
            metrics = {
                "summary": summary,
                "reproduction": {
                    "source_checkpoint": {"sha256": "checkpoint"}
                },
                "protocol": "predict_then_adapt",
                "method": method,
                "target": "cyclegan",
            }
            (destination / "metrics.json").write_text(
                json.dumps(metrics), encoding="utf-8"
            )
            (destination / "online_curve.csv").write_text(
                "batch,domain\n0,cyclegan\n", encoding="utf-8"
            )
            (destination / "batch_stats.csv").write_text(
                "batch,domain,samples\n0,cyclegan,2\n", encoding="utf-8"
            )
            sample_ids = ["a", "b"]
            if mismatched_manifest and method == "tent":
                sample_ids.reverse()
            with (destination / "sample_manifest.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["batch", "domain", "position", "sample_id"],
                )
                writer.writeheader()
                for position, sample_id in enumerate(sample_ids):
                    writer.writerow(
                        {
                            "batch": 0,
                            "domain": "cyclegan",
                            "position": position,
                            "sample_id": sample_id,
                        }
                    )
            aggregate.setdefault(method, {})["cyclegan"] = summary
        (root / "single_target_summary.json").write_text(
            json.dumps(aggregate), encoding="utf-8"
        )

    def test_accepts_complete_equal_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root)
            report = audit_single_target(
                root,
                seed=2,
                methods=("source", "tent"),
                expected_samples={"cyclegan": 2},
                checkpoint_sha256="checkpoint",
            )
        self.assertEqual(report["status"], "completed_and_audited")
        self.assertTrue(report["all_exact_manifest_equality"])

    def test_rejects_manifest_order_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root, mismatched_manifest=True)
            report = audit_single_target(
                root,
                seed=2,
                methods=("source", "tent"),
                expected_samples={"cyclegan": 2},
                checkpoint_sha256="checkpoint",
            )
        self.assertEqual(report["status"], "audit_failed")
        self.assertIn(
            "ordered sample manifests differ across methods for cyclegan",
            report["errors"],
        )


if __name__ == "__main__":
    unittest.main()
