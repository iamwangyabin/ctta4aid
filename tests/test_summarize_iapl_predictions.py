from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_iapl_predictions import (  # noqa: E402
    average_precision,
    prediction_metrics,
)


class SummarizeIAPLPredictionsTests(unittest.TestCase):
    def test_average_precision_groups_equal_scores(self) -> None:
        self.assertAlmostEqual(
            average_precision([1, 0, 1], [0.5, 0.5, 0.1]),
            7 / 12,
        )

    def test_prediction_metrics_audits_padding_and_class_accuracy(self) -> None:
        payload = {
            "domain": "example",
            "dataset_size": 3,
            "world_size": 2,
            "indices": [0, 2, 1, 0],
            "labels": [0, 1, 0, 0],
            "probabilities": [0.1, 0.8, 0.9, 0.1],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "example.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            metrics = prediction_metrics(path)

        self.assertEqual(metrics["samples_with_distributed_padding"], 4)
        self.assertEqual(metrics["unique_sampler_indices"], 3)
        self.assertEqual(metrics["duplicate_sampler_indices"], 1)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["real_accuracy"], 2 / 3)
        self.assertEqual(metrics["fake_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
