from __future__ import annotations

import unittest

from scripts.compare_iapl_predictions import compare_predictions


class IAPLPredictionComparisonTests(unittest.TestCase):
    def test_aligns_padding_by_sampler_index(self) -> None:
        left = {
            "domain": "san",
            "world_size": 2,
            "indices": [0, 2, 1, 0],
            "labels": [0, 1, 0, 0],
            "probabilities": [0.1, 0.9, 0.2, 0.3],
        }
        right = {
            "domain": "san",
            "world_size": 1,
            "indices": [0, 1, 2],
            "labels": [0, 0, 1],
            "probabilities": [0.2, 0.4, 0.8],
        }

        report = compare_predictions(left, right, sample_ids=["a", "b", "c"])

        self.assertEqual(report["left_duplicate_indices"], 1)
        self.assertEqual(report["common_unique_indices"], 3)
        self.assertEqual(report["threshold_disagreements"], 0)
        self.assertEqual(report["by_label"]["0"]["samples"], 2)
        self.assertEqual(report["by_label"]["1"]["samples"], 1)
        self.assertAlmostEqual(report["max_absolute_probability_delta"], 0.2)
        self.assertIn(
            report["largest_probability_deltas"][0]["sample_id"], {"b", "c"}
        )

    def test_rejects_inconsistent_labels(self) -> None:
        left = {
            "domain": "crn",
            "world_size": 1,
            "indices": [0],
            "labels": [0],
            "probabilities": [0.1],
        }
        right = dict(left, labels=[1])
        with self.assertRaisesRegex(ValueError, "Labels differ"):
            compare_predictions(left, right)


if __name__ == "__main__":
    unittest.main()
