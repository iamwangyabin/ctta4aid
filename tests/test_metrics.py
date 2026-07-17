import math
import unittest

import numpy as np

from online_aig_tta.evaluation.metrics import (
    MetricAccumulator,
    binary_metrics,
    continual_forgetting,
)


class MetricsTest(unittest.TestCase):
    def test_binary_metrics(self):
        result = binary_metrics([0, 0, 1, 1], [0.1, 0.4, 0.7, 0.9])
        self.assertEqual(result["auc"], 1.0)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["balanced_accuracy"], 1.0)

    def test_auc_is_nan_for_single_class_batch(self):
        result = binary_metrics([1, 1], [0.8, 0.9])
        self.assertTrue(math.isnan(result["auc"]))

    def test_accumulator_groups_domains(self):
        accumulator = MetricAccumulator()
        accumulator.update(np.array([0, 1]), np.array([0.1, 0.9]), "A")
        accumulator.update(np.array([0, 1]), np.array([0.2, 0.8]), "B")
        summary = accumulator.summary()
        self.assertEqual(summary["overall"]["samples"], 4)
        self.assertEqual(set(summary["by_domain"]), {"A", "B"})
        self.assertEqual(len(accumulator.sliding_curve(window_batches=1)), 2)

    def test_continual_forgetting_uses_fixed_holdout_history(self):
        checkpoints = [
            {"by_domain": {"A": {"auc": 0.90}}},
            {"by_domain": {"A": {"auc": 0.80}, "B": {"auc": 0.75}}},
            {
                "by_domain": {
                    "A": {"auc": 0.70},
                    "B": {"auc": 0.80},
                    "C": {"auc": 0.60},
                }
            },
        ]
        result = continual_forgetting(checkpoints, ["A", "B", "C"])
        self.assertAlmostEqual(result["by_domain"]["A"]["forgetting"], 0.20)
        self.assertAlmostEqual(result["by_domain"]["B"]["forgetting"], 0.0)
        self.assertAlmostEqual(result["by_domain"]["C"]["forgetting"], 0.0)
        self.assertAlmostEqual(result["average"], 0.10)


if __name__ == "__main__":
    unittest.main()
