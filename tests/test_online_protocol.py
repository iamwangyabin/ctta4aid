import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from src.evaluation.online_evaluator import (
    OnlineEvaluator,
    evaluate_without_adaptation,
    save_evaluation,
)
from src.types import AdaptationStats, PredictionBatch, StreamBatch
from run_continual_stream import (
    average_domain_metric,
    final_holdout_stream,
    holdout_matrix_rows,
)
from run_single_target import pairwise_transfer_rows, summarize_pairwise_transfer


class SpyMethod:
    device = "cpu"
    trainable_parameters = 1

    def __init__(self):
        self.events = []
        self.state = 0

    def predict(self, images):
        self.events.append(("predict", self.state, images.copy()))
        probabilities = np.full(len(images), 0.2 + 0.6 * self.state)
        return PredictionBatch(
            logits=np.zeros((len(images), 2)),
            prob_fake=probabilities,
            pred_label=(probabilities >= 0.5).astype(int),
        )

    def adapt(self, images):
        self.events.append(("adapt", self.state, images.copy()))
        self.state += 1
        return AdaptationStats(loss=0.1, selected=len(images))


class OnlineProtocolTest(unittest.TestCase):
    def test_primary_metric_average_weights_targets_equally(self):
        by_domain = {
            "small": {"accuracy": 1.0, "auc": 0.9, "samples": 10},
            "large": {"accuracy": 0.5, "auc": 0.7, "samples": 1000},
        }

        self.assertAlmostEqual(average_domain_metric(by_domain, "accuracy"), 0.75)
        self.assertAlmostEqual(average_domain_metric(by_domain, "auc"), 0.8)

    @unittest.skipIf(torch is None, "PyTorch is required")
    def test_save_evaluation_records_tensor_metadata_without_tensor_values(self):
        result = {
            "summary": {},
            "reproduction": {
                "source_model": {
                    "fishers": {"visual.norm.weight": torch.ones((2, 3))}
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            save_evaluation(result, Path(temporary))
            with (Path(temporary) / "metrics.json").open(encoding="utf-8") as handle:
                payload = json.load(handle)

        self.assertEqual(
            payload["reproduction"]["source_model"]["fishers"][
                "visual.norm.weight"
            ],
            {
                "type": "torch.Tensor",
                "dtype": "torch.float32",
                "shape": [2, 3],
            },
        )

    def test_predict_happens_before_adapt_and_labels_are_not_passed(self):
        method = SpyMethod()
        stream = [
            StreamBatch(
                images=np.array([10, 11]),
                hidden_labels=np.array([0, 1]),
                domain="generator_a",
                sample_ids=["a/10", "a/11"],
            ),
            StreamBatch(
                images=np.array([12, 13]),
                hidden_labels=np.array([1, 1]),
                domain="generator_a",
                sample_ids=["a/12", "a/13"],
            ),
        ]
        result = OnlineEvaluator(curve_window_batches=1).run(method, stream)
        self.assertEqual(
            [event[0] for event in method.events],
            ["predict", "adapt", "predict", "adapt"],
        )
        self.assertEqual(method.events[0][1], 0)
        self.assertEqual(method.events[2][1], 1)
        self.assertEqual(result["summary"]["overall"]["samples"], 4)
        self.assertEqual(
            result["reproduction"]["protocol_wrapper"], "predict_then_adapt"
        )
        self.assertEqual(
            [row["position"] for row in result["sample_manifest"]], [0, 1, 2, 3]
        )

    def test_domain_end_callback_runs_after_last_adaptation_in_each_domain(self):
        method = SpyMethod()
        stream = [
            StreamBatch(np.array([1]), np.array([0]), "A", ["a/1"]),
            StreamBatch(np.array([2]), np.array([1]), "B", ["b/2"]),
        ]
        callback_states = []

        def on_domain_end(current_method, domain):
            callback_states.append((domain, current_method.state))
            return {"by_domain": {}}

        result = OnlineEvaluator().run(method, stream, on_domain_end=on_domain_end)
        self.assertEqual(callback_states, [("A", 1), ("B", 2)])
        self.assertEqual(
            [row["after_domain"] for row in result["domain_end_evaluations"]],
            ["A", "B"],
        )

    def test_final_holdout_is_seeded_and_globally_shuffled(self):
        config = {
            "data": {
                "max_samples_per_class": 10,
                "final_eval_max_samples_per_class": 2,
            }
        }
        with patch(
            "run_continual_stream.build_domain_loader", return_value=[]
        ) as build_loader, patch(
            "run_continual_stream.as_stream", return_value=iter(())
        ):
            list(final_holdout_stream(config, ["A", "B"], seed=7))

        first = build_loader.call_args_list[0]
        second = build_loader.call_args_list[1]
        self.assertTrue(first.kwargs["shuffle"])
        self.assertEqual(first.kwargs["sample_seed"], 7)
        self.assertEqual(first.kwargs["loader_seed"], 1_000_007)
        self.assertEqual(second.kwargs["sample_seed"], 8)
        self.assertEqual(second.kwargs["loader_seed"], 1_000_008)

    def test_final_holdout_uses_locked_manifest_samples_without_shuffling(self):
        config = {
            "data": {
                "max_samples_per_class": 10,
                "final_eval_max_samples_per_class": 2,
            }
        }
        locked_samples = {"A": ["A/a"], "B": ["B/a"]}
        with patch(
            "run_continual_stream.build_domain_loader", return_value=[]
        ) as build_loader, patch(
            "run_continual_stream.as_stream", return_value=iter(())
        ):
            list(
                final_holdout_stream(
                    config,
                    ["A", "B"],
                    seed=7,
                    locked_samples_by_domain=locked_samples,
                )
            )

        first = build_loader.call_args_list[0]
        self.assertFalse(first.kwargs["shuffle"])
        self.assertIsNone(first.kwargs["max_samples_per_class"])
        self.assertEqual(first.kwargs["sample_offset_per_class"], 0)
        self.assertEqual(first.kwargs["locked_sample_ids"], ["A/a"])

    def test_holdout_evaluation_restores_random_state(self):
        method = SpyMethod()
        stream = [
            StreamBatch(
                images=np.array([1, 2]),
                hidden_labels=np.array([0, 1]),
                domain="A",
                sample_ids=["a/1", "a/2"],
            )
        ]
        np.random.seed(123)
        state = np.random.get_state()
        expected_next = np.random.random()
        np.random.set_state(state)

        evaluate_without_adaptation(method, stream, evaluation_seed=999)

        self.assertEqual(np.random.random(), expected_next)

    def test_holdout_matrix_marks_initial_current_past_and_future(self):
        initial = {
            "by_domain": {
                "A": {"auc": 0.5},
                "B": {"auc": 0.6},
            }
        }
        checkpoints = [
            {
                "after_domain": "A",
                "evaluation": {
                    "by_domain": {
                        "A": {"auc": 0.7},
                        "B": {"auc": 0.65},
                    }
                },
            },
            {
                "after_domain": "B",
                "evaluation": {
                    "by_domain": {
                        "A": {"auc": 0.68},
                        "B": {"auc": 0.72},
                    }
                },
            },
        ]

        rows = holdout_matrix_rows(
            checkpoints, ["A", "B"], initial_evaluation=initial
        )

        self.assertEqual(
            [row["temporal_relation"] for row in rows],
            ["initial", "initial", "current", "future", "past", "current"],
        )

    def test_pairwise_transfer_separates_current_and_cross_generator_gain(self):
        initial = {
            "by_domain": {
                "A": {
                    "auc": 0.60,
                    "accuracy": 0.55,
                    "balanced_accuracy": 0.55,
                    "samples": 10,
                },
                "B": {
                    "auc": 0.70,
                    "accuracy": 0.65,
                    "balanced_accuracy": 0.65,
                    "samples": 10,
                },
            }
        }
        adapted = {
            "by_domain": {
                "A": {
                    "auc": 0.65,
                    "accuracy": 0.60,
                    "balanced_accuracy": 0.60,
                    "samples": 10,
                },
                "B": {
                    "auc": 0.68,
                    "accuracy": 0.64,
                    "balanced_accuracy": 0.64,
                    "samples": 10,
                },
            }
        }

        rows = pairwise_transfer_rows(
            method="tent",
            seed=0,
            adapted_on="A",
            initial=initial,
            adapted=adapted,
        )
        summary = summarize_pairwise_transfer(rows)

        self.assertEqual([row["relation"] for row in rows], ["current", "cross_generator"])
        self.assertAlmostEqual(summary["mean_current_auc_delta"], 0.05)
        self.assertAlmostEqual(summary["mean_cross_generator_auc_delta"], -0.02)
        self.assertEqual(summary["cross_generator_negative_transfer_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
