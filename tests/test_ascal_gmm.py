from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ASCALGMMConfigTests(unittest.TestCase):
    def test_seed1_matched_jpeg_configs_are_isolated_and_minimal(self) -> None:
        from src.config import load_config, method_config

        datasets = (
            "genimage",
            "aigc_detection_benchmark",
            "aigi_holmes_p3",
            "opensdid_global",
        )
        removed_gates = {
            "admission_rate_floor",
            "anchor_kappa",
            "gates",
            "lambda_initial",
            "memory_capacity",
            "theta_a",
            "theta_q",
            "window_size",
        }
        for dataset in datasets:
            path = (
                PROJECT_ROOT
                / "configs/experiments/clip_vlm_bias_controlled"
                / f"matched_jpeg_ascal_gmm_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(config["methods"], ["ascal_gmm_static", "ascal_gmm"])
                self.assertEqual(config["seed"], 1)
                self.assertEqual(config["data"]["bias_control_profile"], "matched_jpeg")
                self.assertIn(
                    f"clip_vlm_bias_controlled/matched_jpeg/ascal_gmm/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, "ascal_gmm")
                self.assertTrue(removed_gates.isdisjoint(adaptive))

    def test_source_training_declares_gmm_consumers(self) -> None:
        from src.config import load_config

        config = load_config(
            PROJECT_ROOT / "configs/train/genimage_sd14_clip_vitl14_lora_ascal.yaml"
        )
        self.assertIn("ascal_gmm", config["training"]["intended_methods"])
        self.assertIn("ascal_gmm_static", config["training"]["intended_methods"])


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required to import the methods package")
class AsymmetricPosteriorTests(unittest.TestCase):
    def test_all_components_above_the_lowest_mean_are_fake(self) -> None:
        from src.methods.ascal_gmm import asymmetric_fake_posterior

        mixture = {
            "weights": [0.5, 0.25, 0.25],
            "mus": [-4.0, 0.0, 4.0],
            "sigmas": [0.4, 0.4, 0.4],
        }
        posterior = asymmetric_fake_posterior(np.array([-4.0, 0.0, 4.0]), mixture)
        self.assertLess(float(posterior[0]), 0.01)
        self.assertGreater(float(posterior[1]), 0.99)
        self.assertGreater(float(posterior[2]), 0.99)

    def test_posterior_rejects_unsorted_components(self) -> None:
        from src.methods.ascal_gmm import asymmetric_fake_posterior

        mixture = {
            "weights": [0.5, 0.5],
            "mus": [1.0, -1.0],
            "sigmas": [1.0, 1.0],
        }
        with self.assertRaises(ValueError):
            asymmetric_fake_posterior(np.array([0.0]), mixture)


@unittest.skipUnless(
    TORCH_AVAILABLE and SKLEARN_AVAILABLE,
    "PyTorch and scikit-learn are required",
)
class ASCALGMMMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        import torch.nn as nn

        cls.torch = torch
        cls.nn = nn

    def detector(self):
        torch = self.torch
        nn = self.nn

        class TinyDetector(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def forward(self, images):
                means = images.mean(dim=(2, 3))
                signal = (means[:, 0] - means[:, 1]) * 2.0
                return torch.stack((-signal, signal), dim=1)

        return TinyDetector()

    def anchors(self) -> dict:
        return {
            "temperature": 1.0,
            "real": {"mu": -4.0, "sigma": 1.0},
            "fake": {
                "weights": [0.5, 0.5],
                "mus": [2.0, 5.0],
                "sigmas": [1.0, 1.0],
            },
            "theta_a": 1.0,
            "theta_q": 0.25,
        }

    def method(self, *, adaptation_mode: str = "full"):
        from src.methods.ascal_gmm import ASCALGMM

        return ASCALGMM(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def score_batch(self, scores: np.ndarray):
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        images = self.torch.zeros(len(scores), 3, 2, 2)
        positive = scores >= 0.0
        images[positive, 0] = self.torch.from_numpy(scores[positive] / 4.0).view(
            -1, 1, 1
        )
        images[~positive, 1] = self.torch.from_numpy(-scores[~positive] / 4.0).view(
            -1, 1, 1
        )
        return images

    def test_predict_then_adapt_uses_only_past_batches(self) -> None:
        rng = np.random.default_rng(2)
        scores = np.concatenate(
            [rng.normal(-8.0, 0.25, 64), rng.normal(-2.0, 0.25, 64)]
        )
        method = self.method()
        first = method.predict(self.score_batch(scores))
        expected = 1.0 / (1.0 + np.exp(-scores))
        np.testing.assert_allclose(first.prob_fake.numpy(), expected, atol=1e-6)
        self.assertNotIn("pseudo_label", method._pending)

        stats = method.adapt(self.score_batch(scores))
        self.assertEqual(stats.selected, len(scores))
        self.assertEqual(stats.extra["prediction_mode"], "source_fallback")
        self.assertEqual(stats.extra["total_components"], 2)
        self.assertTrue(stats.extra["mixture_active"])

        calibrated = method.predict(self.score_batch(np.array([-8.0, -2.0])))
        self.assertLess(float(calibrated.prob_fake[0]), 0.1)
        self.assertGreater(float(calibrated.prob_fake[1]), 0.9)

    def test_fake_side_can_gain_multiple_components(self) -> None:
        rng = np.random.default_rng(3)
        scores = np.concatenate(
            [
                rng.normal(-8.0, 0.15, 100),
                rng.normal(-2.0, 0.15, 100),
                rng.normal(4.0, 0.15, 100),
            ]
        )
        method = self.method()
        method.predict(self.score_batch(scores))
        stats = method.adapt(self.score_batch(scores))
        self.assertEqual(stats.extra["total_components"], 3)
        self.assertEqual(stats.extra["fake_components"], 2)

        posterior = method.predict(self.score_batch(np.array([-8.0, -2.0, 4.0])))
        self.assertLess(float(posterior.prob_fake[0]), 0.1)
        self.assertGreater(float(posterior.prob_fake[1]), 0.9)
        self.assertGreater(float(posterior.prob_fake[2]), 0.9)

    def test_unimodal_history_keeps_exact_source_fallback(self) -> None:
        rng = np.random.default_rng(4)
        history = rng.normal(-4.0, 0.6, 300)
        method = self.method()
        method.predict(self.score_batch(history))
        stats = method.adapt(self.score_batch(history))
        self.assertEqual(stats.extra["total_components"], 1)
        self.assertFalse(stats.extra["mixture_active"])

        query = np.array([-3.0, 1.0])
        prediction = method.predict(self.score_batch(query))
        expected = 1.0 / (1.0 + np.exp(-query))
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)

    def test_static_control_never_collects_scores(self) -> None:
        method = self.method(adaptation_mode="static")
        images = self.score_batch(np.array([-2.0, 2.0]))
        method.predict(images)
        stats = method.adapt(images)
        self.assertEqual(stats.selected, 0)
        self.assertEqual(stats.extra["score_samples"], 0)
        self.assertFalse(stats.extra["mixture_active"])

    def test_reset_removes_target_history_and_fit(self) -> None:
        scores = np.concatenate(
            [np.linspace(-6.1, -5.9, 32), np.linspace(-1.1, -0.9, 32)]
        )
        method = self.method()
        images = self.score_batch(scores)
        method.predict(images)
        method.adapt(images)
        self.assertGreater(len(method.score_history), 0)
        method.reset()
        self.assertEqual(method.score_history, [])
        self.assertIsNone(method._mixture)

    def test_adapt_requires_matching_prediction(self) -> None:
        method = self.method()
        with self.assertRaises(RuntimeError):
            method.adapt(self.score_batch(np.array([0.0])))

    def test_method_factory_maps_static_alias(self) -> None:
        from src.methods import build_method

        method = build_method(
            "ascal_gmm_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertEqual(method.adaptation_mode, "static")


if __name__ == "__main__":
    unittest.main()
