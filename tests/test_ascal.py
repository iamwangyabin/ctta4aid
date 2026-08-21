from __future__ import annotations

import importlib.util
import math
import unittest

import numpy as np


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None
TORCHVISION_AVAILABLE = importlib.util.find_spec("torchvision") is not None


class ScoreAnchorMathTests(unittest.TestCase):
    def test_normal_density_matches_known_values(self) -> None:
        from src.methods.ascal import gmm_density, normal_density

        self.assertAlmostEqual(
            float(normal_density(0.0, 0.0, 1.0)), 1.0 / math.sqrt(2.0 * math.pi)
        )
        blended = gmm_density(np.array([0.0]), [1.0], [0.0], [1.0])
        self.assertAlmostEqual(float(blended[0]), 1.0 / math.sqrt(2.0 * math.pi))

    def test_fit_temperature_returns_finite_grid_value(self) -> None:
        from src.methods.ascal import fit_temperature

        scores = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        temperature = fit_temperature(scores, labels)
        self.assertTrue(0.25 <= temperature <= 8.0)
        self.assertTrue(math.isfinite(temperature))

    def test_fit_temperature_requires_both_classes(self) -> None:
        from src.methods.ascal import fit_temperature

        with self.assertRaises(ValueError):
            fit_temperature(np.array([1.0, 2.0]), np.array([1, 1]))

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
    def test_fit_gmm_bic_selects_two_components_for_two_clusters(self) -> None:
        from src.methods.ascal import fit_gmm_bic

        rng = np.random.default_rng(0)
        values = np.concatenate(
            [rng.normal(-5.0, 1.0, 150), rng.normal(5.0, 1.0, 150)]
        )
        result = fit_gmm_bic(values, max_components=4, seed=0)
        self.assertEqual(result["components"], 2)
        self.assertAlmostEqual(sum(result["weights"]), 1.0, places=6)
        self.assertTrue(all(sigma > 0.0 for sigma in result["sigmas"]))

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
    def test_fit_gmm_bic_keeps_single_component_for_one_cluster(self) -> None:
        from src.methods.ascal import fit_gmm_bic

        rng = np.random.default_rng(1)
        values = rng.normal(0.0, 1.0, 300)
        result = fit_gmm_bic(values, max_components=4, seed=0)
        self.assertEqual(result["components"], 1)

    def test_validate_score_anchors_normalizes_and_rejects_bad_blocks(self) -> None:
        from src.methods.ascal import validate_score_anchors

        anchors = {
            "temperature": 1.5,
            "real": {"mu": -2.0, "sigma": 0.5},
            "fake": {"weights": [0.4, 0.6], "mus": [1.0, 3.0], "sigmas": [0.5, 0.7]},
            "theta_a": 0.1,
            "theta_q": 0.2,
        }
        normalized = validate_score_anchors(anchors)
        self.assertEqual(normalized["fake"]["mus"], [1.0, 3.0])
        with self.assertRaises(ValueError):
            validate_score_anchors(None)
        with self.assertRaises(ValueError):
            validate_score_anchors({**anchors, "temperature": 0.0})
        with self.assertRaises(ValueError):
            validate_score_anchors(
                {**anchors, "fake": {"weights": [0.9, 0.9], "mus": [1.0, 3.0], "sigmas": [0.5, 0.7]}}
            )


@unittest.skipUnless(TORCH_AVAILABLE and SKLEARN_AVAILABLE, "PyTorch and scikit-learn are required")
class ASCALMethodTests(unittest.TestCase):
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
            def __init__(self, scale: float = 2.0) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))
                self.scale = scale

            def forward(self, images):
                channel_means = images.mean(dim=(2, 3))
                signal = (channel_means[:, 0] - channel_means[:, 1]) * self.scale
                return torch.stack((-signal, signal), dim=1)

        return TinyDetector()

    def anchors(self) -> dict:
        return {
            "temperature": 1.0,
            "real": {"mu": -4.0, "sigma": 1.0},
            "fake": {"weights": [1.0], "mus": [4.0], "sigmas": [1.0]},
            "theta_a": 10.0,
            "theta_q": 0.3,
            "anchor_kappa": 100.0,
            "fuse_sigma_multiplier": 3.0,
        }

    def config(self, **overrides) -> dict:
        config = {
            "adaptation_mode": "full",
            "score_anchors": self.anchors(),
            "views": 1,
            "memory_capacity": 64,
            "window_size": 16,
            "admission_rate_floor": 0.5,
            "lambda_min_entries_per_class": 2,
            "lambda_stable_windows": 1,
            "lambda_anneal_windows": 2,
        }
        config.update(overrides)
        return config

    def batch(self, fake_count: int, real_count: int, *, strength: float = 1.0, jitter: float = 0.0):
        torch = self.torch
        total = fake_count + real_count
        images = torch.zeros(total, 3, 4, 4)
        for index in range(total):
            value = strength * (1.0 + jitter * index)
            if index < fake_count:
                images[index, 0] = value
            else:
                images[index, 1] = value
        return images

    def method(self, **overrides):
        from src.methods.ascal import ASCAL

        return ASCAL(self.detector(), "cpu", self.config(**overrides))

    def test_missing_anchors_are_rejected(self) -> None:
        from src.methods.ascal import ASCAL

        with self.assertRaises(ValueError):
            ASCAL(self.detector(), "cpu", {"adaptation_mode": "full", "views": 1})

    def test_static_mode_predicts_pure_source_and_skips_adaptation(self) -> None:
        method = self.method(adaptation_mode="static")
        images = self.batch(1, 1)
        prediction = method.predict(images)
        # The batch helper emits fake first and real second.
        expected = 1.0 / (1.0 + np.exp(-np.array([4.0, -4.0])))
        np.testing.assert_allclose(
            prediction.prob_fake.numpy(), expected, rtol=0, atol=1e-6
        )
        stats = method.adapt(images)
        self.assertEqual(stats.selected, 0)
        self.assertEqual(len(method.queue), 0)

    def test_full_mode_initial_prediction_matches_source(self) -> None:
        static = self.method(adaptation_mode="static")
        full = self.method()
        images = self.batch(2, 2)
        np.testing.assert_allclose(
            full.predict(images).prob_fake.numpy(),
            static.predict(images).prob_fake.numpy(),
            rtol=0,
            atol=1e-6,
        )

    def test_prob_fake_matches_returned_logits(self) -> None:
        method = self.method()
        prediction = method.predict(self.batch(2, 2))
        probabilities = prediction.logits.softmax(dim=1)[:, 1]
        np.testing.assert_allclose(
            probabilities.numpy(), prediction.prob_fake.numpy(), rtol=0, atol=1e-5
        )

    def test_adapt_requires_a_matching_predict_call(self) -> None:
        method = self.method()
        with self.assertRaises(RuntimeError):
            method.adapt(self.batch(1, 1))

    def test_admission_filters_low_quality_samples(self) -> None:
        method = self.method()
        images = self.batch(1, 0)
        images = self.torch.cat([images, self.torch.zeros(1, 3, 4, 4)], dim=0)
        method.predict(images)
        stats = method.adapt(images)
        self.assertEqual(stats.selected, 1)
        self.assertEqual(len(method.queue), 1)

    def test_admission_gate_counts_all_arrived_samples_in_window(self) -> None:
        method = self.method(
            window_size=2,
            admission_rate_floor=0.75,
            gates={"bimodality": False, "admission": True, "drift_fuse": False},
        )
        admitted = self.batch(1, 0)
        low_quality = self.torch.zeros(1, 3, 4, 4)
        batch = self.torch.cat([admitted, low_quality], dim=0)
        for _ in range(2):
            images = batch
            method.predict(images)
            stats = method.adapt(images)
        self.assertEqual(stats.extra.get("window_accepted"), False)
        self.assertEqual(stats.extra.get("window_freeze_reason"), "admission_rate")
        self.assertAlmostEqual(float(stats.extra["window_admission_rate"]), 0.5)

    def test_view_variance_feeds_consistency(self) -> None:
        torch = self.torch
        method = self.method()
        images = torch.zeros(2, 3, 3, 4, 4)
        images[0, :, 0] = 1.0
        images[1, 0, 0] = 0.5
        images[1, 1, 0] = 1.0
        images[1, 2, 0] = 1.5
        method.predict(images)
        pending = method._pending
        self.assertAlmostEqual(float(pending["consistency"][0]), 0.0, places=7)
        self.assertGreater(float(pending["consistency"][1]), 0.0)
        self.assertAlmostEqual(float(pending["scores"][0]), 4.0, places=5)

    def test_unimodal_window_is_frozen_by_bimodality_gate(self) -> None:
        method = self.method()
        images = self.batch(16, 0, jitter=0.001)
        method.predict(images)
        stats = method.adapt(images)
        self.assertEqual(stats.extra.get("window_accepted"), False)
        self.assertEqual(stats.extra.get("window_freeze_reason"), "no_bimodal_structure")
        self.assertEqual(method._params["real_mu"], -4.0)
        self.assertEqual(method._frozen_windows, 1)

    def test_huge_kappa_window_is_accepted_but_keeps_anchors(self) -> None:
        method = self.method(anchor_kappa=1e12)
        images = self.batch(8, 8)
        method.predict(images)
        stats = method.adapt(images)
        self.assertEqual(stats.extra.get("window_accepted"), True)
        self.assertAlmostEqual(method._params["real_mu"], -4.0, places=6)
        self.assertAlmostEqual(float(method._params["fake_mus"][0]), 4.0, places=6)

    def test_zero_kappa_moves_parameters_to_window_statistics(self) -> None:
        method = self.method(
            anchor_kappa=0.0, gates={"bimodality": True, "admission": True, "drift_fuse": False}
        )
        images = self.batch(8, 8, strength=2.0)
        method.predict(images)
        method.adapt(images)
        self.assertAlmostEqual(method._params["real_mu"], -8.0, places=2)
        self.assertAlmostEqual(float(method._params["fake_mus"][0]), 8.0, places=2)

    def test_anchored_map_matches_closed_form(self) -> None:
        method = self.method(
            anchor_kappa=4.0,
            gates={"bimodality": True, "admission": True, "drift_fuse": False},
        )
        images = self.batch(8, 8, strength=2.0)
        method.predict(images)
        method.adapt(images)
        quality = 0.5 - 1.0 / (1.0 + math.exp(8.0))
        expected = (4.0 * -4.0 + 8 * quality * -8.0) / (4.0 + 8 * quality)
        self.assertAlmostEqual(method._params["real_mu"], expected, places=3)

    def test_drift_fuse_freezes_windows_beyond_three_sigma(self) -> None:
        method = self.method(anchor_kappa=0.0)
        images = self.batch(8, 8, strength=2.0)
        method.predict(images)
        stats = method.adapt(images)
        self.assertEqual(stats.extra.get("window_accepted"), False)
        self.assertEqual(stats.extra.get("window_freeze_reason"), "drift_fuse")
        self.assertEqual(method._params["real_mu"], -4.0)

    def test_lambda_anneals_only_after_stable_accepted_windows(self) -> None:
        method = self.method()
        images = self.batch(8, 8)
        method.predict(images)
        method.adapt(images)
        self.assertAlmostEqual(method.lambda_value, 0.75)
        method.predict(images)
        method.adapt(images)
        self.assertAlmostEqual(method.lambda_value, 0.5)
        method.predict(images)
        method.adapt(images)
        self.assertAlmostEqual(method.lambda_value, 0.5)

    def test_frozen_window_blocks_lambda_annealing(self) -> None:
        method = self.method()
        images = self.batch(8, 8)
        method.predict(images)
        method.adapt(images)
        self.assertAlmostEqual(method.lambda_value, 0.75)
        unimodal = self.batch(16, 0, jitter=0.001)
        method.predict(unimodal)
        stats = method.adapt(unimodal)
        self.assertEqual(stats.extra.get("window_accepted"), False)
        self.assertAlmostEqual(method.lambda_value, 0.75)
        self.assertEqual(method._consecutive_accepted, 0)

    def test_memory_capacity_trims_oldest_entries(self) -> None:
        method = self.method(memory_capacity=20, window_size=16)
        images = self.batch(8, 8)
        method.predict(images)
        method.adapt(images)
        method.predict(images)
        method.adapt(images)
        self.assertEqual(len(method.queue), 20)

    def test_reset_restores_anchor_state(self) -> None:
        method = self.method(
            anchor_kappa=0.0, gates={"bimodality": True, "admission": True, "drift_fuse": False}
        )
        images = self.batch(8, 8, strength=2.0)
        method.predict(images)
        method.adapt(images)
        self.assertNotAlmostEqual(method._params["real_mu"], -4.0, places=2)
        method.reset()
        self.assertEqual(method._params["real_mu"], -4.0)
        self.assertAlmostEqual(method.lambda_value, 1.0)
        self.assertEqual(len(method.queue), 0)


@unittest.skipUnless(TORCH_AVAILABLE and TORCHVISION_AVAILABLE, "PyTorch and torchvision are required")
class ASCALViewTransformTests(unittest.TestCase):
    def image(self):
        from PIL import Image

        rng = np.random.default_rng(0)
        array = (rng.random((480, 640, 3)) * 255).astype(np.uint8)
        return Image.fromarray(array)

    def transform(self, **overrides):
        from src.data.transforms import CLIP_MEAN, CLIP_STD
        from src.data.views import ASCALViewTransform

        config = {
            "views": 5,
            "image_size": 224,
            "resize_size": 256,
            "mean": CLIP_MEAN,
            "std": CLIP_STD,
        }
        config.update(overrides)
        return ASCALViewTransform(**config)

    def test_stacks_global_local_and_jpeg_views(self) -> None:
        output = self.transform()(self.image())
        self.assertEqual(tuple(output.shape), (5, 3, 224, 224))

    def test_requires_at_least_three_views(self) -> None:
        with self.assertRaises(ValueError):
            self.transform(views=2)

    def test_transform_is_deterministic_under_torch_seed(self) -> None:
        import torch

        transform = self.transform()
        torch.manual_seed(0)
        first = transform(self.image())
        torch.manual_seed(0)
        second = transform(self.image())
        self.assertTrue(torch.equal(first, second))

    def test_jpeg_reencode_preserves_size_and_rgb_mode(self) -> None:
        from src.data.transforms import jpeg_reencode

        image = self.image()
        encoded = jpeg_reencode(image, 75)
        self.assertEqual(encoded.size, image.size)
        self.assertEqual(encoded.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
