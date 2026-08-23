from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

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
                self.assertEqual(
                    config["data"]["bias_control_profile"],
                    "matched_jpeg",
                )
                self.assertIn(
                    f"clip_vlm_bias_controlled/matched_jpeg/ascal_gmm/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, "ascal_gmm")
                self.assertTrue(removed_gates.isdisjoint(adaptive))

    def test_shift_seed1_configs_reuse_the_minimal_detector_setup(self) -> None:
        from src.config import load_config, method_config

        for dataset in (
            "genimage",
            "aigc_detection_benchmark",
            "aigi_holmes_p3",
            "opensdid_global",
        ):
            path = (
                PROJECT_ROOT
                / "configs/experiments/clip_vlm_bias_controlled"
                / f"matched_jpeg_ascal_gmm_shift_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    ["ascal_gmm_shift_static", "ascal_gmm_shift"],
                )
                self.assertEqual(config["seed"], 1)
                self.assertEqual(config["data"]["bias_control_profile"], "matched_jpeg")
                self.assertIn(
                    f"clip_vlm_bias_controlled/matched_jpeg/ascal_gmm_shift/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, "ascal_gmm_shift")
                self.assertNotIn("window_size", adaptive)
                self.assertNotIn("gates", adaptive)

    def test_median_shift_seed1_configs_add_no_stability_knobs(self) -> None:
        from src.config import load_config, method_config

        for dataset in (
            "genimage",
            "aigc_detection_benchmark",
            "aigi_holmes_p3",
            "opensdid_global",
        ):
            path = (
                PROJECT_ROOT
                / "configs/experiments/clip_vlm_bias_controlled"
                / f"matched_jpeg_ascal_gmm_median_shift_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    ["ascal_gmm_median_shift_static", "ascal_gmm_median_shift"],
                )
                self.assertEqual(config["seed"], 1)
                self.assertEqual(config["data"]["bias_control_profile"], "matched_jpeg")
                self.assertIn(
                    "clip_vlm_bias_controlled/matched_jpeg/"
                    f"ascal_gmm_median_shift/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, "ascal_gmm_median_shift")
                for key in ("ema", "momentum", "window_size", "gates", "threshold"):
                    self.assertNotIn(key, adaptive)

    def test_density_shift_seed1_configs_add_no_target_or_semantic_knobs(self) -> None:
        from src.config import load_config, method_config

        for dataset in (
            "genimage",
            "aigc_detection_benchmark",
            "aigi_holmes_p3",
            "opensdid_global",
        ):
            path = (
                PROJECT_ROOT
                / "configs/experiments/clip_vlm_bias_controlled"
                / f"matched_jpeg_ascal_gmm_density_shift_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    ["ascal_gmm_density_shift_static", "ascal_gmm_density_shift"],
                )
                self.assertEqual(config["seed"], 1)
                self.assertEqual(config["data"]["bias_control_profile"], "matched_jpeg")
                self.assertIn(
                    "clip_vlm_bias_controlled/matched_jpeg/"
                    f"ascal_gmm_density_shift/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, "ascal_gmm_density_shift")
                for key in (
                    "ema",
                    "momentum",
                    "window_size",
                    "gates",
                    "threshold",
                    "semantic_features",
                ):
                    self.assertNotIn(key, adaptive)
                self.assertFalse(adaptive["reference"]["semantic_features_used"])

    def test_segmented_shift_seed1_configs_add_no_change_threshold(self) -> None:
        from src.config import load_config, method_config

        for dataset in (
            "genimage",
            "aigc_detection_benchmark",
            "aigi_holmes_p3",
            "opensdid_global",
        ):
            path = (
                PROJECT_ROOT
                / "configs/experiments/clip_vlm_bias_controlled"
                / f"matched_jpeg_ascal_gmm_segmented_shift_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_shift_static",
                        "ascal_gmm_segmented_shift",
                    ],
                )
                self.assertEqual(config["seed"], 1)
                self.assertEqual(config["data"]["bias_control_profile"], "matched_jpeg")
                self.assertIn(
                    "clip_vlm_bias_controlled/matched_jpeg/"
                    f"ascal_gmm_segmented_shift/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, "ascal_gmm_segmented_shift")
                for key in (
                    "change_threshold",
                    "ema",
                    "gates",
                    "momentum",
                    "semantic_features",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                self.assertFalse(adaptive["reference"]["target_labels_used"])
                self.assertFalse(adaptive["reference"]["generator_boundaries_used"])
                self.assertFalse(adaptive["reference"]["semantic_features_used"])

    def test_segmented_shift_continual_configs_preserve_locked_seed1_streams(self) -> None:
        from src.config import load_config

        for dataset in (
            "genimage",
            "aigc_detection_benchmark",
            "aigi_holmes_p3",
            "opensdid_global",
        ):
            path = (
                PROJECT_ROOT
                / "configs/experiments/clip_vlm_bias_controlled"
                / "matched_jpeg_ascal_gmm_segmented_shift_continual_"
                f"{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    ["ascal_gmm_median_shift", "ascal_gmm_segmented_shift"],
                )
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(config["protocol"]["generator_id_available_to_method"])
                self.assertTrue(config["evaluation"]["evaluate_future_generators"])
                self.assertIn("seed1_online_manifest.csv", config["data"]["locked_online_manifest"])
                self.assertIn(
                    "seed1_final_holdout_manifest.csv",
                    config["data"]["locked_final_holdout_manifest"],
                )

    def test_segmented_handoff_continual_configs_add_no_smoothing_knobs(self) -> None:
        from src.config import load_config, method_config

        for dataset in (
            "genimage",
            "aigc_detection_benchmark",
            "aigi_holmes_p3",
            "opensdid_global",
        ):
            path = (
                PROJECT_ROOT
                / "configs/experiments/clip_vlm_bias_controlled"
                / "matched_jpeg_ascal_gmm_segmented_handoff_shift_continual_"
                f"{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_median_shift",
                        "ascal_gmm_segmented_shift",
                        "ascal_gmm_segmented_handoff_shift",
                    ],
                )
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                self.assertIn(
                    "seed1_online_manifest.csv",
                    config["data"]["locked_online_manifest"],
                )
                self.assertIn(
                    "seed1_final_holdout_manifest.csv",
                    config["data"]["locked_final_holdout_manifest"],
                )
                self.assertIn(
                    "ascal_gmm_segmented_handoff_shift_continual/"
                    f"{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(
                    config, "ascal_gmm_segmented_handoff_shift"
                )
                for key in (
                    "change_threshold",
                    "ema",
                    "handoff_length",
                    "momentum",
                    "smoothing_rate",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                self.assertFalse(adaptive["reference"]["target_labels_used"])
                self.assertFalse(
                    adaptive["reference"]["generator_boundaries_used"]
                )
                self.assertFalse(adaptive["reference"]["semantic_features_used"])

    def test_segmented_memory_continual_configs_use_parameter_free_recall(self) -> None:
        from src.config import load_config, method_config

        for dataset in (
            "genimage",
            "aigc_detection_benchmark",
            "aigi_holmes_p3",
            "opensdid_global",
        ):
            path = (
                PROJECT_ROOT
                / "configs/experiments/clip_vlm_bias_controlled"
                / "matched_jpeg_ascal_gmm_segmented_memory_shift_continual_"
                f"{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_shift",
                        "ascal_gmm_segmented_memory_shift",
                    ],
                )
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                self.assertIn(
                    "seed1_online_manifest.csv",
                    config["data"]["locked_online_manifest"],
                )
                self.assertIn(
                    "seed1_final_holdout_manifest.csv",
                    config["data"]["locked_final_holdout_manifest"],
                )
                self.assertIn(
                    "ascal_gmm_segmented_memory_shift_continual/"
                    f"{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(
                    config, "ascal_gmm_segmented_memory_shift"
                )
                for key in (
                    "capacity",
                    "change_threshold",
                    "memory_size",
                    "recall_threshold",
                    "similarity_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                self.assertFalse(adaptive["reference"]["target_labels_used"])
                self.assertFalse(
                    adaptive["reference"]["generator_boundaries_used"]
                )
                self.assertFalse(adaptive["reference"]["semantic_features_used"])
                self.assertFalse(adaptive["reference"]["raw_images_stored"])

    def test_segmented_memory_posterior_configs_add_no_fusion_knobs(self) -> None:
        from src.config import load_config, method_config

        for dataset in (
            "genimage",
            "aigc_detection_benchmark",
            "aigi_holmes_p3",
            "opensdid_global",
        ):
            path = (
                PROJECT_ROOT
                / "configs/experiments/clip_vlm_bias_controlled"
                / "matched_jpeg_ascal_gmm_segmented_memory_posterior_continual_"
                f"{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_memory_shift",
                        "ascal_gmm_segmented_memory_posterior",
                    ],
                )
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                self.assertIn(
                    "seed1_online_manifest.csv",
                    config["data"]["locked_online_manifest"],
                )
                self.assertIn(
                    "seed1_final_holdout_manifest.csv",
                    config["data"]["locked_final_holdout_manifest"],
                )
                self.assertIn(
                    "ascal_gmm_segmented_memory_posterior_continual/"
                    f"{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(
                    config, "ascal_gmm_segmented_memory_posterior"
                )
                for key in (
                    "class_prior",
                    "fusion_weight",
                    "lambda",
                    "posterior_temperature",
                    "recall_threshold",
                    "similarity_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["class_prior_rule"], "equal_real_and_fake_priors"
                )
                self.assertEqual(
                    reference["posterior_pooling"],
                    "evidence_counted_geometric_pool_in_log_odds",
                )
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])

    def test_posterior_projection_configs_preserve_ranking_without_new_knobs(
        self,
    ) -> None:
        from src.config import load_config, method_config

        for dataset in (
            "genimage",
            "aigc_detection_benchmark",
            "aigi_holmes_p3",
            "opensdid_global",
        ):
            path = (
                PROJECT_ROOT
                / "configs/experiments/clip_vlm_bias_controlled"
                / "matched_jpeg_ascal_gmm_segmented_memory_posterior_projection_"
                f"continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_memory_posterior",
                        "ascal_gmm_segmented_memory_posterior_projection",
                    ],
                )
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                self.assertIn(
                    "seed1_online_manifest.csv",
                    config["data"]["locked_online_manifest"],
                )
                self.assertIn(
                    "seed1_final_holdout_manifest.csv",
                    config["data"]["locked_final_holdout_manifest"],
                )
                self.assertIn(
                    "ascal_gmm_segmented_memory_posterior_projection_continual/"
                    f"{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(
                    config, "ascal_gmm_segmented_memory_posterior_projection"
                )
                for key in (
                    "class_prior",
                    "fusion_weight",
                    "lambda",
                    "posterior_temperature",
                    "recall_threshold",
                    "similarity_threshold",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["projection_rule"],
                    "bayes_half_posterior_surface_projected_onto_source_margin",
                )
                self.assertEqual(
                    reference["ranking_rule"],
                    "source_margin_order_preserved_within_each_causal_state",
                )
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])

    def test_source_training_declares_gmm_consumers(self) -> None:
        from src.config import load_config

        config = load_config(
            PROJECT_ROOT / "configs/train/genimage_sd14_clip_vitl14_lora_ascal.yaml"
        )
        self.assertIn("ascal_gmm", config["training"]["intended_methods"])
        self.assertIn("ascal_gmm_static", config["training"]["intended_methods"])
        self.assertIn("ascal_gmm_shift", config["training"]["intended_methods"])
        self.assertIn("ascal_gmm_shift_static", config["training"]["intended_methods"])
        self.assertIn("ascal_gmm_median_shift", config["training"]["intended_methods"])
        self.assertIn(
            "ascal_gmm_median_shift_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_density_shift",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_density_shift_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_shift",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_shift_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_handoff_shift",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_handoff_shift_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_shift",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_shift_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_projection",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_projection_static",
            config["training"]["intended_methods"],
        )


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

    def test_dominant_gap_keeps_contiguous_real_and_fake_blocks(self) -> None:
        from src.methods.ascal_gmm import dominant_gap_boundary

        partition = dominant_gap_boundary(
            {
                "mus": [-10.0, -6.0, 5.0, 9.0],
            }
        )
        self.assertEqual(partition["real_components"], 2)
        self.assertEqual(partition["fake_components"], 2)
        self.assertAlmostEqual(float(partition["decision_boundary"]), -0.5)
        self.assertAlmostEqual(float(partition["dominant_gap"]), 11.0)

    def test_dominant_gap_rejects_unimodal_and_unsorted_inputs(self) -> None:
        from src.methods.ascal_gmm import dominant_gap_boundary

        with self.assertRaises(ValueError):
            dominant_gap_boundary({"mus": [0.0]})
        with self.assertRaises(ValueError):
            dominant_gap_boundary({"mus": [0.0, -1.0]})

    def test_equal_density_boundary_uses_variance_not_only_the_gap_midpoint(self) -> None:
        from src.methods.ascal_gmm import equal_density_boundary

        boundary = equal_density_boundary(
            {
                "weights": [0.5, 0.5],
                "mus": [-2.0, 2.0],
                "sigmas": [0.5, 2.0],
            }
        )
        self.assertTrue(boundary["density_crossing"])
        self.assertAlmostEqual(float(boundary["gap_midpoint"]), 0.0)
        self.assertAlmostEqual(float(boundary["decision_boundary"]), -0.897, delta=0.01)
        self.assertAlmostEqual(float(boundary["density_log_ratio"]), 0.0, places=8)

    def test_joint_density_posterior_supports_multicomponent_class_blocks(self) -> None:
        from src.methods.ascal_gmm import joint_density_fake_posterior

        mixture = {
            "weights": [0.45, 0.05, 0.1, 0.4],
            "mus": [-6.0, -3.0, 2.0, 6.0],
            "sigmas": [0.5, 0.5, 0.5, 0.5],
        }
        posterior = joint_density_fake_posterior(
            np.array([-6.0, -3.0, 2.0, 6.0]), mixture
        )
        self.assertTrue(np.all(posterior[:2] < 0.01))
        self.assertTrue(np.all(posterior[2:] > 0.99))

    def test_joint_density_posterior_uses_equal_class_priors(self) -> None:
        from src.methods.ascal_gmm import joint_density_fake_posterior

        posterior = joint_density_fake_posterior(
            np.array([0.0]),
            {
                "weights": [0.99, 0.01],
                "mus": [-2.0, 2.0],
                "sigmas": [1.0, 1.0],
            },
        )
        self.assertAlmostEqual(float(posterior[0]), 0.5, places=12)

    def test_joint_density_posterior_rejects_invalid_mixtures(self) -> None:
        from src.methods.ascal_gmm import joint_density_fake_posterior

        with self.assertRaises(ValueError):
            joint_density_fake_posterior(
                np.array([0.0]),
                {"weights": [1.0], "mus": [0.0], "sigmas": [1.0]},
            )
        with self.assertRaises(ValueError):
            joint_density_fake_posterior(
                np.array([0.0]),
                {
                    "weights": [0.5, 0.5],
                    "mus": [1.0, -1.0],
                    "sigmas": [1.0, 1.0],
                },
            )


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

    def shift_method(self, *, adaptation_mode: str = "full"):
        from src.methods.ascal_gmm import ASCALGMMShift

        return ASCALGMMShift(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def median_shift_method(self, *, adaptation_mode: str = "full"):
        from src.methods.ascal_gmm import ASCALGMMMedianShift

        return ASCALGMMMedianShift(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def density_shift_method(self, *, adaptation_mode: str = "full"):
        from src.methods.ascal_gmm import ASCALGMMDensityShift

        return ASCALGMMDensityShift(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_shift_method(self, *, adaptation_mode: str = "full"):
        from src.methods.ascal_gmm import ASCALGMMSegmentedShift

        return ASCALGMMSegmentedShift(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_handoff_shift_method(self, *, adaptation_mode: str = "full"):
        from src.methods.ascal_gmm import ASCALGMMSegmentedHandoffShift

        return ASCALGMMSegmentedHandoffShift(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_shift_method(self, *, adaptation_mode: str = "full"):
        from src.methods.ascal_gmm import ASCALGMMSegmentedMemoryShift

        return ASCALGMMSegmentedMemoryShift(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import ASCALGMMSegmentedMemoryPosterior

        return ASCALGMMSegmentedMemoryPosterior(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_projection_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorProjection,
        )

        return ASCALGMMSegmentedMemoryPosteriorProjection(
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

    def test_shift_readout_is_a_monotone_source_score_translation(self) -> None:
        method = self.shift_method()
        method._mixture = {
            "weights": [0.25, 0.25, 0.25, 0.25],
            "mus": [-10.0, -6.0, 5.0, 9.0],
            "sigmas": [1.0, 1.0, 1.0, 1.0],
            "components": 4,
            "bic": 0.0,
        }
        scores = np.array([-8.0, -2.0, 2.0, 7.0])
        prediction = method.predict(self.score_batch(scores))
        expected = 1.0 / (1.0 + np.exp(-(scores + 0.5)))
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)
        self.assertTrue(np.all(np.diff(prediction.prob_fake.numpy()) > 0.0))
        self.assertEqual(method._pending["prediction_real_components"], 2)
        self.assertEqual(method._pending["prediction_fake_components"], 2)
        self.assertAlmostEqual(method._pending["prediction_boundary"], -0.5)

    def test_shift_fit_is_causal_and_records_the_next_boundary(self) -> None:
        rng = np.random.default_rng(5)
        scores = np.concatenate(
            [rng.normal(-8.0, 0.2, 80), rng.normal(-2.0, 0.2, 80)]
        )
        method = self.shift_method()
        first = method.predict(self.score_batch(scores))
        source = 1.0 / (1.0 + np.exp(-scores))
        np.testing.assert_allclose(first.prob_fake.numpy(), source, atol=1e-6)
        stats = method.adapt(self.score_batch(scores))
        self.assertEqual(stats.extra["prediction_mode"], "source_fallback")
        self.assertEqual(stats.extra["total_components"], 2)
        self.assertAlmostEqual(float(stats.extra["decision_boundary"]), -5.0, delta=0.1)

        method.predict(self.score_batch(np.array([-8.0, -2.0])))
        self.assertEqual(method._pending["prediction_mode"], "monotone_gmm_shift")

    def test_method_factory_maps_shift_static_alias(self) -> None:
        from src.methods import build_method

        method = build_method(
            "ascal_gmm_shift_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_median_shift_rejects_a_latest_boundary_outlier(self) -> None:
        method = self.median_shift_method()
        for means in ([-10.0, 0.0], [-8.0, 0.0], [-6.0, 0.0], [10.0, 30.0]):
            method._mixture = {
                "weights": [0.5, 0.5],
                "mus": means,
                "sigmas": [1.0, 1.0],
                "components": 2,
                "bic": 0.0,
            }
            method._after_successful_fit()

        scores = np.array([-4.0, 0.0, 4.0])
        prediction = method.predict(self.score_batch(scores))
        expected = 1.0 / (1.0 + np.exp(-(scores + 3.5)))
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)
        self.assertAlmostEqual(method._pending["prediction_candidate_boundary"], 20.0)
        self.assertAlmostEqual(method._pending["prediction_boundary"], -3.5)

        stats = method._state_stats()
        self.assertEqual(stats["boundary_samples"], 4)
        self.assertAlmostEqual(stats["candidate_boundary"], 20.0)
        self.assertAlmostEqual(stats["stabilized_boundary"], -3.5)
        self.assertAlmostEqual(stats["decision_boundary"], -3.5)

    def test_median_shift_fit_is_causal_and_reset_clears_boundaries(self) -> None:
        rng = np.random.default_rng(6)
        scores = np.concatenate(
            [rng.normal(-8.0, 0.2, 80), rng.normal(-2.0, 0.2, 80)]
        )
        method = self.median_shift_method()
        method.predict(self.score_batch(scores))
        stats = method.adapt(self.score_batch(scores))
        self.assertEqual(stats.extra["prediction_mode"], "source_fallback")
        self.assertEqual(stats.extra["boundary_samples"], 1)
        self.assertAlmostEqual(
            float(stats.extra["decision_boundary"]),
            float(stats.extra["candidate_boundary"]),
        )

        method.predict(self.score_batch(np.array([-8.0, -2.0])))
        self.assertEqual(method._pending["prediction_mode"], "median_gmm_shift")
        method.reset()
        self.assertEqual(method.boundary_history, [])

    def test_method_factory_maps_median_shift_static_alias(self) -> None:
        from src.methods import build_method

        method = build_method(
            "ascal_gmm_median_shift_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_density_shift_uses_the_median_stabilized_density_crossing(self) -> None:
        method = self.density_shift_method()
        method._mixture = {
            "weights": [0.5, 0.5],
            "mus": [-2.0, 2.0],
            "sigmas": [0.5, 2.0],
            "components": 2,
            "bic": 0.0,
        }
        method._after_successful_fit()

        scores = np.array([-2.0, 0.0, 2.0])
        prediction = method.predict(self.score_batch(scores))
        boundary = float(method._pending["prediction_boundary"])
        expected = 1.0 / (1.0 + np.exp(-(scores - boundary)))
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)
        self.assertAlmostEqual(boundary, -0.897, delta=0.01)
        self.assertEqual(method._pending["prediction_mode"], "density_gmm_shift")
        self.assertEqual(
            method.reproduction_metadata["component_semantics"],
            "none_dominant_gap_partition_only",
        )

    def test_method_factory_maps_density_shift_static_alias(self) -> None:
        from src.methods import build_method

        method = build_method(
            "ascal_gmm_density_shift_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_segmented_shift_does_not_split_a_stationary_stream(self) -> None:
        rng = np.random.default_rng(7)
        method = self.segmented_shift_method()
        stats = None
        for _ in range(6):
            scores = np.concatenate(
                [rng.normal(-7.0, 0.25, 48), rng.normal(-2.0, 0.25, 48)]
            )
            rng.shuffle(scores)
            images = self.score_batch(scores)
            method.predict(images)
            stats = method.adapt(images)

        self.assertIsNotNone(stats)
        self.assertEqual(stats.extra["segment_changes"], 0)
        self.assertEqual(stats.extra["segment_batches"], 6)
        self.assertGreater(stats.extra["segment_checks"], 0)

    def test_segmented_shift_uses_one_deterministic_binary_scale(self) -> None:
        method = self.segmented_shift_method()
        expected = {
            3: [],
            4: [2],
            5: [],
            6: [2],
            8: [4],
            10: [2],
            12: [4],
            16: [8],
        }
        for batches, candidates in expected.items():
            with self.subTest(batches=batches):
                method.score_batches = [np.zeros(16)] * batches
                self.assertEqual(method._suffix_candidates(), candidates)

    def test_segmented_shift_forgets_an_obsolete_score_segment_causally(self) -> None:
        rng = np.random.default_rng(8)
        method = self.segmented_shift_method()
        for _ in range(4):
            scores = np.concatenate(
                [rng.normal(-8.0, 0.2, 48), rng.normal(-3.0, 0.2, 48)]
            )
            rng.shuffle(scores)
            images = self.score_batch(scores)
            method.predict(images)
            method.adapt(images)

        first_shifted = np.concatenate(
            [rng.normal(2.0, 0.2, 48), rng.normal(7.0, 0.2, 48)]
        )
        rng.shuffle(first_shifted)
        first_shifted_images = self.score_batch(first_shifted)
        method.predict(first_shifted_images)
        old_boundary = float(method._pending["prediction_boundary"])
        first_stats = method.adapt(first_shifted_images)

        second_shifted = np.concatenate(
            [rng.normal(2.0, 0.2, 48), rng.normal(7.0, 0.2, 48)]
        )
        rng.shuffle(second_shifted)
        second_shifted_images = self.score_batch(second_shifted)
        method.predict(second_shifted_images)
        stats = method.adapt(second_shifted_images)

        self.assertLess(old_boundary, 0.0)
        self.assertFalse(first_stats.extra["segment_changed"])
        self.assertTrue(stats.extra["segment_changed"])
        self.assertEqual(stats.extra["segment_changes"], 1)
        self.assertEqual(stats.extra["segment_batches"], 2)
        self.assertEqual(
            stats.extra["score_samples"],
            len(first_shifted) + len(second_shifted),
        )
        self.assertEqual(stats.extra["boundary_samples"], 1)
        self.assertGreater(float(stats.extra["last_segment_gain"]), 0.0)

        method.predict(self.score_batch(np.array([2.0, 7.0])))
        self.assertGreater(float(method._pending["prediction_boundary"]), 0.0)

    def test_segmented_shift_reset_clears_change_state(self) -> None:
        method = self.segmented_shift_method()
        images = self.score_batch(np.array([-4.0, -3.0, 2.0, 3.0]))
        method.predict(images)
        method.adapt(images)
        method.reset()

        stats = method._state_stats()
        self.assertEqual(stats["segment_changes"], 0)
        self.assertEqual(stats["segment_batches"], 0)
        self.assertEqual(stats["total_score_samples"], 0)
        self.assertIsNone(stats["last_segment_gain"])

    def test_segmented_shift_does_not_reset_to_a_unimodal_suffix(self) -> None:
        rng = np.random.default_rng(9)
        method = self.segmented_shift_method()
        for _ in range(12):
            scores = np.concatenate(
                [rng.normal(-8.0, 0.2, 48), rng.normal(-3.0, 0.2, 48)]
            )
            rng.shuffle(scores)
            images = self.score_batch(scores)
            method.predict(images)
            method.adapt(images)

        stats = None
        for _ in range(2):
            scores = rng.normal(-3.8, 0.2, 96)
            images = self.score_batch(scores)
            method.predict(images)
            stats = method.adapt(images)

        self.assertIsNotNone(stats)
        self.assertFalse(stats.extra["segment_changed"])
        self.assertEqual(stats.extra["segment_changes"], 0)
        self.assertGreater(stats.extra["segment_unimodal_suffixes"], 0)

    def test_segmented_memory_recalls_a_completed_a_b_a_regime(self) -> None:
        rng = np.random.default_rng(10)
        method = self.segmented_memory_shift_method()

        def adapt_regime(low: float, high: float, batches: int):
            stats = None
            for _ in range(batches):
                scores = np.concatenate(
                    [rng.normal(low, 0.2, 48), rng.normal(high, 0.2, 48)]
                )
                rng.shuffle(scores)
                images = self.score_batch(scores)
                method.predict(images)
                stats = method.adapt(images)
            return stats

        adapt_regime(-8.0, -3.0, 4)
        first_change = adapt_regime(2.0, 7.0, 2)
        self.assertTrue(first_change.extra["segment_changed"])
        self.assertEqual(first_change.extra["memory_size"], 1)
        self.assertEqual(first_change.extra["memory_novel_events"], 1)
        self.assertEqual(first_change.extra["memory_recall_events"], 0)
        self.assertIsNone(first_change.extra["active_memory_index"])

        recalled = adapt_regime(-8.0, -3.0, 2)
        self.assertTrue(recalled.extra["segment_changed"])
        self.assertEqual(recalled.extra["segment_changes"], 2)
        self.assertEqual(recalled.extra["memory_size"], 2)
        self.assertEqual(recalled.extra["memory_recall_events"], 1)
        self.assertTrue(recalled.extra["memory_recalled_this_change"])
        self.assertEqual(recalled.extra["active_memory_index"], 0)
        self.assertEqual(recalled.extra["last_recalled_memory_index"], 0)
        self.assertGreater(float(recalled.extra["last_memory_recall_gain"]), 0.0)

        next_images = self.score_batch(np.array([-8.0, -3.0]))
        method.predict(next_images)
        self.assertTrue(method._pending["prediction_memory_recalled"])
        self.assertEqual(method._pending["prediction_memory_index"], 0)
        self.assertEqual(method._pending["prediction_memory_anchor_weight"], 1.0)
        self.assertAlmostEqual(
            float(method._pending["prediction_boundary"]),
            float(method.segment_memories[0]["boundary"]),
        )

    def test_segmented_memory_rejects_a_novel_score_regime(self) -> None:
        rng = np.random.default_rng(11)
        method = self.segmented_memory_shift_method()

        for low, high, batches in (
            (-8.0, -3.0, 4),
            (2.0, 7.0, 2),
            (12.0, 17.0, 2),
        ):
            for _ in range(batches):
                scores = np.concatenate(
                    [rng.normal(low, 0.2, 48), rng.normal(high, 0.2, 48)]
                )
                rng.shuffle(scores)
                images = self.score_batch(scores)
                method.predict(images)
                stats = method.adapt(images)

        self.assertTrue(stats.extra["segment_changed"])
        self.assertEqual(stats.extra["memory_size"], 2)
        self.assertEqual(stats.extra["memory_recall_events"], 0)
        self.assertEqual(stats.extra["memory_novel_events"], 2)
        self.assertFalse(stats.extra["memory_recalled_this_change"])
        self.assertIsNone(stats.extra["active_memory_index"])
        self.assertLess(float(stats.extra["last_memory_recall_gain"]), 0.0)

    def test_segmented_memory_reset_clears_episodic_state(self) -> None:
        method = self.segmented_memory_shift_method()
        method.segment_memories.append(
            {
                "mixture": {},
                "boundary": 1.0,
                "latest_samples": 2,
                "total_samples": 2,
                "visits": 1,
                "recalls": 0,
            }
        )
        method.active_memory_index = 0
        method.recall_anchor_boundary = 1.0
        method.memory_recall_events = 1
        method.reset()

        stats = method._state_stats()
        self.assertEqual(stats["memory_size"], 0)
        self.assertEqual(stats["memory_recall_events"], 0)
        self.assertIsNone(stats["active_memory_index"])
        self.assertIsNone(stats["recall_anchor_boundary"])

    def test_segmented_memory_posterior_reads_both_component_blocks(self) -> None:
        from src.methods.ascal_gmm import joint_density_fake_posterior

        method = self.segmented_memory_posterior_method()
        method._mixture = {
            "weights": [0.4, 0.1, 0.2, 0.3],
            "mus": [-6.0, -3.0, 2.0, 6.0],
            "sigmas": [0.75, 0.75, 0.75, 0.75],
            "components": 4,
            "bic": 0.0,
        }
        scores = np.array([-6.0, -3.0, 2.0, 6.0])
        prediction = method.predict(self.score_batch(scores))
        expected = joint_density_fake_posterior(scores, method._mixture)

        np.testing.assert_allclose(
            prediction.prob_fake.numpy(), expected, atol=1e-6
        )
        self.assertEqual(
            method._pending["prediction_mode"],
            "segmented_memory_joint_density_posterior",
        )
        self.assertEqual(method._pending["prediction_real_components"], 2)
        self.assertEqual(method._pending["prediction_fake_components"], 2)
        self.assertFalse(method._pending["prediction_memory_recalled"])
        self.assertIsNone(method._pending["prediction_boundary"])
        metadata = method.reproduction_metadata
        self.assertIn("not_applied", metadata["boundary_stabilization"])
        self.assertIn("log_odds", metadata["recalled_boundary_rule"])
        self.assertFalse(
            any(
                "additive score boundary" in change
                for change in metadata["intentional_changes"]
            )
        )

    def test_segmented_memory_posterior_pools_recall_in_log_odds(self) -> None:
        from src.methods.ascal_gmm import joint_density_fake_posterior

        method = self.segmented_memory_posterior_method()
        current = {
            "weights": [0.5, 0.5],
            "mus": [-2.0, 2.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 0.0,
        }
        memory = {
            "weights": [0.5, 0.5],
            "mus": [-6.0, -2.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 0.0,
        }
        method._mixture = current
        method.segment_memories = [
            {
                "mixture": memory,
                "boundary": -4.0,
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 1,
            }
        ]
        method.active_memory_index = 0
        method.recall_anchor_boundary = -4.0
        method.boundary_history = [0.0]
        scores = np.array([-2.0])

        first = method.predict(self.score_batch(scores))
        memory_probability = joint_density_fake_posterior(scores, memory)
        np.testing.assert_allclose(
            first.prob_fake.numpy(), memory_probability, atol=1e-6
        )
        self.assertEqual(
            method._pending["prediction_mode"], "recalled_joint_density_posterior"
        )
        self.assertEqual(method._pending["prediction_memory_anchor_weight"], 1.0)

        method.boundary_history.append(0.0)
        second = method.predict(self.score_batch(scores))
        current_probability = joint_density_fake_posterior(scores, current)
        memory_log_odds = np.log(memory_probability / (1.0 - memory_probability))
        current_log_odds = np.log(current_probability / (1.0 - current_probability))
        expected = 1.0 / (1.0 + np.exp(-0.5 * (memory_log_odds + current_log_odds)))
        np.testing.assert_allclose(second.prob_fake.numpy(), expected, atol=1e-6)
        self.assertEqual(method._pending["prediction_memory_anchor_weight"], 0.5)

    def test_posterior_projection_uses_bayes_boundary_and_preserves_order(
        self,
    ) -> None:
        from src.methods.ascal_gmm import equal_density_boundary

        method = self.segmented_memory_posterior_projection_method()
        method._mixture = {
            "weights": [0.5, 0.5],
            "mus": [-2.0, 2.0],
            "sigmas": [0.5, 2.0],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        density = equal_density_boundary(method._mixture)
        boundary = float(density["decision_boundary"])
        self.assertNotAlmostEqual(boundary, 0.0)

        prediction = method.predict(self.score_batch(scores))
        expected = method._source_probability(scores - boundary)
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)
        self.assertTrue(np.all(np.diff(prediction.prob_fake.numpy()) > 0.0))
        self.assertAlmostEqual(method._pending["prediction_boundary"], boundary)
        self.assertAlmostEqual(
            method._pending["prediction_candidate_boundary"], boundary
        )
        self.assertEqual(
            method._pending["prediction_mode"],
            "segmented_memory_posterior_projection",
        )
        metadata = method.reproduction_metadata
        self.assertIn("half_posterior", metadata["projection_rule"])
        self.assertIn("preserved", metadata["ranking_rule"])

    def test_posterior_projection_stores_the_density_boundary_for_recall(self) -> None:
        from src.methods.ascal_gmm import equal_density_boundary

        method = self.segmented_memory_posterior_projection_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-2.0, 2.0],
            "sigmas": [0.5, 2.0],
            "components": 2,
            "bic": 0.0,
        }
        expected = float(equal_density_boundary(mixture)["decision_boundary"])
        stored_index = method._store_completed_segment(mixture, 96)

        self.assertEqual(stored_index, 0)
        self.assertAlmostEqual(method.segment_memories[0]["boundary"], expected)
        self.assertNotAlmostEqual(method.segment_memories[0]["boundary"], 0.0)

    def test_segmented_handoff_keeps_the_first_new_boundary_continuous(self) -> None:
        rng = np.random.default_rng(10)
        method = self.segmented_handoff_shift_method()
        for _ in range(4):
            scores = np.concatenate(
                [rng.normal(-8.0, 0.2, 48), rng.normal(-3.0, 0.2, 48)]
            )
            rng.shuffle(scores)
            images = self.score_batch(scores)
            method.predict(images)
            method.adapt(images)

        stats = None
        boundary_before_change = None
        for _ in range(2):
            scores = np.concatenate(
                [rng.normal(2.0, 0.2, 48), rng.normal(7.0, 0.2, 48)]
            )
            rng.shuffle(scores)
            images = self.score_batch(scores)
            method.predict(images)
            boundary_before_change = float(method._pending["prediction_boundary"])
            stats = method.adapt(images)

        self.assertIsNotNone(stats)
        self.assertTrue(stats.extra["segment_changed"])
        self.assertTrue(stats.extra["handoff_active"])
        self.assertEqual(stats.extra["handoff_boundary_samples"], 1)
        self.assertEqual(stats.extra["handoff_weight"], 0.0)
        self.assertAlmostEqual(
            float(stats.extra["handoff_anchor_boundary"]),
            float(boundary_before_change),
        )
        self.assertAlmostEqual(
            float(stats.extra["decision_boundary"]),
            float(boundary_before_change),
        )

        next_images = self.score_batch(np.array([2.0, 7.0]))
        method.predict(next_images)
        self.assertAlmostEqual(
            float(method._pending["prediction_boundary"]),
            float(boundary_before_change),
        )

    def test_segmented_handoff_uses_parameter_free_evidence_weights(self) -> None:
        method = self.segmented_handoff_shift_method()
        method.handoff_active = True
        method.handoff_anchor_boundary = -4.0
        method._mixture = {
            "weights": [0.5, 0.5],
            "mus": [4.0, 8.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 0.0,
        }

        method.boundary_history = [6.0]
        method.predict(self.score_batch(np.array([1.0])))
        self.assertEqual(method._pending["prediction_handoff_weight"], 0.0)
        self.assertAlmostEqual(method._pending["prediction_boundary"], -4.0)

        method.boundary_history.append(6.0)
        method.predict(self.score_batch(np.array([1.0])))
        self.assertEqual(method._pending["prediction_handoff_weight"], 0.5)
        self.assertAlmostEqual(method._pending["prediction_boundary"], 1.0)

        method.boundary_history.append(6.0)
        method.predict(self.score_batch(np.array([1.0])))
        self.assertAlmostEqual(
            method._pending["prediction_handoff_weight"],
            2.0 / 3.0,
        )
        self.assertAlmostEqual(method._pending["prediction_boundary"], 8.0 / 3.0)

    def test_segmented_handoff_holds_the_last_boundary_when_fit_is_unimodal(self) -> None:
        method = self.segmented_handoff_shift_method()
        method.handoff_active = True
        method.handoff_anchor_boundary = 1.25
        method.last_emitted_boundary = 1.25
        method.boundary_history = [1.25]
        method._mixture = {
            "weights": [1.0],
            "mus": [0.0],
            "sigmas": [1.0],
            "components": 1,
            "bic": 0.0,
        }

        scores = np.array([1.25, 2.25])
        prediction = method.predict(self.score_batch(scores))
        expected = 1.0 / (1.0 + np.exp(-(scores - 1.25)))
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)
        self.assertEqual(method._pending["prediction_mode"], "segmented_handoff_hold")
        self.assertAlmostEqual(method._pending["prediction_boundary"], 1.25)

    def test_segmented_handoff_reset_clears_boundary_state(self) -> None:
        method = self.segmented_handoff_shift_method()
        method.handoff_active = True
        method.handoff_anchor_boundary = 2.0
        method.last_emitted_boundary = 2.0
        method.reset()

        self.assertFalse(method.handoff_active)
        self.assertEqual(method.handoff_anchor_boundary, 0.0)
        self.assertEqual(method.last_emitted_boundary, 0.0)
        self.assertIsNone(method.handoff_start_batch)

    def test_method_factory_maps_segmented_shift_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import ASCALGMMSegmentedShift

        method = build_method(
            "ascal_gmm_segmented_shift_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(method, ASCALGMMSegmentedShift)
        self.assertEqual(method.adaptation_mode, "static")

    def test_method_factory_maps_segmented_handoff_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import ASCALGMMSegmentedHandoffShift

        method = build_method(
            "ascal_gmm_segmented_handoff_shift_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(method, ASCALGMMSegmentedHandoffShift)
        self.assertEqual(method.adaptation_mode, "static")

    def test_method_factory_maps_segmented_memory_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import ASCALGMMSegmentedMemoryShift

        method = build_method(
            "ascal_gmm_segmented_memory_shift_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(method, ASCALGMMSegmentedMemoryShift)
        self.assertEqual(method.adaptation_mode, "static")

    def test_method_factory_maps_segmented_memory_posterior_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import ASCALGMMSegmentedMemoryPosterior

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(method, ASCALGMMSegmentedMemoryPosterior)
        self.assertEqual(method.adaptation_mode, "static")

    def test_method_factory_maps_posterior_projection_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorProjection,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_projection_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(method, ASCALGMMSegmentedMemoryPosteriorProjection)
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_density_shift_with_the_ascal_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import ASCALGMMDensityShift

        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                "ascal_gmm_density_shift_static": {"adaptation_mode": "static"},
            },
        }
        checkpoint_metadata = {
            "lora_rank": 4,
            "score_anchors": self.anchors(),
        }
        with patch(
            "src.cli.common.build_clip_lora_detector",
            return_value=(self.detector(), {"family": "clip_lora_source_detector"}),
        ), patch(
            "src.cli.common.load_checkpoint",
            return_value=checkpoint_metadata,
        ), patch(
            "src.cli.common.checkpoint_sha256",
            return_value="0" * 64,
        ):
            method, _ = build_fresh_method(
                config,
                "ascal_gmm_density_shift_static",
                "cpu",
            )

        self.assertIsInstance(method, ASCALGMMDensityShift)
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_segmented_shift_with_the_ascal_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import ASCALGMMSegmentedShift

        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                "ascal_gmm_segmented_shift_static": {"adaptation_mode": "static"},
            },
        }
        checkpoint_metadata = {
            "lora_rank": 4,
            "score_anchors": self.anchors(),
        }
        with patch(
            "src.cli.common.build_clip_lora_detector",
            return_value=(self.detector(), {"family": "clip_lora_source_detector"}),
        ), patch(
            "src.cli.common.load_checkpoint",
            return_value=checkpoint_metadata,
        ), patch(
            "src.cli.common.checkpoint_sha256",
            return_value="0" * 64,
        ):
            method, _ = build_fresh_method(
                config,
                "ascal_gmm_segmented_shift_static",
                "cpu",
            )

        self.assertIsInstance(method, ASCALGMMSegmentedShift)
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_segmented_handoff_with_the_ascal_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import ASCALGMMSegmentedHandoffShift

        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                "ascal_gmm_segmented_handoff_shift_static": {
                    "adaptation_mode": "static"
                },
            },
        }
        checkpoint_metadata = {
            "lora_rank": 4,
            "score_anchors": self.anchors(),
        }
        with patch(
            "src.cli.common.build_clip_lora_detector",
            return_value=(self.detector(), {"family": "clip_lora_source_detector"}),
        ), patch(
            "src.cli.common.load_checkpoint",
            return_value=checkpoint_metadata,
        ), patch(
            "src.cli.common.checkpoint_sha256",
            return_value="0" * 64,
        ):
            method, _ = build_fresh_method(
                config,
                "ascal_gmm_segmented_handoff_shift_static",
                "cpu",
            )

        self.assertIsInstance(method, ASCALGMMSegmentedHandoffShift)
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_segmented_memory_with_the_ascal_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import ASCALGMMSegmentedMemoryShift

        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                "ascal_gmm_segmented_memory_shift_static": {
                    "adaptation_mode": "static"
                },
            },
        }
        checkpoint_metadata = {
            "lora_rank": 4,
            "score_anchors": self.anchors(),
        }
        with patch(
            "src.cli.common.build_clip_lora_detector",
            return_value=(self.detector(), {"family": "clip_lora_source_detector"}),
        ), patch(
            "src.cli.common.load_checkpoint",
            return_value=checkpoint_metadata,
        ), patch(
            "src.cli.common.checkpoint_sha256",
            return_value="0" * 64,
        ):
            method, _ = build_fresh_method(
                config,
                "ascal_gmm_segmented_memory_shift_static",
                "cpu",
            )

        self.assertIsInstance(method, ASCALGMMSegmentedMemoryShift)
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_segmented_memory_posterior_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import ASCALGMMSegmentedMemoryPosterior

        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                "ascal_gmm_segmented_memory_posterior_static": {
                    "adaptation_mode": "static"
                },
            },
        }
        checkpoint_metadata = {
            "lora_rank": 4,
            "score_anchors": self.anchors(),
        }
        with patch(
            "src.cli.common.build_clip_lora_detector",
            return_value=(self.detector(), {"family": "clip_lora_source_detector"}),
        ), patch(
            "src.cli.common.load_checkpoint",
            return_value=checkpoint_metadata,
        ), patch(
            "src.cli.common.checkpoint_sha256",
            return_value="0" * 64,
        ):
            method, _ = build_fresh_method(
                config,
                "ascal_gmm_segmented_memory_posterior_static",
                "cpu",
            )

        self.assertIsInstance(method, ASCALGMMSegmentedMemoryPosterior)
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_posterior_projection_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorProjection,
        )

        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                "ascal_gmm_segmented_memory_posterior_projection_static": {
                    "adaptation_mode": "static"
                },
            },
        }
        checkpoint_metadata = {
            "lora_rank": 4,
            "score_anchors": self.anchors(),
        }
        with patch(
            "src.cli.common.build_clip_lora_detector",
            return_value=(self.detector(), {"family": "clip_lora_source_detector"}),
        ), patch(
            "src.cli.common.load_checkpoint",
            return_value=checkpoint_metadata,
        ), patch(
            "src.cli.common.checkpoint_sha256",
            return_value="0" * 64,
        ):
            method, _ = build_fresh_method(
                config,
                "ascal_gmm_segmented_memory_posterior_projection_static",
                "cpu",
            )

        self.assertIsInstance(method, ASCALGMMSegmentedMemoryPosteriorProjection)
        self.assertEqual(method.adaptation_mode, "static")


if __name__ == "__main__":
    unittest.main()
