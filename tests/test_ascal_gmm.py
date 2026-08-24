from __future__ import annotations

import copy
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

    def test_current_projection_configs_remove_median_without_new_knobs(self) -> None:
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
                / "matched_jpeg_ascal_gmm_segmented_memory_posterior_current_"
                f"projection_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_memory_posterior_projection",
                        "ascal_gmm_segmented_memory_posterior_current_projection",
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
                    "posterior_current_projection_continual/"
                    f"{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(
                    config,
                    "ascal_gmm_segmented_memory_posterior_current_projection",
                )
                for key in (
                    "change_threshold",
                    "fusion_weight",
                    "lambda",
                    "median_window",
                    "posterior_temperature",
                    "recall_threshold",
                    "smoothing",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(reference["research_name"], "ASCAL-JMP-Current")
                self.assertEqual(reference["research_version"], "R02")
                self.assertEqual(reference["nested_refit_median"], "not_used")
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])

    def test_guarded_projection_configs_add_no_target_hyperparameters(self) -> None:
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
                / "matched_jpeg_ascal_gmm_segmented_memory_posterior_guarded_"
                f"projection_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_memory_posterior_projection",
                        "ascal_gmm_segmented_memory_posterior_guarded_projection",
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
                    "posterior_guarded_projection_continual/"
                    f"{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(
                    config,
                    "ascal_gmm_segmented_memory_posterior_guarded_projection",
                )
                for key in (
                    "change_threshold",
                    "fusion_weight",
                    "lambda",
                    "median_window",
                    "posterior_temperature",
                    "recall_threshold",
                    "smoothing",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(reference["research_name"], "ASCAL-JMP-GuardedScan")
                self.assertEqual(reference["research_version"], "R03")
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])

    def test_support_projection_configs_add_no_target_hyperparameters(self) -> None:
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
                / "matched_jpeg_ascal_gmm_segmented_memory_posterior_support_"
                f"projection_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_memory_posterior_projection",
                        "ascal_gmm_segmented_memory_posterior_support_projection",
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
                    "posterior_support_projection_continual/"
                    f"{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(
                    config,
                    "ascal_gmm_segmented_memory_posterior_support_projection",
                )
                for key in (
                    "change_threshold",
                    "fusion_weight",
                    "lambda",
                    "median_window",
                    "posterior_temperature",
                    "recall_threshold",
                    "smoothing",
                    "support_power",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-SupportMedian"
                )
                self.assertEqual(reference["research_version"], "R04")
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])

    def test_global_residual_configs_use_one_shared_residual_without_knobs(
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
                / "matched_jpeg_ascal_gmm_segmented_memory_posterior_global_"
                f"residual_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_memory_posterior_projection",
                        "ascal_gmm_segmented_memory_posterior_global_residual",
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
                    "posterior_global_residual_continual/"
                    f"{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(
                    config,
                    "ascal_gmm_segmented_memory_posterior_global_residual",
                )
                for key in (
                    "confidence_weight",
                    "fusion_weight",
                    "lambda",
                    "learning_rate",
                    "loss_weight",
                    "memory_capacity",
                    "posterior_temperature",
                    "recall_threshold",
                    "smoothing",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-GlobalResidual"
                )
                self.assertEqual(reference["research_version"], "R05")
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(reference["residual_count"], 1)
                self.assertFalse(reference["adaptive_score_history_stored"])
                self.assertEqual(reference["optimizer"], "none")
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])

    def test_mixture_residual_configs_keep_one_readout_and_bic_modes(self) -> None:
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
                / "matched_jpeg_ascal_gmm_segmented_memory_posterior_mixture_"
                f"residual_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_memory_posterior_projection",
                        "ascal_gmm_segmented_memory_posterior_mixture_residual",
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
                    "posterior_mixture_residual_continual/"
                    f"{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(
                    config,
                    "ascal_gmm_segmented_memory_posterior_mixture_residual",
                )
                for key in (
                    "component_count",
                    "confidence_weight",
                    "fusion_weight",
                    "lambda",
                    "learning_rate",
                    "loss_weight",
                    "memory_capacity",
                    "posterior_temperature",
                    "prototype_count",
                    "recall_threshold",
                    "smoothing",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-MixtureResidual"
                )
                self.assertEqual(reference["research_version"], "R06")
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(reference["residual_count"], 1)
                self.assertIn("bic", reference["fake_component_count_rule"])
                self.assertEqual(
                    reference["seed1_promotion_rule"],
                    "accuracy_noninferiority_0p2pp_auc_gain_0p1pp_and_above_r05",
                )
                self.assertFalse(reference["adaptive_score_history_stored"])
                self.assertEqual(reference["optimizer"], "none")
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])

    def test_real_deviation_configs_use_one_centered_real_anchor(self) -> None:
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
                / "matched_jpeg_ascal_gmm_segmented_memory_posterior_real_"
                f"deviation_residual_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_memory_posterior_projection",
                        "ascal_gmm_segmented_memory_posterior_real_deviation_residual",
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
                    "posterior_real_deviation_residual_continual/"
                    f"{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(
                    config,
                    "ascal_gmm_segmented_memory_posterior_real_deviation_residual",
                )
                for key in (
                    "component_count",
                    "confidence_weight",
                    "fake_component_count",
                    "fusion_weight",
                    "lambda",
                    "learning_rate",
                    "loss_weight",
                    "memory_capacity",
                    "posterior_temperature",
                    "prototype_count",
                    "recall_threshold",
                    "smoothing",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-RealDeviation"
                )
                self.assertEqual(reference["research_version"], "R07")
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(reference["residual_count"], 1)
                self.assertEqual(
                    reference["fake_prototype_rule"],
                    "none_fake_is_open_heterogeneity",
                )
                self.assertEqual(
                    reference["seed1_promotion_rule"],
                    "accuracy_noninferiority_0p2pp_auc_gain_0p1pp_and_above_r06",
                )
                self.assertFalse(reference["adaptive_score_history_stored"])
                self.assertEqual(reference["optimizer"], "none")
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])

    def test_conditional_residual_configs_use_causal_moments_without_knobs(
        self,
    ) -> None:
        from src.config import load_config, method_config

        method_name = "ascal_gmm_segmented_memory_posterior_conditional_residual"
        for dataset in (
            "genimage",
            "aigc_detection_benchmark",
            "aigi_holmes_p3",
            "opensdid_global",
        ):
            path = (
                PROJECT_ROOT
                / "configs/experiments/clip_vlm_bias_controlled"
                / "matched_jpeg_ascal_gmm_segmented_memory_posterior_"
                f"conditional_residual_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_memory_posterior_projection",
                        method_name,
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
                    f"posterior_conditional_residual_continual/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, method_name)
                for key in (
                    "conditional_window",
                    "confidence_weight",
                    "fusion_weight",
                    "lambda",
                    "learning_rate",
                    "loss_weight",
                    "memory_capacity",
                    "posterior_temperature",
                    "recall_threshold",
                    "shrinkage",
                    "smoothing",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-ConditionalResidual"
                )
                self.assertEqual(reference["research_version"], "R08")
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(reference["residual_count"], 1)
                self.assertIn("source", reference["innovation_rule"])
                self.assertEqual(
                    reference["seed1_promotion_rule"],
                    "accuracy_noninferiority_0p2pp_auc_gain_0p1pp_and_above_r06",
                )
                self.assertFalse(reference["adaptive_score_history_stored"])
                self.assertFalse(reference["adaptive_residual_history_stored"])
                self.assertEqual(reference["optimizer"], "none")
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])

    def test_preroute_configs_use_current_batch_likelihood_without_knobs(
        self,
    ) -> None:
        from src.config import load_config, method_config

        method_name = "ascal_gmm_segmented_memory_posterior_preroute"
        for dataset in (
            "genimage",
            "aigc_detection_benchmark",
            "aigi_holmes_p3",
            "opensdid_global",
        ):
            path = (
                PROJECT_ROOT
                / "configs/experiments/clip_vlm_bias_controlled"
                / "matched_jpeg_ascal_gmm_segmented_memory_posterior_"
                f"preroute_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_memory_posterior_projection",
                        method_name,
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
                    f"posterior_preroute_continual/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, method_name)
                for key in (
                    "confidence_threshold",
                    "fusion_weight",
                    "lambda",
                    "memory_capacity",
                    "recall_threshold",
                    "routing_threshold",
                    "similarity_threshold",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(reference["research_name"], "ASCAL-JMP-PreRoute")
                self.assertEqual(reference["research_version"], "R09")
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(
                    reference["routing_candidates"],
                    "active_learning_state_plus_all_completed_memories",
                )
                self.assertEqual(reference["routing_threshold"], "none")
                self.assertFalse(
                    reference["routing_changes_learning_state_during_prediction"]
                )
                self.assertTrue(reference["batch_transductive_prediction"])
                self.assertEqual(
                    reference["seed1_promotion_rule"],
                    "accuracy_noninferiority_0p2pp_auc_gain_0p1pp_and_above_r06",
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
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_current_projection",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_current_projection_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_guarded_projection",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_guarded_projection_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_support_projection",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_support_projection_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_global_residual",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_global_residual_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_mixture_residual",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_mixture_residual_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_real_deviation_residual",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_real_deviation_residual_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_conditional_residual",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_conditional_residual_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_preroute",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_preroute_static",
            config["training"]["intended_methods"],
        )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required to import the methods package")
class AsymmetricPosteriorTests(unittest.TestCase):
    def test_support_weighted_median_matches_equal_votes_and_exact_ties(self) -> None:
        from src.methods.ascal_gmm import support_weighted_median

        self.assertAlmostEqual(
            support_weighted_median([0.0, 10.0, 20.0], [1, 1, 10]),
            20.0,
        )
        self.assertAlmostEqual(
            support_weighted_median([0.0, 10.0, 20.0], [1, 2, 3]),
            15.0,
        )
        self.assertAlmostEqual(
            support_weighted_median([20.0, 0.0, 10.0, 30.0], [1, 1, 1, 1]),
            15.0,
        )
        with self.assertRaises(ValueError):
            support_weighted_median([0.0], [0])

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
                self.classifier = nn.Linear(3, 2, bias=False)
                with torch.no_grad():
                    self.classifier.weight.copy_(
                        torch.tensor(
                            [
                                [-2.0, 2.0, 0.0],
                                [2.0, -2.0, 0.0],
                            ]
                        )
                    )
                self.feature_dim = 3

            def forward_features(self, images):
                return images.mean(dim=(2, 3))

            def forward(self, images):
                return self.classifier(self.forward_features(images))

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

    def segmented_memory_posterior_preroute_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorPreRoute,
        )

        return ASCALGMMSegmentedMemoryPosteriorPreRoute(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_current_projection_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorCurrentProjection,
        )

        return ASCALGMMSegmentedMemoryPosteriorCurrentProjection(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_guarded_projection_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorGuardedProjection,
        )

        return ASCALGMMSegmentedMemoryPosteriorGuardedProjection(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_support_projection_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorSupportProjection,
        )

        return ASCALGMMSegmentedMemoryPosteriorSupportProjection(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_global_residual_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorGlobalResidual,
        )

        return ASCALGMMSegmentedMemoryPosteriorGlobalResidual(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_mixture_residual_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorMixtureResidual,
        )

        return ASCALGMMSegmentedMemoryPosteriorMixtureResidual(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_real_deviation_residual_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorRealDeviationResidual,
        )

        return ASCALGMMSegmentedMemoryPosteriorRealDeviationResidual(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_conditional_residual_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorConditionalResidual,
        )

        return ASCALGMMSegmentedMemoryPosteriorConditionalResidual(
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

    def score_feature_batch(self, scores: np.ndarray, cues: np.ndarray):
        images = self.score_batch(scores)
        cues = np.asarray(cues, dtype=np.float32).reshape(-1)
        if len(cues) != len(images):
            raise ValueError("Feature cues must match source scores")
        images[:, 2] = self.torch.from_numpy(cues).view(-1, 1, 1)
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

    def test_preroute_uses_the_a_expert_on_the_first_returning_a_batch(self) -> None:
        rng = np.random.default_rng(57)
        method = self.segmented_memory_posterior_preroute_method()

        def adapt_regime(low: float, high: float, batches: int) -> None:
            for _ in range(batches):
                scores = np.concatenate(
                    [rng.normal(low, 0.2, 48), rng.normal(high, 0.2, 48)]
                )
                rng.shuffle(scores)
                images = self.score_batch(scores)
                method.predict(images)
                method.adapt(images)

        adapt_regime(-8.0, -3.0, 4)
        adapt_regime(2.0, 7.0, 2)
        self.assertEqual(method.segment_changes, 1)
        self.assertEqual(len(method.segment_memories), 1)
        self.assertIsNone(method.active_memory_index)
        self.assertIsNone(method.recall_anchor_boundary)

        active_before = copy.deepcopy(method._mixture)
        memories_before = copy.deepcopy(method.segment_memories)
        history_before = list(method.boundary_history)
        segment_changes_before = method.segment_changes
        routing_decisions_before = method.routing_decisions
        routing_active_before = method.routing_active_selections
        routing_memory_before = method.routing_memory_selections
        scores = np.concatenate(
            [rng.normal(-8.0, 0.2, 48), rng.normal(-3.0, 0.2, 48)]
        )
        rng.shuffle(scores)
        images = self.score_batch(scores)
        prediction = method.predict(images)

        self.assertEqual(
            method._pending["prediction_routing_expert"], "episodic_memory"
        )
        self.assertEqual(method._pending["prediction_routing_memory_index"], 0)
        self.assertEqual(method._pending["prediction_routing_candidate_count"], 2)
        self.assertFalse(method._pending["prediction_memory_recalled"])
        self.assertIsNone(method._pending["prediction_memory_index"])
        self.assertGreater(
            method._pending["prediction_routing_gain_over_active"], 0.0
        )
        self.assertAlmostEqual(
            method._pending["prediction_boundary"],
            method.segment_memories[0]["boundary"],
        )
        expected = method._source_probability(
            scores - float(method.segment_memories[0]["boundary"])
        )
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)

        self.assertEqual(method._mixture, active_before)
        self.assertEqual(method.segment_memories, memories_before)
        self.assertEqual(method.boundary_history, history_before)
        self.assertEqual(method.segment_changes, segment_changes_before)
        self.assertEqual(method.routing_decisions, routing_decisions_before)

        stats = method.adapt(images)
        self.assertEqual(
            stats.extra["routing_decisions"], routing_decisions_before + 1
        )
        self.assertEqual(
            stats.extra["routing_memory_selections"], routing_memory_before + 1
        )
        self.assertEqual(
            stats.extra["routing_active_selections"], routing_active_before
        )
        self.assertEqual(stats.extra["routing_memory_selection_counts"], [1, 0])
        self.assertEqual(stats.extra["last_routing_expert"], "episodic_memory")
        self.assertEqual(stats.extra["last_routing_memory_index"], 0)
        self.assertTrue(stats.extra["routing_handoff_this_batch"])
        self.assertEqual(stats.extra["routing_handoff_confirmations"], 1)
        self.assertEqual(stats.extra["routing_handoff_rejections"], 0)
        self.assertEqual(stats.extra["active_memory_index"], 0)
        self.assertTrue(stats.extra["memory_recalled_this_change"])
        self.assertGreater(stats.extra["last_routing_handoff_gain"], 0.0)

    def test_preroute_active_choice_is_exact_r01_and_wins_density_ties(self) -> None:
        baseline = self.segmented_memory_posterior_projection_method()
        method = self.segmented_memory_posterior_preroute_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-2.0, 2.0],
            "sigmas": [0.5, 2.0],
            "components": 2,
            "bic": 0.0,
        }
        boundary = float(method._memory_boundary(mixture))
        episode = {
            "mixture": copy.deepcopy(mixture),
            "boundary": boundary,
            "latest_samples": 96,
            "total_samples": 96,
            "visits": 1,
            "recalls": 0,
        }
        baseline._mixture = copy.deepcopy(mixture)
        method._mixture = copy.deepcopy(mixture)
        baseline.boundary_history = [boundary]
        method.boundary_history = [boundary]
        method.segment_memories = [episode]
        scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        images = self.score_batch(scores)

        expected = baseline.predict(images)
        actual = method.predict(images)
        np.testing.assert_allclose(
            actual.prob_fake.numpy(), expected.prob_fake.numpy(), atol=1e-7
        )
        self.assertEqual(
            method._pending["prediction_routing_expert"], "active_learning_state"
        )
        self.assertIsNone(method._pending["prediction_routing_memory_index"])
        self.assertEqual(method._pending["prediction_routing_candidate_count"], 2)
        self.assertAlmostEqual(
            method._pending["prediction_routing_gain_over_active"], 0.0
        )
        self.assertAlmostEqual(method._pending["prediction_routing_margin"], 0.0)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-PreRoute")
        self.assertEqual(metadata["research_version"], "R09")
        self.assertTrue(metadata["batch_transductive_prediction"])
        self.assertIn("after_prediction", metadata["routing_learning_separation"])

    def test_preroute_rejects_a_forced_nearest_memory_for_a_novel_batch(
        self,
    ) -> None:
        rng = np.random.default_rng(58)
        method = self.segmented_memory_posterior_preroute_method()
        active = {
            "weights": [0.5, 0.5],
            "mus": [20.0, 25.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 0.0,
        }
        memory = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 0.0,
        }
        old_scores = np.concatenate(
            [rng.normal(20.0, 0.2, 48), rng.normal(25.0, 0.2, 48)]
        )
        method._mixture = active
        method.score_history = old_scores.tolist()
        method.score_batches = [old_scores.copy()]
        method.total_score_samples = len(old_scores)
        method.total_score_batches = 1
        method.boundary_history = [22.5]
        method.segment_memories = [
            {
                "mixture": memory,
                "boundary": float(method._memory_boundary(memory)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 0,
            }
        ]
        memory_before = copy.deepcopy(method.segment_memories)
        scores = np.concatenate(
            [rng.normal(2.0, 0.2, 48), rng.normal(7.0, 0.2, 48)]
        )
        rng.shuffle(scores)
        images = self.score_batch(scores)

        method.predict(images)
        self.assertEqual(
            method._pending["prediction_routing_expert"], "episodic_memory"
        )
        stats = method.adapt(images)

        self.assertFalse(stats.extra["routing_handoff_this_batch"])
        self.assertEqual(stats.extra["routing_handoff_confirmations"], 0)
        self.assertEqual(stats.extra["routing_handoff_rejections"], 1)
        self.assertIsNone(stats.extra["active_memory_index"])
        self.assertEqual(method.segment_memories, memory_before)
        self.assertLess(stats.extra["last_routing_handoff_gain"], 0.0)

    def test_current_projection_uses_latest_cumulative_fit_not_nested_median(
        self,
    ) -> None:
        from src.methods.ascal_gmm import equal_density_boundary

        method = self.segmented_memory_posterior_current_projection_method()
        method._mixture = {
            "weights": [0.5, 0.5],
            "mus": [-2.0, 2.0],
            "sigmas": [0.5, 2.0],
            "components": 2,
            "bic": 0.0,
        }
        method.boundary_history = [-100.0, 100.0, 100.0]
        candidate = float(
            equal_density_boundary(method._mixture)["decision_boundary"]
        )
        scores = np.array([-2.0, 0.0, 2.0])

        prediction = method.predict(self.score_batch(scores))
        expected = method._source_probability(scores - candidate)
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)
        self.assertAlmostEqual(method._pending["prediction_boundary"], candidate)
        self.assertEqual(
            method._pending["prediction_mode"],
            "segmented_memory_posterior_current_projection",
        )
        stats = method._state_stats()
        self.assertAlmostEqual(stats["nested_boundary_median"], 100.0)
        self.assertAlmostEqual(stats["stabilized_boundary"], candidate)
        self.assertAlmostEqual(stats["current_fit_boundary"], candidate)
        self.assertAlmostEqual(stats["decision_boundary"], candidate)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-Current")
        self.assertEqual(metadata["research_version"], "R02")
        self.assertIn("no_nested_refit_median", metadata["boundary_stabilization"])

    def test_current_projection_keeps_one_vote_memory_recall(self) -> None:
        from src.methods.ascal_gmm import equal_density_boundary

        method = self.segmented_memory_posterior_current_projection_method()
        method._mixture = {
            "weights": [0.5, 0.5],
            "mus": [-2.0, 2.0],
            "sigmas": [0.5, 2.0],
            "components": 2,
            "bic": 0.0,
        }
        candidate = float(
            equal_density_boundary(method._mixture)["decision_boundary"]
        )
        method.recall_anchor_boundary = -4.0
        method.boundary_history = [candidate]
        scores = np.array([0.0])

        method.predict(self.score_batch(scores))
        self.assertAlmostEqual(method._pending["prediction_boundary"], -4.0)
        self.assertEqual(method._pending["prediction_memory_anchor_weight"], 1.0)

        method.boundary_history.append(candidate)
        method.predict(self.score_batch(scores))
        self.assertAlmostEqual(
            method._pending["prediction_boundary"],
            0.5 * (-4.0 + candidate),
        )
        self.assertEqual(method._pending["prediction_memory_anchor_weight"], 0.5)

    def test_guarded_projection_keeps_the_r01_median_prediction(self) -> None:
        method = self.segmented_memory_posterior_guarded_projection_method()
        method._mixture = {
            "weights": [0.5, 0.5],
            "mus": [-2.0, 2.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 0.0,
        }
        method.boundary_history = [-5.0, -5.0, -5.0]
        scores = np.array([-1.0, 0.0, 1.0])

        prediction = method.predict(self.score_batch(scores))
        expected = method._source_probability(scores + 5.0)
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)
        self.assertAlmostEqual(method._pending["prediction_boundary"], -5.0)
        self.assertEqual(
            method._pending["prediction_mode"],
            "segmented_memory_posterior_guarded_projection",
        )

    def test_guarded_projection_keeps_the_ordinary_scheduled_scan_inside_gap(
        self,
    ) -> None:
        method = self.segmented_memory_posterior_guarded_projection_method()
        method._mixture = {
            "weights": [0.5, 0.5],
            "mus": [-2.0, 2.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 0.0,
        }
        method.boundary_history = [0.0, 0.1, -0.1]
        method.score_batches = [np.zeros(2)] * 16

        self.assertEqual(method._suffix_candidates(), [8])
        self.assertFalse(method._guarded_scan_triggered)
        self.assertEqual(method.guarded_scan_events, 0)
        self.assertEqual(method.guarded_scan_candidate_scales, 0)

    def test_guarded_projection_scans_every_dyadic_suffix_outside_gap(self) -> None:
        method = self.segmented_memory_posterior_guarded_projection_method()
        method._mixture = {
            "weights": [0.5, 0.5],
            "mus": [-2.0, 2.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 0.0,
        }
        method.boundary_history = [-5.0, -5.0, -5.0]
        method.score_batches = [np.zeros(2)] * 16

        self.assertEqual(method._suffix_candidates(), [2, 4, 8])
        self.assertTrue(method._guarded_scan_triggered)
        self.assertEqual(method.guarded_scan_events, 1)
        self.assertEqual(method.guarded_scan_candidate_scales, 3)
        self.assertAlmostEqual(method._guarded_scan_boundary, -5.0)
        self.assertAlmostEqual(method._guarded_scan_gap_lower, -2.0)
        self.assertAlmostEqual(method._guarded_scan_gap_upper, 2.0)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-GuardedScan")
        self.assertEqual(metadata["research_version"], "R03")

    def test_support_projection_uses_sample_support_weighted_median(self) -> None:
        method = self.segmented_memory_posterior_support_projection_method()
        method._mixture = {
            "weights": [0.5, 0.5],
            "mus": [-2.0, 2.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 0.0,
        }
        method.boundary_history = [-4.0, 0.0, 2.0]
        method.boundary_support_history = [16, 32, 160]
        scores = np.array([-1.0, 0.0, 1.0])

        prediction = method.predict(self.score_batch(scores))
        expected = method._source_probability(scores - 2.0)
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)
        self.assertAlmostEqual(method._pending["prediction_boundary"], 2.0)
        self.assertEqual(
            method._pending["prediction_mode"],
            "segmented_memory_posterior_support_projection",
        )
        stats = method._state_stats()
        self.assertAlmostEqual(stats["equal_vote_boundary_median"], 0.0)
        self.assertAlmostEqual(stats["support_weighted_boundary"], 2.0)
        self.assertEqual(stats["boundary_support_entries"], 3)
        self.assertEqual(stats["boundary_support_total"], 208)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-SupportMedian")
        self.assertEqual(metadata["research_version"], "R04")

    def test_support_projection_retains_the_r01_recall_vote(self) -> None:
        method = self.segmented_memory_posterior_support_projection_method()
        method._mixture = {
            "weights": [0.5, 0.5],
            "mus": [-2.0, 2.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 0.0,
        }
        method.boundary_history = [-4.0, 0.0, 2.0]
        method.boundary_support_history = [16, 32, 160]
        method.recall_anchor_boundary = -4.0

        method.predict(self.score_batch(np.array([0.0])))

        expected = (1.0 / 3.0) * -4.0 + (2.0 / 3.0) * 2.0
        self.assertAlmostEqual(method._pending["prediction_boundary"], expected)
        self.assertEqual(
            method._pending["prediction_memory_anchor_weight"], 1.0 / 3.0
        )

    def test_support_projection_records_the_samples_summarized_by_each_fit(
        self,
    ) -> None:
        rng = np.random.default_rng(41)
        method = self.segmented_memory_posterior_support_projection_method()
        for _ in range(2):
            scores = np.concatenate(
                [rng.normal(-4.0, 0.2, 32), rng.normal(4.0, 0.2, 32)]
            )
            images = self.score_batch(scores)
            method.predict(images)
            stats = method.adapt(images)

        self.assertEqual(method.boundary_support_history, [64, 128])
        self.assertEqual(stats.extra["boundary_support_entries"], 2)
        self.assertEqual(stats.extra["boundary_support_total"], 192)
        self.assertEqual(stats.extra["boundary_support_latest"], 128)

    def test_support_projection_clears_support_at_a_segment_change(self) -> None:
        method = self.segmented_memory_posterior_support_projection_method()
        old_mixture = {
            "weights": [0.5, 0.5],
            "mus": [-4.0, 0.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 10.0,
        }
        new_mixture = {
            "weights": [0.5, 0.5],
            "mus": [0.0, 4.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 8.0,
        }
        method.boundary_history = [-2.0, -1.5]
        method.boundary_support_history = [64, 128]

        method._on_segment_change(
            old_mixture=old_mixture,
            old_samples=128,
            new_mixture=new_mixture,
            new_scores=np.array([0.0, 0.2, 3.8, 4.0]),
        )

        self.assertEqual(method.boundary_support_history, [])
        self.assertEqual(method.memory_segments_created, 1)

    def test_global_residual_learns_one_complementary_feature_direction(self) -> None:
        rng = np.random.default_rng(52)
        scores = np.concatenate(
            [rng.normal(-4.0, 0.2, 64), rng.normal(4.0, 0.2, 64)]
        )
        cues = np.concatenate([-np.ones(64), np.ones(64)])
        order = rng.permutation(len(scores))
        images = self.score_feature_batch(scores[order], cues[order])
        baseline = self.segmented_memory_posterior_projection_method()
        method = self.segmented_memory_posterior_global_residual_method()

        baseline_first = baseline.predict(images)
        method_first = method.predict(images)
        np.testing.assert_allclose(
            method_first.prob_fake.numpy(),
            baseline_first.prob_fake.numpy(),
            atol=1e-7,
        )
        baseline.adapt(images)
        stats = method.adapt(images)

        np.testing.assert_allclose(method.score_history, scores[order], atol=1e-6)
        np.testing.assert_allclose(method.score_history, baseline.score_history)
        self.assertTrue(stats.extra["residual_updated"])
        self.assertTrue(stats.extra["residual_ready"])
        self.assertEqual(stats.extra["residual_count"], 1)
        self.assertEqual(stats.extra["residual_updates"], 1)
        self.assertGreater(stats.extra["residual_real_support"], 0.0)
        self.assertGreater(stats.extra["residual_fake_support"], 0.0)
        self.assertAlmostEqual(
            float(method.residual_vector @ method._source_feature_direction),
            0.0,
            places=12,
        )

        query = self.score_feature_batch(
            np.array([0.0, 0.0]),
            np.array([-1.0, 1.0]),
        )
        baseline_query = baseline.predict(query)
        method_query = method.predict(query)
        self.assertAlmostEqual(
            float(baseline_query.prob_fake[0]),
            float(baseline_query.prob_fake[1]),
            places=7,
        )
        self.assertLess(
            float(method_query.prob_fake[0]),
            float(method_query.prob_fake[1]),
        )
        self.assertTrue(method._pending["prediction_residual_ready"])
        self.assertGreater(method._pending["prediction_residual_max_abs"], 0.0)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-GlobalResidual")
        self.assertEqual(metadata["research_version"], "R05")
        self.assertEqual(metadata["residual_count"], 1)
        self.assertFalse(metadata["adaptive_score_history_stored"])

    def test_global_residual_is_predict_then_adapt_and_survives_unimodal_batches(
        self,
    ) -> None:
        rng = np.random.default_rng(53)
        method = self.segmented_memory_posterior_global_residual_method()
        scores = np.concatenate(
            [rng.normal(-3.0, 0.1, 64), rng.normal(3.0, 0.1, 64)]
        )
        cues = np.concatenate([-np.ones(64), np.ones(64)])
        images = self.score_feature_batch(scores, cues)

        first = method.predict(images)
        expected = 1.0 / (1.0 + np.exp(-scores))
        np.testing.assert_allclose(first.prob_fake.numpy(), expected, atol=1e-6)
        method.adapt(images)
        learned = method.residual_vector.copy()

        method._mixture = {
            "weights": [1.0],
            "mus": [0.0],
            "sigmas": [1.0],
            "components": 1,
            "bic": 0.0,
        }
        query = self.score_feature_batch(
            np.array([0.0, 0.0]),
            np.array([-1.0, 1.0]),
        )
        prediction = method.predict(query)
        self.assertLess(
            float(prediction.prob_fake[0]),
            float(prediction.prob_fake[1]),
        )
        stats = method.adapt(query)
        self.assertFalse(stats.extra["residual_updated"])
        np.testing.assert_allclose(method.residual_vector, learned)
        self.assertEqual(stats.extra["residual_updates"], 1)

    def test_mixture_residual_preserves_fake_modes_that_global_mean_cancels(
        self,
    ) -> None:
        global_method = self.segmented_memory_posterior_global_residual_method()
        method = self.segmented_memory_posterior_mixture_residual_method()
        mixture = {
            "weights": [1.0 / 3.0] * 3,
            "mus": [-4.0, 2.0, 6.0],
            "sigmas": [0.2, 0.2, 0.2],
            "components": 3,
            "bic": 0.0,
        }
        global_method._mixture = mixture
        method._mixture = mixture

        root_half = 1.0 / np.sqrt(2.0)
        real = np.array([root_half, root_half, 0.0])
        tangent = np.array([0.0, 0.0, 1.0])
        fake_left = 0.5 * real + np.sqrt(0.75) * tangent
        fake_right = 0.5 * real - np.sqrt(0.75) * tangent
        scores = np.repeat(np.array([-4.0, 2.0, 6.0]), 48)
        features = np.concatenate(
            [
                np.repeat(real[None, :], 48, axis=0),
                np.repeat(fake_left[None, :], 48, axis=0),
                np.repeat(fake_right[None, :], 48, axis=0),
            ],
            axis=0,
        )

        self.assertTrue(global_method._update_global_residual(scores, features))
        self.assertTrue(method._update_global_residual(scores, features))
        self.assertLess(float(np.linalg.norm(global_method.residual_vector)), 1e-8)
        self.assertTrue(method._residual_ready())
        stats = method._residual_state_stats()
        self.assertEqual(stats["residual_count"], 1)
        self.assertEqual(stats["residual_fake_prototype_count"], 2)
        self.assertEqual(stats["residual_fake_components_observed"], 2)
        self.assertEqual(stats["residual_multimodal_updates"], 1)
        self.assertEqual(stats["residual_readout_bound"], 2.0)
        residual = method._residual_scores(
            np.stack([real, fake_left, fake_right], axis=0)
        )
        self.assertLess(float(residual[0]), 0.0)
        self.assertGreater(float(residual[1]), 0.0)
        self.assertGreater(float(residual[2]), 0.0)
        self.assertTrue(np.all(np.abs(residual) <= 2.0))

    def test_mixture_residual_is_predict_then_adapt_and_keeps_r01_history(
        self,
    ) -> None:
        rng = np.random.default_rng(54)
        scores = np.concatenate(
            [
                rng.normal(-4.0, 0.15, 64),
                rng.normal(2.0, 0.15, 64),
                rng.normal(6.0, 0.15, 64),
            ]
        )
        cues = np.concatenate(
            [np.zeros(64), np.full(64, 0.8), np.full(64, -2.4)]
        )
        order = rng.permutation(len(scores))
        images = self.score_feature_batch(scores[order], cues[order])
        baseline = self.segmented_memory_posterior_projection_method()
        method = self.segmented_memory_posterior_mixture_residual_method()

        baseline_first = baseline.predict(images)
        method_first = method.predict(images)
        np.testing.assert_allclose(
            method_first.prob_fake.numpy(),
            baseline_first.prob_fake.numpy(),
            atol=1e-7,
        )
        baseline.adapt(images)
        stats = method.adapt(images)

        np.testing.assert_allclose(method.score_history, baseline.score_history)
        self.assertTrue(stats.extra["residual_updated"])
        self.assertTrue(stats.extra["residual_ready"])
        self.assertEqual(stats.extra["residual_count"], 1)
        self.assertEqual(stats.extra["residual_fake_prototype_count"], 2)
        self.assertEqual(method.trainable_parameters, 9)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-MixtureResidual")
        self.assertEqual(metadata["research_version"], "R06")
        self.assertEqual(metadata["residual_count"], 1)
        self.assertFalse(metadata["adaptive_score_history_stored"])

    def test_real_deviation_residual_is_centered_on_the_soft_real_measure(
        self,
    ) -> None:
        method = self.segmented_memory_posterior_real_deviation_residual_method()
        method._mixture = {
            "weights": [0.5, 0.5],
            "mus": [-4.0, 4.0],
            "sigmas": [0.2, 0.2],
            "components": 2,
            "bic": 0.0,
        }
        root_half = 1.0 / np.sqrt(2.0)
        real = np.array([root_half, root_half, 0.0])
        fake = np.array([0.0, 0.0, 1.0])
        scores = np.repeat(np.array([-4.0, 4.0]), 64)
        features = np.concatenate(
            [
                np.repeat(real[None, :], 64, axis=0),
                np.repeat(fake[None, :], 64, axis=0),
            ],
            axis=0,
        )

        self.assertTrue(method._update_global_residual(scores, features))
        residual = method._residual_scores(np.stack([real, fake], axis=0))
        self.assertAlmostEqual(float(residual[0]), 0.0, places=12)
        self.assertAlmostEqual(float(residual[1]), 1.0, places=12)
        self.assertGreaterEqual(float(residual.min()), -0.25)
        self.assertLessEqual(float(residual.max()), 2.0)
        stats = method._residual_state_stats()
        self.assertTrue(stats["residual_ready"])
        self.assertEqual(stats["residual_count"], 1)
        self.assertEqual(stats["residual_fake_prototype_count"], 0)
        self.assertAlmostEqual(stats["residual_real_resultant_length"], 1.0)
        self.assertEqual(stats["residual_readout_lower_bound"], -0.25)
        self.assertEqual(stats["residual_readout_upper_bound"], 2.0)
        self.assertAlmostEqual(
            float(real @ method._source_feature_direction),
            0.0,
            places=12,
        )

    def test_real_deviation_residual_is_causal_and_keeps_r01_history(self) -> None:
        rng = np.random.default_rng(55)
        scores = np.concatenate(
            [rng.normal(-4.0, 0.15, 64), rng.normal(4.0, 0.15, 64)]
        )
        cues = np.concatenate([np.zeros(64), np.ones(64)])
        order = rng.permutation(len(scores))
        images = self.score_feature_batch(scores[order], cues[order])
        baseline = self.segmented_memory_posterior_projection_method()
        method = self.segmented_memory_posterior_real_deviation_residual_method()

        baseline_first = baseline.predict(images)
        method_first = method.predict(images)
        np.testing.assert_allclose(
            method_first.prob_fake.numpy(),
            baseline_first.prob_fake.numpy(),
            atol=1e-7,
        )
        baseline.adapt(images)
        stats = method.adapt(images)

        np.testing.assert_allclose(method.score_history, baseline.score_history)
        self.assertTrue(stats.extra["residual_updated"])
        self.assertTrue(stats.extra["residual_ready"])
        self.assertEqual(stats.extra["residual_updates"], 1)
        self.assertEqual(stats.extra["residual_fake_prototype_count"], 0)
        self.assertEqual(method.trainable_parameters, 3)
        method.predict(images)
        self.assertEqual(
            method._pending["prediction_mode"],
            "segmented_memory_posterior_real_deviation_residual",
        )
        self.assertTrue(method._pending["prediction_residual_ready"])
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-RealDeviation")
        self.assertEqual(metadata["research_version"], "R07")
        self.assertEqual(metadata["residual_count"], 1)
        self.assertEqual(
            metadata["residual_expectation"],
            "zero_under_the_accumulated_soft_real_measure",
        )
        self.assertFalse(metadata["adaptive_score_history_stored"])

    def test_conditional_residual_moments_extract_a_trusted_orthogonal_innovation(
        self,
    ) -> None:
        method = self.segmented_memory_posterior_conditional_residual_method()
        scores = np.tile(np.array([-2.0, -1.0, 1.0, 2.0]), 16)
        noise = np.tile(np.array([1.0, -2.0, 2.0, -1.0]), 16)
        raw_residual = scores + noise

        self.assertTrue(
            method._update_conditional_moments(scores[:32], raw_residual[:32])
        )
        self.assertTrue(
            method._update_conditional_moments(scores[32:], raw_residual[32:])
        )
        state = method._conditional_moment_state()
        self.assertTrue(state["conditional_moments_ready"])
        self.assertTrue(state["conditional_innovation_ready"])
        self.assertEqual(state["conditional_moment_samples"], 64)
        self.assertEqual(state["conditional_moment_updates"], 2)
        self.assertAlmostEqual(state["conditional_projection_slope"], 1.0)
        self.assertAlmostEqual(
            state["conditional_correlation"],
            1.0 / np.sqrt(2.0),
        )
        self.assertAlmostEqual(state["conditional_innovation_rms_source_ratio"], 0.5)

        innovation = method._conditional_innovation(scores, raw_residual)
        self.assertAlmostEqual(float(np.mean(innovation)), 0.0, places=12)
        self.assertAlmostEqual(
            float(np.mean((scores - scores.mean()) * innovation)),
            0.0,
            places=12,
        )
        np.testing.assert_allclose(innovation, 0.5 * noise, atol=1e-12)

        disagreeing = self.segmented_memory_posterior_conditional_residual_method()
        disagreeing._update_conditional_moments(scores, -scores)
        disagreeing_state = disagreeing._conditional_moment_state()
        self.assertEqual(disagreeing_state["conditional_innovation_trust"], 0.0)
        self.assertFalse(disagreeing_state["conditional_innovation_ready"])
        np.testing.assert_array_equal(
            disagreeing._conditional_innovation(scores, -scores),
            np.zeros_like(scores),
        )

    def test_conditional_residual_is_predict_then_adapt_and_keeps_r01_history(
        self,
    ) -> None:
        rng = np.random.default_rng(56)
        scores = np.concatenate(
            [rng.normal(-4.0, 0.2, 64), rng.normal(4.0, 0.2, 64)]
        )
        cues = np.concatenate([-np.ones(64), np.ones(64)])
        order = rng.permutation(len(scores))
        images = self.score_feature_batch(scores[order], cues[order])
        baseline = self.segmented_memory_posterior_projection_method()
        method = self.segmented_memory_posterior_conditional_residual_method()

        baseline_first = baseline.predict(images)
        method_first = method.predict(images)
        np.testing.assert_allclose(
            method_first.prob_fake.numpy(),
            baseline_first.prob_fake.numpy(),
            atol=1e-7,
        )
        baseline.adapt(images)
        first_stats = method.adapt(images)
        self.assertTrue(first_stats.extra["residual_updated"])
        self.assertFalse(first_stats.extra["conditional_moment_updated"])
        self.assertEqual(first_stats.extra["conditional_moment_samples"], 0)

        baseline.predict(images)
        second = method.predict(images)
        self.assertTrue(method._pending["prediction_residual_ready"])
        self.assertFalse(method._pending["prediction_conditional_moments_ready"])
        self.assertEqual(
            method._pending["prediction_conditional_innovation_max_abs"], 0.0
        )
        baseline.adapt(images)
        second_stats = method.adapt(images)
        self.assertTrue(second_stats.extra["conditional_moment_updated"])
        self.assertEqual(second_stats.extra["conditional_moment_samples"], len(scores))
        np.testing.assert_allclose(method.score_history, baseline.score_history)

        third = method.predict(images)
        self.assertTrue(method._pending["prediction_conditional_moments_ready"])
        self.assertTrue(method._pending["prediction_conditional_innovation_ready"])
        self.assertGreater(
            method._pending["prediction_conditional_innovation_trust"], 0.0
        )
        self.assertEqual(
            method._pending["prediction_mode"],
            "segmented_memory_posterior_conditional_residual",
        )
        self.assertFalse(np.array_equal(second.prob_fake.numpy(), third.prob_fake.numpy()))
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-ConditionalResidual")
        self.assertEqual(metadata["research_version"], "R08")
        self.assertEqual(metadata["residual_count"], 1)
        self.assertFalse(metadata["adaptive_score_history_stored"])

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

    def test_method_factory_maps_preroute_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorPreRoute,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_preroute_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(method, ASCALGMMSegmentedMemoryPosteriorPreRoute)
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_current_projection_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorCurrentProjection,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_current_projection_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorCurrentProjection
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_method_factory_maps_guarded_projection_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorGuardedProjection,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_guarded_projection_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorGuardedProjection
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_method_factory_maps_support_projection_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorSupportProjection,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_support_projection_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorSupportProjection
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_method_factory_maps_global_residual_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorGlobalResidual,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_global_residual_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorGlobalResidual
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_mixture_residual_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorMixtureResidual,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_mixture_residual_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorMixtureResidual
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_real_deviation_residual_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorRealDeviationResidual,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_real_deviation_residual_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorRealDeviationResidual
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_conditional_residual_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorConditionalResidual,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_conditional_residual_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorConditionalResidual
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_cli_builds_mixture_residual_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorMixtureResidual,
        )

        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                "ascal_gmm_segmented_memory_posterior_mixture_residual_static": {
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
                "ascal_gmm_segmented_memory_posterior_mixture_residual_static",
                "cpu",
            )

        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorMixtureResidual
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_real_deviation_residual_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorRealDeviationResidual,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_real_deviation_residual_static"
        )
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {method_name: {"adaptation_mode": "static"}},
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorRealDeviationResidual
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_conditional_residual_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorConditionalResidual,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_conditional_residual_static"
        )
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {method_name: {"adaptation_mode": "static"}},
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorConditionalResidual
        )
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

    def test_cli_builds_preroute_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorPreRoute,
        )

        method_name = "ascal_gmm_segmented_memory_posterior_preroute_static"
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {method_name: {"adaptation_mode": "static"}},
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(method, ASCALGMMSegmentedMemoryPosteriorPreRoute)
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_current_projection_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorCurrentProjection,
        )

        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                "ascal_gmm_segmented_memory_posterior_current_projection_static": {
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
                "ascal_gmm_segmented_memory_posterior_current_projection_static",
                "cpu",
            )

        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorCurrentProjection
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_guarded_projection_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorGuardedProjection,
        )

        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                "ascal_gmm_segmented_memory_posterior_guarded_projection_static": {
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
                "ascal_gmm_segmented_memory_posterior_guarded_projection_static",
                "cpu",
            )

        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorGuardedProjection
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_support_projection_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorSupportProjection,
        )

        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                "ascal_gmm_segmented_memory_posterior_support_projection_static": {
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
                "ascal_gmm_segmented_memory_posterior_support_projection_static",
                "cpu",
            )

        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorSupportProjection
        )
        self.assertEqual(method.adaptation_mode, "static")


if __name__ == "__main__":
    unittest.main()
