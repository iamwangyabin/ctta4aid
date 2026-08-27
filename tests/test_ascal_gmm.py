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

    def test_mdl_route_configs_share_one_assignment_without_knobs(self) -> None:
        from src.config import load_config, method_config

        method_name = "ascal_gmm_segmented_memory_posterior_mdl_route"
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
                f"mdl_route_continual_{dataset}_seed1.yaml"
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
                    f"posterior_mdl_route_continual/{dataset}/seed1",
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
                self.assertEqual(reference["research_name"], "ASCAL-JMP-MDLRoute")
                self.assertEqual(reference["research_version"], "R10")
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(
                    reference["assignment_consistency"],
                    "same_admitted_expert_predicts_and_receives_adaptation",
                )
                self.assertEqual(reference["routing_threshold"], "none")
                self.assertFalse(reference["prediction_mutates_experts"])
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

    def test_live_route_configs_expose_one_state_per_expert_without_knobs(
        self,
    ) -> None:
        from src.config import load_config, method_config

        method_name = "ascal_gmm_segmented_memory_posterior_live_route"
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
                f"live_route_continual_{dataset}_seed1.yaml"
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
                    f"posterior_live_route_continual/{dataset}/seed1",
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
                self.assertEqual(reference["research_name"], "ASCAL-JMP-LiveRoute")
                self.assertEqual(reference["research_version"], "R11")
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(
                    reference["routing_candidates"],
                    "active_live_state_plus_archived_non_active_experts",
                )
                self.assertIn("hidden", reference["active_snapshot_rule"])
                self.assertEqual(reference["routing_threshold"], "none")
                self.assertFalse(reference["prediction_mutates_experts"])
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

    def test_ordinal_route_configs_lock_accuracy_and_restore_global_order(
        self,
    ) -> None:
        from src.config import load_config, method_config

        method_name = "ascal_gmm_segmented_memory_posterior_ordinal_route"
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
                f"ordinal_route_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_memory_posterior_live_route",
                        method_name,
                    ],
                )
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                config_dir = Path(config["_config_path"]).parent
                for field in (
                    "locked_online_manifest",
                    "locked_final_holdout_manifest",
                ):
                    manifest = Path(config["data"][field]).expanduser()
                    if not manifest.is_absolute():
                        manifest = config_dir / manifest
                    self.assertTrue(manifest.resolve().is_file())
                self.assertIn(
                    f"posterior_ordinal_route_continual/{dataset}/seed1",
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
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-OrdinalRoute"
                )
                self.assertEqual(reference["research_version"], "R12")
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(
                    reference["accuracy_invariance"],
                    "exact_r11_hard_decision_preservation",
                )
                self.assertEqual(
                    reference["global_rank_coordinate"],
                    "immutable_frozen_source_probability_with_source_temperature",
                )
                self.assertEqual(reference["routing_threshold"], "none")
                self.assertFalse(reference["prediction_mutates_experts"])
                self.assertTrue(reference["batch_transductive_prediction"])
                self.assertEqual(
                    reference["seed1_promotion_rule"],
                    "accuracy_within_0p2pp_of_r11_and_auc_above_r06",
                )
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])

    def test_ordinal_ridge_configs_protect_r12_without_target_knobs(self) -> None:
        from src.config import load_config, method_config

        baseline_name = "ascal_gmm_segmented_memory_posterior_ordinal_route"
        method_name = "ascal_gmm_segmented_memory_posterior_ordinal_ridge"
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
                f"ordinal_ridge_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(config["methods"], [baseline_name, method_name])
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                config_dir = Path(config["_config_path"]).parent
                for field in (
                    "locked_online_manifest",
                    "locked_final_holdout_manifest",
                ):
                    manifest = Path(config["data"][field]).expanduser()
                    if not manifest.is_absolute():
                        manifest = config_dir / manifest
                    self.assertTrue(manifest.resolve().is_file())
                self.assertIn(
                    f"posterior_ordinal_ridge_continual/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, method_name)
                for key in (
                    "confidence_threshold",
                    "fusion_weight",
                    "lambda",
                    "learning_rate",
                    "memory_capacity",
                    "recall_threshold",
                    "ridge_alpha",
                    "similarity_threshold",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-OrdinalRidge"
                )
                self.assertEqual(reference["research_version"], "R15")
                self.assertEqual(reference["protected_base"], "exact_r12_ordinal_route")
                self.assertIn("every_r12", reference["accuracy_invariance"])
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(reference["routing_threshold"], "none")
                self.assertEqual(reference["optimizer"], "none")
                self.assertEqual(reference["learning_rate"], "none")
                self.assertEqual(reference["epochs"], "none")
                self.assertFalse(reference["prediction_mutates_experts"])
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])
                self.assertFalse(reference["raw_features_stored"])

    def test_joint_ridge_configs_unlock_r12_without_target_knobs(self) -> None:
        from src.config import load_config, method_config

        baseline_name = "ascal_gmm_segmented_memory_posterior_ordinal_route"
        method_name = "ascal_gmm_segmented_memory_posterior_joint_ridge"
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
                f"joint_ridge_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(config["methods"], [baseline_name, method_name])
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                config_dir = Path(config["_config_path"]).parent
                for field in (
                    "locked_online_manifest",
                    "locked_final_holdout_manifest",
                ):
                    manifest = Path(config["data"][field]).expanduser()
                    if not manifest.is_absolute():
                        manifest = config_dir / manifest
                    self.assertTrue(manifest.resolve().is_file())
                self.assertIn(
                    f"posterior_joint_ridge_continual/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, method_name)
                for key in (
                    "confidence_threshold",
                    "fusion_weight",
                    "lambda",
                    "learning_rate",
                    "memory_capacity",
                    "recall_threshold",
                    "ridge_alpha",
                    "similarity_threshold",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(reference["research_name"], "ASCAL-JMP-JointRidge")
                self.assertEqual(reference["research_version"], "R16")
                self.assertEqual(
                    reference["protected_initialization"],
                    "exact_r12_ordinal_route",
                )
                self.assertIn("may_change", reference["accuracy_invariance"])
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(reference["routing_threshold"], "none")
                self.assertEqual(reference["optimizer"], "none")
                self.assertEqual(reference["learning_rate"], "none")
                self.assertEqual(reference["epochs"], "none")
                self.assertEqual(reference["residual_target_centering"], "none")
                self.assertEqual(reference["prediction_residual_centering"], "none")
                self.assertIn("bias", reference["residual_intercept"])
                self.assertEqual(
                    reference["routing_score_coordinate"],
                    "immutable_source_score",
                )
                self.assertEqual(
                    reference["gmm_update_score_coordinate"],
                    "immutable_source_score",
                )
                self.assertFalse(reference["prediction_mutates_experts"])
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])
                self.assertFalse(reference["raw_features_stored"])

    def test_pairwise_ridge_configs_target_rank_without_target_knobs(self) -> None:
        from src.config import load_config, method_config

        baseline_name = "ascal_gmm_segmented_memory_posterior_ordinal_route"
        method_name = "ascal_gmm_segmented_memory_posterior_pairwise_ridge"
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
                f"pairwise_ridge_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(config["methods"], [baseline_name, method_name])
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                config_dir = Path(config["_config_path"]).parent
                for field in (
                    "locked_online_manifest",
                    "locked_final_holdout_manifest",
                ):
                    manifest = Path(config["data"][field]).expanduser()
                    if not manifest.is_absolute():
                        manifest = config_dir / manifest
                    self.assertTrue(manifest.resolve().is_file())
                self.assertIn(
                    f"posterior_pairwise_ridge_continual/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, method_name)
                for key in (
                    "confidence_threshold",
                    "fusion_weight",
                    "lambda",
                    "learning_rate",
                    "memory_capacity",
                    "pair_margin",
                    "recall_threshold",
                    "residual_weight",
                    "ridge_alpha",
                    "similarity_threshold",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-PairwiseRidge"
                )
                self.assertEqual(reference["research_version"], "R17")
                self.assertEqual(
                    reference["protected_initialization"],
                    "exact_r12_ordinal_route",
                )
                self.assertEqual(
                    reference["ranker_scope"],
                    "one_global_stream_wide_online_pairwise_ridge",
                )
                self.assertIn("none", reference["ranker_bias"])
                self.assertIn("r12", reference["pair_class_side"])
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(reference["routing_threshold"], "none")
                self.assertEqual(reference["optimizer"], "none")
                self.assertEqual(reference["learning_rate"], "none")
                self.assertEqual(reference["epochs"], "none")
                self.assertEqual(
                    reference["routing_score_coordinate"],
                    "immutable_source_score",
                )
                self.assertFalse(reference["prediction_mutates_ranker"])
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])
                self.assertFalse(reference["raw_features_stored"])
                self.assertFalse(reference["raw_pairs_stored"])

    def test_analytic_expert_configs_use_bounded_soft_labels_without_knobs(
        self,
    ) -> None:
        from src.config import load_config, method_config

        baseline_name = "ascal_gmm_segmented_memory_posterior_ordinal_route"
        method_name = "ascal_gmm_segmented_memory_posterior_analytic_expert"
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
                f"analytic_expert_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(config["methods"], [baseline_name, method_name])
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                config_dir = Path(config["_config_path"]).parent
                for field in (
                    "locked_online_manifest",
                    "locked_final_holdout_manifest",
                ):
                    manifest = Path(config["data"][field]).expanduser()
                    if not manifest.is_absolute():
                        manifest = config_dir / manifest
                    self.assertTrue(manifest.resolve().is_file())
                self.assertIn(
                    f"posterior_analytic_expert_continual/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, method_name)
                for key in (
                    "confidence_threshold",
                    "fusion_weight",
                    "lambda",
                    "learning_rate",
                    "memory_capacity",
                    "recall_threshold",
                    "residual_weight",
                    "ridge_alpha",
                    "similarity_threshold",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-AnalyticExpert"
                )
                self.assertEqual(reference["research_version"], "R18")
                self.assertEqual(
                    reference["protected_initialization"],
                    "exact_r12_ordinal_route_probability",
                )
                self.assertIn("per_r12_gmm_expert", reference["expert_scope"])
                self.assertIn("r12_signed", reference["expert_input"])
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(reference["routing_threshold"], "none")
                self.assertEqual(reference["optimizer"], "none")
                self.assertEqual(reference["learning_rate"], "none")
                self.assertEqual(reference["epochs"], "none")
                self.assertEqual(
                    reference["routing_score_coordinate"],
                    "immutable_source_score",
                )
                self.assertFalse(reference["prediction_mutates_experts"])
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])
                self.assertFalse(reference["raw_features_stored"])

    def test_rms_ridge_expert_configs_use_direct_binary_targets_and_past_rms(
        self,
    ) -> None:
        from src.config import load_config, method_config

        baseline_name = "ascal_gmm_segmented_memory_posterior_ordinal_route"
        method_name = (
            "ascal_gmm_segmented_memory_posterior_rms_ridge_expert"
        )
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
                f"rms_ridge_expert_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(config["methods"], [baseline_name, method_name])
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                config_dir = Path(config["_config_path"]).parent
                for field in (
                    "locked_online_manifest",
                    "locked_final_holdout_manifest",
                ):
                    manifest = Path(config["data"][field]).expanduser()
                    if not manifest.is_absolute():
                        manifest = config_dir / manifest
                    self.assertTrue(manifest.resolve().is_file())
                self.assertIn(
                    f"posterior_rms_ridge_expert_continual/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, method_name)
                for key in (
                    "confidence_threshold",
                    "fusion_weight",
                    "lambda",
                    "learning_rate",
                    "memory_capacity",
                    "min_samples",
                    "recall_threshold",
                    "residual_weight",
                    "ridge_alpha",
                    "similarity_threshold",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-RMSRidgeExpert"
                )
                self.assertEqual(reference["research_version"], "R19")
                self.assertEqual(reference["ridge_target"], "real_or_fake_one_hot_vector")
                self.assertIn("historical_reliability", reference["scale_alignment"])
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(reference["routing_threshold"], "none")
                self.assertEqual(reference["optimizer"], "none")
                self.assertEqual(reference["learning_rate"], "none")
                self.assertEqual(reference["epochs"], "none")
                self.assertEqual(
                    reference["routing_score_coordinate"],
                    "immutable_source_score",
                )
                self.assertFalse(reference["prediction_mutates_experts"])
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])
                self.assertFalse(reference["raw_features_stored"])

    def test_equal_prior_ridge_configs_change_only_the_r19_readout(self) -> None:
        from src.config import load_config, method_config

        baseline_name = "ascal_gmm_segmented_memory_posterior_ordinal_route"
        method_name = (
            "ascal_gmm_segmented_memory_posterior_equal_prior_ridge_expert"
        )
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
                f"equal_prior_ridge_expert_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(config["methods"], [baseline_name, method_name])
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                config_dir = Path(config["_config_path"]).parent
                for field in (
                    "locked_online_manifest",
                    "locked_final_holdout_manifest",
                ):
                    manifest = Path(config["data"][field]).expanduser()
                    if not manifest.is_absolute():
                        manifest = config_dir / manifest
                    self.assertTrue(manifest.resolve().is_file())
                self.assertIn(
                    f"posterior_equal_prior_ridge_expert_continual/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, method_name)
                for key in (
                    "confidence_threshold",
                    "fusion_weight",
                    "lambda",
                    "learning_rate",
                    "memory_capacity",
                    "min_samples",
                    "recall_threshold",
                    "residual_weight",
                    "ridge_alpha",
                    "similarity_threshold",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-EqualPriorRidge"
                )
                self.assertEqual(reference["research_version"], "R20")
                self.assertIn("exact_r19", reference["protected_r19_learning"])
                self.assertIn("midpoint", reference["equal_prior_center"])
                self.assertEqual(
                    reference["class_prior_assumption"], "equal_real_fake_prior"
                )
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(reference["routing_threshold"], "none")
                self.assertEqual(reference["optimizer"], "none")
                self.assertEqual(reference["learning_rate"], "none")
                self.assertEqual(reference["epochs"], "none")
                self.assertEqual(
                    reference["routing_score_coordinate"],
                    "immutable_source_score",
                )
                self.assertFalse(reference["prediction_mutates_experts"])
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])
                self.assertFalse(reference["raw_features_stored"])

    def test_evidence_gated_ridge_configs_change_only_the_r20_readout(
        self,
    ) -> None:
        from src.config import load_config, method_config

        baseline_name = "ascal_gmm_segmented_memory_posterior_ordinal_route"
        method_name = (
            "ascal_gmm_segmented_memory_posterior_evidence_gated_ridge_expert"
        )
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
                f"evidence_gated_ridge_expert_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(config["methods"], [baseline_name, method_name])
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                config_dir = Path(config["_config_path"]).parent
                for field in (
                    "locked_online_manifest",
                    "locked_final_holdout_manifest",
                ):
                    manifest = Path(config["data"][field]).expanduser()
                    if not manifest.is_absolute():
                        manifest = config_dir / manifest
                    self.assertTrue(manifest.resolve().is_file())
                self.assertIn(
                    f"evidence_gated_ridge_expert_continual/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, method_name)
                for key in (
                    "confidence_threshold",
                    "fusion_weight",
                    "lambda",
                    "learning_rate",
                    "memory_capacity",
                    "min_samples",
                    "recall_threshold",
                    "residual_weight",
                    "ridge_alpha",
                    "similarity_threshold",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-EvidenceGatedRidge"
                )
                self.assertEqual(reference["research_version"], "R21")
                self.assertIn("exact_r20", reference["protected_ridge_learning"])
                self.assertIn("exact_r20", reference["protected_equal_prior"])
                self.assertEqual(reference["evidence_threshold"], "none")
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertEqual(reference["routing_threshold"], "none")
                self.assertEqual(reference["optimizer"], "none")
                self.assertEqual(reference["learning_rate"], "none")
                self.assertEqual(reference["epochs"], "none")
                self.assertEqual(
                    reference["routing_score_coordinate"],
                    "immutable_source_score",
                )
                self.assertFalse(reference["prediction_mutates_experts"])
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])
                self.assertFalse(reference["raw_features_stored"])

    def test_feature_routed_trusted_ridge_config_separates_module_roles(
        self,
    ) -> None:
        from src.config import load_config, method_config

        method_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_trusted_ridge"
        )
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_"
            "feature_routed_trusted_ridge_continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [method_name])
        self.assertEqual(config["seed"], 1)
        self.assertFalse(config["protocol"]["reset_between_domains"])
        self.assertFalse(config["protocol"]["generator_id_available_to_method"])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(
            reference["research_name"],
            "ASCAL-JMP-FeatureRoutedTrustedRidge",
        )
        self.assertEqual(reference["research_version"], "R22")
        self.assertEqual(
            reference["routing_coordinate"],
            "l2_normalized_frozen_clip_feature_orthogonal_to_source_binary_head",
        )
        self.assertFalse(reference["routing_score_used"])
        self.assertFalse(reference["gmm_in_final_prediction"])
        self.assertEqual(reference["new_target_hyperparameters"], 0)
        self.assertFalse(reference["target_labels_used"])
        self.assertTrue(reference["semantic_features_used"])
        for key in (
            "confidence_threshold",
            "fusion_weight",
            "learning_rate",
            "memory_capacity",
            "ridge_alpha",
            "routing_threshold",
            "temperature",
        ):
            self.assertNotIn(key, adaptive)

    def test_gaussian_replay_mlp_config_uses_distributional_feature_memory(
        self,
    ) -> None:
        from src.config import load_config, method_config

        static_name = (
            "ascal_gmm_segmented_memory_posterior_"
            "feature_routed_gaussian_replay_mlp_static"
        )
        method_name = static_name.removesuffix("_static")
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_"
            "feature_routed_gaussian_replay_mlp_continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        self.assertEqual(config["seed"], 1)
        self.assertFalse(config["protocol"]["reset_between_domains"])
        self.assertFalse(config["protocol"]["generator_id_available_to_method"])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(reference["research_name"], "ASCAL-JMP-GaussianReplayMLP")
        self.assertEqual(reference["research_version"], "R25")
        self.assertFalse(reference["routing_score_used"])
        self.assertFalse(reference["gmm_in_final_prediction"])
        self.assertFalse(reference["raw_features_stored"])
        self.assertEqual(adaptive["feature_replay_hidden_dim"], 64)
        self.assertEqual(adaptive["feature_replay_learning_rate"], 0.001)
        self.assertEqual(adaptive["feature_replay_seed"], 1)
        self.assertEqual(reference["optimizer_steps_per_predicted_batch"], 1)
        self.assertEqual(reference["target_selected_hyperparameters"], 0)
        self.assertFalse(reference["target_labels_used"])
        self.assertTrue(reference["semantic_features_used"])

    def test_expanded_gaussian_replay_config_uses_distinct_256_draws(
        self,
    ) -> None:
        from src.config import load_config, method_config

        static_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "expanded_gaussian_replay_mlp_static"
        )
        method_name = static_name.removesuffix("_static")
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_feature_"
            "routed_expanded_gaussian_replay_mlp_continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        self.assertEqual(config["seed"], 1)
        self.assertFalse(config["protocol"]["reset_between_domains"])
        self.assertFalse(config["protocol"]["generator_id_available_to_method"])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(
            reference["research_name"],
            "ASCAL-JMP-ExpandedGaussianReplay",
        )
        self.assertEqual(reference["research_version"], "R26")
        self.assertEqual(adaptive["feature_replay_samples_per_update"], 256)
        self.assertEqual(adaptive["feature_replay_seed"], 1)
        self.assertEqual(reference["replay_samples_per_update"], 256)
        self.assertFalse(reference["generated_feature_reuse"])
        self.assertEqual(reference["replay_passes"], 1)
        self.assertEqual(reference["expert_parameter_inheritance"], "none")
        self.assertFalse(reference["routing_score_used"])
        self.assertFalse(reference["gmm_in_final_prediction"])
        self.assertFalse(reference["target_labels_used"])

    def test_shared_gaussian_replay_config_changes_only_head_scope(self) -> None:
        from src.config import load_config, method_config

        static_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "shared_gaussian_replay_mlp_static"
        )
        method_name = static_name.removesuffix("_static")
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_feature_"
            "routed_shared_gaussian_replay_mlp_continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        self.assertEqual(config["seed"], 1)
        self.assertFalse(config["protocol"]["reset_between_domains"])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(reference["research_name"], "ASCAL-JMP-SharedResidualHead")
        self.assertEqual(reference["research_version"], "R27")
        self.assertEqual(
            reference["ablation_variable"],
            "per_expert_residual_heads_replaced_by_one_shared_head",
        )
        self.assertEqual(adaptive["feature_replay_samples_per_update"], 256)
        self.assertEqual(adaptive["feature_replay_hidden_dim"], 64)
        self.assertEqual(adaptive["feature_replay_learning_rate"], 0.001)
        self.assertEqual(adaptive["feature_replay_seed"], 1)
        self.assertEqual(reference["residual_head_count"], 1)
        self.assertFalse(reference["routing_score_used"])
        self.assertFalse(reference["gmm_in_final_prediction"])
        self.assertFalse(reference["target_labels_used"])

    def test_linear_gaussian_replay_config_changes_only_head_architecture(
        self,
    ) -> None:
        from src.config import load_config, method_config

        static_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "linear_gaussian_replay_static"
        )
        method_name = static_name.removesuffix("_static")
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_feature_"
            "routed_linear_gaussian_replay_continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(reference["research_name"], "ASCAL-JMP-LinearResidualHead")
        self.assertEqual(reference["research_version"], "R28")
        self.assertEqual(
            reference["ablation_variable"],
            "per_expert_gelu_mlp_replaced_by_per_expert_linear_residual",
        )
        self.assertEqual(adaptive["feature_replay_samples_per_update"], 256)
        self.assertEqual(adaptive["feature_replay_learning_rate"], 0.001)
        self.assertEqual(adaptive["feature_replay_seed"], 1)
        self.assertFalse(reference["routing_score_used"])
        self.assertFalse(reference["gmm_in_final_prediction"])
        self.assertFalse(reference["target_labels_used"])

    def test_uniform_gaussian_replay_config_removes_only_reliability(self) -> None:
        from src.config import load_config, method_config

        static_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "uniform_gaussian_replay_mlp_static"
        )
        method_name = static_name.removesuffix("_static")
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_feature_"
            "routed_uniform_gaussian_replay_mlp_continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(reference["research_name"], "ASCAL-JMP-UniformConfidence")
        self.assertEqual(reference["research_version"], "R29")
        self.assertEqual(
            reference["ablation_variable"],
            "continuous_gmm_reliability_replaced_by_unit_sample_weight",
        )
        self.assertEqual(adaptive["feature_replay_samples_per_update"], 256)
        self.assertEqual(reference["reliability_rule"], "uniform_one_for_every_hard_pseudo_labeled_sample")
        self.assertFalse(reference["routing_score_used"])
        self.assertFalse(reference["gmm_in_final_prediction"])
        self.assertFalse(reference["target_labels_used"])

    def test_active_gaussian_replay_config_records_segment_recall_leak(
        self,
    ) -> None:
        from src.config import load_config, method_config

        static_name = (
            "ascal_gmm_segmented_memory_posterior_"
            "active_gaussian_replay_mlp_static"
        )
        method_name = static_name.removesuffix("_static")
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_"
            "active_gaussian_replay_mlp_continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(reference["research_name"], "ASCAL-JMP-ActiveOnly")
        self.assertEqual(reference["research_version"], "R30")
        self.assertEqual(
            reference["ablation_variable"],
            "historical_clip_feature_routing_disabled",
        )
        self.assertEqual(adaptive["feature_replay_samples_per_update"], 256)
        self.assertEqual(
            reference["routing_coordinate"],
            "active_candidate_after_inherited_score_segment_recall",
        )
        self.assertFalse(reference["per_batch_historical_feature_candidates"])
        self.assertTrue(reference["segment_change_score_memory_recall"])
        self.assertEqual(
            reference["historical_expert_recall"],
            "segment_change_callback_only",
        )
        self.assertTrue(reference["episodic_memory_updated"])
        self.assertFalse(reference["target_labels_used"])

    def test_no_historical_recall_config_closes_both_recall_paths(self) -> None:
        from src.config import load_config, method_config

        static_name = (
            "ascal_gmm_segmented_memory_posterior_no_historical_recall_"
            "gaussian_replay_mlp_static"
        )
        method_name = static_name.removesuffix("_static")
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_no_"
            "historical_recall_gaussian_replay_mlp_continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(reference["research_name"], "ASCAL-JMP-NoHistoricalRecall")
        self.assertEqual(reference["research_version"], "R34")
        self.assertEqual(
            reference["ablation_variable"],
            "all_historical_expert_recall_disabled",
        )
        self.assertEqual(adaptive["feature_replay_samples_per_update"], 256)
        self.assertFalse(reference["per_batch_historical_feature_candidates"])
        self.assertFalse(reference["segment_change_score_memory_recall"])
        self.assertFalse(reference["historical_expert_recall"])
        self.assertEqual(reference["episodic_memory_role"], "shadow_archive_never_read")
        self.assertFalse(reference["target_labels_used"])

    def test_current_segment_core_config_deletes_shadow_archive(self) -> None:
        from src.config import load_config, method_config

        static_name = "ascal_gmm_current_segment_gaussian_replay_mlp_static"
        method_name = static_name.removesuffix("_static")
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_current_segment_gaussian_replay_mlp_"
            "continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(reference["research_name"], "ASCAL-JMP-CurrentSegmentCore")
        self.assertEqual(reference["research_version"], "R35")
        self.assertEqual(
            reference["ablation_variable"],
            "unused_shadow_archive_deleted",
        )
        self.assertEqual(adaptive["feature_replay_samples_per_update"], 256)
        self.assertFalse(reference["historical_expert_recall"])
        self.assertFalse(reference["episodic_memory_updated"])
        self.assertEqual(reference["episodic_memory_role"], "none")
        self.assertEqual(
            reference["state_scope"],
            "one_current_segment_gmm_feature_distribution_and_mlp",
        )
        self.assertFalse(reference["target_labels_used"])

    def test_global_stream_core_config_removes_only_segment_resets(self) -> None:
        from src.config import load_config, method_config

        static_name = "ascal_gmm_global_stream_gaussian_replay_mlp_static"
        method_name = static_name.removesuffix("_static")
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_global_stream_gaussian_replay_mlp_"
            "continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(reference["research_name"], "ASCAL-JMP-GlobalStreamCore")
        self.assertEqual(reference["research_version"], "R36")
        self.assertEqual(
            reference["ablation_variable"],
            "causal_bic_segment_resets_disabled",
        )
        self.assertEqual(adaptive["feature_replay_samples_per_update"], 256)
        self.assertEqual(reference["segmentation_rule"], "none")
        self.assertEqual(reference["segment_reset_scope"], "none")
        self.assertEqual(
            reference["state_scope"],
            "one_global_stream_gmm_feature_distribution_and_mlp",
        )

    def test_clip_expert_memory_config_disables_only_score_recall(self) -> None:
        from src.config import load_config, method_config

        static_name = (
            "ascal_gmm_segmented_memory_posterior_clip_routed_"
            "gaussian_replay_mlp_static"
        )
        method_name = static_name.removesuffix("_static")
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_clip_routed_"
            "gaussian_replay_mlp_continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(reference["research_name"], "ASCAL-JMP-CLIPExpertMemory")
        self.assertEqual(reference["research_version"], "R37")
        self.assertEqual(
            reference["ablation_variable"],
            "segment_change_score_memory_recall_disabled",
        )
        self.assertEqual(adaptive["feature_replay_samples_per_update"], 256)
        self.assertFalse(reference["routing_score_used"])
        self.assertFalse(reference["segment_change_score_memory_recall"])
        self.assertEqual(
            reference["historical_expert_recall"],
            "clip_feature_route_only",
        )
        self.assertTrue(reference["episodic_memory_updated"])
        self.assertFalse(reference["target_labels_used"])
        self.assertFalse(reference["target_labels_used"])

    def test_current_batch_replay_config_removes_only_gaussian_draws(self) -> None:
        from src.config import load_config, method_config

        static_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "current_batch_replay_mlp_static"
        )
        method_name = static_name.removesuffix("_static")
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_feature_"
            "routed_current_batch_replay_mlp_continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(reference["research_name"], "ASCAL-JMP-CurrentBatchReplay")
        self.assertEqual(reference["research_version"], "R31")
        self.assertEqual(
            reference["ablation_variable"],
            "cumulative_gaussian_draws_replaced_by_current_batch_resampling",
        )
        self.assertEqual(adaptive["feature_replay_samples_per_update"], 256)
        self.assertEqual(reference["feature_memory_role"], "shadow_statistics_not_used_for_training")
        self.assertTrue(reference["replay_replacement"])
        self.assertFalse(reference["routing_score_used"])
        self.assertFalse(reference["target_labels_used"])

    def test_prior_gaussian_replay_config_removes_only_class_balance(self) -> None:
        from src.config import load_config, method_config

        static_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "prior_gaussian_replay_mlp_static"
        )
        method_name = static_name.removesuffix("_static")
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_feature_"
            "routed_prior_gaussian_replay_mlp_continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(reference["research_name"], "ASCAL-JMP-PriorReplay")
        self.assertEqual(reference["research_version"], "R32")
        self.assertEqual(
            reference["ablation_variable"],
            "balanced_replay_replaced_by_accumulated_pseudo_class_mass",
        )
        self.assertEqual(adaptive["feature_replay_samples_per_update"], 256)
        self.assertEqual(reference["replay_class_minimum"], "one_generated_sample_per_class")
        self.assertFalse(reference["routing_score_used"])
        self.assertFalse(reference["target_labels_used"])

    def test_source_replay_config_removes_only_gmm_supervision(self) -> None:
        from src.config import load_config, method_config

        static_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "source_replay_mlp_static"
        )
        method_name = static_name.removesuffix("_static")
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_feature_"
            "routed_source_replay_mlp_continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(reference["research_name"], "ASCAL-JMP-SourceSupervision")
        self.assertEqual(reference["research_version"], "R33")
        self.assertEqual(
            reference["ablation_variable"],
            "expert_gmm_pseudo_supervision_replaced_by_frozen_source_probability",
        )
        self.assertEqual(adaptive["feature_replay_samples_per_update"], 256)
        self.assertEqual(
            reference["gmm_role"], "segmentation_and_expert_identity_only"
        )
        self.assertFalse(reference["gmm_in_feature_supervision"])
        self.assertFalse(reference["routing_score_used"])
        self.assertFalse(reference["target_labels_used"])

    def test_source_ridge_inheritance_config_uses_one_analytic_coordinate(
        self,
    ) -> None:
        from src.config import load_config, method_config

        method_name = (
            "ascal_gmm_segmented_memory_posterior_"
            "feature_routed_source_ridge"
        )
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_"
            "feature_routed_source_ridge_continual_genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(
            config["methods"],
            [f"{method_name}_static", method_name],
        )
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(
            reference["research_name"],
            "ASCAL-JMP-SourceRidgeInheritance",
        )
        self.assertEqual(reference["research_version"], "R23")
        self.assertEqual(adaptive["classifier_feature_normalization"], "l2")
        self.assertIn(
            "ASCAL_ANALYTIC_RIDGE_SOURCE_CHECKPOINT",
            adaptive["source_checkpoint"],
        )
        self.assertEqual(
            reference["source_feature_coordinate"],
            "l2_normalized_clip_lora_feature_plus_constant_bias",
        )
        self.assertFalse(reference["base_probability_in_final_prediction"])
        self.assertEqual(reference["new_target_hyperparameters"], 0)
        for key in (
            "confidence_threshold",
            "fusion_weight",
            "learning_rate",
            "routing_threshold",
            "target_weight",
        ):
            self.assertNotIn(key, adaptive)

        train = load_config(
            PROJECT_ROOT
            / "configs/train/genimage_sd14_clip_vitl14_lora_analytic_ridge.yaml"
        )
        self.assertEqual(train["model"]["classifier_feature_normalization"], "l2")
        self.assertTrue(train["training"]["analytic_ridge"]["enabled"])
        self.assertEqual(train["training"]["analytic_ridge"]["regularization"], 1.0)

    def test_source_ridge_gmm_readout_config_is_an_r23_state_diagnostic(
        self,
    ) -> None:
        from src.config import load_config, method_config

        static_name = (
            "ascal_gmm_segmented_memory_posterior_"
            "feature_routed_source_ridge_static"
        )
        method_name = (
            "ascal_gmm_segmented_memory_posterior_"
            "feature_routed_source_ridge_gmm_readout"
        )
        path = (
            PROJECT_ROOT
            / "configs/experiments/clip_vlm_bias_controlled"
            / "matched_jpeg_ascal_gmm_segmented_memory_posterior_"
            "feature_routed_source_ridge_gmm_readout_continual_"
            "genimage_seed1.yaml"
        )
        config = load_config(path)
        self.assertEqual(config["methods"], [static_name, method_name])
        self.assertEqual(config["seed"], 1)
        self.assertFalse(config["data"]["shuffle"])
        adaptive = method_config(config, method_name)
        reference = adaptive["reference"]
        self.assertEqual(
            reference["research_name"],
            "ASCAL-JMP-SourceRidgeGMMReadout",
        )
        self.assertEqual(reference["research_version"], "R24")
        self.assertTrue(reference["gmm_in_final_prediction"])
        self.assertFalse(reference["expert_ridge_in_final_prediction"])
        self.assertTrue(reference["base_probability_in_final_prediction"])
        self.assertFalse(reference["target_labels_used"])
        self.assertEqual(reference["new_target_hyperparameters"], 0)
        self.assertIn(
            "ASCAL_ANALYTIC_RIDGE_SOURCE_CHECKPOINT",
            adaptive["source_checkpoint"],
        )
        for key in (
            "confidence_threshold",
            "fusion_weight",
            "learning_rate",
            "routing_threshold",
            "target_weight",
        ):
            self.assertNotIn(key, adaptive)

    def test_source_ridge_inheritance_has_all_seed1_dataset_entries(
        self,
    ) -> None:
        from src.config import load_config

        method_name = (
            "ascal_gmm_segmented_memory_posterior_"
            "feature_routed_source_ridge"
        )
        datasets = (
            "genimage",
            "aigc_detection_benchmark",
            "aigi_holmes_p3",
            "opensdid_global",
        )
        for dataset in datasets:
            path = (
                PROJECT_ROOT
                / "configs/experiments/clip_vlm_bias_controlled"
                / (
                    "matched_jpeg_ascal_gmm_segmented_memory_posterior_"
                    "feature_routed_source_ridge_continual_"
                    f"{dataset}_seed1.yaml"
                )
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [f"{method_name}_static", method_name],
                )
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["data"]["shuffle"])
                self.assertIn(dataset, config["output_dir"])
                self.assertIn(
                    dataset,
                    config["data"]["locked_online_manifest"],
                )
                self.assertIn(
                    dataset,
                    config["data"]["locked_final_holdout_manifest"],
                )

    def test_routed_residual_configs_keep_r01_coordinate_and_one_assignment(
        self,
    ) -> None:
        from src.config import load_config, method_config

        method_name = "ascal_gmm_segmented_memory_posterior_routed_residual"
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
                f"routed_residual_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_memory_posterior_mixture_residual",
                        method_name,
                    ],
                )
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                config_dir = Path(config["_config_path"]).parent
                for field in (
                    "locked_online_manifest",
                    "locked_final_holdout_manifest",
                ):
                    manifest = Path(config["data"][field]).expanduser()
                    if not manifest.is_absolute():
                        manifest = config_dir / manifest
                    self.assertTrue(manifest.resolve().is_file())
                self.assertIn(
                    f"posterior_routed_residual_continual/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, method_name)
                for key in (
                    "confidence_threshold",
                    "fusion_weight",
                    "lambda",
                    "memory_capacity",
                    "recall_threshold",
                    "similarity_threshold",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-RoutedResidual"
                )
                self.assertEqual(reference["research_version"], "R13")
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertFalse(reference["score_boundary_routing"])
                self.assertFalse(reference["routing_changes_score_calibrator"])
                self.assertEqual(reference["routing_threshold"], "none")
                self.assertFalse(reference["prediction_mutates_experts"])
                self.assertTrue(reference["batch_transductive_prediction"])
                self.assertEqual(
                    reference["seed1_promotion_rule"],
                    "accuracy_within_0p2pp_of_r11_and_auc_above_r06",
                )
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])
                self.assertFalse(reference["raw_features_stored"])

    def test_routed_ridge_configs_use_exact_online_linear_heads_without_knobs(
        self,
    ) -> None:
        from src.config import load_config, method_config

        method_name = (
            "ascal_gmm_segmented_memory_posterior_routed_ridge_residual"
        )
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
                f"routed_ridge_residual_continual_{dataset}_seed1.yaml"
            )
            with self.subTest(dataset=dataset):
                config = load_config(path)
                self.assertEqual(
                    config["methods"],
                    [
                        "ascal_gmm_segmented_memory_posterior_routed_residual",
                        method_name,
                    ],
                )
                self.assertEqual(config["seed"], 1)
                self.assertFalse(config["protocol"]["reset_between_domains"])
                self.assertFalse(
                    config["protocol"]["generator_id_available_to_method"]
                )
                config_dir = Path(config["_config_path"]).parent
                for field in (
                    "locked_online_manifest",
                    "locked_final_holdout_manifest",
                ):
                    manifest = Path(config["data"][field]).expanduser()
                    if not manifest.is_absolute():
                        manifest = config_dir / manifest
                    self.assertTrue(manifest.resolve().is_file())
                self.assertIn(
                    f"posterior_routed_ridge_residual_continual/{dataset}/seed1",
                    config["output_dir"],
                )
                adaptive = method_config(config, method_name)
                for key in (
                    "confidence_threshold",
                    "fusion_weight",
                    "lambda",
                    "learning_rate",
                    "memory_capacity",
                    "recall_threshold",
                    "ridge_alpha",
                    "similarity_threshold",
                    "target_threshold",
                    "threshold",
                    "window_size",
                ):
                    self.assertNotIn(key, adaptive)
                reference = adaptive["reference"]
                self.assertEqual(
                    reference["research_name"], "ASCAL-JMP-RoutedRidge"
                )
                self.assertEqual(reference["research_version"], "R14")
                self.assertEqual(reference["new_target_hyperparameters"], 0)
                self.assertFalse(reference["score_boundary_routing"])
                self.assertFalse(reference["routing_changes_score_calibrator"])
                self.assertEqual(reference["routing_threshold"], "none")
                self.assertEqual(reference["optimizer"], "none")
                self.assertEqual(reference["learning_rate"], "none")
                self.assertEqual(reference["epochs"], "none")
                self.assertEqual(
                    reference["residual_intercept"],
                    "none_global_shift_is_handled_by_r01_base",
                )
                self.assertEqual(reference["residual_readout_bound"], "none")
                self.assertFalse(reference["prediction_mutates_experts"])
                self.assertFalse(reference["target_labels_used"])
                self.assertFalse(reference["generator_boundaries_used"])
                self.assertFalse(reference["semantic_features_used"])
                self.assertFalse(reference["raw_images_stored"])
                self.assertFalse(reference["raw_features_stored"])

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
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_mdl_route",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_mdl_route_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_live_route",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_live_route_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_ordinal_route",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_ordinal_route_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_ordinal_ridge",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_ordinal_ridge_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_joint_ridge",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_joint_ridge_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_pairwise_ridge",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_pairwise_ridge_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_analytic_expert",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_analytic_expert_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_rms_ridge_expert",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_rms_ridge_expert_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_equal_prior_ridge_expert",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_equal_prior_ridge_expert_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_evidence_gated_ridge_expert",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_evidence_gated_ridge_expert_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_feature_routed_trusted_ridge",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_feature_routed_trusted_ridge_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_"
            "feature_routed_gaussian_replay_mlp",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_"
            "feature_routed_gaussian_replay_mlp_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "expanded_gaussian_replay_mlp",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "expanded_gaussian_replay_mlp_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_clip_routed_"
            "gaussian_replay_mlp",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_clip_routed_"
            "gaussian_replay_mlp_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_routed_residual",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_routed_residual_static",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_routed_ridge_residual",
            config["training"]["intended_methods"],
        )
        self.assertIn(
            "ascal_gmm_segmented_memory_posterior_routed_ridge_residual_static",
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

    def segmented_memory_posterior_mdl_route_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorMDLRoute,
        )

        return ASCALGMMSegmentedMemoryPosteriorMDLRoute(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_live_route_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorLiveRoute,
        )

        return ASCALGMMSegmentedMemoryPosteriorLiveRoute(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_ordinal_route_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorOrdinalRoute,
        )

        return ASCALGMMSegmentedMemoryPosteriorOrdinalRoute(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_ordinal_ridge_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorOrdinalRidge,
        )

        return ASCALGMMSegmentedMemoryPosteriorOrdinalRidge(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_joint_ridge_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorJointRidge,
        )

        return ASCALGMMSegmentedMemoryPosteriorJointRidge(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_pairwise_ridge_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorPairwiseRidge,
        )

        return ASCALGMMSegmentedMemoryPosteriorPairwiseRidge(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_analytic_expert_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorAnalyticExpert,
        )

        return ASCALGMMSegmentedMemoryPosteriorAnalyticExpert(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_rms_ridge_expert_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorRMSRidgeExpert,
        )

        return ASCALGMMSegmentedMemoryPosteriorRMSRidgeExpert(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_equal_prior_ridge_expert_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorEqualPriorRidgeExpert,
        )

        return ASCALGMMSegmentedMemoryPosteriorEqualPriorRidgeExpert(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_evidence_gated_ridge_expert_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorEvidenceGatedRidgeExpert,
        )

        return ASCALGMMSegmentedMemoryPosteriorEvidenceGatedRidgeExpert(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_feature_routed_trusted_ridge_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedTrustedRidge,
        )

        return ASCALGMMSegmentedMemoryPosteriorFeatureRoutedTrustedRidge(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_feature_routed_gaussian_replay_mlp_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedGaussianReplayMLP,
        )

        return ASCALGMMSegmentedMemoryPosteriorFeatureRoutedGaussianReplayMLP(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "feature_replay_hidden_dim": 4,
                "feature_replay_learning_rate": 0.01,
                "feature_replay_seed": 1,
            },
        )

    def segmented_memory_posterior_feature_routed_expanded_gaussian_replay_mlp_method(
        self,
        *,
        adaptation_mode: str = "full",
        replay_samples: int = 8,
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP,
        )

        return (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP(
                self.detector(),
                "cpu",
                {
                    "adaptation_mode": adaptation_mode,
                    "score_anchors": self.anchors(),
                    "feature_replay_hidden_dim": 4,
                    "feature_replay_learning_rate": 0.01,
                    "feature_replay_samples_per_update": replay_samples,
                    "feature_replay_seed": 1,
                },
            )
        )

    def segmented_memory_posterior_feature_routed_shared_gaussian_replay_mlp_method(
        self,
        *,
        adaptation_mode: str = "full",
        replay_samples: int = 8,
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSharedGaussianReplayMLP,
        )

        return ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSharedGaussianReplayMLP(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "feature_replay_hidden_dim": 4,
                "feature_replay_learning_rate": 0.01,
                "feature_replay_samples_per_update": replay_samples,
                "feature_replay_seed": 1,
            },
        )

    def segmented_memory_posterior_feature_routed_linear_gaussian_replay_method(
        self,
        *,
        adaptation_mode: str = "full",
        replay_samples: int = 8,
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedLinearGaussianReplay,
        )

        return ASCALGMMSegmentedMemoryPosteriorFeatureRoutedLinearGaussianReplay(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "feature_replay_hidden_dim": 4,
                "feature_replay_learning_rate": 0.01,
                "feature_replay_samples_per_update": replay_samples,
                "feature_replay_seed": 1,
            },
        )

    def segmented_memory_posterior_feature_routed_uniform_gaussian_replay_method(
        self,
        *,
        adaptation_mode: str = "full",
        replay_samples: int = 8,
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedUniformGaussianReplayMLP,
        )

        return ASCALGMMSegmentedMemoryPosteriorFeatureRoutedUniformGaussianReplayMLP(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "feature_replay_hidden_dim": 4,
                "feature_replay_learning_rate": 0.01,
                "feature_replay_samples_per_update": replay_samples,
                "feature_replay_seed": 1,
            },
        )

    def segmented_memory_posterior_active_gaussian_replay_method(
        self,
        *,
        adaptation_mode: str = "full",
        replay_samples: int = 8,
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorActiveGaussianReplayMLP,
        )

        return ASCALGMMSegmentedMemoryPosteriorActiveGaussianReplayMLP(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "feature_replay_hidden_dim": 4,
                "feature_replay_learning_rate": 0.01,
                "feature_replay_samples_per_update": replay_samples,
                "feature_replay_seed": 1,
            },
        )

    def segmented_memory_posterior_no_historical_recall_method(
        self,
        *,
        adaptation_mode: str = "full",
        replay_samples: int = 8,
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorNoHistoricalRecallGaussianReplayMLP,
        )

        return ASCALGMMSegmentedMemoryPosteriorNoHistoricalRecallGaussianReplayMLP(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "feature_replay_hidden_dim": 4,
                "feature_replay_learning_rate": 0.01,
                "feature_replay_samples_per_update": replay_samples,
                "feature_replay_seed": 1,
            },
        )

    def clip_routed_gaussian_replay_memory_method(
        self,
        *,
        adaptation_mode: str = "full",
        replay_samples: int = 8,
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorCLIPRoutedGaussianReplayMLP,
        )

        return ASCALGMMSegmentedMemoryPosteriorCLIPRoutedGaussianReplayMLP(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "feature_replay_hidden_dim": 4,
                "feature_replay_learning_rate": 0.01,
                "feature_replay_samples_per_update": replay_samples,
                "feature_replay_seed": 1,
            },
        )

    def current_segment_gaussian_replay_method(
        self,
        *,
        adaptation_mode: str = "full",
        replay_samples: int = 8,
    ):
        from src.methods.ascal_gmm import ASCALGMMCurrentSegmentGaussianReplayMLP

        return ASCALGMMCurrentSegmentGaussianReplayMLP(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "feature_replay_hidden_dim": 4,
                "feature_replay_learning_rate": 0.01,
                "feature_replay_samples_per_update": replay_samples,
                "feature_replay_seed": 1,
            },
        )

    def global_stream_gaussian_replay_method(
        self,
        *,
        adaptation_mode: str = "full",
        replay_samples: int = 8,
    ):
        from src.methods.ascal_gmm import ASCALGMMGlobalStreamGaussianReplayMLP

        return ASCALGMMGlobalStreamGaussianReplayMLP(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "feature_replay_hidden_dim": 4,
                "feature_replay_learning_rate": 0.01,
                "feature_replay_samples_per_update": replay_samples,
                "feature_replay_seed": 1,
            },
        )

    def segmented_memory_posterior_current_batch_replay_method(
        self,
        *,
        adaptation_mode: str = "full",
        replay_samples: int = 8,
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedCurrentBatchReplayMLP,
        )

        return ASCALGMMSegmentedMemoryPosteriorFeatureRoutedCurrentBatchReplayMLP(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "feature_replay_hidden_dim": 4,
                "feature_replay_learning_rate": 0.01,
                "feature_replay_samples_per_update": replay_samples,
                "feature_replay_seed": 1,
            },
        )

    def segmented_memory_posterior_prior_gaussian_replay_method(
        self,
        *,
        adaptation_mode: str = "full",
        replay_samples: int = 8,
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedPriorGaussianReplayMLP,
        )

        return ASCALGMMSegmentedMemoryPosteriorFeatureRoutedPriorGaussianReplayMLP(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "feature_replay_hidden_dim": 4,
                "feature_replay_learning_rate": 0.01,
                "feature_replay_samples_per_update": replay_samples,
                "feature_replay_seed": 1,
            },
        )

    def segmented_memory_posterior_source_replay_method(
        self,
        *,
        adaptation_mode: str = "full",
        replay_samples: int = 8,
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceReplayMLP,
        )

        return ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceReplayMLP(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "feature_replay_hidden_dim": 4,
                "feature_replay_learning_rate": 0.01,
                "feature_replay_samples_per_update": replay_samples,
                "feature_replay_seed": 1,
            },
        )

    def source_ridge_detector(self):
        torch = self.torch
        nn = self.nn

        class TinySourceRidgeDetector(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.classifier = nn.Linear(3, 2, bias=True)
                with torch.no_grad():
                    self.classifier.weight.copy_(
                        torch.tensor(
                            [
                                [-2.0, 2.0, 0.0],
                                [2.0, -2.0, 0.0],
                            ]
                        )
                    )
                    self.classifier.bias.copy_(torch.tensor([-0.2, 0.2]))
                self.feature_dim = 3
                self.classifier_feature_normalization = "l2"

            def forward_features(self, images):
                return images.mean(dim=(2, 3))

            def forward_classifier_features(self, features):
                return torch.nn.functional.normalize(features, dim=1)

            def forward(self, images):
                features = self.forward_classifier_features(
                    self.forward_features(images)
                )
                return self.classifier(features)

        return TinySourceRidgeDetector()

    def source_ridge_state(self, model) -> dict:
        from src.models.analytic_ridge import (
            SOURCE_ANALYTIC_RIDGE_PROFILE,
            analytic_ridge_statistics_sha256,
        )

        torch = self.torch
        weights = torch.cat(
            (
                model.classifier.weight.detach().double().t(),
                model.classifier.bias.detach().double()[None, :],
            ),
            dim=0,
        )
        state = {
            "profile": SOURCE_ANALYTIC_RIDGE_PROFILE,
            "feature_coordinate": (
                "l2_normalized_clip_lora_feature_plus_constant_bias"
            ),
            "feature_normalization": "l2",
            "bias_coordinate": True,
            "targets": "two_output_real_fake_one_hot",
            "regularization": 1.0,
            "feature_dim": 3,
            "design_dim": 4,
            "samples": 20,
            "class_mass": torch.tensor([10.0, 10.0], dtype=torch.float64),
            "gram": torch.eye(4, dtype=torch.float64),
            "inverse_gram": torch.eye(4, dtype=torch.float64),
            "cross_covariance": weights.clone(),
            "weights": weights.clone(),
        }
        state["statistics_sha256"] = analytic_ridge_statistics_sha256(state)
        return state

    def segmented_memory_posterior_feature_routed_source_ridge_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidge,
        )

        model = self.source_ridge_detector()
        return ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidge(
            model,
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "source_analytic_ridge": self.source_ridge_state(model),
            },
        )

    def segmented_memory_posterior_feature_routed_source_ridge_gmm_readout_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidgeGMMReadout,
        )

        model = self.source_ridge_detector()
        return ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidgeGMMReadout(
            model,
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
                "source_analytic_ridge": self.source_ridge_state(model),
            },
        )

    def segmented_memory_posterior_routed_residual_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorRoutedResidual,
        )

        return ASCALGMMSegmentedMemoryPosteriorRoutedResidual(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": adaptation_mode,
                "score_anchors": self.anchors(),
            },
        )

    def segmented_memory_posterior_routed_ridge_residual_method(
        self, *, adaptation_mode: str = "full"
    ):
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorRoutedRidgeResidual,
        )

        return ASCALGMMSegmentedMemoryPosteriorRoutedRidgeResidual(
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

    def test_mdl_route_uses_one_confirmed_a_expert_for_predict_and_adapt(
        self,
    ) -> None:
        rng = np.random.default_rng(59)
        method = self.segmented_memory_posterior_mdl_route_method()

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

        active_before = copy.deepcopy(method._mixture)
        memories_before = copy.deepcopy(method.segment_memories)
        boundary_history_before = list(method.boundary_history)
        checks_before = method.routing_handoff_checks
        scores = np.concatenate(
            [rng.normal(-8.0, 0.2, 48), rng.normal(-3.0, 0.2, 48)]
        )
        rng.shuffle(scores)
        images = self.score_batch(scores)
        prediction = method.predict(images)

        pending = method._pending
        self.assertEqual(
            pending["prediction_routing_proposed_expert"], "episodic_memory"
        )
        self.assertEqual(pending["prediction_routing_proposed_memory_index"], 0)
        self.assertTrue(pending["prediction_routing_admission_checked"])
        self.assertTrue(pending["prediction_routing_admission_accepted"])
        self.assertGreater(pending["prediction_routing_admission_gain"], 0.0)
        self.assertEqual(pending["prediction_routing_expert"], "episodic_memory")
        self.assertEqual(pending["prediction_routing_memory_index"], 0)
        self.assertAlmostEqual(
            pending["prediction_boundary"],
            method.segment_memories[0]["boundary"],
        )
        expected = method._source_probability(
            scores - float(method.segment_memories[0]["boundary"])
        )
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)

        self.assertEqual(method._mixture, active_before)
        self.assertEqual(method.segment_memories, memories_before)
        self.assertEqual(method.boundary_history, boundary_history_before)
        self.assertEqual(method.routing_handoff_checks, checks_before)

        stats = method.adapt(images)
        self.assertEqual(stats.extra["routing_handoff_checks"], checks_before + 1)
        self.assertEqual(stats.extra["routing_handoff_confirmations"], 1)
        self.assertEqual(stats.extra["routing_handoff_rejections"], 0)
        self.assertEqual(stats.extra["routing_memory_proposals"], 1)
        self.assertEqual(stats.extra["routing_memory_admission_fallbacks"], 0)
        self.assertEqual(stats.extra["routing_memory_selections"], 1)
        self.assertEqual(stats.extra["active_memory_index"], 0)
        self.assertTrue(stats.extra["routing_handoff_this_batch"])
        self.assertEqual(stats.extra["last_routing_expert"], "episodic_memory")
        self.assertEqual(
            stats.extra["last_routing_proposed_expert"], "episodic_memory"
        )
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-MDLRoute")
        self.assertEqual(metadata["research_version"], "R10")
        self.assertFalse(metadata["prediction_mutates_experts"])

    def test_mdl_route_rejects_novel_memory_before_it_can_predict(self) -> None:
        rng = np.random.default_rng(60)
        method = self.segmented_memory_posterior_mdl_route_method()
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

        prediction = method.predict(images)
        pending = method._pending
        self.assertEqual(
            pending["prediction_routing_proposed_expert"], "episodic_memory"
        )
        self.assertTrue(pending["prediction_routing_admission_checked"])
        self.assertFalse(pending["prediction_routing_admission_accepted"])
        self.assertLess(pending["prediction_routing_admission_gain"], 0.0)
        self.assertEqual(
            pending["prediction_routing_expert"], "active_learning_state"
        )
        self.assertIsNone(pending["prediction_routing_memory_index"])
        expected = method._source_probability(scores - 22.5)
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)
        self.assertEqual(method.segment_memories, memory_before)
        self.assertEqual(method.routing_handoff_checks, 0)

        stats = method.adapt(images)
        self.assertFalse(stats.extra["routing_handoff_this_batch"])
        self.assertEqual(stats.extra["routing_handoff_checks"], 1)
        self.assertEqual(stats.extra["routing_handoff_confirmations"], 0)
        self.assertEqual(stats.extra["routing_handoff_rejections"], 1)
        self.assertEqual(stats.extra["routing_memory_proposals"], 1)
        self.assertEqual(stats.extra["routing_memory_admission_fallbacks"], 1)
        self.assertEqual(stats.extra["routing_memory_selections"], 0)
        self.assertEqual(stats.extra["routing_active_selections"], 1)
        self.assertEqual(stats.extra["last_routing_expert"], "active_learning_state")
        self.assertIsNone(stats.extra["active_memory_index"])
        self.assertEqual(method.segment_memories, memory_before)

    def test_mdl_route_keeps_the_exact_r01_active_prediction_on_ties(self) -> None:
        baseline = self.segmented_memory_posterior_projection_method()
        method = self.segmented_memory_posterior_mdl_route_method()
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
            method._pending["prediction_routing_proposed_expert"],
            "active_learning_state",
        )
        self.assertEqual(
            method._pending["prediction_routing_expert"], "active_learning_state"
        )
        self.assertFalse(method._pending["prediction_routing_admission_checked"])
        self.assertEqual(
            method._pending["prediction_routing_admission_reason"],
            "active_state_wins_deviance",
        )

    def test_live_route_hides_the_active_experts_archived_snapshot(self) -> None:
        rng = np.random.default_rng(61)
        method = self.segmented_memory_posterior_live_route_method()
        active = {
            "weights": [0.5, 0.5],
            "mus": [20.0, 25.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 0.0,
        }
        archived = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        method._mixture = active
        method.boundary_history = [22.5]
        method.active_memory_index = 0
        method.segment_memories = [
            {
                "mixture": archived,
                "boundary": float(method._memory_boundary(archived)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 1,
            }
        ]
        scores = np.concatenate(
            [rng.normal(-8.0, 0.2, 48), rng.normal(-3.0, 0.2, 48)]
        )
        rng.shuffle(scores)
        images = self.score_batch(scores)

        prediction = method.predict(images)
        self.assertEqual(
            method._pending["prediction_routing_proposed_expert"],
            "active_learning_state",
        )
        self.assertEqual(
            method._pending["prediction_routing_expert"], "active_learning_state"
        )
        self.assertEqual(method._pending["prediction_routing_candidate_count"], 1)
        self.assertEqual(
            method._pending["prediction_routing_memory_candidate_count"], 0
        )
        expected = method._source_probability(scores - 22.5)
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)

        stats = method.adapt(images)
        self.assertEqual(stats.extra["routing_memory_proposals"], 0)
        self.assertEqual(stats.extra["routing_memory_selections"], 0)
        self.assertEqual(stats.extra["routing_active_memory_identity_reuses"], 0)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-LiveRoute")
        self.assertEqual(metadata["research_version"], "R11")
        self.assertEqual(
            metadata["expert_identity_rule"], "one_expert_one_routable_live_state"
        )

    def test_live_route_keeps_its_archived_fallback_until_live_gmm_exists(
        self,
    ) -> None:
        rng = np.random.default_rng(62)
        method = self.segmented_memory_posterior_live_route_method()
        archived = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        boundary = float(method._memory_boundary(archived))
        method._mixture = None
        method.active_memory_index = 0
        method.segment_memories = [
            {
                "mixture": archived,
                "boundary": boundary,
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 1,
            }
        ]
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
        self.assertFalse(method._pending["prediction_routing_admission_checked"])
        self.assertEqual(
            method._pending["prediction_routing_admission_reason"],
            "already_active_memory_identity",
        )
        expected = method._source_probability(scores - boundary)
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-6)

        stats = method.adapt(images)
        self.assertEqual(stats.extra["routing_active_memory_identity_reuses"], 1)
        self.assertTrue(stats.extra["mixture_active"])
        self.assertEqual(stats.extra["active_memory_index"], 0)

    def test_live_route_can_handoff_to_a_different_archived_expert(self) -> None:
        rng = np.random.default_rng(63)
        method = self.segmented_memory_posterior_live_route_method()
        returning = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        active = {
            "weights": [0.5, 0.5],
            "mus": [20.0, 25.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        method._mixture = active
        method.boundary_history = [22.5]
        method.active_memory_index = 1
        method.segment_memories = [
            {
                "mixture": returning,
                "boundary": float(method._memory_boundary(returning)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 0,
            },
            {
                "mixture": active,
                "boundary": float(method._memory_boundary(active)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 1,
            },
        ]
        scores = np.concatenate(
            [rng.normal(-8.0, 0.2, 48), rng.normal(-3.0, 0.2, 48)]
        )
        rng.shuffle(scores)
        images = self.score_batch(scores)

        method.predict(images)
        self.assertEqual(method._pending["prediction_routing_candidate_count"], 2)
        self.assertEqual(
            method._pending["prediction_routing_memory_candidate_count"], 1
        )
        self.assertEqual(method._pending["prediction_routing_memory_index"], 0)
        self.assertTrue(method._pending["prediction_routing_admission_accepted"])

        stats = method.adapt(images)
        self.assertTrue(stats.extra["routing_handoff_this_batch"])
        self.assertEqual(stats.extra["active_memory_index"], 0)
        self.assertEqual(stats.extra["routing_handoff_confirmations"], 1)

    def test_ordinal_route_preserves_r11_decisions_and_source_order(self) -> None:
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-1.0, 3.0],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-3.0, -1.0, -0.25, 0.25, 0.75, 1.5, 3.0, 4.0])
        images = self.score_batch(scores)
        live = self.segmented_memory_posterior_live_route_method()
        ordinal = self.segmented_memory_posterior_ordinal_route_method()
        for method in (live, ordinal):
            method._mixture = copy.deepcopy(mixture)
            method.boundary_history = [1.0]

        live_prediction = live.predict(images)
        ordinal_prediction = ordinal.predict(images)
        np.testing.assert_array_equal(
            ordinal_prediction.pred_label.numpy(),
            live_prediction.pred_label.numpy(),
        )
        source_probability = np.clip(
            ordinal._source_probability(scores),
            1e-6,
            1.0 - 1e-6,
        )
        routed_fake = live_prediction.pred_label.numpy().astype(bool)
        expected = 0.5 * (source_probability + routed_fake.astype(np.float64))
        np.testing.assert_allclose(
            ordinal_prediction.prob_fake.numpy(), expected, atol=1e-6
        )
        for routed_label in (False, True):
            selected = routed_fake == routed_label
            ordered = np.argsort(scores[selected])
            probabilities = ordinal_prediction.prob_fake.numpy()[selected][ordered]
            self.assertTrue(np.all(np.diff(probabilities) > 0.0))

        self.assertTrue(ordinal._pending["prediction_ordinal_applied"])
        self.assertGreater(
            ordinal._pending["prediction_ordinal_decision_disagreements"], 0
        )
        stats = ordinal.adapt(images)
        self.assertEqual(stats.extra["ordinal_batches"], 1)
        self.assertEqual(stats.extra["ordinal_samples"], len(scores))
        self.assertEqual(stats.extra["ordinal_label_mismatches"], 0)
        self.assertEqual(
            stats.extra["last_routing_expert"], "active_learning_state"
        )
        metadata = ordinal.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-OrdinalRoute")
        self.assertEqual(metadata["research_version"], "R12")
        self.assertIn("preserves_every_r11", metadata["accuracy_invariance"])

    def test_ordinal_route_keeps_source_fallback_exact(self) -> None:
        method = self.segmented_memory_posterior_ordinal_route_method()
        scores = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        images = self.score_batch(scores)

        prediction = method.predict(images)
        np.testing.assert_allclose(
            prediction.prob_fake.numpy(),
            method._source_probability(scores),
            atol=1e-6,
        )
        self.assertFalse(method._pending["prediction_ordinal_applied"])
        stats = method.adapt(images)
        self.assertEqual(stats.extra["ordinal_batches"], 0)
        self.assertEqual(stats.extra["ordinal_samples"], 0)
        self.assertEqual(stats.extra["ordinal_label_mismatches"], 0)

    def test_ordinal_route_keeps_the_r11_memory_handoff_assignment(self) -> None:
        rng = np.random.default_rng(63)
        method = self.segmented_memory_posterior_ordinal_route_method()
        returning = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        active = {
            "weights": [0.5, 0.5],
            "mus": [20.0, 25.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        method._mixture = active
        method.boundary_history = [22.5]
        method.active_memory_index = 1
        method.segment_memories = [
            {
                "mixture": returning,
                "boundary": float(method._memory_boundary(returning)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 0,
            },
            {
                "mixture": active,
                "boundary": float(method._memory_boundary(active)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 1,
            },
        ]
        scores = np.concatenate(
            [rng.normal(-8.0, 0.2, 48), rng.normal(-3.0, 0.2, 48)]
        )
        rng.shuffle(scores)
        images = self.score_batch(scores)

        method.predict(images)
        self.assertTrue(method._pending["prediction_ordinal_applied"])
        self.assertEqual(method._pending["prediction_routing_memory_index"], 0)
        self.assertTrue(method._pending["prediction_routing_admission_accepted"])

        stats = method.adapt(images)
        self.assertTrue(stats.extra["routing_handoff_this_batch"])
        self.assertEqual(stats.extra["active_memory_index"], 0)
        self.assertEqual(stats.extra["routing_handoff_confirmations"], 1)
        self.assertEqual(stats.extra["ordinal_batches"], 1)
        self.assertEqual(stats.extra["ordinal_label_mismatches"], 0)

    def test_ordinal_ridge_matches_exact_confidence_weighted_ridge(self) -> None:
        method = self.segmented_memory_posterior_ordinal_ridge_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-3.0, 3.0],
            "sigmas": [1.0, 1.0],
            "components": 2,
            "bic": 0.0,
        }
        batches = (
            (
                np.array([-3.0, 0.0, 3.0]),
                np.eye(3, dtype=np.float64),
            ),
            (
                np.array([-2.0, 2.0]),
                np.array(
                    [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]],
                    dtype=np.float64,
                )
                / np.sqrt(2.0),
            ),
        )
        expected_precision = np.eye(3, dtype=np.float64)
        expected_rhs = np.zeros(3, dtype=np.float64)
        state = method._new_ordinal_ridge_state()
        for scores, features in batches:
            posterior, targets, reliability, _ = method._ordinal_ridge_supervision(
                mixture,
                scores,
            )
            expected_precision += features.T @ (
                reliability[:, None] * features
            )
            expected_rhs += features.T @ (reliability * targets)
            self.assertTrue(
                method._update_ordinal_ridge_state(
                    state,
                    mixture,
                    scores,
                    features,
                )
            )
            self.assertTrue(np.allclose(reliability, np.abs(2.0 * posterior - 1.0)))

        expected_weights = np.linalg.solve(expected_precision, expected_rhs)
        np.testing.assert_allclose(state["weights"], expected_weights, atol=1e-12)
        np.testing.assert_allclose(
            state["inverse_gram"],
            np.linalg.inv(expected_precision),
            atol=1e-12,
        )
        self.assertEqual(state["updates"], len(batches))
        self.assertEqual(state["candidate_samples"], 5)
        self.assertEqual(method.ordinal_ridge_solve_failures, 0)
        self.assertEqual(float(np.abs(2.0 * 0.5 - 1.0)), 0.0)

    def test_ordinal_ridge_is_causal_and_preserves_every_r12_decision(self) -> None:
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-4.0, 4.0],
            "sigmas": [0.75, 0.75],
            "components": 2,
            "bic": 0.0,
        }
        baseline = self.segmented_memory_posterior_ordinal_route_method()
        method = self.segmented_memory_posterior_ordinal_ridge_method()
        for candidate in (baseline, method):
            candidate._mixture = copy.deepcopy(mixture)
            candidate.boundary_history = [0.0]

        scores = np.array([-4.0, -3.0, 3.0, 4.0])
        first_images = self.score_feature_batch(
            scores,
            np.array([-1.0, -0.5, 0.5, 1.0]),
        )
        baseline_first = baseline.predict(first_images)
        method_first = method.predict(first_images)
        np.testing.assert_allclose(
            method_first.prob_fake.numpy(),
            baseline_first.prob_fake.numpy(),
            atol=1e-6,
        )
        np.testing.assert_array_equal(
            method_first.pred_label.numpy(),
            baseline_first.pred_label.numpy(),
        )
        self.assertFalse(method._pending["prediction_ordinal_ridge_ready"])
        baseline.adapt(first_images)
        first_stats = method.adapt(first_images)
        self.assertTrue(first_stats.extra["ordinal_ridge_updated"])
        self.assertEqual(first_stats.extra["ordinal_ridge_ready_experts"], 1)

        second_images = self.score_feature_batch(
            scores,
            np.array([1.0, 0.5, -0.5, -1.0]),
        )
        baseline_second = baseline.predict(second_images)
        method_second = method.predict(second_images)
        np.testing.assert_array_equal(
            method_second.pred_label.numpy(),
            baseline_second.pred_label.numpy(),
        )
        self.assertFalse(
            np.allclose(
                method_second.prob_fake.numpy(),
                baseline_second.prob_fake.numpy(),
            )
        )
        self.assertAlmostEqual(
            method._pending["prediction_ordinal_ridge_residual_mean"],
            0.0,
            places=12,
        )
        routed_fake = method_second.pred_label.numpy().astype(bool)
        self.assertTrue(np.all(method_second.prob_fake.numpy()[~routed_fake] < 0.5))
        self.assertTrue(np.all(method_second.prob_fake.numpy()[routed_fake] >= 0.5))
        baseline.adapt(second_images)
        second_stats = method.adapt(second_images)
        np.testing.assert_allclose(method.score_history, baseline.score_history)
        np.testing.assert_allclose(method.boundary_history, baseline.boundary_history)
        self.assertEqual(method.segment_changes, baseline.segment_changes)
        self.assertEqual(second_stats.extra["ordinal_ridge_hard_label_mismatches"], 0)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-OrdinalRidge")
        self.assertEqual(metadata["research_version"], "R15")
        self.assertIn("every_r12", metadata["accuracy_invariance"])

    def test_ordinal_ridge_static_and_unready_paths_are_exact_r12(self) -> None:
        static = self.segmented_memory_posterior_ordinal_ridge_method(
            adaptation_mode="static"
        )
        scores = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        images = self.score_feature_batch(scores, np.linspace(-1.0, 1.0, 5))
        prediction = static.predict(images)
        np.testing.assert_allclose(
            prediction.prob_fake.numpy(),
            static._source_probability(scores),
            atol=1e-6,
        )
        stats = static.adapt(images)
        self.assertEqual(stats.extra["ordinal_ridge_updates"], 0)
        self.assertEqual(stats.extra["ordinal_ridge_expert_count"], 0)
        self.assertEqual(static.trainable_parameters, 0)

    def test_ordinal_ridge_updates_only_the_r12_selected_memory(self) -> None:
        rng = np.random.default_rng(63)
        method = self.segmented_memory_posterior_ordinal_ridge_method()
        returning = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        active = {
            "weights": [0.5, 0.5],
            "mus": [20.0, 25.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        returning_state = method._new_ordinal_ridge_state()
        active_state = method._new_ordinal_ridge_state()
        method._mixture = active
        method.boundary_history = [22.5]
        method.active_memory_index = 1
        method.segment_memories = [
            {
                "mixture": returning,
                "boundary": float(method._memory_boundary(returning)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 0,
                method._ORDINAL_RIDGE_MEMORY_KEY: returning_state,
            },
            {
                "mixture": active,
                "boundary": float(method._memory_boundary(active)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 1,
                method._ORDINAL_RIDGE_MEMORY_KEY: active_state,
            },
        ]
        scores = np.concatenate(
            [rng.normal(-8.0, 0.2, 48), rng.normal(-3.0, 0.2, 48)]
        )
        cues = np.concatenate([-np.ones(48), np.ones(48)])
        order = rng.permutation(len(scores))
        images = self.score_feature_batch(scores[order], cues[order])

        method.predict(images)
        self.assertEqual(method._pending["prediction_routing_expert"], "episodic_memory")
        self.assertEqual(method._pending["prediction_routing_memory_index"], 0)
        stats = method.adapt(images)
        self.assertEqual(returning_state["updates"], 1)
        self.assertEqual(active_state["updates"], 0)
        self.assertTrue(stats.extra["routing_handoff_this_batch"])
        self.assertEqual(stats.extra["active_memory_index"], 0)

    def test_joint_ridge_matches_exact_uncentered_weighted_ridge(self) -> None:
        method = self.segmented_memory_posterior_joint_ridge_method()
        mixture = {
            "weights": [0.4, 0.6],
            "mus": [-3.0, 2.0],
            "sigmas": [1.0, 0.75],
            "components": 2,
            "bic": 0.0,
        }
        batches = (
            (
                np.array([-3.0, -0.5, 2.0]),
                np.array([-2.0, -0.25, 1.0]),
                np.array(
                    [
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
            ),
            (
                np.array([-2.0, 1.5]),
                np.array([-1.0, 0.75]),
                np.array(
                    [[1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 1.0]],
                    dtype=np.float64,
                )
                / np.sqrt(2.0),
            ),
        )
        expected_precision = np.eye(4, dtype=np.float64)
        expected_rhs = np.zeros(4, dtype=np.float64)
        state = method._new_ordinal_ridge_state()
        weighted_target_sum = 0.0
        for scores, base_logits, features in batches:
            _, targets, reliability = method._joint_ridge_supervision(
                mixture,
                scores,
                base_logits,
            )
            expected_precision += features.T @ (
                reliability[:, None] * features
            )
            expected_rhs += features.T @ (reliability * targets)
            weighted_target_sum += float(np.sum(reliability * targets))
            method._pending_joint_ridge_base_logits = base_logits
            self.assertTrue(
                method._update_ordinal_ridge_state(
                    state,
                    mixture,
                    scores,
                    features,
                )
            )

        expected_weights = np.linalg.solve(expected_precision, expected_rhs)
        np.testing.assert_allclose(state["weights"], expected_weights, atol=1e-12)
        np.testing.assert_allclose(
            state["inverse_gram"],
            np.linalg.inv(expected_precision),
            atol=1e-12,
        )
        self.assertNotAlmostEqual(weighted_target_sum, 0.0, places=8)
        self.assertNotAlmostEqual(float(state["weights"][-1]), 0.0, places=8)
        self.assertEqual(method.ordinal_ridge_last_target_center, 0.0)
        self.assertEqual(method.ordinal_ridge_feature_dim, 4)
        self.assertEqual(state["updates"], len(batches))
        self.assertEqual(state["candidate_samples"], 5)
        self.assertEqual(method.ordinal_ridge_solve_failures, 0)

    def test_joint_ridge_starts_at_r12_then_unlocks_ranking_and_labels(self) -> None:
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-4.0, 4.0],
            "sigmas": [0.75, 0.75],
            "components": 2,
            "bic": 0.0,
        }
        baseline = self.segmented_memory_posterior_ordinal_route_method()
        method = self.segmented_memory_posterior_joint_ridge_method()
        for candidate in (baseline, method):
            candidate._mixture = copy.deepcopy(mixture)
            candidate.boundary_history = [0.0]

        scores = np.array([-4.0, -3.0, 3.0, 4.0])
        first_images = self.score_feature_batch(
            scores,
            np.array([-1.0, -1.0, 1.0, 1.0]),
        )
        baseline_first = baseline.predict(first_images)
        method_first = method.predict(first_images)
        np.testing.assert_array_equal(
            method_first.prob_fake.numpy(),
            baseline_first.prob_fake.numpy(),
        )
        np.testing.assert_array_equal(
            method_first.pred_label.numpy(),
            baseline_first.pred_label.numpy(),
        )
        self.assertFalse(method._pending["prediction_joint_ridge_ready"])
        baseline.adapt(first_images)
        first_stats = method.adapt(first_images)
        self.assertTrue(first_stats.extra["joint_ridge_updated"])
        self.assertEqual(first_stats.extra["joint_ridge_ready_experts"], 1)

        for candidate in (baseline, method):
            candidate._mixture = copy.deepcopy(mixture)
            candidate.boundary_history = [0.0]
        second_images = self.score_feature_batch(
            scores,
            np.array([1.0, 1.0, -1.0, -1.0]),
        )
        baseline_second = baseline.predict(second_images)
        method_second = method.predict(second_images)
        label_changes = int(
            np.count_nonzero(
                method_second.pred_label.numpy()
                != baseline_second.pred_label.numpy()
            )
        )
        self.assertGreater(label_changes, 0)
        self.assertEqual(
            method._pending["prediction_joint_ridge_label_changes"],
            label_changes,
        )
        self.assertGreater(
            method._pending["prediction_joint_ridge_residual_max_abs"],
            0.0,
        )
        baseline.adapt(second_images)
        second_stats = method.adapt(second_images)
        np.testing.assert_allclose(method.score_history, baseline.score_history)
        np.testing.assert_allclose(method.boundary_history, baseline.boundary_history)
        self.assertEqual(method.segment_changes, baseline.segment_changes)
        self.assertEqual(second_stats.extra["joint_ridge_label_changes"], label_changes)
        self.assertEqual(
            second_stats.extra["joint_ridge_label_changes"],
            second_stats.extra["joint_ridge_real_to_fake"]
            + second_stats.extra["joint_ridge_fake_to_real"],
        )
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-JointRidge")
        self.assertEqual(metadata["research_version"], "R16")
        self.assertIn("may_cross", metadata["accuracy_invariance"])
        self.assertEqual(metadata["prediction_residual_centering"], "none")

    def test_joint_ridge_static_path_is_exact_r12_source_fallback(self) -> None:
        method = self.segmented_memory_posterior_joint_ridge_method(
            adaptation_mode="static"
        )
        scores = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        images = self.score_feature_batch(scores, np.linspace(-1.0, 1.0, 5))
        prediction = method.predict(images)
        np.testing.assert_allclose(
            prediction.prob_fake.numpy(),
            method._source_probability(scores),
            atol=1e-6,
        )
        stats = method.adapt(images)
        self.assertEqual(stats.extra["joint_ridge_updates"], 0)
        self.assertEqual(stats.extra["joint_ridge_expert_count"], 0)
        self.assertEqual(method.trainable_parameters, 0)

    def test_joint_ridge_updates_only_the_r12_selected_memory(self) -> None:
        rng = np.random.default_rng(63)
        method = self.segmented_memory_posterior_joint_ridge_method()
        returning = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        active = {
            "weights": [0.5, 0.5],
            "mus": [20.0, 25.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        returning_state = method._new_ordinal_ridge_state()
        active_state = method._new_ordinal_ridge_state()
        method._mixture = active
        method.boundary_history = [22.5]
        method.active_memory_index = 1
        method.segment_memories = [
            {
                "mixture": returning,
                "boundary": float(method._memory_boundary(returning)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 0,
                method._ORDINAL_RIDGE_MEMORY_KEY: returning_state,
            },
            {
                "mixture": active,
                "boundary": float(method._memory_boundary(active)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 1,
                method._ORDINAL_RIDGE_MEMORY_KEY: active_state,
            },
        ]
        scores = np.concatenate(
            [rng.normal(-8.0, 0.2, 48), rng.normal(-3.0, 0.2, 48)]
        )
        cues = np.concatenate([-np.ones(48), np.ones(48)])
        order = rng.permutation(len(scores))
        images = self.score_feature_batch(scores[order], cues[order])

        method.predict(images)
        self.assertEqual(method._pending["prediction_routing_expert"], "episodic_memory")
        self.assertEqual(method._pending["prediction_routing_memory_index"], 0)
        stats = method.adapt(images)
        self.assertEqual(returning_state["updates"], 1)
        self.assertEqual(active_state["updates"], 0)
        self.assertTrue(stats.extra["joint_ridge_updated"])
        self.assertTrue(stats.extra["routing_handoff_this_batch"])
        self.assertEqual(stats.extra["active_memory_index"], 0)

    def test_pairwise_ridge_matches_explicit_soft_pair_ridge(self) -> None:
        from src.methods.ascal_gmm import joint_density_fake_posterior

        method = self.segmented_memory_posterior_pairwise_ridge_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-3.0, 2.0],
            "sigmas": [0.75, 0.75],
            "components": 2,
            "bic": 0.0,
        }
        batches = (
            (
                np.array([-3.0, -1.0, 1.0, 2.0]),
                np.array([0, 0, 1, 1]),
                np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
            ),
            (
                np.array([-2.5, 0.5, 2.5]),
                np.array([0, 1, 1]),
                np.array(
                    [
                        [1.0, -1.0, 0.0],
                        [0.0, 1.0, -1.0],
                        [-1.0, 0.0, 1.0],
                    ],
                    dtype=np.float64,
                )
                / np.sqrt(2.0),
            ),
        )
        expected_precision = np.eye(3, dtype=np.float64)
        expected_rhs = np.zeros(3, dtype=np.float64)
        for scores, labels, features in batches:
            posterior = joint_density_fake_posterior(scores, mixture)
            projected = np.where(
                labels.astype(bool),
                np.maximum(posterior, 0.5),
                np.minimum(posterior, 0.5),
            )
            reliability = np.abs(2.0 * projected - 1.0)
            fake_mass = reliability * projected
            real_mass = reliability * (1.0 - projected)
            rows = []
            responses = []
            for fake_index in range(len(scores)):
                for real_index in range(len(scores)):
                    if fake_index == real_index:
                        continue
                    pair_weight = fake_mass[fake_index] * real_mass[real_index]
                    if pair_weight <= np.finfo(np.float64).eps:
                        continue
                    root = np.sqrt(pair_weight)
                    rows.append(
                        root * (features[fake_index] - features[real_index])
                    )
                    responses.append(root)
            explicit_design = np.asarray(rows, dtype=np.float64)
            explicit_response = np.asarray(responses, dtype=np.float64)
            expected_precision += explicit_design.T @ explicit_design
            expected_rhs += explicit_design.T @ explicit_response
            self.assertTrue(
                method._update_pairwise_ridge_state(
                    mixture,
                    scores,
                    labels,
                    features,
                )
            )
            self.assertLessEqual(
                method.pairwise_ridge_last_pair_rank,
                len(scores) - 1,
            )

        state = method._pairwise_ridge_state
        expected_weights = np.linalg.solve(expected_precision, expected_rhs)
        np.testing.assert_allclose(state["weights"], expected_weights, atol=1e-11)
        np.testing.assert_allclose(
            state["inverse_gram"],
            np.linalg.inv(expected_precision),
            atol=1e-11,
        )
        self.assertEqual(state["updates"], len(batches))
        self.assertEqual(state["candidate_samples"], 7)
        self.assertEqual(state["candidate_pairs"], 18)
        self.assertEqual(method.pairwise_ridge_feature_dim, 3)
        self.assertEqual(method.pairwise_ridge_solve_failures, 0)

    def test_pairwise_ridge_projects_gmm_conflicts_to_zero_weight(self) -> None:
        method = self.segmented_memory_posterior_pairwise_ridge_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-3.0, 3.0],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-3.0, 3.0])
        labels = np.array([1, 0])
        features = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

        updated = method._update_pairwise_ridge_state(
            mixture,
            scores,
            labels,
            features,
        )

        self.assertFalse(updated)
        self.assertEqual(method.pairwise_ridge_last_posterior_conflicts, 2)
        self.assertEqual(method.pairwise_ridge_last_reliability, 0.0)
        self.assertEqual(method.pairwise_ridge_last_effective_pair_mass, 0.0)
        self.assertEqual(method._pairwise_ridge_state["updates"], 0)

    def test_pairwise_ridge_starts_at_r12_then_unlocks_sample_decisions(self) -> None:
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        baseline = self.segmented_memory_posterior_ordinal_route_method()
        method = self.segmented_memory_posterior_pairwise_ridge_method()
        for candidate in (baseline, method):
            candidate._mixture = copy.deepcopy(mixture)
            candidate.boundary_history = [-5.5]

        scores = np.array([-8.0, -7.0, -4.0, -3.0])
        first_images = self.score_feature_batch(
            scores,
            np.array([-1.0, -1.0, 1.0, 1.0]),
        )
        baseline_first = baseline.predict(first_images)
        method_first = method.predict(first_images)
        np.testing.assert_array_equal(
            method_first.prob_fake.numpy(),
            baseline_first.prob_fake.numpy(),
        )
        np.testing.assert_array_equal(
            method_first.pred_label.numpy(),
            baseline_first.pred_label.numpy(),
        )
        self.assertFalse(method._pending["prediction_pairwise_ridge_ready"])
        baseline.adapt(first_images)
        first_stats = method.adapt(first_images)
        self.assertTrue(first_stats.extra["pairwise_ridge_updated"])
        self.assertTrue(first_stats.extra["pairwise_ridge_ready"])
        self.assertEqual(first_stats.extra["pairwise_ridge_global_state_count"], 1)

        for candidate in (baseline, method):
            candidate._mixture = copy.deepcopy(mixture)
            candidate.boundary_history = [-5.5]
        second_images = self.score_feature_batch(
            scores,
            np.array([1.0, 1.0, -1.0, -1.0]),
        )
        baseline_second = baseline.predict(second_images)
        method_second = method.predict(second_images)
        label_changes = int(
            np.count_nonzero(
                method_second.pred_label.numpy()
                != baseline_second.pred_label.numpy()
            )
        )
        self.assertGreater(label_changes, 0)
        self.assertEqual(
            method._pending["prediction_pairwise_ridge_label_changes"],
            label_changes,
        )
        self.assertGreater(
            method._pending["prediction_pairwise_ridge_residual_max_abs"],
            0.0,
        )
        baseline.adapt(second_images)
        second_stats = method.adapt(second_images)
        np.testing.assert_allclose(method.score_history, baseline.score_history)
        np.testing.assert_allclose(method.boundary_history, baseline.boundary_history)
        self.assertEqual(method.segment_changes, baseline.segment_changes)
        self.assertEqual(
            second_stats.extra["pairwise_ridge_label_changes"],
            label_changes,
        )
        self.assertEqual(
            second_stats.extra["pairwise_ridge_label_changes"],
            second_stats.extra["pairwise_ridge_real_to_fake"]
            + second_stats.extra["pairwise_ridge_fake_to_real"],
        )
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-PairwiseRidge")
        self.assertEqual(metadata["research_version"], "R17")
        self.assertIn("bias_free", metadata["residual_scope"])
        self.assertIn("never a logit", metadata["intentional_changes"][1])

    def test_pairwise_ridge_static_path_is_exact_r12_source_fallback(self) -> None:
        method = self.segmented_memory_posterior_pairwise_ridge_method(
            adaptation_mode="static"
        )
        scores = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        images = self.score_feature_batch(scores, np.linspace(-1.0, 1.0, 5))
        prediction = method.predict(images)
        np.testing.assert_allclose(
            prediction.prob_fake.numpy(),
            method._source_probability(scores),
            atol=1e-6,
        )
        stats = method.adapt(images)
        self.assertEqual(stats.extra["pairwise_ridge_updates"], 0)
        self.assertEqual(stats.extra["pairwise_ridge_global_state_count"], 0)
        self.assertEqual(method.trainable_parameters, 0)

    def test_analytic_expert_matches_exact_centered_prior_weighted_ridge(
        self,
    ) -> None:
        method = self.segmented_memory_posterior_analytic_expert_method()
        mixture = {
            "weights": [0.4, 0.6],
            "mus": [-3.0, 2.0],
            "sigmas": [1.0, 0.75],
            "components": 2,
            "bic": 0.0,
        }
        batches = (
            (
                np.array([-3.0, -1.5, 2.0]),
                np.array([0, 0, 1]),
                np.array(
                    [
                        [-0.9, 1.0, 0.0, 0.0, 1.0],
                        [-0.5, 0.0, 1.0, 0.0, 1.0],
                        [0.8, 0.0, 0.0, 1.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
            ),
            (
                np.array([-2.5, 1.5]),
                np.array([0, 1]),
                np.array(
                    [
                        [-0.75, 1.0, 1.0, 0.0, 1.0],
                        [0.6, 0.0, 1.0, 1.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
            ),
        )
        dimension = method.ordinal_ridge_feature_dim
        expected_precision = np.eye(dimension, dtype=np.float64)
        expected_prior = np.zeros(dimension, dtype=np.float64)
        expected_prior[0] = 1.0
        expected_rhs = expected_prior.copy()
        state = method._new_ordinal_ridge_state()
        for scores, labels, features in batches:
            _, targets, reliability, _ = method._analytic_expert_supervision(
                mixture,
                scores,
                labels,
            )
            expected_precision += features.T @ (
                reliability[:, None] * features
            )
            expected_rhs += features.T @ (reliability * targets)
            self.assertTrue(
                method._update_analytic_expert_state(
                    state,
                    mixture,
                    scores,
                    labels,
                    features,
                )
            )

        expected_weights = np.linalg.solve(expected_precision, expected_rhs)
        np.testing.assert_allclose(state["weights"], expected_weights, atol=1e-12)
        np.testing.assert_allclose(
            state["inverse_gram"],
            np.linalg.inv(expected_precision),
            atol=1e-12,
        )
        self.assertEqual(state["updates"], len(batches))
        self.assertEqual(state["candidate_samples"], 5)
        self.assertEqual(method.ordinal_ridge_feature_dim, 5)
        self.assertEqual(method.analytic_expert_solve_failures, 0)

    def test_analytic_expert_projects_gmm_conflicts_to_zero_weight(self) -> None:
        method = self.segmented_memory_posterior_analytic_expert_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-3.0, 3.0],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-3.0, 3.0])
        labels = np.array([1, 0])
        features = np.array(
            [[-0.9, 1.0, 0.0, 0.0, 1.0], [0.9, 0.0, 1.0, 0.0, 1.0]]
        )
        state = method._new_ordinal_ridge_state()
        initial_weights = state["weights"].copy()

        updated = method._update_analytic_expert_state(
            state,
            mixture,
            scores,
            labels,
            features,
        )

        self.assertFalse(updated)
        self.assertEqual(method.analytic_expert_last_posterior_conflicts, 2)
        self.assertEqual(method.analytic_expert_last_reliability, 0.0)
        self.assertEqual(state["updates"], 0)
        np.testing.assert_array_equal(state["weights"], initial_weights)

    def test_analytic_expert_starts_exactly_at_r12_then_learns_causally(
        self,
    ) -> None:
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        baseline = self.segmented_memory_posterior_ordinal_route_method()
        method = self.segmented_memory_posterior_analytic_expert_method()
        for candidate in (baseline, method):
            candidate._mixture = copy.deepcopy(mixture)
            candidate.boundary_history = [-5.5]

        scores = np.array([-8.0, -7.0, -4.0, -3.0])
        first_images = self.score_feature_batch(
            scores,
            np.array([-1.0, -1.0, 1.0, 1.0]),
        )
        baseline_first = baseline.predict(first_images)
        method_first = method.predict(first_images)
        np.testing.assert_array_equal(
            method_first.prob_fake.numpy(),
            baseline_first.prob_fake.numpy(),
        )
        np.testing.assert_array_equal(
            method_first.pred_label.numpy(),
            baseline_first.pred_label.numpy(),
        )
        self.assertFalse(method._pending["prediction_analytic_expert_ready"])
        baseline.adapt(first_images)
        first_stats = method.adapt(first_images)
        self.assertTrue(first_stats.extra["analytic_expert_updated"])
        self.assertEqual(first_stats.extra["analytic_expert_ready_experts"], 1)

        for candidate in (baseline, method):
            candidate._mixture = copy.deepcopy(mixture)
            candidate.boundary_history = [-5.5]
        second_images = self.score_feature_batch(
            scores,
            np.array([1.0, 1.0, -1.0, -1.0]),
        )
        baseline_second = baseline.predict(second_images)
        method_second = method.predict(second_images)
        self.assertFalse(
            np.allclose(
                method_second.prob_fake.numpy(),
                baseline_second.prob_fake.numpy(),
            )
        )
        self.assertTrue(method._pending["prediction_analytic_expert_ready"])
        self.assertGreater(
            method._pending["prediction_analytic_expert_correction_max_abs"],
            0.0,
        )
        baseline.adapt(second_images)
        second_stats = method.adapt(second_images)
        np.testing.assert_allclose(method.score_history, baseline.score_history)
        np.testing.assert_allclose(method.boundary_history, baseline.boundary_history)
        self.assertEqual(method.segment_changes, baseline.segment_changes)
        self.assertEqual(
            second_stats.extra["analytic_expert_label_changes"],
            second_stats.extra["analytic_expert_real_to_fake"]
            + second_stats.extra["analytic_expert_fake_to_real"],
        )
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-AnalyticExpert")
        self.assertEqual(metadata["research_version"], "R18")
        self.assertIn("past_soft", metadata["accuracy_invariance"])

    def test_analytic_expert_updates_only_the_r12_selected_memory(self) -> None:
        rng = np.random.default_rng(73)
        method = self.segmented_memory_posterior_analytic_expert_method()
        returning = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        active = {
            "weights": [0.5, 0.5],
            "mus": [20.0, 25.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        returning_state = method._new_ordinal_ridge_state()
        active_state = method._new_ordinal_ridge_state()
        method._mixture = active
        method.boundary_history = [22.5]
        method.active_memory_index = 1
        method.segment_memories = [
            {
                "mixture": returning,
                "boundary": float(method._memory_boundary(returning)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 0,
                method._ORDINAL_RIDGE_MEMORY_KEY: returning_state,
            },
            {
                "mixture": active,
                "boundary": float(method._memory_boundary(active)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 1,
                method._ORDINAL_RIDGE_MEMORY_KEY: active_state,
            },
        ]
        scores = np.concatenate(
            [rng.normal(-8.0, 0.2, 48), rng.normal(-3.0, 0.2, 48)]
        )
        cues = np.concatenate([-np.ones(48), np.ones(48)])
        order = rng.permutation(len(scores))
        images = self.score_feature_batch(scores[order], cues[order])

        method.predict(images)
        self.assertEqual(method._pending["prediction_routing_expert"], "episodic_memory")
        self.assertEqual(method._pending["prediction_routing_memory_index"], 0)
        stats = method.adapt(images)
        self.assertEqual(returning_state["updates"], 1)
        self.assertEqual(active_state["updates"], 0)
        self.assertTrue(stats.extra["analytic_expert_updated"])
        self.assertTrue(stats.extra["routing_handoff_this_batch"])
        self.assertEqual(stats.extra["active_memory_index"], 0)

    def test_analytic_expert_static_path_is_exact_r12_source_fallback(self) -> None:
        method = self.segmented_memory_posterior_analytic_expert_method(
            adaptation_mode="static"
        )
        scores = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        images = self.score_feature_batch(scores, np.linspace(-1.0, 1.0, 5))
        prediction = method.predict(images)
        np.testing.assert_allclose(
            prediction.prob_fake.numpy(),
            method._source_probability(scores),
            atol=1e-6,
        )
        stats = method.adapt(images)
        self.assertEqual(stats.extra["analytic_expert_updates"], 0)
        self.assertEqual(stats.extra["analytic_expert_count"], 0)
        self.assertEqual(method.trainable_parameters, 0)

    def test_rms_ridge_expert_matches_joint_weighted_one_hot_ridge(self) -> None:
        method = self.segmented_memory_posterior_rms_ridge_expert_method()
        mixture = {
            "weights": [0.45, 0.55],
            "mus": [-3.0, 2.0],
            "sigmas": [0.8, 0.7],
            "components": 2,
            "bic": 0.0,
        }
        batches = (
            (
                np.array([-3.5, -2.0, 1.5]),
                np.array([0, 0, 1]),
                np.array(
                    [
                        [1.0, 0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
                np.array([-1.5, -0.8, 1.2]),
            ),
            (
                np.array([-2.5, 2.5]),
                np.array([0, 1]),
                np.array(
                    [
                        [1.0, 1.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0, 1.0],
                    ],
                    dtype=np.float64,
                ),
                np.array([-1.0, 1.8]),
            ),
        )
        dimension = method.ordinal_ridge_feature_dim
        expected_precision = np.eye(dimension, dtype=np.float64)
        expected_cross_covariance = np.zeros((dimension, 2), dtype=np.float64)
        expected_class_mass = np.zeros(2, dtype=np.float64)
        expected_base_square_sum = 0.0
        state = method._new_ordinal_ridge_state()
        for scores, labels, features, base_margins in batches:
            targets, reliability, _, _ = method._rms_ridge_expert_supervision(
                mixture,
                scores,
                labels,
            )
            expected_precision += features.T @ (
                reliability[:, None] * features
            )
            expected_cross_covariance += features.T @ (
                reliability[:, None] * targets
            )
            expected_class_mass += np.bincount(
                labels,
                weights=reliability,
                minlength=2,
            )
            expected_base_square_sum += float(
                np.sum(reliability * base_margins**2)
            )
            self.assertTrue(
                method._update_rms_ridge_expert_state(
                    state,
                    mixture,
                    scores,
                    labels,
                    features,
                    base_margins,
                )
            )

        expected_weights = np.linalg.solve(
            expected_precision,
            expected_cross_covariance,
        )
        np.testing.assert_allclose(state["weights"], expected_weights, atol=1e-12)
        np.testing.assert_allclose(
            state["inverse_gram"],
            np.linalg.inv(expected_precision),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            state["cross_covariance"],
            expected_cross_covariance,
            atol=1e-12,
        )
        np.testing.assert_allclose(state["class_mass"], expected_class_mass)
        self.assertAlmostEqual(
            state["base_margin_square_sum"],
            expected_base_square_sum,
        )
        direction = expected_weights[:, 1] - expected_weights[:, 0]
        covariance = expected_precision - np.eye(dimension)
        expected_ridge_energy = float(direction @ covariance @ direction)
        scales = method._rms_ridge_expert_scales(state)
        self.assertIsNotNone(scales)
        assert scales is not None
        self.assertAlmostEqual(
            scales[0],
            np.sqrt(expected_base_square_sum / expected_class_mass.sum()),
        )
        self.assertAlmostEqual(
            scales[1],
            np.sqrt(expected_ridge_energy / expected_class_mass.sum()),
        )
        self.assertEqual(state["updates"], len(batches))
        self.assertEqual(state["candidate_samples"], 5)
        self.assertEqual(method.ordinal_ridge_feature_dim, 4)
        self.assertEqual(method.rms_ridge_expert_solve_failures, 0)

    def test_rms_ridge_expert_uses_hard_labels_and_zeroes_conflicts(self) -> None:
        method = self.segmented_memory_posterior_rms_ridge_expert_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-3.0, 3.0],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-3.0, 3.0, -3.0, 3.0])
        labels = np.array([0, 1, 1, 0])

        targets, reliability, posterior, conflicts = (
            method._rms_ridge_expert_supervision(mixture, scores, labels)
        )

        np.testing.assert_array_equal(
            targets,
            np.eye(2, dtype=np.float64)[labels],
        )
        self.assertTrue(np.all(reliability[:2] > 0.0))
        np.testing.assert_array_equal(reliability[2:], np.zeros(2))
        self.assertEqual(conflicts, 2)
        self.assertLess(float(posterior[0]), 0.5)
        self.assertGreater(float(posterior[1]), 0.5)

    def test_rms_ridge_expert_starts_at_r12_and_updates_after_prediction(
        self,
    ) -> None:
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        baseline = self.segmented_memory_posterior_ordinal_route_method()
        method = self.segmented_memory_posterior_rms_ridge_expert_method()
        for candidate in (baseline, method):
            candidate._mixture = copy.deepcopy(mixture)
            candidate.boundary_history = [-5.5]

        scores = np.array([-8.0, -7.0, -4.0, -3.0])
        first_images = self.score_feature_batch(
            scores,
            np.array([-1.0, -1.0, 1.0, 1.0]),
        )
        initial_state = method._novel_ordinal_ridge_state
        initial_inverse = initial_state["inverse_gram"].copy()
        baseline_first = baseline.predict(first_images)
        method_first = method.predict(first_images)
        np.testing.assert_array_equal(
            method_first.prob_fake.numpy(),
            baseline_first.prob_fake.numpy(),
        )
        np.testing.assert_array_equal(
            method_first.pred_label.numpy(),
            baseline_first.pred_label.numpy(),
        )
        np.testing.assert_array_equal(initial_state["inverse_gram"], initial_inverse)
        self.assertFalse(method._pending["prediction_rms_ridge_expert_ready"])
        baseline.adapt(first_images)
        first_stats = method.adapt(first_images)
        self.assertTrue(first_stats.extra["rms_ridge_expert_updated"])
        self.assertEqual(first_stats.extra["rms_ridge_expert_ready_experts"], 1)
        self.assertFalse(np.array_equal(initial_state["inverse_gram"], initial_inverse))

        for candidate in (baseline, method):
            candidate._mixture = copy.deepcopy(mixture)
            candidate.boundary_history = [-5.5]
        second_images = self.score_feature_batch(
            scores,
            np.array([1.0, 1.0, -1.0, -1.0]),
        )
        before_predict_weights = initial_state["weights"].copy()
        before_predict_inverse = initial_state["inverse_gram"].copy()
        baseline_second = baseline.predict(second_images)
        method_second = method.predict(second_images)
        self.assertTrue(method._pending["prediction_rms_ridge_expert_ready"])
        self.assertGreater(
            method._pending["prediction_rms_ridge_expert_base_rms"],
            0.0,
        )
        self.assertGreater(
            method._pending["prediction_rms_ridge_expert_ridge_rms"],
            0.0,
        )
        self.assertFalse(
            np.allclose(
                method_second.prob_fake.numpy(),
                baseline_second.prob_fake.numpy(),
            )
        )
        np.testing.assert_array_equal(initial_state["weights"], before_predict_weights)
        np.testing.assert_array_equal(
            initial_state["inverse_gram"],
            before_predict_inverse,
        )
        baseline.adapt(second_images)
        second_stats = method.adapt(second_images)
        self.assertFalse(np.array_equal(initial_state["weights"], before_predict_weights))
        np.testing.assert_allclose(method.score_history, baseline.score_history)
        np.testing.assert_allclose(method.boundary_history, baseline.boundary_history)
        self.assertEqual(method.segment_changes, baseline.segment_changes)
        self.assertEqual(
            second_stats.extra["rms_ridge_expert_label_changes"],
            second_stats.extra["rms_ridge_expert_real_to_fake"]
            + second_stats.extra["rms_ridge_expert_fake_to_real"],
        )
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-RMSRidgeExpert")
        self.assertEqual(metadata["research_version"], "R19")
        self.assertIn("one_hot", metadata["ridge_objective"])
        self.assertIn("historical", metadata["scale_alignment"])

    def test_rms_ridge_expert_normalization_is_invariant_to_ridge_scale(
        self,
    ) -> None:
        method = self.segmented_memory_posterior_rms_ridge_expert_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-3.0, 3.0],
            "sigmas": [0.75, 0.75],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-3.5, -2.5, 2.5, 3.5])
        labels = np.array([0, 0, 1, 1])
        features = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        base_probability = np.array([0.1, 0.2, 0.8, 0.9])
        base_margins = np.log(base_probability / (1.0 - base_probability))
        state = method._new_ordinal_ridge_state()
        self.assertTrue(
            method._update_rms_ridge_expert_state(
                state,
                mixture,
                scores,
                labels,
                features,
                base_margins,
            )
        )
        probability, *_ = method._rms_ridge_expert_probability(
            base_probability,
            features,
            state,
        )
        scaled_state = copy.deepcopy(state)
        scaled_state["weights"] *= 7.0
        scaled_state["cross_covariance"] *= 7.0
        scaled_probability, *_ = method._rms_ridge_expert_probability(
            base_probability,
            features,
            scaled_state,
        )
        np.testing.assert_allclose(
            scaled_probability,
            probability,
            atol=1e-12,
        )

    def test_rms_ridge_expert_requires_reliable_mass_for_both_classes(self) -> None:
        method = self.segmented_memory_posterior_rms_ridge_expert_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-3.0, 3.0],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        state = method._new_ordinal_ridge_state()
        updated = method._update_rms_ridge_expert_state(
            state,
            mixture,
            np.array([-3.5, -2.5]),
            np.array([0, 0]),
            np.array(
                [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            np.array([-1.5, -1.0]),
        )
        self.assertFalse(updated)
        self.assertFalse(method._ordinal_ridge_ready(state))
        self.assertGreater(float(state["class_mass"][0]), 0.0)
        self.assertEqual(float(state["class_mass"][1]), 0.0)

    def test_rms_ridge_expert_updates_only_the_r12_selected_memory(self) -> None:
        rng = np.random.default_rng(73)
        method = self.segmented_memory_posterior_rms_ridge_expert_method()
        returning = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        active = {
            "weights": [0.5, 0.5],
            "mus": [20.0, 25.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        returning_state = method._new_ordinal_ridge_state()
        active_state = method._new_ordinal_ridge_state()
        method._mixture = active
        method.boundary_history = [22.5]
        method.active_memory_index = 1
        method.segment_memories = [
            {
                "mixture": returning,
                "boundary": float(method._memory_boundary(returning)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 0,
                method._ORDINAL_RIDGE_MEMORY_KEY: returning_state,
            },
            {
                "mixture": active,
                "boundary": float(method._memory_boundary(active)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 1,
                method._ORDINAL_RIDGE_MEMORY_KEY: active_state,
            },
        ]
        scores = np.concatenate(
            [rng.normal(-8.0, 0.2, 48), rng.normal(-3.0, 0.2, 48)]
        )
        cues = np.concatenate([-np.ones(48), np.ones(48)])
        order = rng.permutation(len(scores))
        images = self.score_feature_batch(scores[order], cues[order])

        method.predict(images)
        self.assertEqual(
            method._pending["prediction_routing_expert"],
            "episodic_memory",
        )
        self.assertEqual(method._pending["prediction_routing_memory_index"], 0)
        stats = method.adapt(images)
        self.assertEqual(returning_state["updates"], 1)
        self.assertEqual(active_state["updates"], 0)
        self.assertTrue(stats.extra["rms_ridge_expert_updated"])
        self.assertTrue(stats.extra["routing_handoff_this_batch"])
        self.assertEqual(stats.extra["active_memory_index"], 0)

    def test_rms_ridge_expert_static_path_is_exact_r12_source_fallback(self) -> None:
        method = self.segmented_memory_posterior_rms_ridge_expert_method(
            adaptation_mode="static"
        )
        scores = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        images = self.score_feature_batch(scores, np.linspace(-1.0, 1.0, 5))
        prediction = method.predict(images)
        np.testing.assert_allclose(
            prediction.prob_fake.numpy(),
            method._source_probability(scores),
            atol=1e-6,
        )
        stats = method.adapt(images)
        self.assertEqual(stats.extra["rms_ridge_expert_updates"], 0)
        self.assertEqual(stats.extra["rms_ridge_expert_count"], 0)
        self.assertEqual(method.trainable_parameters, 0)

    def test_equal_prior_ridge_centers_exact_historical_class_midpoint(
        self,
    ) -> None:
        r19 = self.segmented_memory_posterior_rms_ridge_expert_method()
        method = self.segmented_memory_posterior_equal_prior_ridge_expert_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-3.0, 3.0],
            "sigmas": [0.75, 0.75],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-4.0, -3.5, -2.5, 2.5, 3.5])
        labels = np.array([0, 0, 0, 1, 1])
        features = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        base_probability = np.array([0.05, 0.1, 0.2, 0.8, 0.9])
        base_margins = np.log(base_probability / (1.0 - base_probability))
        r19_state = r19._new_ordinal_ridge_state()
        state = method._new_ordinal_ridge_state()
        self.assertTrue(
            r19._update_rms_ridge_expert_state(
                r19_state,
                mixture,
                scores,
                labels,
                features,
                base_margins,
            )
        )
        self.assertTrue(
            method._update_rms_ridge_expert_state(
                state,
                mixture,
                scores,
                labels,
                features,
                base_margins,
            )
        )
        for key in (
            "inverse_gram",
            "cross_covariance",
            "weights",
            "class_mass",
        ):
            np.testing.assert_array_equal(state[key], r19_state[key])
        for key in (
            "updates",
            "candidate_samples",
            "effective_support",
            "base_margin_square_sum",
            "posterior_conflicts",
        ):
            self.assertEqual(state[key], r19_state[key])

        _, reliability, _, _ = method._rms_ridge_expert_supervision(
            mixture,
            scores,
            labels,
        )
        direction = state["weights"][:, 1] - state["weights"][:, 0]
        raw_margin = features @ direction
        class_mass = np.bincount(labels, weights=reliability, minlength=2)
        real_centroid = float(
            np.sum(reliability[labels == 0] * raw_margin[labels == 0])
            / class_mass[0]
        )
        fake_centroid = float(
            np.sum(reliability[labels == 1] * raw_margin[labels == 1])
            / class_mass[1]
        )
        expected_center = 0.5 * (real_centroid + fake_centroid)
        expected_energy = float(
            np.sum(reliability * (raw_margin - expected_center) ** 2)
        )
        moments = method._equal_prior_ridge_moments(state)
        self.assertIsNotNone(moments)
        assert moments is not None
        self.assertAlmostEqual(moments[0], expected_center)
        self.assertAlmostEqual(moments[1], fake_centroid - real_centroid)
        self.assertAlmostEqual(moments[2], expected_energy)
        scales = method._rms_ridge_expert_scales(state)
        self.assertIsNotNone(scales)
        assert scales is not None
        self.assertAlmostEqual(
            scales[1],
            np.sqrt(expected_energy / reliability.sum()),
        )

        probability, _, centered_margin, _, _ = (
            method._rms_ridge_expert_probability(
                base_probability,
                features,
                state,
            )
        )
        np.testing.assert_allclose(
            centered_margin,
            raw_margin - expected_center,
            atol=1e-12,
        )
        self.assertTrue(np.all((probability > 0.0) & (probability < 1.0)))
        centered_real = real_centroid - expected_center
        centered_fake = fake_centroid - expected_center
        self.assertAlmostEqual(centered_real + centered_fake, 0.0)

    def test_equal_prior_ridge_preserves_r19_learning_and_causal_prediction(
        self,
    ) -> None:
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        r19 = self.segmented_memory_posterior_rms_ridge_expert_method()
        method = self.segmented_memory_posterior_equal_prior_ridge_expert_method()
        for candidate in (r19, method):
            candidate._mixture = copy.deepcopy(mixture)
            candidate.boundary_history = [-5.5]

        first_images = self.score_feature_batch(
            np.array([-8.5, -7.5, -6.5, -3.0]),
            np.array([-1.0, -0.5, 0.0, 1.0]),
        )
        r19_first = r19.predict(first_images)
        method_first = method.predict(first_images)
        np.testing.assert_array_equal(
            method_first.prob_fake.numpy(),
            r19_first.prob_fake.numpy(),
        )
        r19.adapt(first_images)
        method.adapt(first_images)
        r19_state = r19._novel_ordinal_ridge_state
        state = method._novel_ordinal_ridge_state
        for key in ("inverse_gram", "cross_covariance", "weights", "class_mass"):
            np.testing.assert_array_equal(state[key], r19_state[key])

        for candidate in (r19, method):
            candidate._mixture = copy.deepcopy(mixture)
            candidate.boundary_history = [-5.5]
        second_images = self.score_feature_batch(
            np.array([-8.0, -7.0, -4.0, -3.0]),
            np.array([1.0, 0.5, -0.5, -1.0]),
        )
        before_weights = state["weights"].copy()
        before_inverse = state["inverse_gram"].copy()
        r19_second = r19.predict(second_images)
        method_second = method.predict(second_images)
        self.assertTrue(
            method._pending["prediction_equal_prior_ridge_applied"]
        )
        self.assertTrue(
            np.isfinite(method._pending["prediction_equal_prior_ridge_center"])
        )
        self.assertFalse(
            np.allclose(
                method_second.prob_fake.numpy(),
                r19_second.prob_fake.numpy(),
            )
        )
        np.testing.assert_array_equal(state["weights"], before_weights)
        np.testing.assert_array_equal(state["inverse_gram"], before_inverse)
        r19_stats = r19.adapt(second_images)
        stats = method.adapt(second_images)
        for key in ("inverse_gram", "cross_covariance", "weights", "class_mass"):
            np.testing.assert_array_equal(state[key], r19_state[key])
        np.testing.assert_allclose(method.score_history, r19.score_history)
        np.testing.assert_allclose(method.boundary_history, r19.boundary_history)
        self.assertEqual(method.segment_changes, r19.segment_changes)
        self.assertEqual(
            stats.extra["rms_ridge_expert_updates"],
            r19_stats.extra["rms_ridge_expert_updates"],
        )
        self.assertEqual(
            stats.extra["equal_prior_ridge_ready_experts"],
            stats.extra["rms_ridge_expert_ready_experts"],
        )
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-EqualPriorRidge")
        self.assertEqual(metadata["research_version"], "R20")
        self.assertEqual(metadata["new_persistent_state"], "none_reuse_r19_sufficient_statistics")
        self.assertEqual(metadata["new_target_hyperparameters"], 0)

    def test_equal_prior_ridge_static_path_is_exact_r12_source_fallback(
        self,
    ) -> None:
        method = self.segmented_memory_posterior_equal_prior_ridge_expert_method(
            adaptation_mode="static"
        )
        scores = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        images = self.score_feature_batch(scores, np.linspace(-1.0, 1.0, 5))
        prediction = method.predict(images)
        np.testing.assert_allclose(
            prediction.prob_fake.numpy(),
            method._source_probability(scores),
            atol=1e-6,
        )
        stats = method.adapt(images)
        self.assertEqual(stats.extra["rms_ridge_expert_updates"], 0)
        self.assertEqual(stats.extra["equal_prior_ridge_ready_experts"], 0)
        self.assertEqual(method.trainable_parameters, 0)

    def test_evidence_gated_ridge_uses_inverse_gram_feature_variance_reduction(
        self,
    ) -> None:
        method = self.segmented_memory_posterior_evidence_gated_ridge_expert_method()
        state = method._new_ordinal_ridge_state()
        features = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_array_equal(
            method._inverse_gram_feature_evidence(features, state),
            np.zeros(3),
        )
        arbitrary_feature = np.array([[0.3, 0.4, 0.5, 1.0]], dtype=np.float64)
        np.testing.assert_array_equal(
            method._inverse_gram_feature_evidence(arbitrary_feature, state),
            np.zeros(1),
        )

        state["inverse_gram"] = np.diag([0.25, 0.75, 1.0, 0.0])
        evidence = method._inverse_gram_feature_evidence(features, state)
        np.testing.assert_allclose(evidence, np.array([0.75, 0.25, 0.0]))
        self.assertTrue(np.all((evidence >= 0.0) & (evidence <= 1.0)))

        state["inverse_gram"][-1, -1] = 1000.0
        np.testing.assert_allclose(
            method._inverse_gram_feature_evidence(features, state),
            evidence,
        )
        state["inverse_gram"][0, 0] = 0.1
        increased = method._inverse_gram_feature_evidence(features, state)
        self.assertGreater(increased[0], evidence[0])

    def test_evidence_gated_ridge_preserves_r20_learning_and_is_causal(
        self,
    ) -> None:
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        r20 = self.segmented_memory_posterior_equal_prior_ridge_expert_method()
        method = self.segmented_memory_posterior_evidence_gated_ridge_expert_method()
        for candidate in (r20, method):
            candidate._mixture = copy.deepcopy(mixture)
            candidate.boundary_history = [-5.5]

        first_images = self.score_feature_batch(
            np.array([-8.5, -7.5, -6.5, -3.0]),
            np.array([-1.0, -0.5, 0.0, 1.0]),
        )
        r20_first = r20.predict(first_images)
        method_first = method.predict(first_images)
        np.testing.assert_array_equal(
            method_first.prob_fake.numpy(),
            r20_first.prob_fake.numpy(),
        )
        r20.adapt(first_images)
        method.adapt(first_images)
        r20_state = r20._novel_ordinal_ridge_state
        state = method._novel_ordinal_ridge_state
        for key in ("inverse_gram", "cross_covariance", "weights", "class_mass"):
            np.testing.assert_array_equal(state[key], r20_state[key])

        for candidate in (r20, method):
            candidate._mixture = copy.deepcopy(mixture)
            candidate.boundary_history = [-5.5]
        second_images = self.score_feature_batch(
            np.array([-8.0, -7.0, -4.0, -3.0]),
            np.array([1.0, 0.5, -0.5, -1.0]),
        )
        before = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in state.items()
        }
        method_second = method.predict(second_images)
        self.assertTrue(
            method._pending["prediction_evidence_gated_ridge_applied"]
        )
        features = method._pending_ordinal_ridge_features
        self.assertIsNotNone(features)
        assert features is not None
        evidence = method._inverse_gram_feature_evidence(features, state)
        self.assertTrue(np.all(evidence > 0.0))
        self.assertTrue(np.all(evidence < 1.0))
        self.assertAlmostEqual(
            method._pending["prediction_evidence_gated_ridge_mean"],
            float(np.mean(evidence)),
        )
        scales = method._rms_ridge_expert_scales(state)
        moments = method._equal_prior_ridge_moments(state)
        self.assertIsNotNone(scales)
        self.assertIsNotNone(moments)
        assert scales is not None and moments is not None
        base_rms, ridge_rms, _ = scales
        center, _, _ = moments
        base_margin = method._pending_rms_ridge_expert_base_margins
        self.assertIsNotNone(base_margin)
        assert base_margin is not None
        direction = state["weights"][:, 1] - state["weights"][:, 0]
        ridge_margin = features @ direction - center
        aligned_ridge_margin = ridge_margin * (base_rms / ridge_rms)
        expected_probability = method._stable_sigmoid(
            base_margin + evidence * (aligned_ridge_margin - base_margin)
        )
        np.testing.assert_allclose(
            method_second.prob_fake.numpy(),
            expected_probability,
            atol=1e-7,
        )
        for key, value in before.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(state[key], value)
            else:
                self.assertEqual(state[key], value)

        r20.predict(second_images)
        r20_stats = r20.adapt(second_images)
        stats = method.adapt(second_images)
        for key in ("inverse_gram", "cross_covariance", "weights", "class_mass"):
            np.testing.assert_array_equal(state[key], r20_state[key])
        for key in (
            "updates",
            "candidate_samples",
            "effective_support",
            "base_margin_square_sum",
            "posterior_conflicts",
        ):
            self.assertEqual(state[key], r20_state[key])
        np.testing.assert_allclose(method.score_history, r20.score_history)
        np.testing.assert_allclose(method.boundary_history, r20.boundary_history)
        self.assertEqual(method.segment_changes, r20.segment_changes)
        self.assertEqual(
            stats.extra["rms_ridge_expert_updates"],
            r20_stats.extra["rms_ridge_expert_updates"],
        )
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-EvidenceGatedRidge")
        self.assertEqual(metadata["research_version"], "R21")
        self.assertEqual(metadata["new_persistent_state"], "none_reuse_r20_inverse_gram")
        self.assertEqual(metadata["new_target_hyperparameters"], 0)

    def test_evidence_gated_ridge_zero_evidence_is_exact_r12_probability(
        self,
    ) -> None:
        method = self.segmented_memory_posterior_evidence_gated_ridge_expert_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-3.0, 3.0],
            "sigmas": [0.75, 0.75],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-4.0, -3.0, 2.5, 3.5])
        labels = np.array([0, 0, 1, 1])
        features = np.array(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        base_probability = np.array([0.1, 0.2, 0.8, 0.9])
        base_margin = np.log(base_probability / (1.0 - base_probability))
        state = method._new_ordinal_ridge_state()
        self.assertTrue(
            method._update_rms_ridge_expert_state(
                state,
                mixture,
                scores,
                labels,
                features,
                base_margin,
            )
        )
        state["inverse_gram"] = np.eye(method.ordinal_ridge_feature_dim)
        probability, _, _, _, _ = method._rms_ridge_expert_probability(
            base_probability,
            features,
            state,
        )
        np.testing.assert_array_equal(probability, base_probability)

    def test_evidence_gated_ridge_static_path_is_exact_r12_source_fallback(
        self,
    ) -> None:
        method = self.segmented_memory_posterior_evidence_gated_ridge_expert_method(
            adaptation_mode="static"
        )
        scores = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        images = self.score_feature_batch(scores, np.linspace(-1.0, 1.0, 5))
        prediction = method.predict(images)
        np.testing.assert_allclose(
            prediction.prob_fake.numpy(),
            method._source_probability(scores),
            atol=1e-6,
        )
        stats = method.adapt(images)
        self.assertEqual(stats.extra["rms_ridge_expert_updates"], 0)
        self.assertEqual(stats.extra["equal_prior_ridge_ready_experts"], 0)
        self.assertEqual(method.trainable_parameters, 0)

    def test_feature_routed_trusted_ridge_uses_gmm_labels_as_weighted_targets(
        self,
    ) -> None:
        method = self.segmented_memory_posterior_feature_routed_trusted_ridge_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-3.0, 3.0],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-3.0, 3.0])
        deliberately_wrong_r12_labels = np.array([1, 0])

        targets, reliability, posterior, conflicts = (
            method._rms_ridge_expert_supervision(
                mixture,
                scores,
                deliberately_wrong_r12_labels,
            )
        )

        np.testing.assert_array_equal(targets, np.eye(2, dtype=np.float64))
        self.assertTrue(np.all(reliability > 0.99))
        self.assertLess(float(posterior[0]), 0.5)
        self.assertGreater(float(posterior[1]), 0.5)
        self.assertEqual(conflicts, 0)

    def test_feature_routed_trusted_ridge_routes_only_by_clip_prototypes(
        self,
    ) -> None:
        method = self.segmented_memory_posterior_feature_routed_trusted_ridge_method()
        returning = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        active = {
            "weights": [0.5, 0.5],
            "mus": [20.0, 25.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        returning_state = method._new_ordinal_ridge_state()
        returning_state["route_feature_sum"] = np.array([1.0, 0.0, 0.0])
        returning_state["route_feature_mass"] = 1.0
        active_state = method._new_ordinal_ridge_state()
        active_state["route_feature_sum"] = np.array([0.0, 1.0, 0.0])
        active_state["route_feature_mass"] = 1.0
        method._mixture = active
        method.active_memory_index = 1
        method.segment_memories = [
            {
                "mixture": returning,
                "boundary": float(method._memory_boundary(returning)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 0,
                method._ORDINAL_RIDGE_MEMORY_KEY: returning_state,
            },
            {
                "mixture": active,
                "boundary": float(method._memory_boundary(active)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 1,
                method._ORDINAL_RIDGE_MEMORY_KEY: active_state,
            },
        ]
        method._feature_route_query = np.array([[1.0, 0.0, 0.0]])

        near_returning = method._routing_candidates(np.array([-8.0]))
        near_active_score = method._routing_candidates(np.array([24.0]))

        for candidates in (near_returning, near_active_score):
            winner = min(candidates, key=lambda candidate: candidate["deviance"])
            self.assertEqual(winner["expert"], "episodic_memory")
            self.assertEqual(winner["memory_index"], 0)
            self.assertAlmostEqual(float(winner["feature_similarity"]), 1.0)
        np.testing.assert_allclose(
            [candidate["deviance"] for candidate in near_returning],
            [candidate["deviance"] for candidate in near_active_score],
        )

    def test_feature_route_removes_the_source_binary_direction(self) -> None:
        method = self.segmented_memory_posterior_feature_routed_trusted_ridge_method()
        source_direction = method._ordinal_ridge_source_direction
        features = np.array(
            [
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )

        routed = method._feature_route_coordinates(features)

        np.testing.assert_allclose(routed @ source_direction, 0.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(routed, axis=1), 1.0)

    def test_feature_routed_trusted_ridge_predicts_base_plus_ridge_then_updates(
        self,
    ) -> None:
        method = self.segmented_memory_posterior_feature_routed_trusted_ridge_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        method._mixture = copy.deepcopy(mixture)
        method.boundary_history = [-5.5]
        scores = np.array([-8.0, -7.0, -4.0, -3.0])
        images = self.score_feature_batch(
            scores,
            np.array([-1.0, -0.5, 0.5, 1.0]),
        )
        state = method._novel_ordinal_ridge_state
        before_weights = state["weights"].copy()

        prediction = method.predict(images)
        expected = 0.5 * (method._source_probability(scores) + 0.5)
        np.testing.assert_allclose(prediction.prob_fake.numpy(), expected, atol=1e-7)
        np.testing.assert_array_equal(state["weights"], before_weights)
        self.assertTrue(method._pending["prediction_rms_ridge_expert_applied"])

        stats = method.adapt(images)
        self.assertTrue(stats.extra["rms_ridge_expert_updated"])
        self.assertFalse(np.array_equal(state["weights"], before_weights))
        self.assertGreater(float(state["route_feature_mass"]), 0.0)
        metadata = method.reproduction_metadata
        self.assertEqual(
            metadata["research_name"],
            "ASCAL-JMP-FeatureRoutedTrustedRidge",
        )
        self.assertEqual(
            metadata["routing_coordinate"],
            "l2_normalized_frozen_clip_feature_orthogonal_to_source_binary_head",
        )
        self.assertFalse(metadata["routing_score_used"])
        self.assertFalse(metadata["gmm_in_final_prediction"])

    def test_gaussian_replay_mlp_predicts_then_updates_distribution_and_head(
        self,
    ) -> None:
        method = (
            self.segmented_memory_posterior_feature_routed_gaussian_replay_mlp_method()
        )
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        method._mixture = copy.deepcopy(mixture)
        method.boundary_history = [-5.5]
        scores = np.array([-8.0, -7.0, -4.0, -3.0])
        images = self.score_feature_batch(
            scores,
            np.array([-1.0, -0.5, 0.5, 1.0]),
        )
        state = method._novel_ordinal_ridge_state
        source_probability = method._source_probability(scores)

        first = method.predict(images)
        np.testing.assert_allclose(
            first.prob_fake.numpy(), source_probability, atol=1e-7
        )
        np.testing.assert_array_equal(state["class_samples"], np.zeros(2))
        self.assertIsNone(state["mlp_head"])

        stats = method.adapt(images)
        self.assertTrue(stats.extra["gaussian_replay_updated"])
        np.testing.assert_array_equal(state["class_samples"], np.array([2, 2]))
        self.assertTrue(np.all(state["class_mass"] > 0.0))
        self.assertEqual(state["head_updates"], 1)
        self.assertIsNotNone(state["mlp_head"])
        self.assertGreater(state["generated_samples"], 0)
        self.assertNotIn("raw_features", state)

        method._mixture = copy.deepcopy(mixture)
        method.boundary_history = [-5.5]
        before = [
            parameter.detach().clone()
            for parameter in state["mlp_head"].parameters()
        ]
        second = method.predict(images)
        self.assertTrue(method._pending["prediction_gaussian_replay_applied"])
        self.assertFalse(
            np.allclose(second.prob_fake.numpy(), source_probability, atol=1e-7)
        )
        for expected, parameter in zip(before, state["mlp_head"].parameters()):
            self.assertTrue(self.torch.equal(expected, parameter.detach()))
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-GaussianReplayMLP")
        self.assertEqual(metadata["research_version"], "R25")
        self.assertFalse(metadata["gmm_in_final_prediction"])
        self.assertFalse(metadata["raw_features_stored"])

        method.discard_pending_prediction()

    def test_expanded_gaussian_replay_consumes_each_fresh_draw_once(self) -> None:
        method = (
            self.segmented_memory_posterior_feature_routed_expanded_gaussian_replay_mlp_method(
                replay_samples=8
            )
        )
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        method._mixture = copy.deepcopy(mixture)
        method.boundary_history = [-5.5]
        scores = np.array([-8.0, -7.0, -4.0, -3.0])
        images = self.score_feature_batch(
            scores,
            np.array([-1.0, -0.5, 0.5, 1.0]),
        )
        state = method._novel_ordinal_ridge_state
        source_probability = method._source_probability(scores)

        prediction = method.predict(images)
        np.testing.assert_allclose(
            prediction.prob_fake.numpy(), source_probability, atol=1e-7
        )
        stats = method.adapt(images)

        self.assertTrue(stats.extra["gaussian_replay_updated"])
        self.assertEqual(state["head_updates"], 1)
        self.assertEqual(state["generated_samples"], 8)
        self.assertEqual(state["optimizer_steps"], 2)
        self.assertEqual(stats.extra["gaussian_replay_generated_samples"], 8)
        self.assertEqual(stats.extra["gaussian_replay_optimizer_steps"], 2)
        metadata = method.reproduction_metadata
        self.assertEqual(
            metadata["research_name"],
            "ASCAL-JMP-ExpandedGaussianReplay",
        )
        self.assertEqual(metadata["research_version"], "R26")
        self.assertEqual(metadata["feature_replay_samples_per_update"], 8)
        self.assertFalse(metadata["generated_feature_reuse"])
        self.assertTrue(metadata["frozen_base"])

    def test_shared_gaussian_replay_uses_one_head_for_distinct_experts(self) -> None:
        method = (
            self.segmented_memory_posterior_feature_routed_shared_gaussian_replay_mlp_method(
                replay_samples=8
            )
        )
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-8.0, -7.0, -4.0, -3.0])
        images = self.score_feature_batch(
            scores,
            np.array([-1.0, -0.5, 0.5, 1.0]),
        )
        _, features = method._batch_scores_and_gaussian_replay_features(images)
        method._feature_route_query = None
        first_state = method._novel_ordinal_ridge_state
        second_state = method._new_ordinal_ridge_state()

        self.assertTrue(
            method._update_gaussian_replay_state(
                first_state,
                mixture,
                scores,
                features,
            )
        )
        shared_head = method._shared_gaussian_replay_state["mlp_head"]
        self.assertIsNotNone(shared_head)
        self.assertIsNone(first_state["mlp_head"])
        self.assertEqual(method.trainable_parameters, 21)
        before = [parameter.detach().clone() for parameter in shared_head.parameters()]

        self.assertTrue(
            method._update_gaussian_replay_state(
                second_state,
                mixture,
                scores,
                features,
            )
        )
        self.assertIs(method._shared_gaussian_replay_state["mlp_head"], shared_head)
        self.assertIsNone(second_state["mlp_head"])
        self.assertEqual(method.trainable_parameters, 21)
        self.assertEqual(method._shared_gaussian_replay_state["head_updates"], 2)
        self.assertEqual(method._shared_gaussian_replay_state["generated_samples"], 16)
        self.assertEqual(first_state["head_updates"], 0)
        self.assertEqual(second_state["head_updates"], 0)
        self.assertTrue(
            any(
                not self.torch.equal(expected, parameter.detach())
                for expected, parameter in zip(before, shared_head.parameters())
            )
        )
        first_residual = method._gaussian_replay_residual(first_state, features)
        second_residual = method._gaussian_replay_residual(second_state, features)
        np.testing.assert_array_equal(first_residual, second_residual)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-SharedResidualHead")
        self.assertEqual(metadata["research_version"], "R27")
        self.assertEqual(metadata["head_count"], 1)

    def test_linear_gaussian_replay_keeps_r26_replay_rng_aligned(self) -> None:
        nonlinear = (
            self.segmented_memory_posterior_feature_routed_expanded_gaussian_replay_mlp_method(
                replay_samples=8
            )
        )
        linear = (
            self.segmented_memory_posterior_feature_routed_linear_gaussian_replay_method(
                replay_samples=8
            )
        )
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-8.0, -7.0, -4.0, -3.0])
        images = self.score_feature_batch(
            scores,
            np.array([-1.0, -0.5, 0.5, 1.0]),
        )
        _, nonlinear_features = (
            nonlinear._batch_scores_and_gaussian_replay_features(images)
        )
        _, linear_features = linear._batch_scores_and_gaussian_replay_features(images)
        nonlinear._feature_route_query = None
        linear._feature_route_query = None
        nonlinear_state = nonlinear._novel_ordinal_ridge_state
        linear_state = linear._novel_ordinal_ridge_state

        self.assertTrue(
            nonlinear._update_gaussian_replay_state(
                nonlinear_state,
                mixture,
                scores,
                nonlinear_features,
            )
        )
        self.assertTrue(
            linear._update_gaussian_replay_state(
                linear_state,
                mixture,
                scores,
                linear_features,
            )
        )
        self.assertIsInstance(linear_state["mlp_head"], self.nn.Linear)
        self.assertEqual(linear.trainable_parameters, 4)
        nonlinear_draws, nonlinear_labels = nonlinear._sample_gaussian_replay(
            nonlinear_state,
            8,
        )
        linear_draws, linear_labels = linear._sample_gaussian_replay(
            linear_state,
            8,
        )
        np.testing.assert_array_equal(nonlinear_labels, linear_labels)
        np.testing.assert_array_equal(nonlinear_draws, linear_draws)
        metadata = linear.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-LinearResidualHead")
        self.assertEqual(metadata["research_version"], "R28")
        self.assertEqual(metadata["expert_head_parameters"], 4)

    def test_uniform_gaussian_replay_keeps_labels_but_sets_unit_weight(self) -> None:
        baseline = (
            self.segmented_memory_posterior_feature_routed_expanded_gaussian_replay_mlp_method(
                replay_samples=8
            )
        )
        uniform = (
            self.segmented_memory_posterior_feature_routed_uniform_gaussian_replay_method(
                replay_samples=8
            )
        )
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-8.0, -7.0, -4.0, -3.0])
        baseline_labels, baseline_weights, baseline_posterior = (
            baseline._gaussian_replay_supervision(mixture, scores)
        )
        labels, weights, posterior = uniform._gaussian_replay_supervision(
            mixture,
            scores,
        )
        np.testing.assert_array_equal(labels, baseline_labels)
        np.testing.assert_array_equal(posterior, baseline_posterior)
        np.testing.assert_array_equal(weights, np.ones(scores.size))
        self.assertTrue(np.any(baseline_weights < 1.0))
        metadata = uniform.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-UniformConfidence")
        self.assertEqual(metadata["research_version"], "R29")

    def test_active_gaussian_replay_never_returns_a_memory_candidate(self) -> None:
        method = self.segmented_memory_posterior_active_gaussian_replay_method()
        active = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        memory = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.4, 0.4],
            "components": 2,
            "bic": 0.0,
        }
        method._mixture = copy.deepcopy(active)
        method.boundary_history = [-5.5]
        method.segment_memories = [
            {
                "mixture": copy.deepcopy(memory),
                "boundary": -5.5,
                "samples": 200,
            }
        ]
        candidates = method._routing_candidates(
            np.array([-8.0, -7.0, -4.0, -3.0])
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["expert"], "active_learning_state")
        method._mixture = None
        self.assertEqual(
            method._routing_candidates(np.array([-8.0, -3.0])),
            [],
        )
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-ActiveOnly")
        self.assertEqual(metadata["research_version"], "R30")
        self.assertEqual(
            metadata["historical_expert_recall"],
            "segment_change_callback_only",
        )

    def test_no_historical_recall_rejects_batch_and_segment_memory(self) -> None:
        method = self.segmented_memory_posterior_no_historical_recall_method()
        active = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        memory = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.4, 0.4],
            "components": 2,
            "bic": -100.0,
        }
        method._mixture = copy.deepcopy(active)
        method.boundary_history = [-5.5]
        method.segment_memories = [
            {
                "mixture": copy.deepcopy(memory),
                "boundary": -5.5,
                "latest_samples": 200,
                "total_samples": 200,
                "visits": 1,
                "recalls": 0,
            }
        ]
        candidates = method._routing_candidates(
            np.array([-8.0, -7.0, -4.0, -3.0])
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["expert"], "active_learning_state")
        selected = method._select_recalled_memory(
            np.array([-8.0, -7.0, -4.0, -3.0]),
            active,
            excluded_index=None,
        )
        self.assertIsNone(selected)
        self.assertIsNone(method.last_memory_fixed_score)
        self.assertEqual(method.last_memory_new_bic, 0.0)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-NoHistoricalRecall")
        self.assertEqual(metadata["research_version"], "R34")
        self.assertFalse(metadata["historical_expert_recall"])
        self.assertFalse(metadata["segment_change_score_memory_recall"])

    def test_current_segment_core_discards_completed_state_and_archive(self) -> None:
        method = self.current_segment_gaussian_replay_method()
        old_state = method._novel_ordinal_ridge_state
        old_state["candidate_samples"] = 4
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        stored = method._store_completed_segment(mixture, 200)
        self.assertIsNone(stored)
        self.assertEqual(method.segment_memories, [])
        self.assertIsNot(method._novel_ordinal_ridge_state, old_state)
        self.assertEqual(method.discarded_completed_segment_states, 1)
        method._mixture = copy.deepcopy(mixture)
        method.boundary_history = [-5.5]
        candidates = method._routing_candidates(
            np.array([-8.0, -7.0, -4.0, -3.0])
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["expert"], "active_learning_state")
        self.assertIsNone(candidates[0]["memory_index"])
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-CurrentSegmentCore")
        self.assertEqual(metadata["research_version"], "R35")
        self.assertEqual(metadata["episodic_memory_role"], "none")

    def test_global_stream_core_never_runs_segment_change_scan(self) -> None:
        method = self.global_stream_gaussian_replay_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        method._mixture = copy.deepcopy(mixture)
        method.score_history = [-8.0, -7.0, -4.0, -3.0]
        method.score_batches = [
            np.array([-8.0, -7.0]),
            np.array([-4.0, -3.0]),
        ]
        old_state = method._novel_ordinal_ridge_state
        method._detect_segment_change()
        self.assertIs(method._novel_ordinal_ridge_state, old_state)
        self.assertEqual(method.segment_changes, 0)
        self.assertEqual(method.segment_checks, 0)
        self.assertEqual(method.discarded_completed_segment_states, 0)
        self.assertIsNone(method.last_segment_gain)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-GlobalStreamCore")
        self.assertEqual(metadata["research_version"], "R36")
        self.assertEqual(metadata["segmentation_rule"], "none")

    def test_clip_expert_memory_routes_by_feature_but_never_score_recalls(
        self,
    ) -> None:
        method = self.clip_routed_gaussian_replay_memory_method()
        returning = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": -100.0,
        }
        active = {
            "weights": [0.5, 0.5],
            "mus": [20.0, 25.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        returning_state = method._new_ordinal_ridge_state()
        returning_state["route_feature_sum"] = np.array([1.0, 0.0, 0.0])
        returning_state["route_feature_mass"] = 1.0
        active_state = method._new_ordinal_ridge_state()
        active_state["route_feature_sum"] = np.array([0.0, 1.0, 0.0])
        active_state["route_feature_mass"] = 1.0
        method._mixture = copy.deepcopy(active)
        method.active_memory_index = 1
        method.segment_memories = [
            {
                "mixture": copy.deepcopy(returning),
                "boundary": float(method._memory_boundary(returning)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 0,
                method._ORDINAL_RIDGE_MEMORY_KEY: returning_state,
            },
            {
                "mixture": copy.deepcopy(active),
                "boundary": float(method._memory_boundary(active)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 1,
                method._ORDINAL_RIDGE_MEMORY_KEY: active_state,
            },
        ]
        method._feature_route_query = np.array([[1.0, 0.0, 0.0]])
        candidates = method._routing_candidates(np.array([24.0]))
        winner = min(candidates, key=lambda candidate: candidate["deviance"])
        self.assertEqual(winner["expert"], "episodic_memory")
        self.assertEqual(winner["memory_index"], 0)
        self.assertAlmostEqual(float(winner["feature_similarity"]), 1.0)

        selected = method._select_recalled_memory(
            np.array([-8.0, -7.0, -4.0, -3.0]),
            returning,
            excluded_index=0,
        )
        self.assertIsNone(selected)
        self.assertIsNone(method.last_memory_fixed_score)
        self.assertEqual(method.last_memory_new_bic, -100.0)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-CLIPExpertMemory")
        self.assertEqual(metadata["research_version"], "R37")
        self.assertFalse(metadata["segment_change_score_memory_recall"])
        self.assertEqual(metadata["historical_expert_recall"], "clip_feature_route_only")

    def test_current_batch_replay_uses_only_arrived_features(self) -> None:
        method = self.segmented_memory_posterior_current_batch_replay_method(
            replay_samples=8
        )
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-8.0, -7.0, -4.0, -3.0])
        images = self.score_feature_batch(
            scores,
            np.array([-1.0, -0.5, 0.5, 1.0]),
        )
        _, features = method._batch_scores_and_gaussian_replay_features(images)
        method._feature_route_query = None
        labels, reliability, _ = method._gaussian_replay_supervision(
            mixture,
            scores,
        )
        replay = method._gaussian_replay_training_samples(
            method._novel_ordinal_ridge_state,
            8,
            features,
            labels,
            reliability,
        )
        self.assertIsNotNone(replay)
        replay_features, replay_labels = replay
        self.assertEqual(replay_features.shape, (8, 3))
        np.testing.assert_array_equal(
            np.bincount(replay_labels.astype(np.int64), minlength=2),
            np.array([4, 4]),
        )
        for row in replay_features:
            self.assertTrue(np.any(np.all(features == row[None, :], axis=1)))

        state = method._novel_ordinal_ridge_state
        self.assertTrue(
            method._update_gaussian_replay_state(
                state,
                mixture,
                scores,
                features,
            )
        )
        updates = int(state["head_updates"])
        one_class_scores = np.array([-9.0, -8.5, -8.0, -7.5])
        one_class_images = self.score_feature_batch(
            one_class_scores,
            np.array([-1.0, -0.5, 0.5, 1.0]),
        )
        _, one_class_features = method._batch_scores_and_gaussian_replay_features(
            one_class_images
        )
        method._feature_route_query = None
        method._update_gaussian_replay_state(
            state,
            mixture,
            one_class_scores,
            one_class_features,
        )
        self.assertEqual(state["head_updates"], updates)
        self.assertEqual(method.current_batch_replay_skipped_updates, 1)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-CurrentBatchReplay")
        self.assertEqual(metadata["research_version"], "R31")

    def test_prior_gaussian_replay_follows_accumulated_class_mass(self) -> None:
        method = self.segmented_memory_posterior_prior_gaussian_replay_method(
            replay_samples=8
        )
        state = method._novel_ordinal_ridge_state
        state["class_samples"] = np.array([90, 10], dtype=np.int64)
        state["class_mass"] = np.array([90.0, 10.0], dtype=np.float64)
        state["feature_mean"][0] = np.array([-1.0, 0.0, 0.0])
        state["feature_mean"][1] = np.array([1.0, 0.0, 0.0])
        state["feature_m2"][:] = 1.0
        self.assertEqual(
            method._gaussian_replay_class_counts(state, 256),
            (230, 26),
        )
        features, labels = method._sample_gaussian_replay(state, 256)
        self.assertEqual(features.shape, (256, 3))
        np.testing.assert_array_equal(
            np.bincount(labels.astype(np.int64), minlength=2),
            np.array([230, 26]),
        )
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-PriorReplay")
        self.assertEqual(metadata["research_version"], "R32")

    def test_source_replay_uses_frozen_source_probability_for_supervision(
        self,
    ) -> None:
        baseline = (
            self.segmented_memory_posterior_feature_routed_expanded_gaussian_replay_mlp_method(
                replay_samples=8
            )
        )
        method = self.segmented_memory_posterior_source_replay_method(
            replay_samples=8
        )
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-7.5, -3.5],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-8.0, -7.0, -4.0, -3.0])
        labels, reliability, probability = method._gaussian_replay_supervision(
            mixture,
            scores,
        )
        expected_probability = np.asarray(method._source_probability(scores))
        np.testing.assert_allclose(probability, expected_probability)
        np.testing.assert_array_equal(
            labels,
            (expected_probability >= 0.5).astype(np.int64),
        )
        np.testing.assert_allclose(
            reliability,
            np.abs(2.0 * expected_probability - 1.0),
        )
        _, _, gmm_posterior = baseline._gaussian_replay_supervision(
            mixture,
            scores,
        )
        self.assertFalse(np.allclose(probability, gmm_posterior))
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-SourceSupervision")
        self.assertEqual(metadata["research_version"], "R33")
        self.assertFalse(metadata["gmm_in_feature_supervision"])

    def test_source_ridge_experts_clone_complete_source_statistics(self) -> None:
        method = (
            self.segmented_memory_posterior_feature_routed_source_ridge_method()
        )
        source = method._source_analytic_ridge
        first = method._novel_ordinal_ridge_state
        second = method._new_ordinal_ridge_state()

        np.testing.assert_array_equal(first["weights"], source["weights"])
        np.testing.assert_array_equal(
            first["inverse_gram"], source["inverse_gram"]
        )
        np.testing.assert_array_equal(
            first["cross_covariance"], source["cross_covariance"]
        )
        np.testing.assert_array_equal(first["class_mass"], source["class_mass"])
        self.assertEqual(first["source_prior_samples"], source["samples"])
        second["weights"][0, 0] += 1.0
        self.assertFalse(np.array_equal(first["weights"], second["weights"]))
        np.testing.assert_array_equal(source["weights"], first["weights"])

    def test_source_ridge_is_exact_at_birth_then_updates_one_classifier(self) -> None:
        method = (
            self.segmented_memory_posterior_feature_routed_source_ridge_method()
        )
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-3.6, 4.4],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        method._mixture = copy.deepcopy(mixture)
        method.boundary_history = [0.4]
        images = self.score_batch(np.array([-4.0, -3.0, 3.0, 4.0]))
        source_scores = method._batch_scores(images)
        state = method._novel_ordinal_ridge_state
        source_weights = state["weights"].copy()

        prediction = method.predict(images)

        np.testing.assert_allclose(
            prediction.prob_fake.numpy(),
            method._source_probability(source_scores),
            atol=1e-6,
        )
        stats = method.adapt(images)
        self.assertTrue(stats.extra["rms_ridge_expert_updated"])
        self.assertFalse(np.array_equal(state["weights"], source_weights))
        self.assertGreater(float(state["target_effective_support"]), 0.0)
        self.assertEqual(
            method.reproduction_metadata["research_name"],
            "ASCAL-JMP-SourceRidgeInheritance",
        )
        self.assertFalse(
            method.reproduction_metadata["base_probability_in_final_prediction"]
        )

    def test_source_ridge_gmm_readout_uses_r12_and_keeps_r23_shadow_state(self) -> None:
        r23 = self.segmented_memory_posterior_feature_routed_source_ridge_method()
        r24 = (
            self.segmented_memory_posterior_feature_routed_source_ridge_gmm_readout_method()
        )
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-3.6, 4.4],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        for method in (r23, r24):
            method._mixture = copy.deepcopy(mixture)
            method.boundary_history = [0.4]
        images = self.score_feature_batch(
            np.array([-4.0, -3.0, 3.0, 4.0]),
            np.array([-0.75, -0.25, 0.25, 0.75]),
        )
        scores = r24._batch_scores(images)

        r23_prediction = r23.predict(images)
        r24_prediction = r24.predict(images)

        r24_labels = r24_prediction.pred_label.numpy().astype(np.float64)
        expected_r12 = 0.5 * (r24._source_probability(scores) + r24_labels)
        np.testing.assert_allclose(
            r24_prediction.prob_fake.numpy(),
            expected_r12,
            atol=1e-7,
        )
        self.assertTrue(r24._pending["prediction_ordinal_applied"])
        self.assertEqual(
            r24._pending["prediction_rms_ridge_expert_label_changes"],
            0,
        )
        self.assertFalse(
            np.allclose(
                r24_prediction.prob_fake.numpy(),
                r23_prediction.prob_fake.numpy(),
            )
        )

        r23_stats = r23.adapt(images)
        r24_stats = r24.adapt(images)
        self.assertTrue(r23_stats.extra["rms_ridge_expert_updated"])
        self.assertTrue(r24_stats.extra["rms_ridge_expert_updated"])
        r23_state = r23._novel_ordinal_ridge_state
        r24_state = r24._novel_ordinal_ridge_state
        for key in (
            "weights",
            "inverse_gram",
            "cross_covariance",
            "class_mass",
            "target_class_mass",
            "route_feature_sum",
        ):
            np.testing.assert_array_equal(r24_state[key], r23_state[key])
        for key in (
            "updates",
            "candidate_samples",
            "effective_support",
            "target_effective_support",
            "route_feature_mass",
        ):
            self.assertEqual(r24_state[key], r23_state[key])
        metadata = r24.reproduction_metadata
        self.assertEqual(metadata["research_version"], "R24")
        self.assertTrue(metadata["gmm_in_final_prediction"])
        self.assertFalse(metadata["expert_ridge_in_final_prediction"])

    def test_routed_residual_static_path_is_exact_source(self) -> None:
        method = self.segmented_memory_posterior_routed_residual_method(
            adaptation_mode="static"
        )
        scores = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        cues = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        images = self.score_feature_batch(scores, cues)

        prediction = method.predict(images)
        np.testing.assert_allclose(
            prediction.prob_fake.numpy(),
            method._source_probability(scores),
            atol=1e-6,
        )
        self.assertEqual(
            method._pending["prediction_residual_routing_candidate_count"], 0
        )
        self.assertFalse(method._pending["prediction_routed_residual_ready"])
        stats = method.adapt(images)
        self.assertEqual(stats.extra["routed_residual_updates"], 0)
        self.assertEqual(method.trainable_parameters, 0)

    def test_routed_residual_uses_memory_features_without_routing_r01_boundary(
        self,
    ) -> None:
        rng = np.random.default_rng(63)
        method = self.segmented_memory_posterior_routed_residual_method()
        returning = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        active = {
            "weights": [0.5, 0.5],
            "mus": [20.0, 25.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        returning_state = method._new_routed_residual_state()
        returning_state["real_sum"][2] = -1.0
        returning_state["real_support"] = 1.0
        returning_state["fake_component_sums"][0, 2] = 1.0
        returning_state["fake_component_supports"][0] = 1.0
        active_state = method._new_routed_residual_state()
        method._mixture = active
        method.boundary_history = [22.5]
        method.active_memory_index = 1
        method.segment_memories = [
            {
                "mixture": returning,
                "boundary": float(method._memory_boundary(returning)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 0,
                method._RESIDUAL_MEMORY_KEY: returning_state,
            },
            {
                "mixture": active,
                "boundary": float(method._memory_boundary(active)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 1,
                method._RESIDUAL_MEMORY_KEY: active_state,
            },
        ]
        real_scores = rng.normal(-8.0, 0.2, 48)
        fake_scores = rng.normal(-3.0, 0.2, 48)
        scores = np.concatenate([real_scores, fake_scores])
        cues = np.concatenate([-np.ones(48), np.ones(48)])
        order = rng.permutation(len(scores))
        images = self.score_feature_batch(scores[order], cues[order])

        method.predict(images)
        self.assertEqual(
            method._pending["prediction_residual_routing_expert"],
            "episodic_memory",
        )
        self.assertEqual(
            method._pending["prediction_residual_routing_memory_index"], 0
        )
        self.assertTrue(
            method._pending["prediction_residual_routing_admission_accepted"]
        )
        self.assertAlmostEqual(method._pending["prediction_boundary"], 22.5)
        self.assertNotAlmostEqual(
            method._pending["prediction_boundary"],
            float(method._memory_boundary(returning)),
        )
        self.assertGreater(
            method._pending["prediction_routed_residual_max_abs"], 0.0
        )

        stats = method.adapt(images)
        self.assertEqual(method.active_memory_index, 1)
        self.assertEqual(returning_state["updates"], 1)
        self.assertEqual(active_state["updates"], 0)
        self.assertEqual(stats.extra["routed_residual_memory_selections"], 1)
        self.assertEqual(stats.extra["routed_residual_admission_accepts"], 1)
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-RoutedResidual")
        self.assertEqual(metadata["research_version"], "R13")
        self.assertFalse(metadata["score_boundary_routing"])
        self.assertFalse(metadata["prediction_mutates_experts"])

    def test_routed_residual_is_causal_and_never_rewrites_source_scores(self) -> None:
        method = self.segmented_memory_posterior_routed_residual_method()
        method._mixture = {
            "weights": [0.5, 0.5],
            "mus": [-4.0, 4.0],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        method.boundary_history = [0.0]
        scores = np.array([-4.0, -3.5, 3.5, 4.0])
        cues = np.array([-1.0, -1.0, 1.0, 1.0])
        images = self.score_feature_batch(scores, cues)

        first = method.predict(images)
        np.testing.assert_allclose(
            first.prob_fake.numpy(),
            method._source_probability(scores),
            atol=1e-6,
        )
        self.assertFalse(method._pending["prediction_routed_residual_ready"])
        first_stats = method.adapt(images)
        self.assertTrue(first_stats.extra["routed_residual_updated"])

        second = method.predict(images)
        self.assertTrue(method._pending["prediction_routed_residual_ready"])
        self.assertGreater(
            method._pending["prediction_routed_residual_max_abs"], 0.0
        )
        self.assertFalse(
            np.allclose(second.prob_fake.numpy(), first.prob_fake.numpy())
        )
        method.adapt(images)
        np.testing.assert_allclose(method.score_history, np.tile(scores, 2))

    def test_routed_residual_keeps_the_r01_score_state_trajectory(self) -> None:
        rng = np.random.default_rng(65)
        baseline = self.segmented_memory_posterior_projection_method()
        method = self.segmented_memory_posterior_routed_residual_method()
        batches = [
            np.concatenate(
                [rng.normal(-8.0, 0.2, 8), rng.normal(-3.0, 0.2, 8)]
            ),
            np.concatenate(
                [rng.normal(-8.0, 0.2, 8), rng.normal(-3.0, 0.2, 8)]
            ),
            np.concatenate(
                [rng.normal(4.0, 0.2, 8), rng.normal(9.0, 0.2, 8)]
            ),
            np.concatenate(
                [rng.normal(4.0, 0.2, 8), rng.normal(9.0, 0.2, 8)]
            ),
        ]
        for scores in batches:
            cues = np.sign(scores - np.median(scores))
            baseline_images = self.score_feature_batch(scores, cues)
            method_images = self.score_feature_batch(scores, cues)
            baseline.predict(baseline_images)
            method.predict(method_images)
            baseline.adapt(baseline_images)
            method.adapt(method_images)

            np.testing.assert_allclose(method.score_history, baseline.score_history)
            np.testing.assert_allclose(
                method.boundary_history,
                baseline.boundary_history,
            )
            self.assertEqual(
                method.active_memory_index,
                baseline.active_memory_index,
            )
            self.assertEqual(
                len(method.segment_memories),
                len(baseline.segment_memories),
            )
            self.assertEqual(method.segment_changes, baseline.segment_changes)
            self.assertEqual(
                method._mixture["components"],
                baseline._mixture["components"],
            )
            np.testing.assert_allclose(
                method._mixture["weights"], baseline._mixture["weights"]
            )
            np.testing.assert_allclose(
                method._mixture["mus"],
                baseline._mixture["mus"],
            )
            np.testing.assert_allclose(
                method._mixture["sigmas"], baseline._mixture["sigmas"]
            )

    def test_routed_ridge_matches_the_exact_weighted_ridge_solution(self) -> None:
        from src.methods.ascal_gmm import joint_density_fake_posterior

        method = self.segmented_memory_posterior_routed_ridge_residual_method()
        mixture = {
            "weights": [0.5, 0.5],
            "mus": [-3.0, 3.0],
            "sigmas": [0.8, 0.8],
            "components": 2,
            "bic": 0.0,
        }
        scores = np.array([-3.5, -2.5, 2.5, 3.5])
        features = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        features[-1] /= np.linalg.norm(features[-1])
        posterior = joint_density_fake_posterior(scores, mixture)
        targets = 2.0 * posterior - 1.0
        reliability = np.abs(targets)
        expected_precision = np.eye(3) + features.T @ (
            reliability[:, None] * features
        )
        expected_rhs = features.T @ (reliability * targets)
        expected_weights = np.linalg.solve(expected_precision, expected_rhs)

        state = method._new_routed_residual_state()
        for indices in (slice(0, 2), slice(2, 4)):
            self.assertTrue(
                method._update_routed_residual_state(
                    state,
                    mixture,
                    scores[indices],
                    features[indices],
                )
            )
        np.testing.assert_allclose(state["weights"], expected_weights, atol=1e-12)
        np.testing.assert_allclose(
            state["inverse_gram"],
            np.linalg.inv(expected_precision),
            atol=1e-12,
        )
        self.assertEqual(state["updates"], 2)
        self.assertEqual(state["candidate_samples"], len(scores))
        self.assertEqual(method.routed_ridge_solve_failures, 0)

    def test_routed_ridge_static_path_is_exact_source(self) -> None:
        method = self.segmented_memory_posterior_routed_ridge_residual_method(
            adaptation_mode="static"
        )
        scores = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
        cues = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        images = self.score_feature_batch(scores, cues)

        prediction = method.predict(images)
        np.testing.assert_allclose(
            prediction.prob_fake.numpy(),
            method._source_probability(scores),
            atol=1e-6,
        )
        self.assertEqual(
            method._pending["prediction_residual_routing_candidate_count"], 0
        )
        self.assertFalse(method._pending["prediction_routed_residual_ready"])
        stats = method.adapt(images)
        self.assertEqual(stats.extra["routed_residual_updates"], 0)
        self.assertEqual(stats.extra["routed_ridge_head_count"], 0)
        self.assertEqual(method.trainable_parameters, 0)

    def test_routed_ridge_is_causal_and_zero_initialized(self) -> None:
        method = self.segmented_memory_posterior_routed_ridge_residual_method()
        method._mixture = {
            "weights": [0.5, 0.5],
            "mus": [-4.0, 4.0],
            "sigmas": [0.5, 0.5],
            "components": 2,
            "bic": 0.0,
        }
        method.boundary_history = [0.0]
        scores = np.array([-4.0, -3.5, 3.5, 4.0])
        cues = np.array([-1.0, -1.0, 1.0, 1.0])
        images = self.score_feature_batch(scores, cues)

        first = method.predict(images)
        np.testing.assert_allclose(
            first.prob_fake.numpy(),
            method._source_probability(scores),
            atol=1e-6,
        )
        self.assertFalse(method._pending["prediction_routed_residual_ready"])
        first_stats = method.adapt(images)
        self.assertTrue(first_stats.extra["routed_residual_updated"])
        self.assertEqual(first_stats.extra["routed_ridge_head_count"], 1)
        self.assertNotIn("routed_residual_readout_bound", first_stats.extra)
        self.assertNotIn(
            "routed_residual_fake_prototype_count", first_stats.extra
        )
        self.assertEqual(method.trainable_parameters, 3)

        second = method.predict(images)
        self.assertTrue(method._pending["prediction_routed_residual_ready"])
        self.assertGreater(
            method._pending["prediction_routed_residual_max_abs"], 0.0
        )
        self.assertFalse(
            np.allclose(second.prob_fake.numpy(), first.prob_fake.numpy())
        )
        method.adapt(images)
        np.testing.assert_allclose(method.score_history, np.tile(scores, 2))
        metadata = method.reproduction_metadata
        self.assertEqual(metadata["research_name"], "ASCAL-JMP-RoutedRidge")
        self.assertEqual(metadata["research_version"], "R14")
        self.assertEqual(metadata["optimizer"], "none_closed_form_recursive_ridge")
        self.assertEqual(metadata["new_target_hyperparameters"], 0)

    def test_routed_ridge_updates_only_the_prediction_selected_expert(self) -> None:
        rng = np.random.default_rng(67)
        method = self.segmented_memory_posterior_routed_ridge_residual_method()
        returning = {
            "weights": [0.5, 0.5],
            "mus": [-8.0, -3.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        active = {
            "weights": [0.5, 0.5],
            "mus": [20.0, 25.0],
            "sigmas": [0.25, 0.25],
            "components": 2,
            "bic": 0.0,
        }
        returning_state = method._new_routed_residual_state()
        returning_state["weights"][2] = 0.5
        returning_state["updates"] = 1
        returning_state["candidate_samples"] = 16
        returning_state["effective_support"] = 8.0
        active_state = method._new_routed_residual_state()
        method._mixture = active
        method.boundary_history = [22.5]
        method.active_memory_index = 1
        method.segment_memories = [
            {
                "mixture": returning,
                "boundary": float(method._memory_boundary(returning)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 0,
                method._RESIDUAL_MEMORY_KEY: returning_state,
            },
            {
                "mixture": active,
                "boundary": float(method._memory_boundary(active)),
                "latest_samples": 96,
                "total_samples": 96,
                "visits": 1,
                "recalls": 1,
                method._RESIDUAL_MEMORY_KEY: active_state,
            },
        ]
        scores = np.concatenate(
            [rng.normal(-8.0, 0.2, 48), rng.normal(-3.0, 0.2, 48)]
        )
        cues = np.concatenate([-np.ones(48), np.ones(48)])
        order = rng.permutation(len(scores))
        images = self.score_feature_batch(scores[order], cues[order])

        method.predict(images)
        self.assertEqual(
            method._pending["prediction_residual_routing_expert"],
            "episodic_memory",
        )
        self.assertEqual(
            method._pending["prediction_residual_routing_memory_index"], 0
        )
        self.assertGreater(
            method._pending["prediction_routed_residual_max_abs"], 0.0
        )
        stats = method.adapt(images)

        self.assertEqual(returning_state["updates"], 2)
        self.assertEqual(active_state["updates"], 0)
        self.assertEqual(stats.extra["routed_residual_memory_selections"], 1)
        self.assertEqual(stats.extra["routed_residual_admission_accepts"], 1)

    def test_routed_ridge_keeps_the_exact_r13_routing_and_score_trajectory(
        self,
    ) -> None:
        rng = np.random.default_rng(69)
        baseline = self.segmented_memory_posterior_routed_residual_method()
        method = self.segmented_memory_posterior_routed_ridge_residual_method()
        for center in (-6.0, -6.0, 6.0, 6.0):
            scores = np.concatenate(
                [
                    rng.normal(center - 2.0, 0.2, 8),
                    rng.normal(center + 2.0, 0.2, 8),
                ]
            )
            cues = np.sign(scores - np.median(scores))
            baseline_images = self.score_feature_batch(scores, cues)
            method_images = self.score_feature_batch(scores, cues)
            baseline.predict(baseline_images)
            method.predict(method_images)
            baseline_stats = baseline.adapt(baseline_images)
            method_stats = method.adapt(method_images)

            np.testing.assert_allclose(method.score_history, baseline.score_history)
            np.testing.assert_allclose(
                method.boundary_history, baseline.boundary_history
            )
            self.assertEqual(method.segment_changes, baseline.segment_changes)
            self.assertEqual(
                method_stats.extra["last_routed_residual_expert"],
                baseline_stats.extra["last_routed_residual_expert"],
            )
            self.assertEqual(
                method_stats.extra["last_routed_residual_memory_index"],
                baseline_stats.extra["last_routed_residual_memory_index"],
            )

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

    def test_method_factory_maps_mdl_route_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorMDLRoute,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_mdl_route_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(method, ASCALGMMSegmentedMemoryPosteriorMDLRoute)
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_live_route_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorLiveRoute,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_live_route_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(method, ASCALGMMSegmentedMemoryPosteriorLiveRoute)
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_ordinal_route_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorOrdinalRoute,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_ordinal_route_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(method, ASCALGMMSegmentedMemoryPosteriorOrdinalRoute)
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_ordinal_ridge_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorOrdinalRidge,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_ordinal_ridge_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorOrdinalRidge,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_joint_ridge_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorJointRidge,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_joint_ridge_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorJointRidge,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_pairwise_ridge_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorPairwiseRidge,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_pairwise_ridge_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorPairwiseRidge,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_analytic_expert_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorAnalyticExpert,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_analytic_expert_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorAnalyticExpert,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_rms_ridge_expert_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorRMSRidgeExpert,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_rms_ridge_expert_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorRMSRidgeExpert,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_equal_prior_ridge_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorEqualPriorRidgeExpert,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_equal_prior_ridge_expert_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorEqualPriorRidgeExpert,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_evidence_gated_ridge_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorEvidenceGatedRidgeExpert,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_evidence_gated_ridge_expert_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorEvidenceGatedRidgeExpert,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_feature_routed_trusted_ridge_static_alias(
        self,
    ) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedTrustedRidge,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_feature_routed_trusted_ridge_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedTrustedRidge,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_gaussian_replay_mlp_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedGaussianReplayMLP,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_"
            "feature_routed_gaussian_replay_mlp_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_expanded_gaussian_replay_static_alias(
        self,
    ) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "expanded_gaussian_replay_mlp_static",
            self.detector(),
            "cpu",
            {
                "score_anchors": self.anchors(),
                "feature_replay_samples_per_update": 8,
            },
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_shared_gaussian_replay_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSharedGaussianReplayMLP,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "shared_gaussian_replay_mlp_static",
            self.detector(),
            "cpu",
            {
                "score_anchors": self.anchors(),
                "feature_replay_samples_per_update": 8,
            },
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSharedGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_linear_gaussian_replay_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedLinearGaussianReplay,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "linear_gaussian_replay_static",
            self.detector(),
            "cpu",
            {
                "score_anchors": self.anchors(),
                "feature_replay_samples_per_update": 8,
            },
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedLinearGaussianReplay,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_uniform_gaussian_replay_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedUniformGaussianReplayMLP,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "uniform_gaussian_replay_mlp_static",
            self.detector(),
            "cpu",
            {
                "score_anchors": self.anchors(),
                "feature_replay_samples_per_update": 8,
            },
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedUniformGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_active_gaussian_replay_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorActiveGaussianReplayMLP,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_"
            "active_gaussian_replay_mlp_static",
            self.detector(),
            "cpu",
            {
                "score_anchors": self.anchors(),
                "feature_replay_samples_per_update": 8,
            },
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorActiveGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_no_historical_recall_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorNoHistoricalRecallGaussianReplayMLP,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_no_historical_recall_"
            "gaussian_replay_mlp_static",
            self.detector(),
            "cpu",
            {
                "score_anchors": self.anchors(),
                "feature_replay_samples_per_update": 8,
            },
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorNoHistoricalRecallGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_current_segment_core_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import ASCALGMMCurrentSegmentGaussianReplayMLP

        method = build_method(
            "ascal_gmm_current_segment_gaussian_replay_mlp_static",
            self.detector(),
            "cpu",
            {
                "score_anchors": self.anchors(),
                "feature_replay_samples_per_update": 8,
            },
        )
        self.assertIsInstance(method, ASCALGMMCurrentSegmentGaussianReplayMLP)
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_global_stream_core_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import ASCALGMMGlobalStreamGaussianReplayMLP

        method = build_method(
            "ascal_gmm_global_stream_gaussian_replay_mlp_static",
            self.detector(),
            "cpu",
            {
                "score_anchors": self.anchors(),
                "feature_replay_samples_per_update": 8,
            },
        )
        self.assertIsInstance(method, ASCALGMMGlobalStreamGaussianReplayMLP)
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_clip_expert_memory_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorCLIPRoutedGaussianReplayMLP,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_clip_routed_"
            "gaussian_replay_mlp_static",
            self.detector(),
            "cpu",
            {
                "score_anchors": self.anchors(),
                "feature_replay_samples_per_update": 8,
            },
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorCLIPRoutedGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_current_batch_replay_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedCurrentBatchReplayMLP,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "current_batch_replay_mlp_static",
            self.detector(),
            "cpu",
            {
                "score_anchors": self.anchors(),
                "feature_replay_samples_per_update": 8,
            },
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedCurrentBatchReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_prior_gaussian_replay_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedPriorGaussianReplayMLP,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "prior_gaussian_replay_mlp_static",
            self.detector(),
            "cpu",
            {
                "score_anchors": self.anchors(),
                "feature_replay_samples_per_update": 8,
            },
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedPriorGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_source_replay_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceReplayMLP,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "source_replay_mlp_static",
            self.detector(),
            "cpu",
            {
                "score_anchors": self.anchors(),
                "feature_replay_samples_per_update": 8,
            },
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_feature_routed_source_ridge_static_alias(
        self,
    ) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidge,
        )

        model = self.source_ridge_detector()
        method = build_method(
            "ascal_gmm_segmented_memory_posterior_feature_routed_source_ridge_static",
            model,
            "cpu",
            {
                "score_anchors": self.anchors(),
                "source_analytic_ridge": self.source_ridge_state(model),
            },
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidge,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_source_ridge_gmm_readout(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidgeGMMReadout,
        )

        model = self.source_ridge_detector()
        method = build_method(
            "ascal_gmm_segmented_memory_posterior_"
            "feature_routed_source_ridge_gmm_readout",
            model,
            "cpu",
            {
                "score_anchors": self.anchors(),
                "source_analytic_ridge": self.source_ridge_state(model),
            },
        )
        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidgeGMMReadout,
        )
        self.assertEqual(method.adaptation_mode, "full")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_routed_residual_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorRoutedResidual,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_routed_residual_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorRoutedResidual
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(method.trainable_parameters, 0)

    def test_method_factory_maps_routed_ridge_static_alias(self) -> None:
        from src.methods import build_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorRoutedRidgeResidual,
        )

        method = build_method(
            "ascal_gmm_segmented_memory_posterior_routed_ridge_residual_static",
            self.detector(),
            "cpu",
            {"score_anchors": self.anchors()},
        )
        self.assertIsInstance(
            method, ASCALGMMSegmentedMemoryPosteriorRoutedRidgeResidual
        )
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

    def test_cli_builds_mdl_route_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorMDLRoute,
        )

        method_name = "ascal_gmm_segmented_memory_posterior_mdl_route_static"
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

        self.assertIsInstance(method, ASCALGMMSegmentedMemoryPosteriorMDLRoute)
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_live_route_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorLiveRoute,
        )

        method_name = "ascal_gmm_segmented_memory_posterior_live_route_static"
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

        self.assertIsInstance(method, ASCALGMMSegmentedMemoryPosteriorLiveRoute)
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_ordinal_route_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorOrdinalRoute,
        )

        method_name = "ascal_gmm_segmented_memory_posterior_ordinal_route_static"
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
            method, ASCALGMMSegmentedMemoryPosteriorOrdinalRoute
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_ordinal_ridge_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorOrdinalRidge,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_ordinal_ridge_static"
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
            method,
            ASCALGMMSegmentedMemoryPosteriorOrdinalRidge,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_joint_ridge_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorJointRidge,
        )

        method_name = "ascal_gmm_segmented_memory_posterior_joint_ridge_static"
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
            method,
            ASCALGMMSegmentedMemoryPosteriorJointRidge,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_pairwise_ridge_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorPairwiseRidge,
        )

        method_name = "ascal_gmm_segmented_memory_posterior_pairwise_ridge_static"
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
            method,
            ASCALGMMSegmentedMemoryPosteriorPairwiseRidge,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_analytic_expert_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorAnalyticExpert,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_analytic_expert_static"
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
            method,
            ASCALGMMSegmentedMemoryPosteriorAnalyticExpert,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_rms_ridge_expert_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorRMSRidgeExpert,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_rms_ridge_expert_static"
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
            method,
            ASCALGMMSegmentedMemoryPosteriorRMSRidgeExpert,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_equal_prior_ridge_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorEqualPriorRidgeExpert,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_"
            "equal_prior_ridge_expert_static"
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
            method,
            ASCALGMMSegmentedMemoryPosteriorEqualPriorRidgeExpert,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_evidence_gated_ridge_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorEvidenceGatedRidgeExpert,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_"
            "evidence_gated_ridge_expert_static"
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
            method,
            ASCALGMMSegmentedMemoryPosteriorEvidenceGatedRidgeExpert,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_feature_routed_trusted_ridge_with_lora_profile(
        self,
    ) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedTrustedRidge,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_"
            "feature_routed_trusted_ridge_static"
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
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedTrustedRidge,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_expanded_gaussian_replay_with_lora_profile(
        self,
    ) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "expanded_gaussian_replay_mlp_static"
        )
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                method_name: {
                    "adaptation_mode": "static",
                    "feature_replay_samples_per_update": 256,
                }
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_shared_gaussian_replay_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSharedGaussianReplayMLP,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "shared_gaussian_replay_mlp_static"
        )
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                method_name: {
                    "adaptation_mode": "static",
                    "feature_replay_samples_per_update": 256,
                }
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSharedGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_linear_gaussian_replay_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedLinearGaussianReplay,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "linear_gaussian_replay_static"
        )
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                method_name: {
                    "adaptation_mode": "static",
                    "feature_replay_samples_per_update": 256,
                }
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedLinearGaussianReplay,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_uniform_gaussian_replay_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedUniformGaussianReplayMLP,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "uniform_gaussian_replay_mlp_static"
        )
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                method_name: {
                    "adaptation_mode": "static",
                    "feature_replay_samples_per_update": 256,
                }
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedUniformGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_active_gaussian_replay_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorActiveGaussianReplayMLP,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_"
            "active_gaussian_replay_mlp_static"
        )
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                method_name: {
                    "adaptation_mode": "static",
                    "feature_replay_samples_per_update": 256,
                }
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorActiveGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_no_historical_recall_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorNoHistoricalRecallGaussianReplayMLP,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_no_historical_recall_"
            "gaussian_replay_mlp_static"
        )
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                method_name: {
                    "adaptation_mode": "static",
                    "feature_replay_samples_per_update": 256,
                }
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorNoHistoricalRecallGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_current_segment_core_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import ASCALGMMCurrentSegmentGaussianReplayMLP

        method_name = "ascal_gmm_current_segment_gaussian_replay_mlp_static"
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                method_name: {
                    "adaptation_mode": "static",
                    "feature_replay_samples_per_update": 256,
                }
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(method, ASCALGMMCurrentSegmentGaussianReplayMLP)
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_clip_expert_memory_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorCLIPRoutedGaussianReplayMLP,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_clip_routed_"
            "gaussian_replay_mlp_static"
        )
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                method_name: {
                    "adaptation_mode": "static",
                    "feature_replay_samples_per_update": 256,
                }
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorCLIPRoutedGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_global_stream_core_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import ASCALGMMGlobalStreamGaussianReplayMLP

        method_name = "ascal_gmm_global_stream_gaussian_replay_mlp_static"
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                method_name: {
                    "adaptation_mode": "static",
                    "feature_replay_samples_per_update": 256,
                }
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(method, ASCALGMMGlobalStreamGaussianReplayMLP)
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_current_batch_replay_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedCurrentBatchReplayMLP,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "current_batch_replay_mlp_static"
        )
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                method_name: {
                    "adaptation_mode": "static",
                    "feature_replay_samples_per_update": 256,
                }
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedCurrentBatchReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_prior_gaussian_replay_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedPriorGaussianReplayMLP,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "prior_gaussian_replay_mlp_static"
        )
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                method_name: {
                    "adaptation_mode": "static",
                    "feature_replay_samples_per_update": 256,
                }
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedPriorGaussianReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_source_replay_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceReplayMLP,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_feature_routed_"
            "source_replay_mlp_static"
        )
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal.pt",
                "lora_rank": 4,
            },
            "method_configs": {
                method_name: {
                    "adaptation_mode": "static",
                    "feature_replay_samples_per_update": 256,
                }
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
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceReplayMLP,
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_feature_routed_source_ridge_with_complete_prior(
        self,
    ) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidge,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_"
            "feature_routed_source_ridge_static"
        )
        model = self.source_ridge_detector()
        source_state = self.source_ridge_state(model)
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal-ridge.pt",
                "classifier_feature_normalization": "l2",
                "lora_rank": 4,
            },
            "method_configs": {method_name: {"adaptation_mode": "static"}},
        }
        checkpoint_metadata = {
            "lora_rank": 4,
            "score_anchors": self.anchors(),
            "source_analytic_ridge": source_state,
        }
        with patch(
            "src.cli.common.build_clip_lora_detector",
            return_value=(model, {"family": "clip_lora_source_detector"}),
        ), patch(
            "src.cli.common.load_checkpoint",
            return_value=checkpoint_metadata,
        ), patch(
            "src.cli.common.checkpoint_sha256",
            return_value="0" * 64,
        ):
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidge,
        )
        self.assertEqual(method.adaptation_mode, "static")
        self.assertEqual(
            method.reproduction_metadata["source_ridge_statistics_sha256"],
            source_state["statistics_sha256"],
        )

    def test_cli_builds_source_ridge_gmm_readout_with_complete_prior(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidgeGMMReadout,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_"
            "feature_routed_source_ridge_gmm_readout"
        )
        model = self.source_ridge_detector()
        source_state = self.source_ridge_state(model)
        config = {
            "model": {"family": "clip_vlm_main"},
            "method_defaults": {
                "checkpoint": "/tmp/clip.pt",
                "source_checkpoint": "/tmp/ascal-ridge.pt",
                "classifier_feature_normalization": "l2",
                "lora_rank": 4,
            },
            "method_configs": {method_name: {"adaptation_mode": "full"}},
        }
        checkpoint_metadata = {
            "lora_rank": 4,
            "score_anchors": self.anchors(),
            "source_analytic_ridge": source_state,
        }
        with patch(
            "src.cli.common.build_clip_lora_detector",
            return_value=(model, {"family": "clip_lora_source_detector"}),
        ), patch(
            "src.cli.common.load_checkpoint",
            return_value=checkpoint_metadata,
        ), patch(
            "src.cli.common.checkpoint_sha256",
            return_value="0" * 64,
        ):
            method, _ = build_fresh_method(config, method_name, "cpu")

        self.assertIsInstance(
            method,
            ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidgeGMMReadout,
        )
        self.assertEqual(method.adaptation_mode, "full")
        self.assertEqual(
            method.reproduction_metadata["source_ridge_statistics_sha256"],
            source_state["statistics_sha256"],
        )

    def test_cli_builds_routed_residual_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorRoutedResidual,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_routed_residual_static"
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
            method, ASCALGMMSegmentedMemoryPosteriorRoutedResidual
        )
        self.assertEqual(method.adaptation_mode, "static")

    def test_cli_builds_routed_ridge_with_lora_profile(self) -> None:
        from src.cli.common import build_fresh_method
        from src.methods.ascal_gmm import (
            ASCALGMMSegmentedMemoryPosteriorRoutedRidgeResidual,
        )

        method_name = (
            "ascal_gmm_segmented_memory_posterior_routed_ridge_residual_static"
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
            method, ASCALGMMSegmentedMemoryPosteriorRoutedRidgeResidual
        )
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
