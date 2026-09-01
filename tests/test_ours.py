from __future__ import annotations

import copy
import importlib.util
import re
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


def score_anchors() -> dict:
    return {
        "temperature": 1.0,
        "real": {"mu": -4.0, "sigma": 1.0},
        "fake": {
            "weights": [0.5, 0.5],
            "mus": [2.0, 5.0],
            "sigmas": [1.0, 1.0],
        },
    }


class OursPublicSurfaceTests(unittest.TestCase):
    def test_only_retained_ours_interfaces_are_configured(self) -> None:
        config = load_config(PROJECT_ROOT / "configs/methods/ours.yaml")
        self.assertEqual(config["methods"], ["ours_static", "ours"])
        self.assertEqual(
            set(config["method_configs"]),
            {"ours_static", "ours_no_calibrated_readout", "ours"},
        )
        self.assertEqual(
            config["method_configs"]["ours_no_calibrated_readout"]["readout_mode"],
            "base",
        )
        self.assertEqual(config["method_configs"]["ours"]["readout_mode"], "calibrated")

    def test_legacy_custom_method_files_and_entrypoints_are_gone(self) -> None:
        forbidden = (
            "configs/methods/ascal*.yaml",
            "configs/methods/pound*.yaml",
            "configs/experiments/**/*ascal*.yaml",
            "configs/experiments/**/*pound*.yaml",
            "src/methods/ascal*.py",
            "src/methods/pound*.py",
            "src/models/analytic_ridge.py",
            "src/models/poundnet.py",
        )
        for pattern in forbidden:
            with self.subTest(pattern=pattern):
                self.assertEqual(list(PROJECT_ROOT.glob(pattern)), [])

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
    def test_ours_module_exports_one_method_class_without_version_classes(self) -> None:
        import src.methods.ours as ours_module

        self.assertEqual(
            ours_module.__all__,
            [
                "Ours",
                "binary_score",
                "fit_gaussian_ml",
                "fit_gmm_bic",
                "fit_temperature",
                "validate_score_anchors",
            ],
        )
        public_classes = {
            name
            for name, value in vars(ours_module).items()
            if isinstance(value, type)
            and value.__module__ == ours_module.__name__
            and not name.startswith("_")
        }
        self.assertEqual(public_classes, {"Ours"})
        self.assertFalse(
            any(re.fullmatch(r"R\d+", name) for name in vars(ours_module))
        )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
    def test_factory_has_no_legacy_aliases(self) -> None:
        from src.methods import build_method

        for name in ("r37", "r47", "ascal", "ascal_gmm", "pound_tta"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                build_method(name, object(), "cpu", {})

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
    def test_factory_routes_both_readouts_to_the_same_class(self) -> None:
        from src.methods import build_method

        sentinel = object()
        for name, expected_mode, expected_adaptation in (
            ("ours", "calibrated", None),
            ("ours_no_calibrated_readout", "base", None),
            ("ours_static", "calibrated", "static"),
        ):
            with self.subTest(name=name), patch(
                "src.methods.Ours", return_value=sentinel
            ) as constructor:
                result = build_method(name, object(), "cpu", {})
                self.assertIs(result, sentinel)
                passed_config = constructor.call_args.args[2]
                self.assertEqual(passed_config["readout_mode"], expected_mode)
                if expected_adaptation is None:
                    self.assertNotIn("adaptation_mode", passed_config)
                else:
                    self.assertEqual(
                        passed_config["adaptation_mode"], expected_adaptation
                    )

    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
    def test_factory_rejects_cross_wired_retained_settings(self) -> None:
        from src.methods import build_method

        with self.assertRaises(ValueError):
            build_method("ours", object(), "cpu", {"readout_mode": "base"})
        with self.assertRaises(ValueError):
            build_method(
                "ours_no_calibrated_readout",
                object(),
                "cpu",
                {"readout_mode": "calibrated"},
            )
        with self.assertRaises(ValueError):
            build_method(
                "ours_static", object(), "cpu", {"adaptation_mode": "full"}
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required to import the methods package")
class OursAnchorTests(unittest.TestCase):
    def test_temperature_and_gaussian_anchor_helpers_are_finite(self) -> None:
        from src.methods.ours import fit_gaussian_ml, fit_temperature

        scores = np.array([-4.0, -3.0, -2.0, 2.0, 3.0, 4.0])
        labels = np.array([0, 0, 0, 1, 1, 1])
        temperature = fit_temperature(scores, labels)
        mu, sigma = fit_gaussian_ml(scores[:3])
        self.assertGreater(temperature, 0.0)
        self.assertTrue(np.isfinite(temperature))
        self.assertAlmostEqual(mu, -3.0)
        self.assertGreater(sigma, 0.0)

    def test_score_anchor_validation_normalizes_the_retained_schema(self) -> None:
        from src.methods.ours import validate_score_anchors

        anchors = validate_score_anchors(score_anchors())
        self.assertEqual(anchors["temperature"], 1.0)
        self.assertEqual(anchors["fake"]["weights"], [0.5, 0.5])
        with self.assertRaises(ValueError):
            validate_score_anchors({**score_anchors(), "temperature": 0.0})

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
    def test_bic_selects_two_well_separated_score_modes(self) -> None:
        from src.methods.ours import fit_gmm_bic

        rng = np.random.default_rng(4)
        values = np.concatenate(
            (rng.normal(-3.0, 0.1, 128), rng.normal(3.0, 0.1, 128))
        )
        mixture = fit_gmm_bic(values, max_components=3, seed=0)
        self.assertEqual(mixture["components"], 2)
        self.assertLess(mixture["mus"][0], 0.0)
        self.assertGreater(mixture["mus"][1], 0.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class OursReadoutTests(unittest.TestCase):
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

    def method(self, readout_mode: str):
        from src.methods.ours import Ours

        return Ours(
            self.detector(),
            "cpu",
            {
                "adaptation_mode": "full",
                "readout_mode": readout_mode,
                "score_anchors": score_anchors(),
                "feature_replay_hidden_dim": 4,
                "feature_replay_learning_rate": 0.01,
                "feature_replay_samples_per_update": 8,
                "feature_replay_seed": 1,
                **(
                    {"feature_residual_scale": 0.75}
                    if readout_mode == "calibrated"
                    else {}
                ),
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
        images[:, 2] = self.torch.from_numpy(
            np.asarray(cues, dtype=np.float32)
        ).view(-1, 1, 1)
        return images

    def test_both_readouts_are_the_same_exact_method_type(self) -> None:
        from src.methods import build_method
        from src.methods.ours import Ours

        final = build_method(
            "ours",
            self.detector(),
            "cpu",
            {"score_anchors": score_anchors(), "feature_replay_samples_per_update": 8},
        )
        ablation = build_method(
            "ours_no_calibrated_readout",
            self.detector(),
            "cpu",
            {"score_anchors": score_anchors(), "feature_replay_samples_per_update": 8},
        )
        self.assertIs(type(final), Ours)
        self.assertIs(type(ablation), Ours)
        self.assertEqual(final.readout_mode, "calibrated")
        self.assertEqual(ablation.readout_mode, "base")

    def test_final_readout_fixes_scale_and_refits_only_the_intercept(self) -> None:
        import torch
        import torch.nn.functional as functional

        method = self.method("calibrated")
        state = method._new_ordinal_ridge_state()
        head = method._ensure_gaussian_replay_head(state)
        features = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ]
        )
        source_margin = torch.tensor([-1.0, 0.5, -0.25, 1.25])
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        with torch.no_grad():
            head[2].weight.fill_(0.25)
            head[2].bias.fill_(0.4)
        parameters_before = [
            parameter.detach().clone() for parameter in head.parameters()
        ]

        method._after_gaussian_replay_head_update(
            state, head, features, source_margin, labels
        )

        for before, after in zip(parameters_before, head.parameters(), strict=True):
            self.assertTrue(torch.equal(before, after))
        calibrated_bias = state["calibrated_prediction_bias"]
        with torch.no_grad():
            hidden = head[1](head[0](features))
            feature_residual = functional.linear(
                hidden, head[2].weight, bias=None
            ).reshape(-1)
            balance = torch.mean(
                torch.sigmoid(source_margin + 0.75 * feature_residual + calibrated_bias)
                - labels
            )
        self.assertLess(abs(float(balance.item())), 1e-6)
        self.assertLess(abs(method.intercept_refit_last_balance_error), 1e-6)

        feature_values = features.numpy().astype(np.float64)
        full_residual = method._gaussian_replay_residual(state, feature_values)
        learned_bias = float(head[2].bias.detach().item())
        expected = calibrated_bias + 0.75 * (full_residual - learned_bias)
        actual = method._gaussian_replay_prediction_residual(
            state, feature_values, np.zeros(4), {}
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)
        self.assertEqual(method.reproduction_metadata["internal_version"], "R47")

    def test_base_readout_has_no_shrinkage_or_intercept_refit(self) -> None:
        import torch

        method = self.method("base")
        state = method._new_ordinal_ridge_state()
        head = method._ensure_gaussian_replay_head(state)
        features = torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32
        )
        source_margin = torch.tensor([-1.0, 1.0])
        labels = torch.tensor([0.0, 1.0])
        method._after_gaussian_replay_head_update(
            state, head, features, source_margin, labels
        )
        self.assertNotIn("calibrated_prediction_bias", state)
        self.assertEqual(method.intercept_refit_updates, 0)
        feature_values = features.numpy().astype(np.float64)
        expected = method._gaussian_replay_residual(state, feature_values)
        actual = method._gaussian_replay_prediction_residual(
            state, feature_values, np.zeros(2), {}
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(method.reproduction_metadata["internal_version"], "R37")

    def test_calibrated_scale_cannot_be_retuned(self) -> None:
        from src.methods.ours import Ours

        with self.assertRaises(ValueError):
            Ours(
                self.detector(),
                "cpu",
                {
                    "score_anchors": score_anchors(),
                    "feature_replay_samples_per_update": 8,
                    "readout_mode": "calibrated",
                    "feature_residual_scale": 0.5,
                },
            )

    @unittest.skipUnless(SKLEARN_AVAILABLE, "scikit-learn is required")
    def test_predict_then_adapt_updates_one_gaussian_replay_expert(self) -> None:
        method = self.method("base")
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
        images = self.score_feature_batch(scores, np.array([-1.0, -0.5, 0.5, 1.0]))
        state = method._novel_ordinal_ridge_state
        source_probability = method._source_probability(scores)

        first = method.predict(images)
        np.testing.assert_allclose(first.prob_fake.numpy(), source_probability, atol=1e-7)
        np.testing.assert_array_equal(state["class_samples"], np.zeros(2))
        self.assertIsNone(state["mlp_head"])

        stats = method.adapt(images)
        self.assertTrue(stats.extra["gaussian_replay_updated"])
        np.testing.assert_array_equal(state["class_samples"], np.array([2, 2]))
        self.assertEqual(state["head_updates"], 1)
        self.assertEqual(state["generated_samples"], 8)

        method._mixture = copy.deepcopy(mixture)
        method.boundary_history = [-5.5]
        second = method.predict(images)
        self.assertTrue(method._pending["prediction_gaussian_replay_applied"])
        self.assertFalse(
            np.allclose(second.prob_fake.numpy(), source_probability, atol=1e-7)
        )


if __name__ == "__main__":
    unittest.main()
