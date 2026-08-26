from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class SourceAnalyticRidgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        import torch.nn as nn

        cls.torch = torch
        cls.nn = nn

    def model(self):
        torch = self.torch
        nn = self.nn

        class TinyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.feature_dim = 2
                self.classifier_feature_normalization = "l2"
                self.classifier = nn.Linear(2, 2)

            def forward_features(self, values):
                return values.float()

            def forward_classifier_features(self, features):
                return torch.nn.functional.normalize(features, dim=1)

            def forward(self, values):
                features = self.forward_classifier_features(
                    self.forward_features(values)
                )
                return self.classifier(features)

        return TinyModel()

    def test_fit_retains_complete_statistics_and_installs_exact_head(self) -> None:
        import numpy as np

        from src.models.analytic_ridge import (
            fit_source_analytic_ridge,
            install_source_analytic_ridge,
            source_analytic_ridge_arrays,
        )

        torch = self.torch
        model = self.model()
        features = torch.tensor(
            [[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [0.0, 2.0]]
        )
        labels = torch.tensor([0, 0, 1, 1])
        loader = [(features, labels, ["a", "b", "c", "d"])]

        state = fit_source_analytic_ridge(
            model, loader, "cpu", regularization=1.0
        )
        arrays = source_analytic_ridge_arrays(state, expected_feature_dim=2)
        install_source_analytic_ridge(model, state)

        self.assertEqual(arrays["samples"], 4)
        np.testing.assert_array_equal(arrays["class_mass"], [2.0, 2.0])
        np.testing.assert_allclose(
            arrays["gram"] @ arrays["weights"],
            arrays["cross_covariance"],
        )
        installed = np.concatenate(
            (
                model.classifier.weight.detach().numpy().T,
                model.classifier.bias.detach().numpy()[None, :],
            ),
            axis=0,
        )
        np.testing.assert_allclose(installed, arrays["weights"], atol=1e-7)
        predictions = model(features).argmax(dim=1)
        self.assertTrue(torch.equal(predictions, labels))

    def test_statistics_hash_detects_modified_state(self) -> None:
        from src.models.analytic_ridge import (
            fit_source_analytic_ridge,
            source_analytic_ridge_arrays,
        )

        torch = self.torch
        model = self.model()
        loader = [
            (
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([0, 1]),
                ["a", "b"],
            )
        ]
        state = fit_source_analytic_ridge(model, loader, "cpu")
        state["weights"] = state["weights"].clone()
        state["weights"][0, 0] += 0.1
        with self.assertRaises(ValueError):
            source_analytic_ridge_arrays(state, expected_feature_dim=2)


if __name__ == "__main__":
    unittest.main()
