from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class PoundTTATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional

        cls.torch = torch
        cls.nn = nn
        cls.functional = functional

    def detector(self):
        torch = self.torch
        nn = self.nn
        functional = self.functional

        class TinyPoundNet(nn.Module):
            feature_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(()))

            def forward_pound_features(self, images, *, semantic_temperature):
                del semantic_temperature
                channel_means = images.mean(dim=(2, 3))
                signal = channel_means[:, 0] - channel_means[:, 1]
                source_logits = torch.stack((-signal, signal), dim=1)
                features = functional.normalize(
                    torch.stack(
                        (signal, channel_means[:, 2], torch.ones_like(signal), -signal),
                        dim=1,
                    ),
                    dim=1,
                )
                semantic_keys = functional.normalize(
                    torch.stack(
                        (
                            torch.ones_like(signal),
                            signal.abs() + 0.1,
                            torch.zeros_like(signal),
                            torch.zeros_like(signal),
                        ),
                        dim=1,
                    ),
                    dim=1,
                )
                residuals = functional.normalize(
                    torch.stack(
                        (
                            -signal,
                            signal,
                            torch.full_like(signal, 0.1),
                            torch.zeros_like(signal),
                        ),
                        dim=1,
                    ),
                    dim=1,
                )
                return {
                    "source_logits": source_logits,
                    "features": features,
                    "semantic_keys": semantic_keys,
                    "residuals": residuals,
                    "conditional_margin": 2.0 * signal,
                    "semantic_routing": torch.ones(len(signal), 1),
                }

            def forward(self, images):
                return self.forward_pound_features(
                    images, semantic_temperature=1.0
                )["source_logits"]

        return TinyPoundNet()

    def batch(self):
        images = self.torch.zeros(2, 3, 4, 4)
        images[0, 0] = 1.0
        images[1, 1] = 1.0
        return images

    def config(self):
        return {
            "memory_size": 4,
            "min_memory_per_class": 1,
            "support_saturation": 1,
            "candidate_queue_size": 8,
            "promotion_delay": 1,
            "candidate_reliability": 0.0,
            "promotion_reliability": 0.0,
            "min_vote_ratio": 0.5,
            "guard_fraction": 0.0,
            "use_horizontal_flip": False,
            "use_scale_view": False,
            "adapter_update_frequency": 100,
            "gate_threshold": 0.0,
            "max_adaptation_weight": 1.0,
        }

    def test_static_mode_is_the_exact_poundnet_prediction(self) -> None:
        from src.methods.pound_tta import PoundTTA

        detector = self.detector()
        method = PoundTTA(
            detector, "cpu", {**self.config(), "adaptation_mode": "static"}
        )
        images = self.batch()
        expected = detector(images)
        prediction = method.predict(images)
        self.assertTrue(self.torch.equal(prediction.logits, expected))
        stats = method.adapt(images)
        self.assertEqual(stats.selected, 0)
        self.assertEqual(method.trainable_parameters, 0)

    def test_candidates_are_delayed_until_after_preupdate_prediction(self) -> None:
        from src.methods.pound_tta import PoundTTA

        method = PoundTTA(self.detector(), "cpu", self.config())
        images = self.batch()
        source_logits = method.model(images)

        first = method.predict(images)
        self.assertTrue(self.torch.equal(first.logits, source_logits))
        self.assertEqual([len(method.memory[label]) for label in (0, 1)], [0, 0])
        self.assertEqual(len(method.candidate_queue), 0)
        first_stats = method.adapt(images)
        self.assertEqual(first_stats.extra["candidates_promoted"], 0)
        self.assertEqual(len(method.candidate_queue), 2)

        second = method.predict(images)
        self.assertTrue(self.torch.equal(second.logits, source_logits))
        self.assertEqual([len(method.memory[label]) for label in (0, 1)], [0, 0])
        second_stats = method.adapt(images)
        self.assertEqual(second_stats.extra["candidates_promoted"], 2)
        self.assertEqual([len(method.memory[label]) for label in (0, 1)], [1, 1])

        third = method.predict(images)
        self.assertFalse(self.torch.equal(third.logits, source_logits))
        self.assertGreater(method._last_gate_mean, 0.0)

    def test_memory_is_class_reserved_and_resettable(self) -> None:
        from src.methods.pound_tta import PoundTTA

        method = PoundTTA(self.detector(), "cpu", self.config())
        images = self.batch()
        for _ in range(8):
            method.predict(images)
            method.adapt(images)
        self.assertLessEqual(len(method.memory[0]), 2)
        self.assertLessEqual(len(method.memory[1]), 2)
        self.assertGreater(sum(map(len, method.memory.values())), 0)
        method.reset()
        self.assertEqual([len(method.memory[label]) for label in (0, 1)], [0, 0])
        self.assertEqual(len(method.candidate_queue), 0)

    def test_pound_midpoint_is_orthogonal_to_authenticity_difference(self) -> None:
        from src.models.poundnet import pound_prompt_components

        paired = self.functional.normalize(self.torch.randn(7, 2, 8), dim=-1)
        midpoints, directions, differences = pound_prompt_components(paired)
        dot = (midpoints * differences).sum(dim=1)
        self.assertTrue(self.torch.allclose(dot, self.torch.zeros_like(dot), atol=1e-6))
        self.assertTrue(
            self.torch.allclose(
                directions.norm(dim=1), self.torch.ones(7), atol=1e-6
            )
        )

    def test_registry_accepts_paper_and_explicit_method_names(self) -> None:
        from src.methods import PoundTTA, build_method

        for name in ("ours", "pound_tta", "ours_static", "pound_tta_static"):
            with self.subTest(name=name):
                method = build_method(name, self.detector(), "cpu", self.config())
                self.assertIsInstance(method, PoundTTA)
                expected_mode = "static" if "static" in name else "full"
                self.assertEqual(method.adaptation_mode, expected_mode)


if __name__ == "__main__":
    unittest.main()
