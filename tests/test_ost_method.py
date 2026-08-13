from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class OSTMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch

        cls.torch = torch

    def tiny_meta_detector(self):
        import torch.nn as nn
        import torch.nn.functional as functional

        class TinyMetaDetector(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(self_outer.torch.randn(2, 3) * 0.1)
                self.bias = nn.Parameter(self_outer.torch.zeros(2))
                self.running_mean = nn.Parameter(
                    self_outer.torch.zeros(3), requires_grad=False
                )

            def forward(
                self,
                x,
                num_step,
                params=None,
                training=False,
                backup_running_statistics=False,
            ):
                del num_step, training, backup_running_statistics
                parameters = params or dict(self.named_parameters())
                features = x.mean(dim=(2, 3))
                logits = functional.linear(
                    features, parameters["weight"], parameters["bias"]
                )
                return logits, features

            def restore_backup_stats(self) -> None:
                return None

        self_outer = self
        return TinyMetaDetector()

    def test_ost_uses_official_fast_weights_without_target_labels(self) -> None:
        from src.methods.ost import OST

        class TemplateSampler:
            def __init__(self, torch_module) -> None:
                self.torch = torch_module
                self.calls = 0

            def sample(self):
                self.calls += 1
                return self.torch.full((3, 4, 4), -0.25), 0, f"source-{self.calls}"

            def reset(self) -> None:
                self.calls = 0

        model = self.tiny_meta_detector()
        initial = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
        }
        method = OST(
            model,
            "cpu",
            {
                "image_size": 4,
                "steps": 1,
                "task_learning_rate": 0.0005,
                "second_order": False,
                "alpha_min": 0.7,
                "alpha_max": 0.7,
            },
        )
        sampler = TemplateSampler(self.torch)
        method.set_template_sampler(sampler)
        self.assertFalse(method.model.running_mean.requires_grad)
        prediction = method.predict(self.torch.randn(2, 3, 4, 4))

        self.assertEqual(tuple(prediction.logits.shape), (2, 2))
        self.assertEqual(sampler.calls, 2)
        self.assertEqual(method.core.__class__.__module__, "src.official.ost.runtime")
        stats = method.adapt(None)
        self.assertEqual(stats.selected, 4)
        self.assertEqual(stats.extra["optimizer_updates"], 2)
        self.assertEqual(stats.extra["template_sample_ids"], ["source-1", "source-2"])
        for name, parameter in method.model.named_parameters():
            self.assertTrue(self.torch.equal(parameter, initial[name]))

    def test_ost_requires_a_labeled_source_template_sampler(self) -> None:
        from src.methods.ost import OST

        method = OST(
            self.tiny_meta_detector(),
            "cpu",
            {"image_size": 4, "steps": 1, "second_order": False},
        )
        with self.assertRaisesRegex(RuntimeError, "source template"):
            method.predict(self.torch.randn(1, 3, 4, 4))

    def test_ost_static_mode_uses_the_same_checkpoint_without_fast_weights(self) -> None:
        from src.methods.ost import OST

        method = OST(
            self.tiny_meta_detector(),
            "cpu",
            {
                "adaptation_mode": "static",
                "image_size": 4,
                "steps": 1,
                "second_order": False,
            },
        )
        prediction = method.predict(self.torch.randn(2, 3, 4, 4))
        stats = method.adapt(None)

        self.assertEqual(tuple(prediction.logits.shape), (2, 2))
        self.assertEqual(method.protocol_name, "predict_only")
        self.assertIsNone(method.core)
        self.assertEqual(method.trainable_parameters, 0)
        self.assertEqual(stats.selected, 0)
        self.assertEqual(stats.extra["optimizer_updates"], 0)

    def test_ost_meta_training_updates_the_initialization_through_fast_weights(self) -> None:
        from src.official.ost import OSTMetaTrainingCore

        model = self.tiny_meta_detector()
        trainer = OSTMetaTrainingCore(
            model,
            "cpu",
            task_learning_rate=0.0005,
            outer_learning_rate=0.001,
            second_order=False,
        )
        initial_weight = model.weight.detach().clone()
        support = self.torch.randn(4, 3, 4, 4)
        support_labels = self.torch.tensor([0, 1, 0, 1])
        query = self.torch.randn(4, 3, 4, 4)
        query_labels = self.torch.tensor([0, 1, 1, 0])

        result = trainer.train_step(support, support_labels, query, query_labels)

        self.assertTrue(self.torch.isfinite(result["support_loss"]))
        self.assertTrue(self.torch.isfinite(result["query_loss"]))
        self.assertEqual(tuple(result["query_scores"].shape), (4, 2))
        self.assertFalse(self.torch.equal(model.weight.detach(), initial_weight))

    def test_ost_training_episode_has_labeled_support_and_query_pairs(self) -> None:
        from train_source import _ost_episode

        images = self.torch.zeros(2, 3, 4, 4)
        labels = self.torch.tensor([0, 1])
        templates = self.torch.ones(2, 3, 4, 4)
        template_labels = self.torch.tensor([1, 0])
        support, support_labels, query, query_labels = _ost_episode(
            images,
            labels,
            templates,
            template_labels,
            alpha_min=0.7,
            alpha_max=0.7,
        )

        self.assertEqual(tuple(support.shape), (4, 3, 4, 4))
        self.assertEqual(tuple(query.shape), (4, 3, 4, 4))
        self.assertEqual(support_labels.tolist()[-2:], [1, 1])
        self.assertEqual(query_labels.tolist(), [0, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()
