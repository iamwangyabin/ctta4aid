from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("torchvision") is not None
)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch and torchvision are required")
class IAPLMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        import torch.nn as nn

        cls.torch = torch
        cls.nn = nn

    def model(self):
        nn = self.nn

        class PromptLearner(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ctx = nn.Parameter(self_outer.torch.tensor([[0.2]]))

        class TinyIAPLModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.prompt_learner = PromptLearner()
                self.frozen_scale = nn.Parameter(self_outer.torch.tensor(1.0))

            def freeze_tta(self) -> None:
                self.requires_grad_(False)
                self.prompt_learner.ctx.requires_grad_(True)

            def forward(self, images):
                score = images.mean(dim=(1, 2, 3)) * self.frozen_scale
                logits = (score + self.prompt_learner.ctx.mean()).unsqueeze(1)
                if self.training:
                    return [logits, score.unsqueeze(1), logits]
                return logits

        self_outer = self
        return TinyIAPLModel()

    def config(self):
        return {
            "views": 4,
            "steps": 1,
            "selection_fraction": 0.5,
            "optimal_input_selection": True,
            "lr": 0.005,
            "weight_decay": 0.01,
            "amp": False,
            "image_size": 8,
            "resize_size": 8,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        }

    def test_iapl_is_registered_as_a_framework_method(self) -> None:
        from src.methods import IAPL, build_method

        method = build_method("iapl", self.model(), "cpu", self.config())
        self.assertIsInstance(method, IAPL)
        self.assertEqual(method.protocol_name, "episodic_adapt_then_predict")

    def test_predict_adapts_each_image_and_returns_common_binary_output(self) -> None:
        from src.methods.iapl import IAPL

        method = IAPL(self.model(), "cpu", self.config())
        initial_prompt = method._initial_prompt.clone()
        images = self.torch.randn(2, 4, 3, 8, 8)

        prediction = method.predict(images)
        stats = method.adapt(images)

        self.assertEqual(tuple(prediction.logits.shape), (2, 2))
        self.assertEqual(tuple(prediction.prob_fake.shape), (2,))
        self.assertEqual(stats.selected, 4)
        self.assertEqual(stats.extra["optimizer_updates"], 2)
        self.assertTrue(stats.extra["adaptation_inside_predict"])
        self.assertTrue(self.torch.isfinite(self.torch.tensor(stats.loss)))

        method.reset()
        self.assertTrue(
            self.torch.equal(method.model.prompt_learner.ctx, initial_prompt)
        )

    def test_static_and_views_only_modes_do_not_update_the_prompt(self) -> None:
        from src.methods.iapl import IAPL

        images = self.torch.randn(2, 4, 3, 8, 8)
        for mode, protocol in (
            ("static", "predict_only"),
            ("views_only", "multiview_predict_only"),
        ):
            with self.subTest(mode=mode):
                config = {**self.config(), "adaptation_mode": mode}
                method = IAPL(self.model(), "cpu", config)
                initial_prompt = method._initial_prompt.clone()
                prediction = method.predict(images)
                stats = method.adapt(images)

                self.assertEqual(tuple(prediction.logits.shape), (2, 2))
                self.assertEqual(method.protocol_name, protocol)
                self.assertEqual(method.trainable_parameters, 0)
                self.assertEqual(stats.selected, 0)
                self.assertEqual(stats.extra["optimizer_updates"], 0)
                self.assertFalse(stats.extra["adaptation_inside_predict"])
                self.assertTrue(
                    self.torch.equal(method.model.prompt_learner.ctx, initial_prompt)
                )

    def test_view_transform_returns_global_plus_local_views(self) -> None:
        from PIL import Image

        from src.data.views import GlobalLocalViewTransform

        transform = GlobalLocalViewTransform(
            views=4,
            image_size=8,
            resize_size=10,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )
        output = transform(Image.new("RGB", (12, 12), color="white"))
        self.assertEqual(tuple(output.shape), (4, 3, 8, 8))


if __name__ == "__main__":
    unittest.main()
