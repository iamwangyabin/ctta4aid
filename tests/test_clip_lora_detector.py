from __future__ import annotations

import importlib.util
import io
import unittest
from collections import OrderedDict
from unittest.mock import patch


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is required")
class ClipLoRADetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        import torch.nn as nn

        cls.torch = torch
        cls.nn = nn

    def fake_clip(self):
        nn = self.nn

        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.attn = nn.MultiheadAttention(8, 2)
                self.mlp = nn.Sequential(
                    OrderedDict(
                        [("c_fc", nn.Linear(8, 16)), ("c_proj", nn.Linear(16, 8))]
                    )
                )

            def forward(self, values):
                return values + self.mlp(values)

        class Transformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.resblocks = nn.Sequential(Block(), Block())

        class Visual(nn.Module):
            output_dim = 8

            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 8, kernel_size=1)
                self.transformer = Transformer()

            def forward(self, images):
                values = self.conv1(images).mean(dim=(2, 3))
                for block in self.transformer.resblocks:
                    values = block(values)
                return values

        class FakeClip(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.visual = Visual()

        return FakeClip()

    def build(self, **overrides):
        from src.models.clip_lora_detector import build_clip_lora_detector

        config = {"image_size": 8, "resize_size": 8, "lora_rank": 2, "lora_alpha": 4}
        config.update(overrides)
        metadata = {"image_size": 8, "architecture": "ViT-L/14"}
        with patch(
            "src.models.clip_lora_detector.load_openai_clip_model",
            return_value=(self.fake_clip(), metadata),
        ):
            return build_clip_lora_detector(config, device="cpu")

    def test_lora_linear_zero_init_matches_base_and_freezes_base(self) -> None:
        from src.models.clip_lora_detector import _lora_linear_class

        torch = self.torch
        nn = self.nn
        torch.manual_seed(0)
        base = nn.Linear(8, 4)
        wrapped = _lora_linear_class()(base, rank=2, alpha=2.0)
        values = torch.randn(3, 8)
        self.assertTrue(torch.equal(wrapped(values), base(values)))
        self.assertFalse(wrapped.base.weight.requires_grad)
        self.assertTrue(wrapped.lora_a.requires_grad)
        self.assertTrue(wrapped.lora_b.requires_grad)
        wrapped(values).sum().backward()
        self.assertIsNone(wrapped.base.weight.grad)
        self.assertIsNotNone(wrapped.lora_a.grad)

    def test_lora_linear_rejects_invalid_rank_and_alpha(self) -> None:
        from src.models.clip_lora_detector import _lora_linear_class

        nn = self.nn
        with self.assertRaises(ValueError):
            _lora_linear_class()(nn.Linear(8, 4), rank=0, alpha=1.0)
        with self.assertRaises(ValueError):
            _lora_linear_class()(nn.Linear(8, 4), rank=2, alpha=0.0)

    def test_injection_wraps_only_mlp_projections(self) -> None:
        model, metadata = self.build()
        self.assertEqual(metadata["lora_injected_layers"], 4)
        self.assertEqual(
            metadata["source_setup"], "lora_binary_detector_from_fixed_clip_vitl14"
        )
        for block in model.clip.visual.transformer.resblocks:
            self.assertTrue(hasattr(block.mlp.c_fc, "lora_a"))
            self.assertTrue(hasattr(block.mlp.c_proj, "lora_a"))
            self.assertFalse(hasattr(block.attn.out_proj, "lora_a"))

    def test_only_lora_and_head_parameters_are_trainable(self) -> None:
        model, _metadata = self.build()
        trainable = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        for name in trainable:
            self.assertTrue(
                ".lora_a" in name or ".lora_b" in name or name.startswith("classifier."),
                msg=f"unexpected trainable parameter {name}",
            )

    def test_lora_branch_changes_output_only_after_training(self) -> None:
        torch = self.torch
        torch.manual_seed(0)
        model, _metadata = self.build()
        model.eval()
        images = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            initial = model(images)
        for module in model.modules():
            if hasattr(module, "lora_b"):
                torch.nn.init.constant_(module.lora_b, 0.1)
        with torch.no_grad():
            changed = model(images)
        self.assertFalse(torch.allclose(initial, changed))

    def test_state_dict_roundtrip_restores_exact_outputs(self) -> None:
        torch = self.torch
        torch.manual_seed(0)
        model, _metadata = self.build()
        for module in model.modules():
            if hasattr(module, "lora_b"):
                torch.nn.init.normal_(module.lora_b, std=0.01)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        torch.manual_seed(1234)
        reloaded, _ = self.build()
        buffer.seek(0)
        reloaded.load_state_dict(torch.load(buffer, weights_only=True))
        model.eval()
        reloaded.eval()
        images = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(images), reloaded(images)))

    def test_configure_requires_injected_lora(self) -> None:
        from src.models.clip_lora_detector import configure_clip_lora_trainable_parameters

        with self.assertRaises(RuntimeError):
            configure_clip_lora_trainable_parameters(self.nn.Linear(4, 2))


if __name__ == "__main__":
    unittest.main()
