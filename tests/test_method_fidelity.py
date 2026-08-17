from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch


TORCH_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("torchvision") is not None
)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch and torchvision are required")
class MethodFidelityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import torch
        import torch.nn as nn

        cls.torch = torch
        cls.nn = nn

    def detector(self):
        nn = self.nn

        class TinyDetector(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
                self.bn = nn.BatchNorm2d(4)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.head = nn.Linear(4, 2)

            def forward(self, images):
                return self.classifier(self.forward_features(images))

            def forward_features(self, images):
                return self.pool(self.bn(self.conv(images))).flatten(1)

            @property
            def classifier(self):
                return self.head

        return TinyDetector()

    def clip_vlm(self):
        nn = self.nn
        torch = self.torch

        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.ln_1 = nn.LayerNorm(4)
                self.ln_2 = nn.LayerNorm(4)

            def forward(self, values):
                return self.ln_2(torch.tanh(self.ln_1(values)))

        class Transformer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.resblocks = nn.ModuleList([Block() for _ in range(4)])

        class Visual(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(3, 4, kernel_size=1)
                self.ln_pre = nn.LayerNorm(4)
                self.transformer = Transformer()
                self.ln_post = nn.LayerNorm(4)

            def forward(self, images):
                values = self.conv(images).mean(dim=(2, 3))
                values = self.ln_pre(values)
                for block in self.transformer.resblocks:
                    values = block(values)
                return self.ln_post(values)

        class Clip(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.visual = Visual()

        class TinyClipVLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.clip = Clip()
                self.head = nn.Linear(4, 2)

            def forward(self, images):
                return self.head(self.clip.visual(images))

        return TinyClipVLM()

    def test_clip_source_detector_keeps_only_visual_tower_and_binary_head(self) -> None:
        from src.models.clip_detector import build_clip_source_detector

        nn = self.nn

        class Visual(nn.Module):
            output_dim = 4

            def __init__(self) -> None:
                super().__init__()
                self.conv1 = nn.Conv2d(3, 4, kernel_size=1)

            def forward(self, images):
                return self.conv1(images).mean(dim=(2, 3))

        class FakeClip(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.visual = Visual()
                self.text = nn.Linear(4, 4)

        metadata = {"image_size": 8, "architecture": "ViT-L/14"}
        with patch(
            "src.models.clip_detector.load_openai_clip_model",
            return_value=(FakeClip(), metadata),
        ):
            model, result_metadata = build_clip_source_detector(
                {"image_size": 8, "resize_size": 8, "trainable_scope": "full"},
                device="cpu",
            )

        names = [name for name, _parameter in model.named_parameters()]
        self.assertTrue(any(name.startswith("clip.visual") for name in names))
        self.assertTrue(any(name.startswith("classifier") for name in names))
        self.assertFalse(any(name.startswith("clip.text") for name in names))
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        self.assertEqual(
            result_metadata["source_setup"],
            "shared_source_trained_clip_vitl14_binary_detector",
        )
        self.assertEqual(tuple(model(self.torch.randn(2, 3, 8, 8)).shape), (2, 2))

    def test_tent_updates_only_batch_norm_affine_parameters(self) -> None:
        from src.methods.tent import Tent

        method = Tent(
            self.detector(),
            "cpu",
            {"optimizer": "sgd", "learning_rate": 0.001, "momentum": 0.9},
        )
        self.assertEqual(method.core.__class__.__module__, "src.official.tent")
        self.assertTrue(method.model.bn.weight.requires_grad)
        self.assertTrue(method.model.bn.bias.requires_grad)
        self.assertFalse(method.model.conv.weight.requires_grad)
        self.assertFalse(method.model.head.weight.requires_grad)
        self.assertIsNone(method.model.bn.running_mean)
        self.assertIsNone(method.model.bn.running_var)
        optimizer_ids = {
            id(parameter)
            for group in method.optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(
            optimizer_ids, {id(method.model.bn.weight), id(method.model.bn.bias)}
        )
        method.reset()
        self.assertIsNone(method.model.bn.running_mean)

    def test_full_eata_rejects_missing_fisher_information(self) -> None:
        from src.methods.eata import EATA

        with self.assertRaisesRegex(RuntimeError, "Fisher"):
            EATA(self.detector(), "cpu", {"optimizer": "sgd"})

    def test_eata_binary_port_adapts_without_labels(self) -> None:
        from src.methods.eata import EATA

        method = EATA(
            self.detector(),
            "cpu",
            {
                "optimizer": "sgd",
                "learning_rate": 0.001,
                "require_fisher": False,
                "entropy_margin": 10.0,
            },
        )
        self.assertEqual(method.core.__class__.__module__, "src.official.eata")
        stats = method.adapt(self.torch.randn(4, 3, 8, 8))
        self.assertEqual(stats.selected, 4)
        self.assertFalse(stats.extra["fisher_enabled"])

    def test_eata_layernorm_mapping_uses_matching_fisher_names(self) -> None:
        from src.methods.eata import EATA
        from src.methods.utils import select_clip_visual_norm_parameters

        model = self.clip_vlm()
        parameters, names = select_clip_visual_norm_parameters(model)
        fishers = {
            name: [self.torch.zeros_like(parameter), parameter.detach().clone()]
            for name, parameter in zip(names, parameters, strict=True)
        }
        method = EATA(
            model,
            "cpu",
            {
                "optimizer": "adam",
                "lr": 0.001,
                "clip_visual_layernorm": True,
                "entropy_margin": 10.0,
                "require_fisher": True,
            },
            fishers=fishers,
        )
        self.assertEqual(
            method.normalization_mapping,
            "BatchNorm_affine_to_CLIP_visual_LayerNorm_affine",
        )
        stats = method.adapt(self.torch.randn(4, 3, 8, 8))
        self.assertTrue(stats.extra["fisher_enabled"])

    def test_tent_layernorm_mapping_and_sar_adapt_only_clip_visual_normalization(
        self,
    ) -> None:
        from src.methods import build_method
        from src.methods.sar import SAR
        from src.methods.tent import Tent
        from src.methods.tent_ln import TentLayerNorm

        images = self.torch.randn(4, 3, 8, 8)
        mapped_tent = build_method(
            "tent",
            self.clip_vlm(),
            "cpu",
            {
                "optimizer": "adam",
                "lr": 0.001,
                "steps": 1,
                "clip_visual_layernorm": True,
            },
        )
        self.assertIsInstance(mapped_tent, Tent)
        self.assertEqual(
            mapped_tent.normalization_mapping,
            "BatchNorm_affine_to_CLIP_visual_LayerNorm_affine",
        )
        self.assertTrue(
            all(
                name.startswith("clip.visual.")
                for name in mapped_tent.official_parameter_names
            )
        )
        mapped_head_before = mapped_tent.model.head.weight.detach().clone()
        mapped_tent.predict(images)
        mapped_tent_stats = mapped_tent.adapt(images)
        self.assertEqual(mapped_tent_stats.selected, 4)
        self.assertTrue(
            self.torch.equal(mapped_tent.model.head.weight, mapped_head_before)
        )

        tent = build_method(
            "tent_ln",
            self.clip_vlm(),
            "cpu",
            {"optimizer": "adam", "lr": 0.001, "steps": 1},
        )
        self.assertIsInstance(tent, TentLayerNorm)
        self.assertTrue(
            all(name.startswith("clip.visual.") for name in tent.official_parameter_names)
        )
        head_before = tent.model.head.weight.detach().clone()
        tent.predict(images)
        tent_stats = tent.adapt(images)
        self.assertEqual(tent_stats.selected, 4)
        self.assertTrue(self.torch.equal(tent.model.head.weight, head_before))

        sar = build_method(
            "sar",
            self.clip_vlm(),
            "cpu",
            {
                "margin": 10.0,
                "reset_constant": -1.0,
                "exclude_last_visual_blocks": 3,
            },
        )
        self.assertIsInstance(sar, SAR)
        self.assertNotIn("clip.visual.ln_post.weight", sar.official_parameter_names)
        self.assertTrue(
            all("resblocks.1" not in name for name in sar.official_parameter_names)
        )
        sar_head_before = sar.model.head.weight.detach().clone()
        sar.predict(images)
        sar_stats = sar.adapt(images)
        self.assertGreater(sar_stats.extra["first_reliable"], 0)
        self.assertGreater(sar_stats.selected, 0)
        self.assertFalse(sar_stats.extra["model_recovered"])
        self.assertTrue(self.torch.equal(sar.model.head.weight, sar_head_before))

    def test_dynaprompt_keeps_trainable_prompt_state_fp32(self) -> None:
        from src.methods.dynaprompt import DynaPrompt

        nn = self.nn
        torch = self.torch

        class StubPromptModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = nn.Linear(2, 2, bias=False).half()
                self.prompt_learner = nn.Module()
                self.prompt_learner.register_parameter(
                    "ctx", nn.Parameter(torch.ones(2, 2, dtype=torch.float16))
                )

            def reset(self) -> None:
                pass

        with patch(
            "src.official.dynaprompt.build_prompt_model",
            return_value=StubPromptModel(),
        ):
            method = DynaPrompt(
                SimpleNamespace(clip=object()),
                "cpu",
                {"views": 10, "selection_fraction": 0.2, "amp": False},
            )

        self.assertEqual(method.model.prompt_learner.ctx.dtype, torch.float32)
        self.assertFalse(method.model.backbone.weight.requires_grad)
        self.assertTrue(method.model.prompt_learner.ctx.requires_grad)

    def test_cotta_runs_official_teacher_augmentation_update(self) -> None:
        from src.methods.cotta import CoTTA

        method = CoTTA(
            self.detector(),
            "cpu",
            {
                "optimizer": "adam",
                "learning_rate": 0.001,
                "augmentations": 2,
                "image_size": 8,
                "restore_probability": 0.0,
                "anchor_confidence": 2.0,
            },
        )
        self.assertEqual(method.core.__class__.__module__, "src.official.cotta")
        images = self.torch.randn(2, 3, 8, 8)
        augmented = method._augment(images)
        self.assertEqual(tuple(augmented.shape), tuple(images.shape))
        prediction = method.predict(images)
        self.assertEqual(tuple(prediction.logits.shape), (2, 2))
        self.assertIsNotNone(method._pending_teacher_target)
        stats = method.adapt(images)
        self.assertIsNone(method._pending_teacher_target)
        self.assertTrue(self.torch.isfinite(self.torch.tensor(stats.loss)))
        self.assertEqual(stats.selected, 2)

    def test_t2a_repaired_losses_and_bernoulli_labels_are_well_formed(self) -> None:
        from src.methods.t2a import T2A
        from src.official.t2a_losses import (
            complementary_labels,
            compute_noise_tolerant_negative_loss,
        )

        method = T2A(
            self.detector(),
            "cpu",
            {
                "optimizer": "adam",
                "learning_rate": 0.0001,
                "noise_type": "bernoulli",
                "gradient_masking": True,
            },
        )
        self.assertEqual(method.core.__class__.__module__, "src.official.t2a")
        images = self.torch.randn(4, 3, 8, 8)
        logits = method.model(images)
        pseudo_labels = logits.argmax(dim=1)
        noisy_labels = complementary_labels(logits, "bernoulli")
        self.assertEqual(tuple(noisy_labels.shape), (4,))
        self.assertTrue(self.torch.all(noisy_labels != pseudo_labels))
        loss = compute_noise_tolerant_negative_loss(
            logits, noise_type="bernoulli", gamma=2.0, alpha=1.0, beta=1.0
        )
        self.assertTrue(self.torch.isfinite(loss))
        state_before_predict = {
            name: tensor.detach().clone()
            for name, tensor in method.model.state_dict().items()
        }
        method.predict(images)
        for name, tensor in method.model.state_dict().items():
            self.assertTrue(
                self.torch.equal(tensor, state_before_predict[name]),
                f"predict mutated {name}",
            )
        stats = method.adapt(images)
        self.assertTrue(self.torch.isfinite(self.torch.tensor(stats.loss)))

    def test_normalized_input_transform_round_trips_identity_pixels(self) -> None:
        from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD
        from src.methods.utils import NormalizedInputTransform

        class RecordingIdentity:
            seen = None

            def __call__(self, images):
                self.seen = images.detach().clone()
                return images

        pixel_transform = RecordingIdentity()
        pixels = self.torch.rand(2, 3, 8, 8)
        mean = self.torch.tensor(IMAGENET_MEAN)[None, :, None, None]
        std = self.torch.tensor(IMAGENET_STD)[None, :, None, None]
        normalized = (pixels - mean) / std
        transformed = NormalizedInputTransform(pixel_transform)(normalized)

        self.assertTrue(self.torch.allclose(pixel_transform.seen, pixels, atol=1e-6))
        self.assertTrue(self.torch.allclose(transformed, normalized, atol=1e-6))

    def test_rotta_runs_robust_bn_cstu_and_teacher_student_update(self) -> None:
        from src.methods.rotta import RoTTA
        from src.official.rotta import RobustBN2d

        method = RoTTA(
            self.detector(),
            "cpu",
            {
                "optimizer": "adam",
                "lr": 0.001,
                "memory_size": 4,
                "update_frequency": 2,
                "num_classes": 2,
                "image_size": 8,
            },
        )
        self.assertEqual(method.core.__class__.__module__, "src.official.rotta")
        self.assertIsInstance(method.model.bn, RobustBN2d)
        self.assertEqual(
            set(method.official_parameter_names), {"bn.weight", "bn.bias"}
        )
        images = self.torch.randn(2, 3, 8, 8)
        prediction = method.predict(images)
        self.assertEqual(tuple(prediction.logits.shape), (2, 2))
        stats = method.adapt(images)
        self.assertEqual(stats.selected, 2)
        self.assertEqual(stats.extra["memory_occupancy"], 2)
        self.assertEqual(stats.extra["optimizer_updates"], 1)
        self.assertTrue(self.torch.isfinite(self.torch.tensor(stats.loss)))
        method.reset()
        self.assertEqual(method.core.mem.get_occupancy(), 0)
        self.assertEqual(method.core.current_instance, 0)
        self.assertEqual(
            method.core.transform.__class__.__name__, "NormalizedInputTransform"
        )

    def test_lame_runs_official_parameter_free_laplacian_output_update(self) -> None:
        from src.methods.lame import LAME

        method = LAME(
            self.detector(),
            "cpu",
            {"affinity": "rbf", "knn": 5, "max_steps": 100},
        )
        before = {
            name: parameter.detach().clone()
            for name, parameter in method.model.named_parameters()
        }
        prediction = method.predict(self.torch.randn(4, 3, 8, 8))
        probabilities = prediction.logits.softmax(dim=1)
        self.assertEqual(tuple(prediction.logits.shape), (4, 2))
        self.assertTrue(
            self.torch.allclose(
                probabilities.sum(dim=1), self.torch.ones(4), atol=1e-6
            )
        )
        stats = method.adapt(self.torch.randn(4, 3, 8, 8))
        self.assertEqual(stats.selected, 0)
        self.assertFalse(stats.extra["state_update"])
        self.assertEqual(method.trainable_parameters, 0)
        for name, parameter in method.model.named_parameters():
            self.assertTrue(self.torch.equal(parameter, before[name]))

        singleton = method.predict(self.torch.randn(1, 3, 8, 8))
        self.assertEqual(tuple(singleton.logits.shape), (1, 2))
        self.assertTrue(method.last_batch_guarded)


if __name__ == "__main__":
    unittest.main()
