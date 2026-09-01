"""LoRA-adapted binary detector initialized from the fixed OpenAI CLIP backbone.

This is the Ours source model: the CLIP ViT-L/14 visual tower stays frozen
except for small-rank LoRA branches on each transformer block's MLP
projections, plus a supervised binary head. Attention ``out_proj`` is
intentionally not wrapped: ``nn.MultiheadAttention`` consumes
``out_proj.weight``/``out_proj.bias`` through the functional path, so
replacing that module with a wrapper would silently change nothing.
"""

from __future__ import annotations

from typing import Any

from src.data.transforms import CLIP_MEAN, CLIP_STD, build_clip_eval_transform

from .clip_vlm import load_openai_clip_model


OURS_LORA_TARGET_SUFFIXES = ("mlp.c_fc", "mlp.c_proj")


def _lora_linear_class() -> Any:
    import math

    import torch
    import torch.nn as nn

    class LoRALinear(nn.Module):
        """A frozen Linear plus a trainable low-rank residual branch."""

        def __init__(self, base: Any, rank: int, alpha: float) -> None:
            super().__init__()
            if not isinstance(base, nn.Linear):
                raise TypeError("LoRALinear can only wrap nn.Linear modules")
            if rank < 1:
                raise ValueError("LoRA rank must be positive")
            if alpha <= 0.0:
                raise ValueError("LoRA alpha must be positive")
            self.base = base
            self.rank = int(rank)
            self.alpha = float(alpha)
            self.scaling = self.alpha / self.rank
            # B starts at zero so the wrapped layer initially equals the frozen
            # base layer and training never destabilizes the pretrained model.
            self.lora_a = nn.Parameter(torch.empty(self.rank, base.in_features))
            self.lora_b = nn.Parameter(torch.zeros(base.out_features, self.rank))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
            self.base.weight.requires_grad_(False)
            if self.base.bias is not None:
                self.base.bias.requires_grad_(False)

        @property
        def in_features(self) -> int:
            return self.base.in_features

        @property
        def out_features(self) -> int:
            return self.base.out_features

        def forward(self, inputs: Any) -> Any:
            delta = (inputs @ self.lora_a.t() @ self.lora_b.t()) * self.scaling
            return self.base(inputs) + delta

    return LoRALinear


def inject_visual_lora(visual: Any, *, rank: int, alpha: float) -> list[str]:
    """Attach LoRA branches to every visual transformer block MLP projection."""

    LoRALinear = _lora_linear_class()
    resblocks = getattr(getattr(visual, "transformer", None), "resblocks", None)
    if resblocks is None:
        raise TypeError(
            "Expected an OpenAI CLIP visual tower with transformer.resblocks"
        )
    injected: list[str] = []
    for index, block in enumerate(resblocks):
        mlp = getattr(block, "mlp", None)
        if mlp is None:
            raise TypeError(
                f"Visual transformer block {index} has no MLP to attach LoRA to"
            )
        for name in ("c_fc", "c_proj"):
            original = getattr(mlp, name, None)
            if original is None:
                raise TypeError(f"Visual block {index} MLP is missing {name}")
            setattr(mlp, name, LoRALinear(original, rank, alpha))
            injected.append(f"transformer.resblocks.{index}.mlp.{name}")
    return injected


def _clip_lora_detector_class() -> Any:
    import torch
    import torch.nn as nn

    class VisualTower(nn.Module):
        """Keep only the pretrained CLIP visual tower in the detector state."""

        def __init__(self, visual: Any) -> None:
            super().__init__()
            self.visual = visual

    class CLIPLoRADetector(nn.Module):
        """Frozen CLIP visual tower + LoRA branches + supervised binary head."""

        def __init__(
            self,
            clip_model: Any,
            *,
            image_size: int,
            resize_size: int,
            lora_rank: int,
            lora_alpha: float,
            classifier_feature_normalization: str,
        ) -> None:
            super().__init__()
            # The task-trained detector never calls CLIP's text tower. Retaining
            # it would falsely expose unused text parameters to adaptation.
            self.clip = VisualTower(clip_model.visual.float())
            self.lora_rank = int(lora_rank)
            self.lora_alpha = float(lora_alpha)
            self.classifier_feature_normalization = str(
                classifier_feature_normalization
            ).lower()
            if self.classifier_feature_normalization not in {"none", "l2"}:
                raise ValueError(
                    "classifier_feature_normalization must be none or l2"
                )
            self.injected_lora_layers = inject_visual_lora(
                self.clip.visual, rank=self.lora_rank, alpha=self.lora_alpha
            )
            self.classifier = nn.Linear(int(self.clip.visual.output_dim), 2)
            self.input_transform = build_clip_eval_transform(
                image_size, resize_size=resize_size
            )
            self.input_mean = CLIP_MEAN
            self.input_std = CLIP_STD
            self.feature_dim = int(self.clip.visual.output_dim)

        @property
        def dtype(self) -> Any:
            return self.clip.visual.conv1.weight.dtype

        def encode_image(self, images: Any) -> Any:
            return self.clip.visual(images.to(dtype=self.dtype))

        def forward_features(self, images: Any) -> Any:
            return self.encode_image(images).float()

        @property
        def clip_visual_transformer_depth(self) -> int:
            resblocks = getattr(
                getattr(self.clip.visual, "transformer", None),
                "resblocks",
                None,
            )
            if resblocks is None:
                raise TypeError(
                    "CLIP multilayer features require visual transformer blocks"
                )
            return len(resblocks)

        def forward_multilayer_features(
            self,
            images: Any,
            layers: Any,
        ) -> tuple[Any, Any]:
            """Return the final feature and selected projected CLS features.

            Layer numbers are one-based. Each captured block output is passed
            through the visual tower's frozen ``ln_post`` and ``proj`` so all
            selected layers share the detector head's 768-dimensional CLIP
            coordinate. The normal final feature is produced by the unchanged
            visual forward and is returned separately for source scoring.
            """

            try:
                selected = tuple(int(layer) for layer in layers)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "CLIP multilayer feature layers must be integers"
                ) from exc
            if (
                not selected
                or any(layer < 1 for layer in selected)
                or selected != tuple(sorted(set(selected)))
            ):
                raise ValueError(
                    "CLIP multilayer feature layers must be unique positive "
                    "integers in increasing order"
                )

            visual = self.clip.visual
            resblocks = getattr(
                getattr(visual, "transformer", None),
                "resblocks",
                None,
            )
            if resblocks is None:
                raise TypeError(
                    "CLIP multilayer features require visual transformer blocks"
                )
            depth = len(resblocks)
            if selected[-1] > depth:
                raise ValueError(
                    f"CLIP multilayer feature layer {selected[-1]} exceeds "
                    f"visual depth {depth}"
                )
            ln_post = getattr(visual, "ln_post", None)
            if not callable(ln_post):
                raise TypeError("CLIP multilayer features require visual ln_post")
            projection = getattr(visual, "proj", None)
            batch = int(images.shape[0])
            captured: dict[int, Any] = {}
            handles = []

            def capture(layer: int) -> Any:
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    sequence = output[0] if isinstance(output, (tuple, list)) else output
                    if sequence.ndim != 3 or int(sequence.shape[1]) != batch:
                        raise ValueError(
                            "CLIP visual block output must have token-batch-width "
                            "shape for multilayer feature capture"
                        )
                    feature = ln_post(sequence[0])
                    if projection is not None:
                        feature = feature @ projection
                    captured[layer] = feature.float()

                return hook

            try:
                for layer in selected:
                    handles.append(
                        resblocks[layer - 1].register_forward_hook(capture(layer))
                    )
                final_features = self.forward_features(images)
            finally:
                for handle in handles:
                    handle.remove()

            missing = [layer for layer in selected if layer not in captured]
            if missing:
                raise RuntimeError(
                    f"CLIP multilayer feature hooks missed layers {missing}"
                )
            # Reuse the normal visual output for the final block so its slice is
            # exactly the coordinate used by the frozen source classifier.
            if selected[-1] == depth:
                captured[depth] = final_features
            concatenated = torch.cat([captured[layer] for layer in selected], dim=1)
            expected_dim = len(selected) * self.feature_dim
            if concatenated.shape != (batch, expected_dim):
                raise ValueError(
                    "CLIP multilayer feature concatenation has an invalid shape"
                )
            return final_features, concatenated

        def forward_classifier_features(self, features: Any) -> Any:
            if self.classifier_feature_normalization == "l2":
                return torch.nn.functional.normalize(features.float(), dim=1)
            return features.float()

        def forward(self, images: Any) -> Any:
            features = self.forward_features(images)
            return self.classifier(self.forward_classifier_features(features))

    return CLIPLoRADetector


def configure_clip_lora_trainable_parameters(model: Any) -> list[str]:
    """Freeze the pretrained tower; train only LoRA branches and the head."""

    model.requires_grad_(False)
    for name, parameter in model.named_parameters():
        if ".lora_a" in name or ".lora_b" in name or name.startswith("classifier."):
            parameter.requires_grad_(True)
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not any(".lora_a" in name for name in trainable):
        raise RuntimeError("LoRA injection produced no trainable LoRA parameters")
    return trainable


def build_clip_lora_detector(
    config: dict[str, Any], *, device: str | Any = "cpu"
) -> tuple[Any, dict[str, Any]]:
    """Construct the Ours LoRA source detector before loading its checkpoint."""

    clip_model, metadata = load_openai_clip_model(config, device=device)
    image_size = int(metadata["image_size"])
    lora_rank = int(config.get("lora_rank", 4))
    lora_alpha = float(config.get("lora_alpha", 8.0))
    detector = _clip_lora_detector_class()(
        clip_model,
        image_size=image_size,
        resize_size=int(config.get("resize_size", round(image_size / 0.875))),
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        classifier_feature_normalization=str(
            config.get("classifier_feature_normalization", "none")
        ),
    ).to(device)
    trainable_names = configure_clip_lora_trainable_parameters(detector)
    return detector, {
        **metadata,
        "source_setup": "lora_binary_detector_from_fixed_clip_vitl14",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_target_suffixes": list(OURS_LORA_TARGET_SUFFIXES),
        "lora_injected_layers": len(detector.injected_lora_layers),
        "classifier_feature_normalization": (
            detector.classifier_feature_normalization
        ),
        "source_training_trainable_parameters": trainable_names,
    }
