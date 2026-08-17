from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.data.transforms import build_clip_eval_transform


OPENAI_CLIP_COMMIT = "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6"
OPENAI_CLIP_VIT_L14_SHA256 = "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BINARY_CLASS_PROMPTS = (
    ("a real photograph",),
    ("an AI-generated image",),
)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _normalize_architecture(value: object) -> str:
    normalized = str(value).lower().replace("_", "").replace("-", "").replace("/", "")
    if normalized != "vitl14":
        raise ValueError("The CLIP VLM main track currently pins OpenAI CLIP ViT-L/14")
    return "ViT-L/14"


def _binary_class_prompts(config: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    configured = config.get("class_prompts", DEFAULT_BINARY_CLASS_PROMPTS)
    if not isinstance(configured, (list, tuple)) or len(configured) != 2:
        raise ValueError("CLIP VLM class_prompts must contain real and fake prompt lists")
    prompts: list[tuple[str, ...]] = []
    for index, group in enumerate(configured):
        if not isinstance(group, (list, tuple)) or not group:
            raise ValueError(f"CLIP VLM class_prompts[{index}] must be a non-empty prompt list")
        normalized = tuple(str(prompt).strip() for prompt in group)
        if any(not prompt for prompt in normalized):
            raise ValueError("CLIP VLM prompts must be non-empty strings")
        prompts.append(normalized)
    return prompts[0], prompts[1]


def _binary_vlm_class() -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    from src.official.openai_clip import tokenize

    class CLIPBinaryVLM(nn.Module):
        """OpenAI CLIP with fixed, pre-registered binary text prototypes."""

        def __init__(
            self,
            clip_model: Any,
            class_prompts: tuple[tuple[str, ...], tuple[str, ...]],
            input_transform: Any,
            *,
            dynamic_text_features: bool,
        ) -> None:
            super().__init__()
            self.clip = clip_model
            self.class_prompts = class_prompts
            self.class_names = ("real", "fake")
            self.input_transform = input_transform
            self.dynamic_text_features = dynamic_text_features
            for index, prompts in enumerate(self.class_prompts):
                self.register_buffer(
                    f"_class_prompt_tokens_{index}",
                    tokenize(list(prompts)).to(self.clip.logit_scale.device),
                    persistent=False,
                )
            text_features, text_pre_features = self._encode_class_prompts(with_grad=False)
            self.register_buffer("text_features", text_features)
            self.register_buffer("text_pre_features", text_pre_features)

        @property
        def dtype(self) -> Any:
            return self.clip.dtype

        @property
        def feature_dim(self) -> int:
            return int(self.text_features.shape[-1])

        def _encode_class_prompts(self, *, with_grad: bool) -> tuple[Any, Any]:
            class_features = []
            class_pre_features = []
            context = nullcontext() if with_grad else torch.no_grad()
            with context:
                for prompts in self.class_prompts:
                    index = len(class_features)
                    tokenized = getattr(self, f"_class_prompt_tokens_{index}")
                    encoded = self.clip.encode_text(tokenized)
                    normalized = functional.normalize(encoded, dim=-1)
                    class_pre_features.append(encoded.mean(dim=0))
                    class_features.append(functional.normalize(normalized.mean(dim=0), dim=0))
            return torch.stack(class_features), torch.stack(class_pre_features)

        def _active_text_features(self) -> tuple[Any, Any]:
            if self.dynamic_text_features:
                return self._encode_class_prompts(with_grad=torch.is_grad_enabled())
            return self.text_features, self.text_pre_features

        def _classify(self, features: Any, text_features: Any) -> Any:
            normalized = functional.normalize(features, dim=-1)
            prototypes = text_features.to(dtype=normalized.dtype)
            return self.clip.logit_scale.exp() * normalized @ prototypes.t()

        def encode_image(self, images: Any) -> Any:
            return self.clip.encode_image(images.type(self.dtype))

        def forward_features(self, images: Any) -> Any:
            return functional.normalize(self.encode_image(images), dim=-1)

        def classifier(self, features: Any) -> Any:
            text_features, _ = self._active_text_features()
            return self._classify(features, text_features)

        def forward_with_features(self, images: Any) -> tuple[Any, Any, Any, Any, Any]:
            pre_features = self.encode_image(images)
            features = functional.normalize(pre_features, dim=-1)
            text_features, text_pre_features = self._active_text_features()
            logits = self._classify(features, text_features)
            return (
                logits,
                features,
                text_features,
                pre_features,
                text_pre_features,
            )

        def forward(self, images: Any) -> Any:
            return self.classifier(self.forward_features(images))

    return CLIPBinaryVLM


def load_openai_clip_model(
    config: dict[str, Any], *, device: str | Any = "cpu"
) -> tuple[Any, dict[str, Any]]:
    """Load and validate the pinned local OpenAI CLIP ViT-L/14 checkpoint."""

    import torch

    from src.official.openai_clip import load as load_clip

    architecture = _normalize_architecture(config.get("architecture", "ViT-L/14"))
    checkpoint_value = config.get("checkpoint")
    if not checkpoint_value:
        raise ValueError("CLIP VLM model config requires checkpoint")
    checkpoint = _resolve_path(str(checkpoint_value))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"CLIP checkpoint does not exist: {checkpoint}")

    torch_load = torch.load

    def compatible_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return torch_load(*args, **kwargs)

    with patch.object(torch, "load", compatible_torch_load):
        clip_model, _ = load_clip(str(checkpoint), device=device, jit=False)

    expected_size = int(getattr(clip_model.visual, "input_resolution", 224))
    image_size = int(config.get("image_size", expected_size))
    if image_size != expected_size:
        raise ValueError(
            f"{architecture} expects image_size={expected_size}, got {image_size}"
        )
    return clip_model, {
        "official_commit": OPENAI_CLIP_COMMIT,
        "architecture": architecture,
        "checkpoint_sha256_expected": OPENAI_CLIP_VIT_L14_SHA256,
        "checkpoint_path": str(checkpoint),
        "image_size": image_size,
    }


def build_clip_vlm_detector(
    config: dict[str, Any], *, device: str | Any = "cpu"
) -> tuple[Any, dict[str, Any]]:
    """Build a CLIP-native binary VLM with a method-specific text classifier."""

    clip_model, base_metadata = load_openai_clip_model(config, device=device)
    clip_model.eval()
    class_prompts = _binary_class_prompts(config)
    dynamic_text_features = bool(config.get("dynamic_text_features", False))
    detector = _binary_vlm_class()(
        clip_model,
        class_prompts,
        build_clip_eval_transform(
            int(base_metadata["image_size"]),
            resize_size=int(
                config.get("resize_size", round(int(base_metadata["image_size"]) / 0.875))
            ),
        ),
        dynamic_text_features=dynamic_text_features,
    )
    metadata = {
        **base_metadata,
        "source_setup": "frozen_openai_clip_zero_shot_binary_prompts",
        "class_prompts": [list(group) for group in class_prompts],
        "prompt_profile": config.get("prompt_profile"),
        "dynamic_text_features": dynamic_text_features,
    }
    return detector.to(device), metadata
