"""Task-trained binary detector initialized from the fixed OpenAI CLIP backbone."""

from __future__ import annotations

from typing import Any

from src.data.transforms import CLIP_MEAN, CLIP_STD, build_clip_eval_transform

from .clip_vlm import load_openai_clip_model


def _clip_source_detector_class() -> Any:
    import torch.nn as nn

    class VisualTower(nn.Module):
        """Keep only the pretrained CLIP visual tower in the detector state."""

        def __init__(self, visual: Any) -> None:
            super().__init__()
            self.visual = visual

    class CLIPSourceDetector(nn.Module):
        """A supervised real/fake head over the OpenAI CLIP visual encoder."""

        def __init__(self, clip_model: Any, *, image_size: int, resize_size: int) -> None:
            super().__init__()
            # The task-trained detector never calls CLIP's text tower. Retaining
            # it would falsely expose unused text parameters to full-model CTTA.
            self.clip = VisualTower(clip_model.visual.float())
            self.classifier = nn.Linear(int(self.clip.visual.output_dim), 2)
            self.input_transform = build_clip_eval_transform(image_size, resize_size=resize_size)
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

        def forward(self, images: Any) -> Any:
            return self.classifier(self.forward_features(images))

    return CLIPSourceDetector


def configure_clip_source_trainable_parameters(model: Any, scope: str) -> list[str]:
    """Select the supervised source-training scope without changing the backbone."""

    normalized = scope.lower().replace("-", "_")
    if normalized in {"full", "visual_and_head"}:
        model.requires_grad_(False)
        model.clip.visual.requires_grad_(True)
        model.classifier.requires_grad_(True)
    elif normalized in {"linear", "linear_head", "head_only"}:
        model.requires_grad_(False)
        model.classifier.requires_grad_(True)
    else:
        raise ValueError(
            "CLIP source trainable_scope must be full or linear_head, "
            f"got {scope!r}"
        )
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def build_clip_source_detector(
    config: dict[str, Any], *, device: str | Any = "cpu"
) -> tuple[Any, dict[str, Any]]:
    """Construct the shared source-domain detector before loading its checkpoint."""

    clip_model, metadata = load_openai_clip_model(config, device=device)
    image_size = int(metadata["image_size"])
    detector = _clip_source_detector_class()(
        clip_model,
        image_size=image_size,
        resize_size=int(config.get("resize_size", round(image_size / 0.875))),
    ).to(device)
    trainable_scope = str(config.get("trainable_scope", "full"))
    trainable_names = configure_clip_source_trainable_parameters(detector, trainable_scope)
    return detector, {
        **metadata,
        "source_setup": "shared_source_trained_clip_vitl14_binary_detector",
        "source_training_scope": trainable_scope,
        "source_training_trainable_parameters": trainable_names,
    }
