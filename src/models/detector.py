from __future__ import annotations

from pathlib import Path
from typing import Any


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to construct detectors") from exc
    return torch, nn


class BinaryDetector:
    """Factory wrapper documented for typing; instances are real ``nn.Module`` objects."""


def build_detector(
    architecture: str = "resnet50",
    *,
    pretrained: bool = True,
    pretrained_weights: str = "default",
    num_classes: int = 2,
) -> Any:
    _, nn = _torch_modules()
    try:
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError("torchvision is required to construct detectors") from exc

    architecture = architecture.lower().replace("-", "_")

    class Detector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if architecture == "resnet50":
                weights = _resolve_weights(
                    models.ResNet50_Weights, pretrained, pretrained_weights
                )
                network = models.resnet50(weights=weights)
                feature_dim = network.fc.in_features
                network.fc = nn.Identity()
            elif architecture in {"efficientnet_b4", "efficientnetb4"}:
                weights = _resolve_weights(
                    models.EfficientNet_B4_Weights, pretrained, pretrained_weights
                )
                network = models.efficientnet_b4(weights=weights)
                feature_dim = network.classifier[-1].in_features
                network.classifier = nn.Identity()
            else:
                raise ValueError(f"Unsupported architecture: {architecture}")
            self.backbone = network
            self.classifier = nn.Linear(feature_dim, num_classes)
            self.feature_dim = feature_dim
            self.architecture = architecture

        def forward_features(self, images: Any) -> Any:
            return self.backbone(images)

        def forward(self, images: Any) -> Any:
            return self.classifier(self.forward_features(images))

    return Detector()


def _resolve_weights(weights_enum: Any, pretrained: bool, name: str) -> Any:
    if not pretrained:
        return None
    normalized = name.lower().replace("-", "_")
    if normalized == "default":
        return weights_enum.DEFAULT
    member_name = normalized.upper()
    try:
        return weights_enum[member_name]
    except KeyError as exc:
        choices = ", ".join(weight.name.lower() for weight in weights_enum)
        raise ValueError(
            f"Unknown pretrained weights {name!r}; expected default or one of {choices}"
        ) from exc


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(
    model: Any, checkpoint_path: str | Path, *, device: str | Any = "cpu", strict: bool = True
) -> dict[str, Any]:
    torch, _ = _torch_modules()
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        metadata = {key: value for key, value in checkpoint.items() if key != "model"}
    else:
        state_dict = checkpoint
        metadata = {}
    model.load_state_dict(_clean_state_dict(state_dict), strict=strict)
    return metadata


def save_checkpoint(model: Any, path: str | Path, **metadata: Any) -> None:
    torch, _ = _torch_modules()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **metadata}, output)
