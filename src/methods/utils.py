from __future__ import annotations

import math
from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NormalizedInputTransform:
    """Run a pixel-space transform around the model's normalized input."""

    def __init__(
        self,
        pixel_transform: Any,
        *,
        mean: tuple[float, ...] = IMAGENET_MEAN,
        std: tuple[float, ...] = IMAGENET_STD,
    ) -> None:
        self.pixel_transform = pixel_transform
        self.mean = mean
        self.std = std

    def __call__(self, images: Any) -> Any:
        import torch

        mean = torch.as_tensor(
            self.mean, device=images.device, dtype=images.dtype
        )[None, :, None, None]
        std = torch.as_tensor(
            self.std, device=images.device, dtype=images.dtype
        )[None, :, None, None]
        pixels = (images * std + mean).clamp(0.0, 1.0)
        augmented = self.pixel_transform(pixels)
        return (augmented - mean) / std


@contextmanager
def preserve_batch_norm_buffers(model: Any):
    """Keep a train-mode prediction from persisting BatchNorm statistics."""
    import torch
    import torch.nn as nn

    states = []
    for module in model.modules():
        if not isinstance(module, nn.modules.batchnorm._BatchNorm):
            continue
        states.append(
            (
                module,
                None if module.running_mean is None else module.running_mean.detach().clone(),
                None if module.running_var is None else module.running_var.detach().clone(),
                (
                    None
                    if module.num_batches_tracked is None
                    else module.num_batches_tracked.detach().clone()
                ),
            )
        )
    try:
        yield
    finally:
        with torch.no_grad():
            for module, running_mean, running_var, num_batches_tracked in states:
                _restore_buffer(module, "running_mean", running_mean)
                _restore_buffer(module, "running_var", running_var)
                _restore_buffer(module, "num_batches_tracked", num_batches_tracked)


def _restore_buffer(module: Any, name: str, saved: Any) -> None:
    current = getattr(module, name)
    if saved is None:
        setattr(module, name, None)
    elif current is None:
        setattr(module, name, saved)
    else:
        current.copy_(saved)


def entropy_from_logits(logits: Any) -> Any:
    probabilities = logits.softmax(dim=1)
    return -(probabilities * logits.log_softmax(dim=1)).sum(dim=1)


def configure_batch_norm(model: Any, *, train_all: bool = False) -> list[Any]:
    """Use test-batch statistics and return the parameters eligible for updates."""
    import torch.nn as nn

    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(train_all)

    selected = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.train()
            module.track_running_stats = False
            if module.affine:
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
                selected.extend([module.weight, module.bias])

    if train_all:
        selected = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not selected:
        raise RuntimeError("The selected TTA method requires BatchNorm layers, but none were found")
    return selected


def select_clip_visual_norm_parameters(
    model: Any,
    *,
    exclude_last_blocks: int = 0,
    exclude_output_norm: bool = False,
) -> tuple[list[Any], list[str]]:
    """Select OpenAI CLIP ViT visual LayerNorm affine parameters for TTA."""

    import torch.nn as nn

    clip = getattr(model, "clip", None)
    visual = getattr(clip, "visual", None)
    if visual is None:
        raise TypeError("LayerNorm adaptation requires an OpenAI CLIP visual tower")

    blocks = tuple(getattr(getattr(visual, "transformer", None), "resblocks", ()))
    excluded_blocks = set(
        range(max(0, len(blocks) - exclude_last_blocks), len(blocks))
    )
    parameters: list[Any] = []
    names: list[str] = []
    seen: set[int] = set()
    for module_name, module in visual.named_modules():
        if not isinstance(module, nn.LayerNorm):
            continue
        if exclude_output_norm and module_name == "ln_post":
            continue
        parts = module_name.split(".")
        if (
            len(parts) >= 3
            and parts[0] == "transformer"
            and parts[1] == "resblocks"
            and parts[2].isdigit()
            and int(parts[2]) in excluded_blocks
        ):
            continue
        module.requires_grad_(True)
        for parameter_name, parameter in module.named_parameters(recurse=False):
            if parameter_name not in {"weight", "bias"} or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            parameters.append(parameter)
            names.append(f"clip.visual.{module_name}.{parameter_name}")
    if not parameters:
        raise RuntimeError("No CLIP visual LayerNorm affine parameters were found")
    return parameters, names


def model_input_normalization(model: Any) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the model's input normalization for pixel-space augmentations."""

    mean = tuple(float(value) for value in getattr(model, "input_mean", IMAGENET_MEAN))
    std = tuple(float(value) for value in getattr(model, "input_std", IMAGENET_STD))
    if len(mean) != 3 or len(std) != 3:
        raise ValueError("Input normalization must provide three mean and std values")
    return mean, std


def build_optimizer(parameters: Iterable[Any], config: dict[str, Any]) -> Any:
    import torch.optim as optim

    name = str(config.get("optimizer", "adam")).lower()
    learning_rate = float(config.get("lr", config.get("learning_rate", 1e-4)))
    weight_decay = float(config.get("wd", config.get("weight_decay", 0.0)))
    beta1 = float(config.get("beta1", config.get("beta", 0.9)))
    beta2 = float(config.get("beta2", 0.999))
    if name == "adam":
        return optim.Adam(
            parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return optim.AdamW(
            parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            weight_decay=weight_decay,
        )
    if name == "sgd":
        return optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=float(config.get("momentum", 0.9)),
            dampening=float(config.get("dampening", 0.0)),
            weight_decay=weight_decay,
            nesterov=bool(config.get("nesterov", False)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def soft_cross_entropy(logits: Any, target_probabilities: Any) -> Any:
    return -(target_probabilities * logits.log_softmax(dim=1)).sum(dim=1).mean()


def binary_entropy_margin(default_fraction: float = 0.8) -> float:
    return math.log(2.0) * default_fraction
