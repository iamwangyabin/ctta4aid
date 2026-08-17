"""LayerNorm Tent baseline using the ViT pathway retained by the SAR release."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.types import AdaptationStats

from .base import TTAMethod
from .utils import build_optimizer


def select_clip_visual_norm_parameters(
    model: Any,
    *,
    exclude_last_blocks: int = 0,
    exclude_output_norm: bool = False,
) -> tuple[list[Any], list[str]]:
    """Select visual LayerNorm affine parameters for CLIP ViT adaptation."""

    import torch.nn as nn

    clip = getattr(model, "clip", None)
    visual = getattr(clip, "visual", None)
    if visual is None:
        raise TypeError("LayerNorm adaptation requires a CLIP visual tower")

    blocks = getattr(getattr(visual, "transformer", None), "resblocks", ())
    excluded_blocks = set(
        range(max(0, len(blocks) - exclude_last_blocks), len(blocks))
    )
    parameters: list[Any] = []
    names: list[str] = []
    seen: set[int] = set()
    norm_types = (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)
    for module_name, module in visual.named_modules():
        if not isinstance(module, norm_types):
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
        raise RuntimeError("No CLIP visual normalization affine parameters were found")
    return parameters, names


class TentLayerNorm(TTAMethod):
    """Tent entropy minimization on CLIP ViT visual LayerNorm affine parameters."""

    protocol_name = "predict_then_adapt"

    def __init__(
        self, model: Any, device: Any, config: dict[str, Any] | None = None
    ) -> None:
        config = dict(config or {})
        config.setdefault("capture_initial_state", False)
        super().__init__(model, device, config)
        from src.official import sar as official_sar

        self.model.train()
        self.model.requires_grad_(False)
        self.official_module = official_sar
        self._adaptation_parameters, self.official_parameter_names = (
            select_clip_visual_norm_parameters(self.model)
        )
        self.optimizer = build_optimizer(self._adaptation_parameters, self.config)
        self._initial_adaptation_state = [
            parameter.detach().clone() for parameter in self._adaptation_parameters
        ]
        self._initial_optimizer_state = deepcopy(self.optimizer.state_dict())
        self.steps = int(self.config.get("steps", 1))
        if self.steps < 1:
            raise ValueError("Tent-LN steps must be positive")

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "vendored_public_layernorm_path_with_framework_wrapper",
            "official_commit": "20f6e24b17525f34503510afccedc0629b67b7c4",
            "official_core": "src.official.sar",
            "protocol_wrapper": self.protocol_name,
            "intentional_changes": [
                "uses the SAR release's LayerNorm-capable Tent path rather than "
                "claiming a BatchNorm-only Tent replication",
                "selection is scoped to the shared CLIP visual tower",
                "the framework owns Arrow I/O and evaluator serialization",
            ],
        }

    def adapt(self, images: Any) -> AdaptationStats:
        import torch

        device_images = images.to(self.device, non_blocking=True)
        losses: list[float] = []
        for _ in range(self.steps):
            logits = self.model(device_images)
            loss = self.official_module.softmax_entropy(logits).mean()
            if not torch.isfinite(loss):
                self.optimizer.zero_grad(set_to_none=True)
                continue
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.detach().item()))
        return AdaptationStats(
            loss=(sum(losses) / len(losses)) if losses else None,
            selected=int(device_images.shape[0]) if losses else 0,
            extra={"steps": self.steps, "updated": bool(losses)},
        )

    def reset(self) -> None:
        import torch

        with torch.no_grad():
            for parameter, initial in zip(
                self._adaptation_parameters, self._initial_adaptation_state, strict=True
            ):
                parameter.copy_(initial)
        self.optimizer.load_state_dict(deepcopy(self._initial_optimizer_state))
