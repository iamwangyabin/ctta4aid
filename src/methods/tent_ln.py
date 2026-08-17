"""LayerNorm Tent baseline using the ViT pathway retained by the SAR release."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.types import AdaptationStats

from .base import TTAMethod
from .utils import build_optimizer, select_clip_visual_norm_parameters


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
