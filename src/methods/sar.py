"""SAR wrapper for CLIP ViT visual LayerNorm test-time adaptation."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any

from src.types import AdaptationStats

from .base import TTAMethod
from .utils import select_clip_visual_norm_parameters


class SAR(TTAMethod):
    """Sharpness-aware reliable entropy minimization on a CLIP visual tower."""

    protocol_name = "predict_then_adapt"

    def __init__(
        self, model: Any, device: Any, config: dict[str, Any] | None = None
    ) -> None:
        import torch

        config = dict(config or {})
        config.setdefault("capture_initial_state", False)
        super().__init__(model, device, config)
        from src.official import sar as official_sar

        self.model.train()
        self.model.requires_grad_(False)
        self.official_module = official_sar
        self._adaptation_parameters, self.official_parameter_names = (
            select_clip_visual_norm_parameters(
                self.model,
                exclude_last_blocks=int(
                    self.config.get("exclude_last_visual_blocks", 3)
                ),
                exclude_output_norm=bool(
                    self.config.get("exclude_visual_output_norm", True)
                ),
            )
        )
        self.optimizer = official_sar.SAM(
            self._adaptation_parameters,
            torch.optim.SGD,
            lr=float(self.config.get("lr", 0.00025)),
            momentum=float(self.config.get("momentum", 0.9)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
            rho=float(self.config.get("rho", 0.05)),
            adaptive=bool(self.config.get("adaptive", False)),
        )
        self._initial_adaptation_state = [
            parameter.detach().clone() for parameter in self._adaptation_parameters
        ]
        self._initial_optimizer_state = deepcopy(self.optimizer.state_dict())
        self.margin = float(self.config.get("margin", 0.4 * math.log(2.0)))
        self.reset_constant = float(self.config.get("reset_constant", 0.2))
        self.steps = int(self.config.get("steps", 1))
        if self.steps < 1:
            raise ValueError("SAR steps must be positive")
        self.ema: float | None = None

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "vendored_official_core_with_vitl14_protocol_wrapper",
            "official_commit": "20f6e24b17525f34503510afccedc0629b67b7c4",
            "official_core": "src.official.sar",
            "protocol_wrapper": self.protocol_name,
            "intentional_changes": [
                "the official ViT LayerNorm path is mapped to OpenAI CLIP's visual tower",
                "the final three CLIP visual residual blocks and ln_post follow the "
                "published ViT exclusion intent",
                "an empty reliable-set guard restores SAM parameters instead of "
                "propagating a NaN update",
                "the framework owns Arrow I/O and evaluator serialization",
            ],
        }

    def adapt(self, images: Any) -> AdaptationStats:
        device_images = images.to(self.device, non_blocking=True)
        losses: list[float] = []
        first_selected = 0
        second_selected = 0
        recovered = False
        for _ in range(self.steps):
            (
                _outputs,
                self.ema,
                reset,
                first_count,
                second_count,
                loss,
            ) = self.official_module.forward_and_adapt_sar(
                device_images,
                self.model,
                self.optimizer,
                margin=self.margin,
                reset_constant=self.reset_constant,
                ema=self.ema,
            )
            first_selected += first_count
            second_selected += second_count
            if loss is not None:
                losses.append(loss)
            if reset:
                self.reset()
                recovered = True
                break
        return AdaptationStats(
            loss=(sum(losses) / len(losses)) if losses else None,
            selected=second_selected,
            extra={
                "steps": self.steps,
                "first_reliable": first_selected,
                "second_reliable": second_selected,
                "ema": self.ema,
                "model_recovered": recovered,
            },
        )

    def reset(self) -> None:
        import torch

        with torch.no_grad():
            for parameter, initial in zip(
                self._adaptation_parameters, self._initial_adaptation_state, strict=True
            ):
                parameter.copy_(initial)
        self.optimizer.load_state_dict(deepcopy(self._initial_optimizer_state))
        self.ema = None
