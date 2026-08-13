from __future__ import annotations

from typing import Any

from src.types import AdaptationStats

from .base import TTAMethod


class SourceOnly(TTAMethod):
    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(model, device, config)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def adapt(self, images: Any) -> AdaptationStats:
        return AdaptationStats(selected=0)

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "common_control",
            "protocol_wrapper": "predict_only",
            "intentional_changes": [],
        }
