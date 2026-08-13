"""Framework wrapper around the pinned authors' TENT core."""

from __future__ import annotations

from typing import Any

from src.types import AdaptationStats

from .base import TTAMethod
from .utils import build_optimizer


class Tent(TTAMethod):
    """Expose official TENT through the common Predict-Then-Adapt interface."""

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(model, device, config)
        from src.official import tent as official_tent

        self.official_module = official_tent
        official_tent.configure_model(self.model)
        parameters, self.official_parameter_names = official_tent.collect_params(self.model)
        if not parameters:
            raise RuntimeError("Official TENT requires at least one BatchNorm2d layer")
        self.optimizer = build_optimizer(parameters, self.config)
        self.core = official_tent.Tent(
            self.model,
            self.optimizer,
            steps=int(self.config.get("steps", 1)),
            episodic=bool(self.config.get("episodic", False)),
        )

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "vendored_official_core_with_protocol_wrapper",
            "official_commit": "e9e926a668d85244c66a6d5c006efbd2b82e83e8",
            "official_core": "src.official.tent",
            "numerical_validation": "not_run_requires_official_data_and_weights",
            "protocol_wrapper": "predict_then_adapt",
            "intentional_changes": ["framework separates prediction from official adaptation"],
        }

    def adapt(self, images: Any) -> AdaptationStats:
        images = images.to(self.device, non_blocking=True)
        outputs = self.core(images)
        loss = self.official_module.softmax_entropy(outputs.detach()).mean(0)
        return AdaptationStats(
            loss=float(loss.cpu()),
            selected=int(images.shape[0]),
        )

    def reset(self) -> None:
        self.core.reset()
