from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from src.types import AdaptationStats, PredictionBatch


class TTAMethod(ABC):
    """Common stateful interface for predict-then-adapt experiments."""

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config or {}
        self.optimizer: Any = None
        self._initial_model_state = (
            deepcopy(self.model.state_dict())
            if bool(self.config.get("capture_initial_state", True))
            else None
        )
        self._initial_optimizer_state: dict[str, Any] | None = None

    def _capture_optimizer_state(self) -> None:
        if self.optimizer is not None:
            self._initial_optimizer_state = deepcopy(self.optimizer.state_dict())

    def _prediction_model(self) -> Any:
        return self.model

    def predict(self, images: Any) -> PredictionBatch:
        import torch

        model = self._prediction_model()
        with torch.no_grad():
            logits = model(images.to(self.device, non_blocking=True))
            probabilities = logits.softmax(dim=1)
        return PredictionBatch(
            logits=logits.detach(),
            prob_fake=probabilities[:, 1].detach(),
            pred_label=probabilities.argmax(dim=1).detach(),
        )

    @abstractmethod
    def adapt(self, images: Any) -> AdaptationStats:
        """Update from images only. Ground-truth labels must never enter here."""

    def reset(self) -> None:
        if self._initial_model_state is None:
            raise RuntimeError("This method was configured without an initial model snapshot")
        self.model.load_state_dict(deepcopy(self._initial_model_state))
        if self.optimizer is not None and self._initial_optimizer_state is not None:
            self.optimizer.load_state_dict(deepcopy(self._initial_optimizer_state))

    def discard_pending_prediction(self) -> None:
        """Clear prediction-only caches when no matching adapt call will follow."""

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "project_implementation",
            "protocol_wrapper": "predict_then_adapt",
            "intentional_changes": [],
        }
