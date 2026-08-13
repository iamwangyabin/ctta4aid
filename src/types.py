from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PredictionBatch:
    """Prediction returned by every TTA method.

    Tensor-like types are intentionally kept as ``Any`` so the light-weight
    metric and protocol tests can run without importing PyTorch.
    """

    logits: Any
    prob_fake: Any
    pred_label: Any


@dataclass(frozen=True)
class StreamBatch:
    """One deployment batch; labels remain hidden from adaptation methods."""

    images: Any
    hidden_labels: Any
    domain: str
    sample_ids: list[str] = field(default_factory=list)


@dataclass
class AdaptationStats:
    loss: float | None = None
    selected: int | None = None
    elapsed_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
