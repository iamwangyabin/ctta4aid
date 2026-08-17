"""Pinned TDA cache core adapted from the authors' public implementation.

Vendored from kdiAAA/TDA commit e697fb0c8078cdeff93daa56bcf8860702542069
(`tda_runner.py`). Copyright (c) 2024 Adilbek Karmanov. MIT License.

Only the cache update and cache-logit routines are retained here. The project
wrapper supplies Arrow streams, binary text prototypes, device placement, and
metric serialization.
"""

from __future__ import annotations

import math
from typing import Any


def avg_entropy(outputs: Any) -> Any:
    """Author implementation of entropy after averaging normalized logits."""

    import torch

    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    averaged = logits.logsumexp(dim=0) - math.log(logits.shape[0])
    minimum = torch.finfo(averaged.dtype).min
    averaged = torch.clamp(averaged, min=minimum)
    return -(averaged * torch.exp(averaged)).sum(dim=-1)


def process_features(logits: Any, image_features: Any) -> tuple[Any, Any, Any, Any, int, int]:
    """Apply TDA's confidence selection for the current online batch."""

    import torch

    if image_features.size(0) > 1:
        batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        selected_count = max(1, int(batch_entropy.size(0) * 0.1))
        selected_idx = torch.argsort(batch_entropy, descending=False)[:selected_count]
        selected_logits = logits[selected_idx]
        cache_feature = image_features[selected_idx].mean(0).unsqueeze(0)
        loss = avg_entropy(selected_logits)
        probability_map = selected_logits.softmax(1).mean(0).unsqueeze(0)
        prediction = int(selected_logits.mean(0).argmax().item())
    else:
        selected_count = 1
        cache_feature = image_features
        loss = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        probability_map = logits.softmax(1)
        prediction = int(logits.argmax(dim=1)[0].item())
    return cache_feature, logits, loss, probability_map, prediction, selected_count


def update_cache(
    cache: dict[int, list[tuple[Any, float, Any | None]]],
    prediction: int,
    feature: Any,
    loss: Any,
    shot_capacity: int,
    probability_map: Any | None = None,
) -> None:
    """Retain TDA's lowest-entropy positive or negative cache entries."""

    item = (
        feature.detach(),
        float(loss.detach().mean().item()),
        None if probability_map is None else probability_map.detach(),
    )
    entries = cache.setdefault(prediction, [])
    if len(entries) < shot_capacity:
        entries.append(item)
    elif item[1] < entries[-1][1]:
        entries[-1] = item
    entries.sort(key=lambda candidate: candidate[1])


def compute_cache_logits(
    image_features: Any,
    cache: dict[int, list[tuple[Any, float, Any | None]]],
    *,
    alpha: float,
    beta: float,
    classes: int,
    negative_mask_thresholds: tuple[float, float] | None = None,
) -> Any:
    """Compute the positive or negative feature-cache contribution from TDA."""

    import torch
    import torch.nn.functional as functional

    keys = []
    values = []
    for class_index in sorted(cache):
        for feature, _loss, probability_map in cache[class_index]:
            keys.append(feature)
            values.append(class_index if negative_mask_thresholds is None else probability_map)
    if not keys:
        return image_features.new_zeros((image_features.shape[0], classes))

    cache_keys = torch.cat(keys, dim=0).permute(1, 0).to(
        device=image_features.device, dtype=image_features.dtype
    )
    if negative_mask_thresholds is None:
        cache_values = functional.one_hot(
            torch.tensor(values, device=image_features.device, dtype=torch.long),
            num_classes=classes,
        ).to(dtype=image_features.dtype)
    else:
        lower, upper = negative_mask_thresholds
        probability_maps = torch.cat(values, dim=0).to(image_features.device)
        cache_values = ((probability_maps > lower) & (probability_maps < upper)).to(
            dtype=image_features.dtype
        )

    affinity = image_features @ cache_keys
    cache_logits = (-(beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits
