"""Closed-set CLIPTTA loss retained from the authors' public code.

Derived from MarcLafon/cliptta commit
ef0e6797f7618959ca85be36816a5e01299a522f (`ttavlm/methods/cliptta_otsu.py`
and `ttavlm/lib/softmax_entropy.py`). The upstream repository did not declare
a repository-wide software license at import time.
"""

from __future__ import annotations

from typing import Any


def softmax_entropy(logits: Any, eps: float = 1e-6) -> Any:
    return -((logits + eps).softmax(1) * (logits + eps).log_softmax(1)).sum(1)


def softmax_mean_entropy(logits: Any, eps: float = 1e-6) -> Any:
    probabilities = (logits + eps).softmax(1).mean(0)
    return -(probabilities * (probabilities + eps).log()).sum()


def cliptta_loss(
    image_features: Any,
    class_prototypes: Any,
    *,
    logit_scale: float,
    beta_tta: float,
    beta_reg: float,
    use_softmax_entropy: bool,
    use_tent: bool,
) -> tuple[Any, dict[str, Any]]:
    """Compute CLIPTTA's closed-set pre-training-consistency objective."""

    import torch
    import torch.nn.functional as functional

    similarity = image_features @ class_prototypes.t()
    pseudo_labels = similarity.topk(1, 1, True, True)[1][:, 0]
    pseudo_text_features = class_prototypes[pseudo_labels]
    logits_per_image = logit_scale * image_features @ pseudo_text_features.t()
    logits_per_text = logits_per_image

    if use_tent:
        tta_loss = softmax_entropy(logit_scale * similarity).mean(0)
    elif use_softmax_entropy:
        tta_loss = (
            softmax_entropy(logits_per_image).mean(0)
            + softmax_entropy(logits_per_text).mean(0)
        ) / 2
    else:
        targets = torch.eye(logits_per_image.shape[0], device=logits_per_image.device)
        tta_loss = (
            functional.cross_entropy(logits_per_image, targets, reduction="none").mean(0)
            + functional.cross_entropy(logits_per_text, targets, reduction="none").mean(0)
        ) / 2
    regularization = softmax_mean_entropy(logit_scale * similarity)
    loss = beta_tta * tta_loss - beta_reg * regularization
    return loss, {"tta_loss": tta_loss, "regularization_loss": regularization}
