"""Pinned BATCLIP loss core adapted from the authors' public implementation.

Vendored from sarthaxxxxx/BATCLIP commit
ba2e3381873ef58e76a90148ee3835864349e985 (`classification/methods/ours.py`
and `classification/utils/losses.py`). The repository declares the MIT License.
"""

from __future__ import annotations

from typing import Any


def entropy(logits: Any) -> Any:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


def image_to_text_loss(logits: Any, image_features: Any, text_features: Any) -> Any:
    """BATCLIP's pseudo-class image-to-text alignment objective."""

    import torch

    labels = torch.argmax(logits.softmax(1), dim=1)
    loss = 0.0
    unique_labels = torch.unique(labels, sorted=True)
    for label in unique_labels.tolist():
        mean_features = image_features[labels == label].mean(0).type(text_features.dtype)
        distance = torch.matmul(
            mean_features.unsqueeze(0), text_features[label].unsqueeze(0).t()
        ).mean()
        loss += distance
    return loss / len(unique_labels)


def inter_mean_loss(logits: Any, image_features: Any) -> Any:
    """BATCLIP's inter-pseudo-class feature separation objective."""

    import torch

    labels = torch.argmax(logits.softmax(1), dim=1)
    means = []
    for label in torch.unique(labels, sorted=True).tolist():
        mean = image_features[labels == label].mean(0)
        means.append(mean / mean.norm())
    similarity = torch.matmul(torch.stack(means), torch.stack(means).t())
    loss = 1 - similarity
    loss.fill_diagonal_(0)
    return loss.sum()


def batclip_loss(
    logits: Any, image_features: Any, text_features: Any
) -> tuple[Any, dict[str, Any]]:
    """Return the exact objective assembled by BATCLIP's `OURS` method."""

    entropy_loss = entropy(logits).mean(0)
    image_text = image_to_text_loss(logits, image_features, text_features)
    separation = inter_mean_loss(logits, image_features)
    return (
        entropy_loss - image_text - separation,
        {
            "entropy_loss": entropy_loss,
            "image_to_text_loss": image_text,
            "inter_mean_loss": separation,
        },
    )
