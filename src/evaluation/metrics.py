from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def binary_metrics(
    labels: Any,
    prob_fake: Any,
    *,
    threshold: float = 0.5,
    calibration_bins: int = 15,
) -> dict[str, float | int]:
    labels_array = to_numpy(labels).astype(np.int64).reshape(-1)
    probability_array = to_numpy(prob_fake).astype(np.float64).reshape(-1)
    if labels_array.shape != probability_array.shape:
        raise ValueError(
            f"Shape mismatch: labels={labels_array.shape}, probabilities={probability_array.shape}"
        )
    predictions = (probability_array >= threshold).astype(np.int64)
    accuracy = float(accuracy_score(labels_array, predictions))
    real_mask = labels_array == 0
    fake_mask = labels_array == 1
    real_accuracy = (
        float(np.mean(predictions[real_mask] == 0)) if np.any(real_mask) else math.nan
    )
    fake_accuracy = (
        float(np.mean(predictions[fake_mask] == 1)) if np.any(fake_mask) else math.nan
    )
    clipped = np.clip(probability_array, 1e-7, 1.0 - 1e-7)
    nll_terms = labels_array * np.log(clipped) + (1 - labels_array) * np.log(
        1 - clipped
    )
    nll = float(-np.mean(nll_terms))
    if np.unique(labels_array).size < 2:
        auc = math.nan
        average_precision = math.nan
        balanced_accuracy = accuracy
    else:
        auc = float(roc_auc_score(labels_array, probability_array))
        average_precision = float(
            average_precision_score(labels_array, probability_array)
        )
        balanced_accuracy = float(balanced_accuracy_score(labels_array, predictions))
    return {
        "auc": auc,
        "average_precision": average_precision,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "real_accuracy": real_accuracy,
        "fake_accuracy": fake_accuracy,
        "predicted_fake_rate": float(np.mean(predictions)),
        "brier_score": float(np.mean((probability_array - labels_array) ** 2)),
        "nll": nll,
        "ece": expected_calibration_error(
            labels_array, probability_array, bins=calibration_bins
        ),
        "samples": int(labels_array.size),
    }


def expected_calibration_error(
    labels: Any, probabilities: Any, *, bins: int = 15
) -> float:
    """Return equal-width binary ECE for the predicted fake probability."""
    if bins < 1:
        raise ValueError("bins must be positive")
    labels_array = to_numpy(labels).astype(np.float64).reshape(-1)
    probability_array = to_numpy(probabilities).astype(np.float64).reshape(-1)
    if labels_array.shape != probability_array.shape:
        raise ValueError(
            "Labels and probabilities must contain the same number of samples"
        )
    if labels_array.size == 0:
        return math.nan

    boundaries = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.minimum(
        np.searchsorted(boundaries, probability_array, side="right") - 1,
        bins - 1,
    )
    bin_ids = np.maximum(bin_ids, 0)
    error = 0.0
    for bin_index in range(bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        error += float(np.mean(mask)) * abs(
            float(np.mean(probability_array[mask])) - float(np.mean(labels_array[mask]))
        )
    return error


class MetricAccumulator:
    def __init__(self, *, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.labels: list[np.ndarray] = []
        self.probabilities: list[np.ndarray] = []
        self.domains: list[str] = []
        self.batch_sizes: list[int] = []

    def update(self, labels: Any, prob_fake: Any, domain: str) -> None:
        label_array = to_numpy(labels).astype(np.int64).reshape(-1)
        probability_array = to_numpy(prob_fake).astype(np.float64).reshape(-1)
        if label_array.shape != probability_array.shape:
            raise ValueError("Labels and predictions must contain the same number of samples")
        self.labels.append(label_array)
        self.probabilities.append(probability_array)
        self.domains.append(domain)
        self.batch_sizes.append(int(label_array.size))

    def summary(self) -> dict[str, Any]:
        if not self.labels:
            return {"overall": {}, "by_domain": {}}
        overall = binary_metrics(
            np.concatenate(self.labels),
            np.concatenate(self.probabilities),
            threshold=self.threshold,
        )
        grouped_labels: dict[str, list[np.ndarray]] = defaultdict(list)
        grouped_probabilities: dict[str, list[np.ndarray]] = defaultdict(list)
        for labels, probabilities, domain in zip(
            self.labels, self.probabilities, self.domains, strict=True
        ):
            grouped_labels[domain].append(labels)
            grouped_probabilities[domain].append(probabilities)
        by_domain = {
            domain: binary_metrics(
                np.concatenate(grouped_labels[domain]),
                np.concatenate(grouped_probabilities[domain]),
                threshold=self.threshold,
            )
            for domain in grouped_labels
        }
        return {"overall": overall, "by_domain": by_domain}

    def sliding_curve(self, window_batches: int = 20) -> list[dict[str, Any]]:
        curve = []
        for index in range(len(self.labels)):
            start = max(0, index + 1 - window_batches)
            metrics = binary_metrics(
                np.concatenate(self.labels[start : index + 1]),
                np.concatenate(self.probabilities[start : index + 1]),
                threshold=self.threshold,
            )
            curve.append(
                {
                    "batch": index,
                    "domain": self.domains[index],
                    **metrics,
                }
            )
        return curve


def continual_forgetting(
    checkpoint_metrics: list[dict[str, Any]], domain_order: list[str]
) -> dict[str, Any]:
    """Compute forgetting from repeated evaluation on fixed domain holdouts."""
    if not checkpoint_metrics:
        return {"by_domain": {}, "average": math.nan}

    final_by_domain = checkpoint_metrics[-1].get("by_domain", {})
    by_domain: dict[str, dict[str, float]] = {}
    for introduced_at, domain in enumerate(domain_order):
        if domain not in final_by_domain:
            continue
        final_auc = float(final_by_domain[domain]["auc"])
        history = []
        for checkpoint in checkpoint_metrics[introduced_at:]:
            metrics = checkpoint.get("by_domain", {}).get(domain)
            if metrics is None:
                continue
            auc = float(metrics["auc"])
            if math.isfinite(auc):
                history.append(auc)
        forgetting = (
            max(history) - final_auc
            if history and math.isfinite(final_auc)
            else math.nan
        )
        by_domain[domain] = {
            "best_auc": max(history) if history else math.nan,
            "final_auc": final_auc,
            "forgetting": forgetting,
        }

    # The last domain has no later adaptation step in which it could be
    # forgotten, so standard average forgetting excludes it.
    average_domains = set(domain_order[:-1])
    valid = [
        metrics["forgetting"]
        for domain, metrics in by_domain.items()
        if domain in average_domains and math.isfinite(metrics["forgetting"])
    ]
    return {
        "by_domain": by_domain,
        "average": float(np.mean(valid)) if valid else math.nan,
    }


def temporal_generalization(
    initial_metrics: dict[str, Any],
    checkpoint_metrics: list[dict[str, Any]],
    domain_order: list[str],
) -> dict[str, Any]:
    """Separate current adaptation from transfer to future generators.

    Checkpoint ``t`` is recorded after adapting to ``domain_order[t]``. Domains
    after ``t`` are therefore still unseen by the online adaptation stream.
    """
    initial_by_domain = initial_metrics.get("by_domain", {})
    current_by_domain: dict[str, dict[str, float]] = {}
    future_by_checkpoint: list[dict[str, Any]] = []
    current_deltas: list[float] = []
    future_deltas: list[float] = []

    for checkpoint_index, checkpoint in enumerate(checkpoint_metrics):
        if checkpoint_index >= len(domain_order):
            break
        checkpoint_by_domain = checkpoint.get("by_domain", {})
        current_domain = domain_order[checkpoint_index]
        current_initial = initial_by_domain.get(current_domain)
        current_metrics = checkpoint_by_domain.get(current_domain)
        if current_initial is not None and current_metrics is not None:
            current_delta = float(current_metrics["auc"]) - float(
                current_initial["auc"]
            )
            if math.isfinite(current_delta):
                current_deltas.append(current_delta)
            current_by_domain[current_domain] = {
                "initial_auc": float(current_initial["auc"]),
                "current_auc": float(current_metrics["auc"]),
                "auc_delta": current_delta,
            }

        checkpoint_future_deltas = []
        for future_domain in domain_order[checkpoint_index + 1 :]:
            initial = initial_by_domain.get(future_domain)
            future = checkpoint_by_domain.get(future_domain)
            if initial is None or future is None:
                continue
            delta = float(future["auc"]) - float(initial["auc"])
            if math.isfinite(delta):
                checkpoint_future_deltas.append(delta)
                future_deltas.append(delta)

        future_by_checkpoint.append(
            {
                "checkpoint": checkpoint_index,
                "after_domain": current_domain,
                "future_domains": len(checkpoint_future_deltas),
                "mean_auc_delta": (
                    float(np.mean(checkpoint_future_deltas))
                    if checkpoint_future_deltas
                    else math.nan
                ),
                "negative_transfer_rate": (
                    float(np.mean(np.asarray(checkpoint_future_deltas) < 0.0))
                    if checkpoint_future_deltas
                    else math.nan
                ),
            }
        )

    return {
        "definition": "checkpoint_auc_minus_initial_auc_on_fixed_holdouts",
        "current_by_domain": current_by_domain,
        "mean_current_auc_delta": (
            float(np.mean(current_deltas)) if current_deltas else math.nan
        ),
        "future_by_checkpoint": future_by_checkpoint,
        "mean_future_auc_delta": (
            float(np.mean(future_deltas)) if future_deltas else math.nan
        ),
        "future_negative_transfer_rate": (
            float(np.mean(np.asarray(future_deltas) < 0.0))
            if future_deltas
            else math.nan
        ),
    }
