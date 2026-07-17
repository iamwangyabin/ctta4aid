from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def binary_metrics(
    labels: Any, prob_fake: Any, *, threshold: float = 0.5
) -> dict[str, float | int]:
    labels_array = to_numpy(labels).astype(np.int64).reshape(-1)
    probability_array = to_numpy(prob_fake).astype(np.float64).reshape(-1)
    if labels_array.shape != probability_array.shape:
        raise ValueError(
            f"Shape mismatch: labels={labels_array.shape}, probabilities={probability_array.shape}"
        )
    predictions = (probability_array >= threshold).astype(np.int64)
    accuracy = float(accuracy_score(labels_array, predictions))
    if np.unique(labels_array).size < 2:
        auc = math.nan
        balanced_accuracy = accuracy
    else:
        auc = float(roc_auc_score(labels_array, probability_array))
        balanced_accuracy = float(balanced_accuracy_score(labels_array, predictions))
    return {
        "auc": auc,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "samples": int(labels_array.size),
    }


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
