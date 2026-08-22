"""Causal asymmetric GMM calibration over a frozen detector's scores.

The lowest-mean target component represents the compact real distribution;
every higher-mean component belongs to the heterogeneous fake distribution.
BIC decides whether the arrived unlabeled scores support one or more total
components. A one-component fit is treated as insufficient evidence, so the
method falls back exactly to the source detector until a split is supported.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.types import AdaptationStats, PredictionBatch

from .ascal import binary_score, fit_gmm_bic, validate_score_anchors
from .base import TTAMethod


def asymmetric_fake_posterior(scores: Any, mixture: dict[str, Any]) -> Any:
    """Sum responsibilities of every component except the lowest-mean one."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    weights = np.asarray(mixture["weights"], dtype=np.float64).reshape(-1)
    mus = np.asarray(mixture["mus"], dtype=np.float64).reshape(-1)
    sigmas = np.asarray(mixture["sigmas"], dtype=np.float64).reshape(-1)
    if weights.size < 2 or not (weights.size == mus.size == sigmas.size):
        raise ValueError("Asymmetric fake posterior requires at least two components")
    if np.any(weights <= 0.0) or np.any(sigmas <= 0.0):
        raise ValueError("Mixture weights and sigmas must be positive")
    if np.any(np.diff(mus) < 0.0):
        raise ValueError("Mixture components must be sorted by mean")

    z = (values[:, None] - mus[None, :]) / sigmas[None, :]
    log_joint = (
        np.log(weights)[None, :]
        - np.log(sigmas)[None, :]
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * z * z
    )
    log_joint -= np.max(log_joint, axis=1, keepdims=True)
    joint = np.exp(log_joint)
    return joint[:, 1:].sum(axis=1) / joint.sum(axis=1)


class ASCALGMM(TTAMethod):
    """Unlabeled online calibration with one real mode and BIC-selected fake modes."""

    protocol_name = "predict_then_adapt"

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        config.setdefault("capture_initial_state", False)
        super().__init__(model, device, config)
        self.model.eval()
        self.model.requires_grad_(False)

        self.adaptation_mode = str(self.config.get("adaptation_mode", "full")).lower()
        if self.adaptation_mode not in {"static", "full"}:
            raise ValueError("ASCAL-GMM adaptation_mode must be static or full")

        self.anchors = validate_score_anchors(self.config.get("score_anchors"))
        self.temperature = float(self.config.get("temperature", self.anchors["temperature"]))
        if self.temperature <= 0.0:
            raise ValueError("ASCAL-GMM temperature must be positive")

        source_fake_mean = float(
            np.dot(self.anchors["fake"]["weights"], self.anchors["fake"]["mus"])
        )
        if source_fake_mean <= float(self.anchors["real"]["mu"]):
            raise ValueError("ASCAL-GMM requires source scores to increase toward fake")

        # The source fit supplies only a target-independent complexity ceiling.
        # BIC still chooses the target component count from unlabeled target scores.
        self.max_fake_components = len(self.anchors["fake"]["mus"])
        self.max_total_components = 1 + self.max_fake_components
        self._pending: dict[str, Any] | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        self.score_history: list[float] = []
        self._mixture: dict[str, Any] | None = None
        self._refits = 0
        self._fit_failures = 0
        self._pending = None

    @property
    def trainable_parameters(self) -> int:
        return 0

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "project_method_over_lora_source_detector",
            "protocol_wrapper": self.protocol_name,
            "adaptation_mode": self.adaptation_mode,
            "adaptive_role": "unlabeled_score_mixture_calibration_only",
            "score_orientation": "higher_source_margin_means_fake",
            "component_rule": (
                "lowest_mean_component_is_real_and_all_higher_components_are_fake"
            ),
            "selection_rule": "bic_over_all_causally_arrived_target_scores",
            "source_anchor_role": (
                "temperature_fallback_orientation_and_target_independent_component_cap_only"
            ),
            "max_fake_components": self.max_fake_components,
            "intentional_changes": [
                "the detector stays frozen during deployment",
                "all arrived scores enter the fit without pseudo-label admission",
                "one selected component means insufficient evidence and exact source fallback",
                "predictions use only the mixture fitted after earlier batches",
            ],
        }

    def _batch_scores(self, images: Any) -> Any:
        import torch

        if images.dim() == 5:
            batch, views = int(images.shape[0]), int(images.shape[1])
            flat = images.reshape(batch * views, *images.shape[2:])
        elif images.dim() == 4:
            batch, views = int(images.shape[0]), 1
            flat = images
        else:
            raise ValueError(
                "ASCAL-GMM expects (B, C, H, W) or (B, V, C, H, W) images"
            )
        with torch.no_grad():
            logits = self.model(flat.to(self.device, non_blocking=True))
        margins = (
            binary_score(logits)
            .view(batch, views)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        return margins.mean(axis=1)

    def _source_probability(self, scores: Any) -> Any:
        scaled = np.asarray(scores, dtype=np.float64) / self.temperature
        return 1.0 / (1.0 + np.exp(-np.clip(scaled, -60.0, 60.0)))

    def _mixture_active(self) -> bool:
        return self._mixture is not None and int(self._mixture["components"]) >= 2

    def predict(self, images: Any) -> PredictionBatch:
        import torch

        scores = self._batch_scores(images)
        if self.adaptation_mode == "full" and self._mixture_active():
            probability = asymmetric_fake_posterior(scores, self._mixture)
            prediction_mode = "target_mixture"
        else:
            probability = self._source_probability(scores)
            prediction_mode = "source_fallback"
        probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
        logit_margin = np.log(probability / (1.0 - probability))
        logits = torch.from_numpy(
            np.stack([-0.5 * logit_margin, 0.5 * logit_margin], axis=1)
        ).float()
        prob_fake = torch.from_numpy(probability).float()
        self._pending = {"scores": scores, "prediction_mode": prediction_mode}
        return PredictionBatch(
            logits=logits,
            prob_fake=prob_fake,
            pred_label=(prob_fake >= 0.5).long(),
        )

    def discard_pending_prediction(self) -> None:
        self._pending = None

    def _state_stats(self) -> dict[str, Any]:
        mixture = self._mixture
        components = int(mixture["components"]) if mixture is not None else 0
        return {
            "score_samples": len(self.score_history),
            "mixture_active": components >= 2,
            "total_components": components,
            "fake_components": max(0, components - 1),
            "component_means": [] if mixture is None else list(mixture["mus"]),
            "component_weights": [] if mixture is None else list(mixture["weights"]),
            "bic": None if mixture is None else float(mixture["bic"]),
            "refits": self._refits,
            "fit_failures": self._fit_failures,
        }

    def adapt(self, images: Any) -> AdaptationStats:
        del images
        if self.adaptation_mode == "static":
            self._pending = None
            return AdaptationStats(
                selected=0,
                extra={"adaptation_mode": "static", **self._state_stats()},
            )
        if self._pending is None:
            raise RuntimeError("ASCAL-GMM adapt requires a matching predict call")

        scores = np.asarray(self._pending["scores"], dtype=np.float64).reshape(-1)
        prediction_mode = str(self._pending["prediction_mode"])
        self._pending = None
        self.score_history.extend(float(score) for score in scores)

        fit_error = None
        if len(self.score_history) >= 2:
            try:
                self._mixture = fit_gmm_bic(
                    self.score_history,
                    max_components=min(
                        self.max_total_components, len(self.score_history)
                    ),
                    seed=0,
                )
                self._refits += 1
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                self._fit_failures += 1
                fit_error = f"{type(exc).__name__}: {exc}"

        extra = {
            "adaptation_mode": "full",
            "prediction_mode": prediction_mode,
            **self._state_stats(),
        }
        if fit_error is not None:
            extra["fit_error"] = fit_error
        return AdaptationStats(loss=None, selected=int(scores.size), extra=extra)

    def reset(self) -> None:
        self._reset_state()


__all__ = ["ASCALGMM", "asymmetric_fake_posterior"]
