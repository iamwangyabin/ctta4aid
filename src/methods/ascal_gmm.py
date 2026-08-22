"""Causal asymmetric GMM calibration over a frozen detector's scores.

BIC decides whether arrived unlabeled scores support one or more components.
The original readout treats the lowest component as real and sums all other
responsibilities as fake. The monotone-shift readout instead finds the largest
gap between component means, uses it as the real/fake block boundary, and only
shifts the source score. Its median-shift variant stabilizes that boundary with
the cumulative median of all causal boundary estimates. The density-shift
variant keeps the same non-semantic block partition but replaces the gap
midpoint with the equal-prior density crossing of the two normalized blocks.
The segmented variant uses BIC to forget an obsolete score segment after an
unlabeled distribution change. A one-component fit falls back to the source
detector.
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


def dominant_gap_boundary(mixture: dict[str, Any]) -> dict[str, float | int]:
    """Split sorted components at their largest mean gap and return its midpoint."""

    mus = np.asarray(mixture["mus"], dtype=np.float64).reshape(-1)
    if mus.size < 2:
        raise ValueError("Dominant-gap calibration requires at least two components")
    if np.any(np.diff(mus) < 0.0):
        raise ValueError("Mixture components must be sorted by mean")
    gaps = np.diff(mus)
    split = int(np.argmax(gaps)) + 1
    return {
        "decision_boundary": float(0.5 * (mus[split - 1] + mus[split])),
        "dominant_gap": float(gaps[split - 1]),
        "real_components": split,
        "fake_components": int(mus.size - split),
    }


def _log_mixture_density(
    value: float,
    weights: Any,
    mus: Any,
    sigmas: Any,
) -> float:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    mus = np.asarray(mus, dtype=np.float64).reshape(-1)
    sigmas = np.asarray(sigmas, dtype=np.float64).reshape(-1)
    terms = (
        np.log(weights)
        - np.log(sigmas)
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * ((float(value) - mus) / sigmas) ** 2
    )
    maximum = float(np.max(terms))
    return maximum + float(np.log(np.exp(terms - maximum).sum()))


def equal_density_boundary(mixture: dict[str, Any]) -> dict[str, Any]:
    """Find the equal-prior real/fake density crossing inside the dominant gap."""

    partition = dominant_gap_boundary(mixture)
    weights = np.asarray(mixture["weights"], dtype=np.float64).reshape(-1)
    mus = np.asarray(mixture["mus"], dtype=np.float64).reshape(-1)
    sigmas = np.asarray(mixture["sigmas"], dtype=np.float64).reshape(-1)
    if not (weights.size == mus.size == sigmas.size):
        raise ValueError("Density boundary requires matching mixture arrays")
    if np.any(weights <= 0.0) or np.any(sigmas <= 0.0):
        raise ValueError("Density boundary requires positive weights and sigmas")

    split = int(partition["real_components"])
    real_weights = weights[:split] / weights[:split].sum()
    fake_weights = weights[split:] / weights[split:].sum()

    def log_ratio(value: float) -> float:
        return _log_mixture_density(
            value,
            real_weights,
            mus[:split],
            sigmas[:split],
        ) - _log_mixture_density(
            value,
            fake_weights,
            mus[split:],
            sigmas[split:],
        )

    left = float(mus[split - 1])
    right = float(mus[split])
    midpoint = float(partition["decision_boundary"])
    grid = np.linspace(left, right, 257, dtype=np.float64)
    ratios = np.asarray([log_ratio(float(value)) for value in grid])
    exact = np.flatnonzero(np.isclose(ratios, 0.0, atol=1e-12, rtol=0.0))
    crossing_found = False
    if exact.size:
        candidates = grid[exact]
        boundary = float(candidates[np.argmin(np.abs(candidates - midpoint))])
        crossing_found = True
    else:
        changes = np.flatnonzero(np.signbit(ratios[:-1]) != np.signbit(ratios[1:]))
        if changes.size:
            index = int(
                min(
                    changes,
                    key=lambda candidate: abs(
                        float(0.5 * (grid[candidate] + grid[candidate + 1]))
                        - midpoint
                    ),
                )
            )
            lower = float(grid[index])
            upper = float(grid[index + 1])
            lower_ratio = float(ratios[index])
            for _ in range(60):
                candidate = 0.5 * (lower + upper)
                candidate_ratio = log_ratio(candidate)
                if (candidate_ratio < 0.0) == (lower_ratio < 0.0):
                    lower = candidate
                    lower_ratio = candidate_ratio
                else:
                    upper = candidate
            boundary = float(0.5 * (lower + upper))
            crossing_found = True
        else:
            boundary = midpoint

    return {
        **partition,
        "decision_boundary": boundary,
        "gap_midpoint": midpoint,
        "density_crossing": crossing_found,
        "density_log_ratio": float(log_ratio(boundary)),
    }


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

    def _prediction_batch(
        self,
        scores: Any,
        probability: Any,
        **pending_state: Any,
    ) -> PredictionBatch:
        import torch

        probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
        logit_margin = np.log(probability / (1.0 - probability))
        logits = torch.from_numpy(
            np.stack([-0.5 * logit_margin, 0.5 * logit_margin], axis=1)
        ).float()
        prob_fake = torch.from_numpy(probability).float()
        self._pending = {"scores": scores, **pending_state}
        return PredictionBatch(
            logits=logits,
            prob_fake=prob_fake,
            pred_label=(prob_fake >= 0.5).long(),
        )

    def predict(self, images: Any) -> PredictionBatch:
        scores = self._batch_scores(images)
        if self.adaptation_mode == "full" and self._mixture_active():
            probability = asymmetric_fake_posterior(scores, self._mixture)
            prediction_mode = "target_mixture"
        else:
            probability = self._source_probability(scores)
            prediction_mode = "source_fallback"
        return self._prediction_batch(
            scores,
            probability,
            prediction_mode=prediction_mode,
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

    def _after_successful_fit(self) -> None:
        """Allow readout variants to record state before adaptation is reported."""

    def _append_scores(self, scores: Any) -> None:
        self.score_history.extend(float(score) for score in scores)

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
        prediction_state = {
            key: value for key, value in self._pending.items() if key != "scores"
        }
        self._pending = None
        self._append_scores(scores)

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
                self._after_successful_fit()
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                self._fit_failures += 1
                fit_error = f"{type(exc).__name__}: {exc}"

        extra = {
            "adaptation_mode": "full",
            **prediction_state,
            **self._state_stats(),
        }
        if fit_error is not None:
            extra["fit_error"] = fit_error
        return AdaptationStats(loss=None, selected=int(scores.size), extra=extra)

    def reset(self) -> None:
        self._reset_state()


class ASCALGMMShift(ASCALGMM):
    """Use the target mixture only to shift the frozen detector's score boundary."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": "unlabeled_monotone_score_shift_only",
                "component_rule": (
                    "largest_adjacent_mean_gap_separates_the_real_and_fake_blocks"
                ),
                "prediction_rule": "sigmoid((source_margin-target_boundary)/source_temperature)",
                "ranking_rule": "source_margin_order_is_preserved_within_each_causal_state",
                "intentional_changes": [
                    "the detector stays frozen during deployment",
                    "all arrived scores enter the fit without pseudo-label admission",
                    "one selected component means insufficient evidence and exact source fallback",
                    "the target mixture changes only one additive score boundary",
                    "predictions use only the boundary fitted after earlier batches",
                ],
            }
        )
        return metadata

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        if not self._mixture_active():
            stats.update(
                {
                    "decision_boundary": None,
                    "dominant_gap": None,
                    "real_components": 0,
                    "fake_components": 0,
                }
            )
            return stats
        stats.update(self._candidate_partition())
        return stats

    def _candidate_partition(self) -> dict[str, Any]:
        if self._mixture is None:
            raise RuntimeError("ASCAL-GMM boundary requires an active mixture")
        return dominant_gap_boundary(self._mixture)

    def predict(self, images: Any) -> PredictionBatch:
        scores = self._batch_scores(images)
        if self.adaptation_mode == "full" and self._mixture_active():
            partition = self._candidate_partition()
            boundary = float(partition["decision_boundary"])
            probability = self._source_probability(scores - boundary)
            pending_state = {
                "prediction_mode": "monotone_gmm_shift",
                "prediction_boundary": boundary,
                "prediction_real_components": int(partition["real_components"]),
                "prediction_fake_components": int(partition["fake_components"]),
            }
        else:
            probability = self._source_probability(scores)
            pending_state = {
                "prediction_mode": "source_fallback",
                "prediction_boundary": 0.0,
                "prediction_real_components": 0,
                "prediction_fake_components": 0,
            }
        return self._prediction_batch(scores, probability, **pending_state)


class ASCALGMMMedianShift(ASCALGMMShift):
    """Stabilize causal GMM boundaries with their cumulative median."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.boundary_history: list[float] = []

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": "unlabeled_median_stabilized_monotone_score_shift",
                "boundary_stabilization": (
                    "cumulative_median_of_all_causal_dominant_gap_boundaries"
                ),
                "prediction_rule": (
                    "sigmoid((source_margin-causal_boundary_median)"
                    "/source_temperature)"
                ),
                "hyperparameter_rule": "no_new_stabilization_hyperparameters",
            }
        )
        return metadata

    def _after_successful_fit(self) -> None:
        if self._mixture_active():
            partition = self._candidate_partition()
            self.boundary_history.append(float(partition["decision_boundary"]))

    @property
    def _prediction_mode_name(self) -> str:
        return "median_gmm_shift"

    def _stabilized_boundary(self, candidate: float) -> float:
        if not self.boundary_history:
            return float(candidate)
        return float(np.median(np.asarray(self.boundary_history, dtype=np.float64)))

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        candidate = stats["decision_boundary"]
        historical = (
            None
            if not self.boundary_history
            else float(np.median(np.asarray(self.boundary_history, dtype=np.float64)))
        )
        stats.update(
            {
                "candidate_boundary": candidate,
                "stabilized_boundary": historical,
                "decision_boundary": (
                    None
                    if candidate is None
                    else self._stabilized_boundary(float(candidate))
                ),
                "boundary_samples": len(self.boundary_history),
            }
        )
        return stats

    def predict(self, images: Any) -> PredictionBatch:
        scores = self._batch_scores(images)
        if self.adaptation_mode == "full" and self._mixture_active():
            partition = self._candidate_partition()
            candidate = float(partition["decision_boundary"])
            boundary = self._stabilized_boundary(candidate)
            probability = self._source_probability(scores - boundary)
            pending_state = {
                "prediction_mode": self._prediction_mode_name,
                "prediction_boundary": boundary,
                "prediction_candidate_boundary": candidate,
                "prediction_real_components": int(partition["real_components"]),
                "prediction_fake_components": int(partition["fake_components"]),
            }
        else:
            probability = self._source_probability(scores)
            pending_state = {
                "prediction_mode": "source_fallback",
                "prediction_boundary": 0.0,
                "prediction_candidate_boundary": None,
                "prediction_real_components": 0,
                "prediction_fake_components": 0,
            }
        return self._prediction_batch(scores, probability, **pending_state)


class ASCALGMMDensityShift(ASCALGMMMedianShift):
    """Use a median-stabilized equal-density boundary without semantic features."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "unlabeled_median_stabilized_equal_density_score_shift"
                ),
                "component_semantics": "none_dominant_gap_partition_only",
                "boundary_rule": (
                    "equal_prior_crossing_of_normalized_real_and_fake_block_densities"
                ),
                "boundary_fallback": (
                    "dominant_gap_midpoint_when_no_crossing_exists_inside_the_gap"
                ),
                "hyperparameter_rule": "no_new_target_hyperparameters",
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "density_gmm_shift"

    def _candidate_partition(self) -> dict[str, Any]:
        if self._mixture is None:
            raise RuntimeError("ASCAL-GMM density boundary requires an active mixture")
        return equal_density_boundary(self._mixture)


def _gmm_parameter_count(mixture: dict[str, Any]) -> int:
    """Return the free-parameter count of a one-dimensional full-covariance GMM."""

    components = int(mixture["components"])
    if components < 1:
        raise ValueError("GMM parameter counting requires at least one component")
    return 3 * components - 1


def _segmented_bic(
    left: dict[str, Any],
    left_samples: int,
    right: dict[str, Any],
    right_samples: int,
) -> float:
    """Score two independent GMM segments with one BIC change-point penalty."""

    if left_samples < 2 or right_samples < 2:
        raise ValueError("Segmented BIC requires at least two samples on each side")
    total_samples = left_samples + right_samples
    left_parameters = _gmm_parameter_count(left)
    right_parameters = _gmm_parameter_count(right)
    negative_twice_log_likelihood = (
        float(left["bic"]) - left_parameters * math.log(left_samples)
        + float(right["bic"])
        - right_parameters * math.log(right_samples)
    )
    return float(
        negative_twice_log_likelihood
        + (left_parameters + right_parameters + 1) * math.log(total_samples)
    )


class ASCALGMMSegmentedShift(ASCALGMMMedianShift):
    """Reset score calibration at causal, unlabeled BIC-selected change points."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.score_batches: list[np.ndarray] = []
        self.total_score_samples = 0
        self.total_score_batches = 0
        self.segment_changes = 0
        self.segment_checks = 0
        self.segment_candidate_fits = 0
        self.segment_fit_failures = 0
        self.segment_unimodal_suffixes = 0
        self.segment_unstable_suffixes = 0
        self.segment_discarded_samples = 0
        self.last_segment_gain: float | None = None
        self.last_change_batch: int | None = None
        self.last_change_suffix_batches: int | None = None
        self._segment_changed = False

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "unlabeled_bic_segmented_median_stabilized_score_shift"
                ),
                "segmentation_rule": (
                    "binary_scheduled_bic_comparison_with_an_active_unsplit_causal_suffix"
                ),
                "change_point_penalty": (
                    "one_additional_bic_parameter_no_tuned_change_threshold"
                ),
                "segment_reset_scope": (
                    "score_mixture_and_boundary_history_only_detector_remains_frozen"
                ),
                "generator_boundaries_used": False,
                "semantic_features_used": False,
                "hyperparameter_rule": "no_new_target_hyperparameters",
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_median_gmm_shift"

    def _append_scores(self, scores: Any) -> None:
        values = np.asarray(scores, dtype=np.float64).reshape(-1)
        super()._append_scores(values)
        self.score_batches.append(values.copy())
        self.total_score_samples += int(values.size)
        self.total_score_batches += 1
        self._segment_changed = False

    def _suffix_candidates(self) -> list[int]:
        segment_batches = len(self.score_batches)
        if segment_batches < 4 or segment_batches % 2:
            return []

        largest = segment_batches // 2
        suffix_batches = segment_batches & -segment_batches
        while suffix_batches > largest:
            suffix_batches //= 2
        return [max(2, suffix_batches)]

    def _detect_segment_change(self) -> None:
        if self._mixture is None:
            return
        candidates = self._suffix_candidates()
        if not candidates:
            return

        self.segment_checks += 1
        values = np.asarray(self.score_history, dtype=np.float64)
        full_bic = float(self._mixture["bic"])
        best: dict[str, Any] | None = None
        for suffix_batches in candidates:
            suffix_samples = sum(
                int(batch.size) for batch in self.score_batches[-suffix_batches:]
            )
            left_values = values[:-suffix_samples]
            right_values = values[-suffix_samples:]
            if left_values.size < 2 or right_values.size < 2:
                continue
            self.segment_candidate_fits += 2
            try:
                left = fit_gmm_bic(
                    left_values,
                    max_components=min(self.max_total_components, left_values.size),
                    seed=0,
                )
                right = fit_gmm_bic(
                    right_values,
                    max_components=min(self.max_total_components, right_values.size),
                    seed=0,
                )
                split_bic = _segmented_bic(
                    left,
                    int(left_values.size),
                    right,
                    int(right_values.size),
                )
            except (FloatingPointError, RuntimeError, ValueError):
                self.segment_fit_failures += 1
                continue
            if int(right["components"]) < 2:
                self.segment_unimodal_suffixes += 1
                continue
            gain = full_bic - split_bic
            if gain > 0.0:
                half_batches = suffix_batches // 2
                half_samples = sum(
                    int(batch.size) for batch in self.score_batches[-half_batches:]
                )
                older_right = right_values[:-half_samples]
                newer_right = right_values[-half_samples:]
                self.segment_candidate_fits += 2
                try:
                    older_mixture = fit_gmm_bic(
                        older_right,
                        max_components=min(
                            self.max_total_components, older_right.size
                        ),
                        seed=0,
                    )
                    newer_mixture = fit_gmm_bic(
                        newer_right,
                        max_components=min(
                            self.max_total_components, newer_right.size
                        ),
                        seed=0,
                    )
                    internal_split_bic = _segmented_bic(
                        older_mixture,
                        int(older_right.size),
                        newer_mixture,
                        int(newer_right.size),
                    )
                except (FloatingPointError, RuntimeError, ValueError):
                    self.segment_fit_failures += 1
                    continue
                if internal_split_bic < float(right["bic"]):
                    self.segment_unstable_suffixes += 1
                    continue
            if best is None or gain > float(best["gain"]):
                best = {
                    "gain": float(gain),
                    "right": right,
                    "suffix_batches": suffix_batches,
                    "suffix_samples": suffix_samples,
                }

        self.last_segment_gain = None if best is None else float(best["gain"])
        if best is None or float(best["gain"]) <= 0.0:
            return

        suffix_batches = int(best["suffix_batches"])
        suffix_samples = int(best["suffix_samples"])
        discarded_samples = len(self.score_history) - suffix_samples
        self.score_batches = self.score_batches[-suffix_batches:]
        self.score_history = values[-suffix_samples:].tolist()
        self._mixture = best["right"]
        self.boundary_history = []
        self.segment_changes += 1
        self.segment_discarded_samples += discarded_samples
        self.last_change_batch = self.total_score_batches
        self.last_change_suffix_batches = suffix_batches
        self._segment_changed = True

    def _after_successful_fit(self) -> None:
        self._detect_segment_change()
        super()._after_successful_fit()

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(
            {
                "segment_changed": self._segment_changed,
                "segment_changes": self.segment_changes,
                "segment_batches": len(self.score_batches),
                "total_score_batches": self.total_score_batches,
                "total_score_samples": self.total_score_samples,
                "segment_checks": self.segment_checks,
                "segment_candidate_fits": self.segment_candidate_fits,
                "segment_fit_failures": self.segment_fit_failures,
                "segment_unimodal_suffixes": self.segment_unimodal_suffixes,
                "segment_unstable_suffixes": self.segment_unstable_suffixes,
                "segment_discarded_samples": self.segment_discarded_samples,
                "last_segment_gain": self.last_segment_gain,
                "last_change_batch": self.last_change_batch,
                "last_change_suffix_batches": self.last_change_suffix_batches,
            }
        )
        return stats


__all__ = [
    "ASCALGMM",
    "ASCALGMMDensityShift",
    "ASCALGMMMedianShift",
    "ASCALGMMSegmentedShift",
    "ASCALGMMShift",
    "asymmetric_fake_posterior",
    "dominant_gap_boundary",
    "equal_density_boundary",
]
