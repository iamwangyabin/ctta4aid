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
unlabeled distribution change. Its handoff variant keeps the last emitted
boundary as a one-count anchor so the new segment cannot introduce an immediate
score-coordinate jump; after that first change, an uncertain one-component fit
holds the last emitted boundary instead of jumping back to the source detector.
The segmented-memory variant instead stores each completed score regime as a
compact GMM episode and uses predictive description length to recall a matching
episode when a previously observed regime returns. Its joint-density posterior
variant reads the two dominant-gap blocks as equal-prior class-conditional
densities and pools a recalled episode with current evidence in log-odds space.
Its posterior-projection variant keeps only the equal-density decision surface
from that Bayes model and applies it as a monotone shift of the source score.
Its current-projection variant recognizes that each active-segment GMM refit
already contains all causal segment scores, so it removes the redundant median
over nested refits while retaining one-vote episodic recall.
Its guarded-scan variant instead retains the robust median and only expands the
MDL change-point search to every dyadic suffix when that deployed boundary has
left the dominant gap of the current target GMM.
Its support-median variant keeps the R01 trajectory and gives each nested GMM
boundary vote weight equal to the causal segment samples summarized by that
fit, replacing equal-fit voting without adding a target hyperparameter.
Its global-residual variant keeps the immutable source-margin GMM and R01
boundary trajectory, then estimates one stream-wide feature residual from
soft, source-score-derived class prototypes. The residual never rewrites the
historical score coordinate and is updated only after each batch is predicted.
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


def _joint_density_log_odds(scores: Any, mixture: dict[str, Any]) -> Any:
    """Return log p(score|fake) - log p(score|real) for two GMM blocks."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    weights = np.asarray(mixture["weights"], dtype=np.float64).reshape(-1)
    mus = np.asarray(mixture["mus"], dtype=np.float64).reshape(-1)
    sigmas = np.asarray(mixture["sigmas"], dtype=np.float64).reshape(-1)
    if weights.size < 2 or not (weights.size == mus.size == sigmas.size):
        raise ValueError(
            "Joint-density posterior requires at least two matching components"
        )
    if np.any(weights <= 0.0) or np.any(sigmas <= 0.0):
        raise ValueError("Joint-density posterior requires positive weights and sigmas")
    if np.any(np.diff(mus) < 0.0):
        raise ValueError("Mixture components must be sorted by mean")

    partition = dominant_gap_boundary(mixture)
    split = int(partition["real_components"])
    real_weights = weights[:split] / weights[:split].sum()
    fake_weights = weights[split:] / weights[split:].sum()

    def log_density(
        block_weights: np.ndarray,
        block_mus: np.ndarray,
        block_sigmas: np.ndarray,
    ) -> np.ndarray:
        z = (values[:, None] - block_mus[None, :]) / block_sigmas[None, :]
        log_joint = (
            np.log(block_weights)[None, :]
            - np.log(block_sigmas)[None, :]
            - 0.5 * math.log(2.0 * math.pi)
            - 0.5 * z * z
        )
        return np.logaddexp.reduce(log_joint, axis=1)

    log_real = log_density(real_weights, mus[:split], sigmas[:split])
    log_fake = log_density(fake_weights, mus[split:], sigmas[split:])
    return log_fake - log_real


def joint_density_fake_posterior(scores: Any, mixture: dict[str, Any]) -> Any:
    """Return the equal-prior Bayes posterior of the dominant-gap fake block."""

    log_odds = _joint_density_log_odds(scores, mixture)
    return 1.0 / (1.0 + np.exp(-np.clip(log_odds, -60.0, 60.0)))


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


def support_weighted_median(values: Any, supports: Any) -> float:
    """Return a deterministic weighted median, averaging an exact middle tie."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    supports = np.asarray(supports, dtype=np.float64).reshape(-1)
    if values.size < 1 or values.size != supports.size:
        raise ValueError(
            "Support-weighted median requires matching non-empty arrays"
        )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(supports)):
        raise ValueError("Support-weighted median requires finite values")
    if np.any(supports <= 0.0):
        raise ValueError("Support-weighted median requires positive supports")

    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(supports[order])
    midpoint = 0.5 * float(cumulative[-1])
    index = int(np.searchsorted(cumulative, midpoint, side="left"))
    if cumulative[index] == midpoint and index + 1 < ordered_values.size:
        return float(0.5 * (ordered_values[index] + ordered_values[index + 1]))
    return float(ordered_values[index])


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


def _fixed_gmm_deviance(values: Any, mixture: dict[str, Any]) -> float:
    """Return -2 log likelihood for values under a fixed one-dimensional GMM."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(mixture["weights"], dtype=np.float64).reshape(-1)
    mus = np.asarray(mixture["mus"], dtype=np.float64).reshape(-1)
    sigmas = np.asarray(mixture["sigmas"], dtype=np.float64).reshape(-1)
    if values.size < 1:
        raise ValueError("Fixed GMM scoring requires at least one sample")
    if not (weights.size == mus.size == sigmas.size):
        raise ValueError("Fixed GMM scoring requires matching mixture arrays")
    if np.any(weights <= 0.0) or np.any(sigmas <= 0.0):
        raise ValueError("Fixed GMM scoring requires positive weights and sigmas")

    z = (values[:, None] - mus[None, :]) / sigmas[None, :]
    log_joint = (
        np.log(weights)[None, :]
        - np.log(sigmas)[None, :]
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * z * z
    )
    log_density = np.logaddexp.reduce(log_joint, axis=1)
    return float(-2.0 * log_density.sum())


def _copy_gmm(mixture: dict[str, Any]) -> dict[str, Any]:
    """Copy the serializable fields needed for an episodic score model."""

    return {
        "weights": [float(value) for value in mixture["weights"]],
        "mus": [float(value) for value in mixture["mus"]],
        "sigmas": [float(value) for value in mixture["sigmas"]],
        "components": int(mixture["components"]),
        "bic": float(mixture["bic"]),
    }


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

    def _on_segment_change(
        self,
        *,
        old_mixture: dict[str, Any],
        old_samples: int,
        new_mixture: dict[str, Any],
        new_scores: np.ndarray,
    ) -> None:
        """Allow variants to retain state before the obsolete segment is reset."""

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
                    "left": left,
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
        self._on_segment_change(
            old_mixture=best["left"],
            old_samples=discarded_samples,
            new_mixture=best["right"],
            new_scores=values[-suffix_samples:].copy(),
        )
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


class ASCALGMMSegmentedMemoryShift(ASCALGMMSegmentedShift):
    """Recall completed score regimes with predictive description length."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.segment_memories: list[dict[str, Any]] = []
        self.active_memory_index: int | None = None
        self.recall_anchor_boundary: float | None = None
        self.recall_start_batch: int | None = None
        self.memory_segments_created = 0
        self.memory_segments_updated = 0
        self.memory_recall_events = 0
        self.memory_novel_events = 0
        self.memory_comparisons = 0
        self.last_recalled_memory_index: int | None = None
        self.last_memory_fixed_score: float | None = None
        self.last_memory_new_bic: float | None = None
        self.last_memory_identity_penalty: float | None = None
        self.last_memory_recall_gain: float | None = None
        self._memory_recalled_this_change = False

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "unlabeled_bic_segmented_episodic_score_memory_shift"
                ),
                "long_term_memory": (
                    "one_fixed_gmm_boundary_count_summary_per_discovered_score_regime"
                ),
                "recall_rule": (
                    "minimum_fixed_episode_deviance_plus_model_identity_code_"
                    "versus_new_segment_bic"
                ),
                "memory_identity_code": "two_log_number_of_eligible_episodes",
                "recalled_boundary_rule": (
                    "one_episode_anchor_vote_then_current_segment_evidence"
                ),
                "memory_merge_rule": (
                    "a_recalled_episode_is_replaced_by_its_latest_completed_visit"
                ),
                "raw_images_stored": False,
                "target_labels_used": False,
                "generator_boundaries_used": False,
                "semantic_features_used": False,
                "hyperparameter_rule": (
                    "no_memory_capacity_similarity_threshold_or_recall_weight"
                ),
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_gmm_shift"

    def _append_scores(self, scores: Any) -> None:
        super()._append_scores(scores)
        self._memory_recalled_this_change = False

    def _memory_boundary(self, mixture: dict[str, Any]) -> float:
        return float(dominant_gap_boundary(mixture)["decision_boundary"])

    def _store_completed_segment(
        self,
        mixture: dict[str, Any],
        samples: int,
    ) -> int | None:
        finalized_index = self.active_memory_index
        if int(mixture["components"]) < 2:
            return finalized_index

        boundary = self._memory_boundary(mixture)
        if finalized_index is None:
            self.segment_memories.append(
                {
                    "mixture": _copy_gmm(mixture),
                    "boundary": boundary,
                    "latest_samples": int(samples),
                    "total_samples": int(samples),
                    "visits": 1,
                    "recalls": 0,
                }
            )
            self.memory_segments_created += 1
            return len(self.segment_memories) - 1

        if not 0 <= finalized_index < len(self.segment_memories):
            raise RuntimeError("Active ASCAL-GMM memory index is out of range")
        episode = self.segment_memories[finalized_index]
        episode.update(
            {
                "mixture": _copy_gmm(mixture),
                "boundary": boundary,
                "latest_samples": int(samples),
                "total_samples": int(episode["total_samples"]) + int(samples),
                "visits": int(episode["visits"]) + 1,
            }
        )
        self.memory_segments_updated += 1
        return finalized_index

    def _select_recalled_memory(
        self,
        scores: np.ndarray,
        new_mixture: dict[str, Any],
        *,
        excluded_index: int | None,
    ) -> int | None:
        eligible = [
            index
            for index, episode in enumerate(self.segment_memories)
            if index != excluded_index
            and int(episode["mixture"]["components"]) >= 2
        ]
        self.last_memory_fixed_score = None
        self.last_memory_new_bic = float(new_mixture["bic"])
        self.last_memory_identity_penalty = None
        self.last_memory_recall_gain = None
        if not eligible:
            return None

        identity_penalty = 2.0 * math.log(len(eligible))
        self.last_memory_identity_penalty = float(identity_penalty)
        best_index = None
        best_score = math.inf
        for index in eligible:
            self.memory_comparisons += 1
            episode_score = _fixed_gmm_deviance(
                scores,
                self.segment_memories[index]["mixture"],
            ) + identity_penalty
            if episode_score < best_score:
                best_score = float(episode_score)
                best_index = index

        self.last_memory_fixed_score = float(best_score)
        self.last_memory_recall_gain = float(new_mixture["bic"]) - float(best_score)
        if best_score < float(new_mixture["bic"]):
            return best_index
        return None

    def _on_segment_change(
        self,
        *,
        old_mixture: dict[str, Any],
        old_samples: int,
        new_mixture: dict[str, Any],
        new_scores: np.ndarray,
    ) -> None:
        finalized_index = self._store_completed_segment(old_mixture, old_samples)
        recalled_index = self._select_recalled_memory(
            new_scores,
            new_mixture,
            excluded_index=finalized_index,
        )
        self.active_memory_index = recalled_index
        self.recall_anchor_boundary = None
        self.recall_start_batch = None
        self.last_recalled_memory_index = recalled_index
        if recalled_index is None:
            self.memory_novel_events += 1
            return

        episode = self.segment_memories[recalled_index]
        episode["recalls"] = int(episode["recalls"]) + 1
        self.recall_anchor_boundary = float(episode["boundary"])
        self.recall_start_batch = self.total_score_batches
        self.memory_recall_events += 1
        self._memory_recalled_this_change = True

    def _recall_anchor_weight(self) -> float:
        if self.recall_anchor_boundary is None:
            return 0.0
        evidence = len(self.boundary_history)
        if evidence <= 1:
            return 1.0
        return 1.0 / float(evidence)

    def _stabilized_boundary(self, candidate: float) -> float:
        target = super()._stabilized_boundary(candidate)
        if self.recall_anchor_boundary is None:
            return target
        anchor_weight = self._recall_anchor_weight()
        return float(
            anchor_weight * self.recall_anchor_boundary
            + (1.0 - anchor_weight) * target
        )

    def predict(self, images: Any) -> PredictionBatch:
        prediction = super().predict(images)
        if self._pending is not None:
            self._pending.update(
                {
                    "prediction_memory_index": self.active_memory_index,
                    "prediction_memory_recalled": (
                        self.recall_anchor_boundary is not None
                    ),
                    "prediction_memory_anchor_boundary": self.recall_anchor_boundary,
                    "prediction_memory_anchor_weight": self._recall_anchor_weight(),
                }
            )
        return prediction

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(
            {
                "memory_size": len(self.segment_memories),
                "memory_segments_created": self.memory_segments_created,
                "memory_segments_updated": self.memory_segments_updated,
                "memory_recall_events": self.memory_recall_events,
                "memory_novel_events": self.memory_novel_events,
                "memory_comparisons": self.memory_comparisons,
                "memory_recalled_this_change": self._memory_recalled_this_change,
                "active_memory_index": self.active_memory_index,
                "recall_anchor_boundary": self.recall_anchor_boundary,
                "recall_anchor_weight": self._recall_anchor_weight(),
                "recall_start_batch": self.recall_start_batch,
                "last_recalled_memory_index": self.last_recalled_memory_index,
                "last_memory_fixed_score": self.last_memory_fixed_score,
                "last_memory_new_bic": self.last_memory_new_bic,
                "last_memory_identity_penalty": self.last_memory_identity_penalty,
                "last_memory_recall_gain": self.last_memory_recall_gain,
                "memory_boundaries": [
                    float(episode["boundary"])
                    for episode in self.segment_memories
                ],
                "memory_visits": [
                    int(episode["visits"]) for episode in self.segment_memories
                ],
                "memory_recalls": [
                    int(episode["recalls"]) for episode in self.segment_memories
                ],
            }
        )
        return stats


class ASCALGMMSegmentedMemoryPosteriorProjection(ASCALGMMSegmentedMemoryShift):
    """Project a joint-density Bayes boundary onto the monotone source score."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "unlabeled_bic_segmented_episodic_joint_density_"
                    "boundary_projection"
                ),
                "research_name": "ASCAL-JMP-Median",
                "research_version": "R01",
                "component_rule": (
                    "largest_adjacent_mean_gap_separates_multicomponent_real_"
                    "and_fake_blocks"
                ),
                "class_density_rule": (
                    "mixture_weights_are_normalized_independently_inside_each_block"
                ),
                "class_prior_rule": "equal_real_and_fake_priors",
                "boundary_rule": (
                    "equal_prior_crossing_of_normalized_real_and_fake_block_densities"
                ),
                "boundary_fallback": (
                    "dominant_gap_midpoint_when_no_crossing_exists_inside_the_gap"
                ),
                "projection_rule": (
                    "retain_the_bayes_half_posterior_decision_surface_and_project_"
                    "onto_the_source_margin"
                ),
                "prediction_rule": (
                    "sigmoid((source_margin-projected_bayes_boundary)"
                    "/source_temperature)"
                ),
                "ranking_rule": (
                    "source_margin_order_is_preserved_within_each_causal_state"
                ),
                "recalled_boundary_rule": (
                    "one_episode_projected_bayes_boundary_vote_then_current_evidence"
                ),
                "hyperparameter_rule": (
                    "no_class_prior_fusion_weight_posterior_temperature_or_target_threshold"
                ),
                "intentional_changes": [
                    "the detector stays frozen during deployment",
                    "all arrived scores enter the fit without pseudo-label admission",
                    "one selected component means insufficient evidence and exact source fallback",
                    "joint densities determine only their equal-prior decision boundary",
                    "the final readout remains monotone in the frozen detector score",
                    "a recalled projected boundary contributes one decaying evidence vote",
                    "predictions use only GMMs fitted after earlier batches",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_projection"

    def _candidate_partition(self) -> dict[str, Any]:
        if self._mixture is None:
            raise RuntimeError(
                "ASCAL-GMM posterior projection requires an active mixture"
            )
        return equal_density_boundary(self._mixture)

    def _memory_boundary(self, mixture: dict[str, Any]) -> float:
        return float(equal_density_boundary(mixture)["decision_boundary"])


class ASCALGMMSegmentedMemoryPosteriorCurrentProjection(
    ASCALGMMSegmentedMemoryPosteriorProjection
):
    """Use the latest cumulative-segment density boundary without median lag."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "unlabeled_bic_segmented_episodic_current_joint_density_"
                    "boundary_projection"
                ),
                "research_name": "ASCAL-JMP-Current",
                "research_version": "R02",
                "boundary_stabilization": (
                    "no_nested_refit_median_the_latest_active_segment_gmm_"
                    "already_contains_all_causally_arrived_segment_scores"
                ),
                "current_evidence_rule": (
                    "use_the_equal_density_boundary_of_the_latest_cumulative_"
                    "active_segment_gmm"
                ),
                "recalled_boundary_rule": (
                    "one_episode_projected_bayes_boundary_vote_then_latest_"
                    "cumulative_segment_boundary"
                ),
                "hyperparameter_rule": (
                    "removes_nested_median_and_adds_no_target_hyperparameters"
                ),
                "intentional_changes": [
                    "the detector stays frozen during deployment",
                    "all arrived scores enter the fit without pseudo-label admission",
                    "one selected component means insufficient evidence and exact source fallback",
                    "joint densities determine only their equal-prior decision boundary",
                    "the latest GMM fit already accumulates the current segment history",
                    "nested cumulative-fit boundaries are not aggregated a second time",
                    "a recalled projected boundary contributes one decaying evidence vote",
                    "predictions use only GMMs fitted after earlier batches",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_current_projection"

    def _stabilized_boundary(self, candidate: float) -> float:
        target = float(candidate)
        if self.recall_anchor_boundary is None:
            return target
        anchor_weight = self._recall_anchor_weight()
        return float(
            anchor_weight * self.recall_anchor_boundary
            + (1.0 - anchor_weight) * target
        )

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        nested_median = stats.get("stabilized_boundary")
        candidate = stats.get("candidate_boundary")
        stats.update(
            {
                "nested_boundary_median": nested_median,
                "stabilized_boundary": candidate,
                "current_fit_boundary": candidate,
            }
        )
        return stats


class ASCALGMMSegmentedMemoryPosteriorGuardedProjection(
    ASCALGMMSegmentedMemoryPosteriorProjection
):
    """Run a complete dyadic MDL scan only on density-inconsistent boundaries."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.guarded_scan_events = 0
        self.guarded_scan_candidate_scales = 0
        self._guarded_scan_triggered = False
        self._guarded_scan_boundary: float | None = None
        self._guarded_scan_gap_lower: float | None = None
        self._guarded_scan_gap_upper: float | None = None

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "unlabeled_density_guarded_complete_dyadic_mdl_segmented_"
                    "episodic_joint_density_boundary_projection"
                ),
                "research_name": "ASCAL-JMP-GuardedScan",
                "research_version": "R03",
                "boundary_stabilization": (
                    "retain_the_cumulative_median_of_causal_active_segment_"
                    "density_boundaries"
                ),
                "segmentation_guard": (
                    "expand_the_mdl_search_only_when_the_deployed_boundary_"
                    "leaves_the_current_dominant_component_gap"
                ),
                "guarded_candidate_scales": (
                    "every_power_of_two_suffix_from_two_batches_to_half_the_"
                    "active_segment"
                ),
                "change_evidence": (
                    "the_existing_segmented_bic_and_stable_multimodal_suffix_"
                    "rules_remain_unchanged"
                ),
                "hyperparameter_rule": (
                    "uses_gmm_gap_feasibility_and_mdl_only_no_new_target_"
                    "hyperparameters"
                ),
                "intentional_changes": [
                    "the R01 median boundary and monotone source-score readout remain unchanged",
                    "the ordinary binary-scheduled MDL check remains unchanged",
                    "an out-of-gap deployed boundary triggers a complete dyadic suffix scan",
                    "every candidate still pays the existing BIC change-point penalty",
                    "the detector stays frozen and predictions use only earlier-batch state",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_guarded_projection"

    def _append_scores(self, scores: Any) -> None:
        super()._append_scores(scores)
        self._guarded_scan_triggered = False
        self._guarded_scan_boundary = None
        self._guarded_scan_gap_lower = None
        self._guarded_scan_gap_upper = None

    def _suffix_candidates(self) -> list[int]:
        scheduled = super()._suffix_candidates()
        if not scheduled or not self._mixture_active():
            return scheduled

        partition = self._candidate_partition()
        candidate = float(partition["decision_boundary"])
        deployed = self._stabilized_boundary(candidate)
        split = int(partition["real_components"])
        mus = np.asarray(self._mixture["mus"], dtype=np.float64).reshape(-1)
        lower = float(mus[split - 1])
        upper = float(mus[split])
        self._guarded_scan_boundary = float(deployed)
        self._guarded_scan_gap_lower = lower
        self._guarded_scan_gap_upper = upper
        if lower <= deployed <= upper:
            return scheduled

        largest = len(self.score_batches) // 2
        suffix_batches = 2
        candidates = []
        while suffix_batches <= largest:
            candidates.append(suffix_batches)
            suffix_batches *= 2
        self._guarded_scan_triggered = True
        self.guarded_scan_events += 1
        self.guarded_scan_candidate_scales += len(candidates)
        return candidates

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(
            {
                "guarded_scan_triggered": self._guarded_scan_triggered,
                "guarded_scan_events": self.guarded_scan_events,
                "guarded_scan_candidate_scales": self.guarded_scan_candidate_scales,
                "guarded_scan_boundary": self._guarded_scan_boundary,
                "guarded_scan_gap_lower": self._guarded_scan_gap_lower,
                "guarded_scan_gap_upper": self._guarded_scan_gap_upper,
            }
        )
        return stats


class ASCALGMMSegmentedMemoryPosteriorSupportProjection(
    ASCALGMMSegmentedMemoryPosteriorProjection
):
    """Weight robust nested-fit boundary votes by their causal sample support."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.boundary_support_history: list[int] = []

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "unlabeled_sample_support_weighted_median_segmented_"
                    "episodic_joint_density_boundary_projection"
                ),
                "research_name": "ASCAL-JMP-SupportMedian",
                "research_version": "R04",
                "boundary_stabilization": (
                    "weighted_median_of_causal_active_segment_density_"
                    "boundaries"
                ),
                "boundary_vote_support": (
                    "number_of_causal_active_segment_scores_summarized_by_"
                    "each_nested_gmm_fit"
                ),
                "weighted_median_tie_rule": (
                    "midpoint_of_the_two_adjacent_values_when_support_is_"
                    "split_exactly_in_half"
                ),
                "recalled_boundary_rule": (
                    "retain_the_r01_one_episode_boundary_vote_and_fit_count_"
                    "decay"
                ),
                "hyperparameter_rule": (
                    "sample_support_is_observed_evidence_no_new_target_"
                    "hyperparameters"
                ),
                "intentional_changes": [
                    "the R01 detector segmentation memory and readout remain unchanged",
                    "each nested boundary vote is weighted by its observed sample support",
                    "the robust weighted median still prevents a latest-fit overwrite",
                    "episodic recall remains one vote with the original fit-count decay",
                    "predictions use only GMMs fitted after earlier batches",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_support_projection"

    def _on_segment_change(
        self,
        *,
        old_mixture: dict[str, Any],
        old_samples: int,
        new_mixture: dict[str, Any],
        new_scores: np.ndarray,
    ) -> None:
        super()._on_segment_change(
            old_mixture=old_mixture,
            old_samples=old_samples,
            new_mixture=new_mixture,
            new_scores=new_scores,
        )
        self.boundary_support_history = []

    def _after_successful_fit(self) -> None:
        super()._after_successful_fit()
        boundary_count = len(self.boundary_history)
        support_count = len(self.boundary_support_history)
        if boundary_count == support_count:
            return
        if boundary_count != support_count + 1:
            raise RuntimeError(
                "ASCAL support history lost alignment with boundary history"
            )
        self.boundary_support_history.append(len(self.score_history))

    def _support_weighted_boundary(self, candidate: float) -> float:
        if not self.boundary_history:
            return float(candidate)
        if len(self.boundary_history) != len(self.boundary_support_history):
            raise RuntimeError(
                "ASCAL support history must match the boundary history"
            )
        return support_weighted_median(
            self.boundary_history,
            self.boundary_support_history,
        )

    def _stabilized_boundary(self, candidate: float) -> float:
        target = self._support_weighted_boundary(candidate)
        if self.recall_anchor_boundary is None:
            return target
        anchor_weight = self._recall_anchor_weight()
        return float(
            anchor_weight * self.recall_anchor_boundary
            + (1.0 - anchor_weight) * target
        )

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        equal_vote_boundary = stats.get("stabilized_boundary")
        candidate = stats.get("candidate_boundary")
        support_boundary = (
            None
            if candidate is None
            else self._support_weighted_boundary(float(candidate))
        )
        stats.update(
            {
                "equal_vote_boundary_median": equal_vote_boundary,
                "stabilized_boundary": support_boundary,
                "support_weighted_boundary": support_boundary,
                "boundary_support_entries": len(self.boundary_support_history),
                "boundary_support_total": sum(self.boundary_support_history),
                "boundary_support_latest": (
                    None
                    if not self.boundary_support_history
                    else self.boundary_support_history[-1]
                ),
            }
        )
        return stats


class ASCALGMMSegmentedMemoryPosteriorGlobalResidual(
    ASCALGMMSegmentedMemoryPosteriorProjection
):
    """Add one causal prototype residual without changing score history."""

    def _reset_state(self) -> None:
        super()._reset_state()
        classifier = getattr(self.model, "classifier", None)
        weight = getattr(classifier, "weight", None)
        if weight is None or weight.ndim != 2 or int(weight.shape[0]) != 2:
            raise TypeError(
                "ASCAL global residual requires a two-class linear classifier"
            )
        direction = (
            weight[1].detach().float().cpu().numpy()
            - weight[0].detach().float().cpu().numpy()
        ).astype(np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if not math.isfinite(direction_norm) or direction_norm <= 0.0:
            raise ValueError(
                "ASCAL global residual requires a nonzero source score direction"
            )
        self._source_feature_direction = direction / direction_norm
        self.residual_feature_dim = int(direction.size)
        self.residual_real_sum = np.zeros(self.residual_feature_dim, dtype=np.float64)
        self.residual_fake_sum = np.zeros(self.residual_feature_dim, dtype=np.float64)
        self.residual_real_support = 0.0
        self.residual_fake_support = 0.0
        self.residual_vector: np.ndarray | None = None
        self.residual_updates = 0
        self.residual_candidate_samples = 0
        self.residual_last_reliability = 0.0
        self.residual_last_real_support = 0.0
        self.residual_last_fake_support = 0.0
        self._pending_residual_features: np.ndarray | None = None

    @property
    def trainable_parameters(self) -> int:
        if self.adaptation_mode == "static":
            return 0
        return self.residual_feature_dim

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "immutable_source_score_gmm_with_one_causal_global_"
                    "prototype_rank_residual"
                ),
                "research_name": "ASCAL-JMP-GlobalResidual",
                "research_version": "R05",
                "immutable_history_coordinate": "frozen_source_logit_margin_only",
                "base_prediction_rule": (
                    "exact_r01_median_stabilized_posterior_boundary_projection"
                ),
                "residual_count": 1,
                "residual_scope": "one_shared_vector_for_the_entire_continual_stream",
                "residual_input": (
                    "l2_normalized_frozen_features_orthogonal_to_the_source_"
                    "classifier_direction"
                ),
                "residual_teacher": (
                    "equal_prior_joint_density_posterior_computed_only_from_"
                    "the_current_immutable_source_score_gmm"
                ),
                "reliability_rule": "absolute_centered_soft_posterior_no_threshold",
                "residual_update": (
                    "cumulative_soft_real_and_fake_feature_prototypes_no_"
                    "optimizer_or_learning_rate"
                ),
                "residual_readout": (
                    "cosine_similarity_to_fake_prototype_minus_cosine_"
                    "similarity_to_real_prototype"
                ),
                "prediction_rule": "r01_base_logit_plus_one_global_residual",
                "adaptive_score_history_stored": False,
                "raw_images_stored": False,
                "target_labels_used": False,
                "generator_boundaries_used": False,
                "semantic_features_used": False,
                "hyperparameter_rule": (
                    "no_residual_learning_rate_loss_weight_confidence_threshold_"
                    "memory_capacity_or_per_domain_router"
                ),
                "intentional_changes": [
                    "the R01 source-score GMM segmentation memory and boundary trajectory remain unchanged",
                    "only immutable source margins are appended to score history",
                    "one zero-initialized stream-wide feature residual is estimated after prediction",
                    "soft GMM evidence updates cumulative class prototypes without hard admission thresholds",
                    "source-head orthogonalization prevents the residual from relearning the source margin",
                    "the adaptive residual never teaches or rewrites its own pseudo labels",
                    "predictions use only GMM and residual state learned from earlier batches",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_global_residual"

    def _batch_scores_and_residual_features(self, images: Any) -> tuple[Any, Any]:
        import torch

        if images.dim() == 5:
            batch, views = int(images.shape[0]), int(images.shape[1])
            flat = images.reshape(batch * views, *images.shape[2:])
        elif images.dim() == 4:
            batch, views = int(images.shape[0]), 1
            flat = images
        else:
            raise ValueError(
                "ASCAL global residual expects (B, C, H, W) or "
                "(B, V, C, H, W) images"
            )
        forward_features = getattr(self.model, "forward_features", None)
        classifier = getattr(self.model, "classifier", None)
        if not callable(forward_features) or not callable(classifier):
            raise TypeError(
                "ASCAL global residual requires forward_features and classifier"
            )
        with torch.no_grad():
            features = forward_features(flat.to(self.device, non_blocking=True))
            logits = classifier(features)
        margins = (
            binary_score(logits)
            .view(batch, views)
            .mean(dim=1)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        feature_values = (
            features.detach()
            .float()
            .view(batch, views, -1)
            .mean(dim=1)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        if int(feature_values.shape[1]) != self.residual_feature_dim:
            raise ValueError(
                "ASCAL global residual feature dimension does not match the source head"
            )
        direction = self._source_feature_direction
        feature_values -= (feature_values @ direction)[:, None] * direction[None, :]
        norms = np.linalg.norm(feature_values, axis=1, keepdims=True)
        feature_values = np.divide(
            feature_values,
            norms,
            out=np.zeros_like(feature_values),
            where=norms > np.finfo(np.float64).eps,
        )
        return margins, feature_values

    def _residual_scores(self, features: np.ndarray) -> np.ndarray:
        if self.residual_vector is None:
            return np.zeros(int(features.shape[0]), dtype=np.float64)
        return np.asarray(features @ self.residual_vector, dtype=np.float64)

    def _residual_state_stats(self) -> dict[str, Any]:
        return {
            "residual_ready": self.residual_vector is not None,
            "residual_count": 1,
            "residual_feature_dim": self.residual_feature_dim,
            "residual_updates": self.residual_updates,
            "residual_candidate_samples": self.residual_candidate_samples,
            "residual_real_support": self.residual_real_support,
            "residual_fake_support": self.residual_fake_support,
            "residual_norm": (
                0.0
                if self.residual_vector is None
                else float(np.linalg.norm(self.residual_vector))
            ),
            "residual_last_reliability": self.residual_last_reliability,
            "residual_last_real_support": self.residual_last_real_support,
            "residual_last_fake_support": self.residual_last_fake_support,
        }

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(self._residual_state_stats())
        return stats

    def predict(self, images: Any) -> PredictionBatch:
        scores, features = self._batch_scores_and_residual_features(images)
        memory_weight = self._recall_anchor_weight()
        if self.adaptation_mode == "full" and self._mixture_active():
            partition = self._candidate_partition()
            candidate = float(partition["decision_boundary"])
            boundary = self._stabilized_boundary(candidate)
            base_logit = (scores - boundary) / self.temperature
            base_mode = self._prediction_mode_name
            real_components = int(partition["real_components"])
            fake_components = int(partition["fake_components"])
        else:
            candidate = None
            boundary = 0.0
            base_logit = scores / self.temperature
            base_mode = "source_fallback"
            real_components = 0
            fake_components = 0

        residual_scores = self._residual_scores(features)
        final_logit = base_logit + residual_scores
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(final_logit, -60.0, 60.0))
        )
        self._pending_residual_features = features
        return self._prediction_batch(
            scores,
            probability,
            prediction_mode=base_mode,
            prediction_boundary=boundary,
            prediction_candidate_boundary=candidate,
            prediction_real_components=real_components,
            prediction_fake_components=fake_components,
            prediction_memory_index=self.active_memory_index,
            prediction_memory_recalled=self.recall_anchor_boundary is not None,
            prediction_memory_anchor_boundary=self.recall_anchor_boundary,
            prediction_memory_anchor_weight=memory_weight,
            prediction_residual_ready=self.residual_vector is not None,
            prediction_residual_mean=float(np.mean(residual_scores)),
            prediction_residual_abs_mean=float(np.mean(np.abs(residual_scores))),
            prediction_residual_max_abs=float(np.max(np.abs(residual_scores))),
        )

    def _update_global_residual(
        self,
        scores: np.ndarray,
        features: np.ndarray,
    ) -> bool:
        if self._mixture is None or not self._mixture_active():
            return False
        posterior = joint_density_fake_posterior(scores, self._mixture)
        reliability = np.abs(2.0 * posterior - 1.0)
        fake_weights = reliability * posterior
        real_weights = reliability * (1.0 - posterior)
        fake_support = float(fake_weights.sum())
        real_support = float(real_weights.sum())
        self.residual_candidate_samples += int(scores.size)
        self.residual_last_reliability = float(np.mean(reliability))
        self.residual_last_real_support = real_support
        self.residual_last_fake_support = fake_support
        epsilon = np.finfo(np.float64).eps
        if fake_support <= epsilon or real_support <= epsilon:
            return False

        self.residual_fake_sum += np.sum(fake_weights[:, None] * features, axis=0)
        self.residual_real_sum += np.sum(real_weights[:, None] * features, axis=0)
        self.residual_fake_support += fake_support
        self.residual_real_support += real_support
        fake_mean = self.residual_fake_sum / self.residual_fake_support
        real_mean = self.residual_real_sum / self.residual_real_support
        fake_norm = float(np.linalg.norm(fake_mean))
        real_norm = float(np.linalg.norm(real_mean))
        if fake_norm <= epsilon or real_norm <= epsilon:
            return False
        self.residual_vector = fake_mean / fake_norm - real_mean / real_norm
        self.residual_updates += 1
        return True

    def adapt(self, images: Any) -> AdaptationStats:
        if self._pending is None:
            return super().adapt(images)
        scores = np.asarray(self._pending["scores"], dtype=np.float64).reshape(-1)
        features = self._pending_residual_features
        self._pending_residual_features = None
        stats = super().adapt(images)
        residual_updated = False
        if self.adaptation_mode == "full":
            if features is None or int(features.shape[0]) != int(scores.size):
                raise RuntimeError(
                    "ASCAL global residual lost its matching prediction features"
                )
            residual_updated = self._update_global_residual(scores, features)
        stats.extra.update(
            {
                **self._residual_state_stats(),
                "residual_updated": residual_updated,
            }
        )
        return stats

    def discard_pending_prediction(self) -> None:
        super().discard_pending_prediction()
        self._pending_residual_features = None


class ASCALGMMSegmentedMemoryPosterior(ASCALGMMSegmentedMemoryShift):
    """Read current and recalled score regimes as class-density posteriors."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "unlabeled_bic_segmented_episodic_joint_density_posterior"
                ),
                "component_rule": (
                    "largest_adjacent_mean_gap_separates_multicomponent_real_"
                    "and_fake_blocks"
                ),
                "class_density_rule": (
                    "mixture_weights_are_normalized_independently_inside_each_block"
                ),
                "class_prior_rule": "equal_real_and_fake_priors",
                "prediction_rule": (
                    "bayes_posterior_from_real_and_fake_block_joint_densities"
                ),
                "recalled_posterior_rule": (
                    "one_episode_log_likelihood_ratio_vote_then_current_evidence"
                ),
                "posterior_pooling": "evidence_counted_geometric_pool_in_log_odds",
                "boundary_stabilization": (
                    "not_applied_to_posterior_readout_boundary_history_counts_"
                    "current_segment_evidence_only"
                ),
                "recalled_boundary_rule": (
                    "not_applied_recalled_gmm_votes_in_log_odds_instead"
                ),
                "ranking_rule": (
                    "density_likelihood_ratio_is_not_constrained_to_preserve_source_order"
                ),
                "hyperparameter_rule": (
                    "no_class_prior_fusion_weight_or_posterior_temperature"
                ),
                "intentional_changes": [
                    "the detector stays frozen during deployment",
                    "all arrived scores enter the fit without pseudo-label admission",
                    "one selected component means insufficient evidence and exact source fallback",
                    "dominant-gap blocks become independently normalized class densities",
                    "a recalled GMM contributes one decaying log-likelihood-ratio vote",
                    "predictions use only GMMs fitted after earlier batches",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_joint_density_posterior"

    def _recalled_mixture(self) -> dict[str, Any] | None:
        if self.recall_anchor_boundary is None or self.active_memory_index is None:
            return None
        if not 0 <= self.active_memory_index < len(self.segment_memories):
            raise RuntimeError("Active ASCAL-GMM posterior memory is out of range")
        mixture = self.segment_memories[self.active_memory_index]["mixture"]
        if int(mixture["components"]) < 2:
            return None
        return mixture

    def predict(self, images: Any) -> PredictionBatch:
        scores = self._batch_scores(images)
        memory_mixture = None
        memory_weight = 0.0
        if self.adaptation_mode == "full" and self._mixture_active():
            if self._mixture is None:
                raise RuntimeError("ASCAL-GMM posterior requires an active mixture")
            partition = dominant_gap_boundary(self._mixture)
            current_log_odds = _joint_density_log_odds(scores, self._mixture)
            log_odds = current_log_odds
            memory_mixture = self._recalled_mixture()
            memory_weight = self._recall_anchor_weight()
            if memory_mixture is not None and memory_weight > 0.0:
                memory_log_odds = _joint_density_log_odds(scores, memory_mixture)
                log_odds = (
                    memory_weight * memory_log_odds
                    + (1.0 - memory_weight) * current_log_odds
                )
                prediction_mode = "recalled_joint_density_posterior"
            else:
                prediction_mode = self._prediction_mode_name
            probability = 1.0 / (
                1.0 + np.exp(-np.clip(log_odds, -60.0, 60.0))
            )
            current_density = equal_density_boundary(self._mixture)
            pending_state = {
                "prediction_mode": prediction_mode,
                "prediction_boundary": None,
                "prediction_current_density_boundary": float(
                    current_density["decision_boundary"]
                ),
                "prediction_real_components": int(partition["real_components"]),
                "prediction_fake_components": int(partition["fake_components"]),
            }
        else:
            probability = self._source_probability(scores)
            pending_state = {
                "prediction_mode": "source_fallback",
                "prediction_boundary": 0.0,
                "prediction_current_density_boundary": None,
                "prediction_real_components": 0,
                "prediction_fake_components": 0,
            }

        pending_state.update(
            {
                "prediction_memory_index": self.active_memory_index,
                "prediction_memory_recalled": memory_mixture is not None,
                "prediction_memory_anchor_boundary": self.recall_anchor_boundary,
                "prediction_memory_anchor_weight": memory_weight,
                "prediction_memory_density_boundary": (
                    None
                    if memory_mixture is None
                    else float(
                        equal_density_boundary(memory_mixture)["decision_boundary"]
                    )
                ),
            }
        )
        return self._prediction_batch(scores, probability, **pending_state)

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(
            {
                "joint_density_posterior_active": (
                    self.adaptation_mode == "full" and self._mixture_active()
                ),
                "posterior_memory_weight": self._recall_anchor_weight(),
            }
        )
        return stats


class ASCALGMMSegmentedHandoffShift(ASCALGMMSegmentedShift):
    """Keep the emitted score boundary continuous after a segment change."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.handoff_active = False
        self.handoff_anchor_boundary = 0.0
        self.handoff_start_batch: int | None = None
        self.last_emitted_boundary = 0.0

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "unlabeled_bic_segmented_continuous_handoff_score_shift"
                ),
                "boundary_handoff": (
                    "anchor_weight_1_over_j_and_segment_median_weight_"
                    "j_minus_1_over_j"
                ),
                "handoff_anchor": "last_boundary_emitted_before_the_change",
                "handoff_evidence_count": (
                    "active_gmm_boundary_estimates_in_the_new_segment"
                ),
                "uncertain_fit_behavior": (
                    "hold_last_emitted_boundary_after_the_first_segment_change"
                ),
                "prediction_rule": (
                    "sigmoid((source_margin-continuous_segment_boundary)"
                    "/source_temperature)"
                ),
                "hyperparameter_rule": (
                    "no_target_smoothing_rate_window_or_handoff_length"
                ),
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_handoff_gmm_shift"

    def _handoff_weight(self) -> float:
        if not self.handoff_active:
            return 1.0
        samples = len(self.boundary_history)
        if samples <= 1:
            return 0.0
        return float(samples - 1) / float(samples)

    def _stabilized_boundary(self, candidate: float) -> float:
        target = super()._stabilized_boundary(candidate)
        if not self.handoff_active:
            return target
        weight = self._handoff_weight()
        return float(
            self.handoff_anchor_boundary
            + weight * (target - self.handoff_anchor_boundary)
        )

    def _detect_segment_change(self) -> None:
        previous_changes = self.segment_changes
        anchor = float(self.last_emitted_boundary)
        super()._detect_segment_change()
        if self.segment_changes > previous_changes:
            self.handoff_active = True
            self.handoff_anchor_boundary = anchor
            self.handoff_start_batch = self.total_score_batches

    def predict(self, images: Any) -> PredictionBatch:
        scores = self._batch_scores(images)
        if self.adaptation_mode == "full" and self._mixture_active():
            partition = self._candidate_partition()
            candidate = float(partition["decision_boundary"])
            target = super()._stabilized_boundary(candidate)
            boundary = self._stabilized_boundary(candidate)
            probability = self._source_probability(scores - boundary)
            pending_state = {
                "prediction_mode": self._prediction_mode_name,
                "prediction_boundary": boundary,
                "prediction_candidate_boundary": candidate,
                "prediction_handoff_target_boundary": target,
                "prediction_handoff_weight": self._handoff_weight(),
                "prediction_real_components": int(partition["real_components"]),
                "prediction_fake_components": int(partition["fake_components"]),
            }
        elif self.adaptation_mode == "full" and self.handoff_active:
            boundary = float(self.last_emitted_boundary)
            probability = self._source_probability(scores - boundary)
            pending_state = {
                "prediction_mode": "segmented_handoff_hold",
                "prediction_boundary": boundary,
                "prediction_candidate_boundary": None,
                "prediction_handoff_target_boundary": None,
                "prediction_handoff_weight": self._handoff_weight(),
                "prediction_real_components": 0,
                "prediction_fake_components": 0,
            }
        else:
            boundary = 0.0
            probability = self._source_probability(scores)
            pending_state = {
                "prediction_mode": "source_fallback",
                "prediction_boundary": boundary,
                "prediction_candidate_boundary": None,
                "prediction_handoff_target_boundary": None,
                "prediction_handoff_weight": 0.0,
                "prediction_real_components": 0,
                "prediction_fake_components": 0,
            }
        prediction = self._prediction_batch(scores, probability, **pending_state)
        self.last_emitted_boundary = float(boundary)
        return prediction

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(
            {
                "handoff_active": self.handoff_active,
                "handoff_anchor_boundary": (
                    self.handoff_anchor_boundary if self.handoff_active else None
                ),
                "handoff_start_batch": self.handoff_start_batch,
                "handoff_boundary_samples": (
                    len(self.boundary_history) if self.handoff_active else 0
                ),
                "handoff_weight": self._handoff_weight(),
                "handoff_target_boundary": stats.get("stabilized_boundary"),
                "last_emitted_boundary": self.last_emitted_boundary,
            }
        )
        return stats


__all__ = [
    "ASCALGMM",
    "ASCALGMMDensityShift",
    "ASCALGMMMedianShift",
    "ASCALGMMSegmentedHandoffShift",
    "ASCALGMMSegmentedMemoryPosterior",
    "ASCALGMMSegmentedMemoryPosteriorCurrentProjection",
    "ASCALGMMSegmentedMemoryPosteriorGuardedProjection",
    "ASCALGMMSegmentedMemoryPosteriorProjection",
    "ASCALGMMSegmentedMemoryPosteriorSupportProjection",
    "ASCALGMMSegmentedMemoryShift",
    "ASCALGMMSegmentedShift",
    "ASCALGMMShift",
    "asymmetric_fake_posterior",
    "dominant_gap_boundary",
    "equal_density_boundary",
    "joint_density_fake_posterior",
    "support_weighted_median",
]
