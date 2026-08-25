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
Its pre-route variant uses current-batch predictive likelihood to choose that
active model or any completed episodic model before prediction, then lets the
same choice propose an MDL-confirmed learning-state handoff after prediction.
Its MDL-route variant moves that same parameter-free confirmation ahead of the
readout, so one accepted expert assignment governs both prediction and learning.
Its live-route variant further removes the active expert's archived snapshot
from routing whenever that expert already has an eligible live GMM.
Its ordinal-route variant lets that routed expert retain the binary decision
while an immutable source probability supplies a globally comparable order
inside the selected real or fake half interval. Its ordinal-ridge variant keeps
those decisions exact and replaces only that within-interval order with the
source logit plus one centered online weighted-ridge residual per expert.
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
Its mixture-residual variant keeps one real prototype but preserves every
BIC-selected fake score component as a separate feature prototype. A single
rank residual compares the real prototype with the closest persistent fake
prototype, avoiding the unimodal fake-feature assumption without adding a
target-selected component count.
Its routed-residual variant keeps the R01 score boundary trajectory global,
but lets the current unlabeled batch select one active or episodic GMM before
reading a compact bank of expert-specific feature prototypes. The selected
expert alone supplies and receives the causal residual, so routing can improve
rank evidence without translating the score coordinate between experts.
Its real-deviation variant removes the fake prototype entirely. It estimates
one centered squared-distance residual around the soft real feature mean, so
heterogeneous fake evidence need not share a mode or teach the residual.
Its conditional-residual variant instead keeps the simplest global prototype
score and adds only its causally estimated innovation beyond the immutable
source margin, with scale and trust derived from online second moments.
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


def _block_component_responsibilities(
    scores: Any,
    weights: Any,
    mus: Any,
    sigmas: Any,
) -> np.ndarray:
    """Return normalized responsibilities inside one ordered GMM block."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    mus = np.asarray(mus, dtype=np.float64).reshape(-1)
    sigmas = np.asarray(sigmas, dtype=np.float64).reshape(-1)
    if weights.size < 1 or not (weights.size == mus.size == sigmas.size):
        raise ValueError("GMM block responsibilities require matching components")
    if np.any(weights <= 0.0) or np.any(sigmas <= 0.0):
        raise ValueError("GMM block responsibilities require positive parameters")
    if np.any(np.diff(mus) < 0.0):
        raise ValueError("GMM block components must be sorted by mean")

    z = (values[:, None] - mus[None, :]) / sigmas[None, :]
    log_joint = (
        np.log(weights / weights.sum())[None, :]
        - np.log(sigmas)[None, :]
        - 0.5 * math.log(2.0 * math.pi)
        - 0.5 * z * z
    )
    log_joint -= np.max(log_joint, axis=1, keepdims=True)
    joint = np.exp(log_joint)
    return joint / joint.sum(axis=1, keepdims=True)


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


class ASCALGMMSegmentedMemoryPosteriorPreRoute(
    ASCALGMMSegmentedMemoryPosteriorProjection
):
    """Route the current batch to a frozen score expert before prediction."""

    _ROUTING_PENDING_FIELDS = (
        "prediction_routing_expert",
        "prediction_routing_memory_index",
        "prediction_routing_candidate_count",
        "prediction_routing_selected_deviance",
        "prediction_routing_active_deviance",
        "prediction_routing_second_deviance",
        "prediction_routing_margin",
        "prediction_routing_gain_over_active",
    )

    def _reset_state(self) -> None:
        super()._reset_state()
        self.routing_decisions = 0
        self.routing_active_selections = 0
        self.routing_memory_selections = 0
        self.routing_source_fallback_selections = 0
        self.routing_expert_evaluations = 0
        self.routing_score_samples = 0
        self.routing_memory_selection_counts: dict[int, int] = {}
        self.routing_handoff_checks = 0
        self.routing_handoff_confirmations = 0
        self.routing_handoff_rejections = 0
        self.routing_handoff_fit_failures = 0
        self.routing_handoff_unimodal_batches = 0
        self.last_routing_expert: str | None = None
        self.last_routing_memory_index: int | None = None
        self.last_routing_candidate_count = 0
        self.last_routing_selected_deviance: float | None = None
        self.last_routing_active_deviance: float | None = None
        self.last_routing_second_deviance: float | None = None
        self.last_routing_margin: float | None = None
        self.last_routing_gain_over_active: float | None = None
        self.last_routing_handoff_memory_index: int | None = None
        self.last_routing_handoff_fixed_score: float | None = None
        self.last_routing_handoff_new_bic: float | None = None
        self.last_routing_handoff_identity_penalty: float | None = None
        self.last_routing_handoff_gain: float | None = None
        self._routing_handoff_this_batch = False

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "unlabeled_batch_likelihood_prerouted_episodic_"
                    "joint_density_boundary_projection"
                ),
                "research_name": "ASCAL-JMP-PreRoute",
                "research_version": "R09",
                "routing_coordinate": "immutable_frozen_source_logit_margin",
                "routing_candidates": (
                    "active_learning_gmm_plus_every_completed_episodic_gmm"
                ),
                "routing_rule": (
                    "minimum_fixed_predictive_deviance_on_the_current_"
                    "unlabeled_batch"
                ),
                "routing_tie_break": "active_state_then_oldest_memory_index",
                "routing_granularity": "one_current_unlabeled_stream_batch",
                "routing_readout": (
                    "active_uses_the_exact_r01_deployed_boundary_and_memory_"
                    "uses_its_frozen_projected_bayes_boundary"
                ),
                "routing_learning_separation": (
                    "prediction_only_proposes_an_expert_then_parameter_free_mdl_"
                    "confirms_any_active_learning_state_handoff_after_prediction"
                ),
                "routing_adaptation_rule": (
                    "a_selected_memory_starts_a_new_visit_only_when_its_fixed_"
                    "deviance_plus_identity_code_beats_a_current_batch_gmm_bic"
                ),
                "expert_update_rule": (
                    "a_confirmed_memory_anchors_the_new_active_visit_and_is_"
                    "replaced_by_that_visit_when_the_stream_later_leaves_it"
                ),
                "ranking_rule": (
                    "source_margin_order_is_preserved_within_each_batch_"
                    "while_expert_boundaries_align_scores_across_batches"
                ),
                "batch_transductive_prediction": True,
                "target_labels_used": False,
                "generator_boundaries_used": False,
                "semantic_features_used": False,
                "hyperparameter_rule": (
                    "no_routing_threshold_similarity_metric_fusion_weight_"
                    "memory_capacity_or_new_target_hyperparameter"
                ),
                "intentional_changes": [
                    "the detector and every routing expert stay frozen during prediction",
                    "the current batch only selects among experts fitted after earlier batches",
                    "an unconfirmed routing choice cannot modify or corrupt a memory expert",
                    "a confirmed memory route starts its active visit only after prediction",
                    "novel-state learning otherwise retains the R01 segmented adaptation path",
                    "a likelihood tie keeps the active state and therefore the R01 prediction",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_preroute"

    def _routing_candidates(self, scores: np.ndarray) -> list[dict[str, Any]]:
        candidates = []
        if self._mixture_active() and self._mixture is not None:
            active_partition = self._candidate_partition()
            active_candidate = float(active_partition["decision_boundary"])
            candidates.append(
                {
                    "expert": "active_learning_state",
                    "memory_index": None,
                    "mixture": self._mixture,
                    "boundary": self._stabilized_boundary(active_candidate),
                    "candidate_boundary": active_candidate,
                    "real_components": int(active_partition["real_components"]),
                    "fake_components": int(active_partition["fake_components"]),
                    "deviance": _fixed_gmm_deviance(scores, self._mixture),
                }
            )
        for index, episode in enumerate(self.segment_memories):
            mixture = episode["mixture"]
            if int(mixture["components"]) < 2:
                continue
            partition = equal_density_boundary(mixture)
            candidates.append(
                {
                    "expert": "episodic_memory",
                    "memory_index": index,
                    "mixture": mixture,
                    "boundary": float(episode["boundary"]),
                    "candidate_boundary": float(episode["boundary"]),
                    "real_components": int(partition["real_components"]),
                    "fake_components": int(partition["fake_components"]),
                    "deviance": _fixed_gmm_deviance(scores, mixture),
                }
            )
        return candidates

    def predict(self, images: Any) -> PredictionBatch:
        scores = self._batch_scores(images)
        candidates: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        active_deviance = None
        second_deviance = None
        routing_margin = None
        routing_gain = None
        memory_candidate_count = 0
        if self.adaptation_mode == "full":
            candidates = self._routing_candidates(scores)
        if candidates:
            selected = min(candidates, key=lambda candidate: candidate["deviance"])
            ordered_deviances = sorted(float(item["deviance"]) for item in candidates)
            active = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate["expert"] == "active_learning_state"
                ),
                None,
            )
            active_deviance = (
                None if active is None else float(active["deviance"])
            )
            memory_candidate_count = sum(
                candidate["expert"] == "episodic_memory"
                for candidate in candidates
            )
            if len(ordered_deviances) > 1:
                second_deviance = float(ordered_deviances[1])
                routing_margin = second_deviance - float(selected["deviance"])
            if active_deviance is not None:
                routing_gain = active_deviance - float(selected["deviance"])
            boundary = float(selected["boundary"])
            probability = self._source_probability(scores - boundary)
            prediction_mode = self._prediction_mode_name
            candidate_boundary = float(selected["candidate_boundary"])
            real_components = int(selected["real_components"])
            fake_components = int(selected["fake_components"])
        else:
            boundary = 0.0
            probability = self._source_probability(scores)
            prediction_mode = "source_fallback"
            candidate_boundary = None
            real_components = 0
            fake_components = 0

        selected_expert = None if selected is None else str(selected["expert"])
        routed_memory_index = (
            None if selected is None else selected["memory_index"]
        )
        return self._prediction_batch(
            scores,
            probability,
            prediction_mode=prediction_mode,
            prediction_boundary=boundary,
            prediction_candidate_boundary=candidate_boundary,
            prediction_real_components=real_components,
            prediction_fake_components=fake_components,
            prediction_memory_index=self.active_memory_index,
            prediction_memory_recalled=self.recall_anchor_boundary is not None,
            prediction_memory_anchor_boundary=self.recall_anchor_boundary,
            prediction_memory_anchor_weight=self._recall_anchor_weight(),
            prediction_routing_expert=selected_expert,
            prediction_routing_memory_index=routed_memory_index,
            prediction_routing_candidate_count=len(candidates),
            prediction_routing_memory_candidate_count=memory_candidate_count,
            prediction_routing_selected_deviance=(
                None if selected is None else float(selected["deviance"])
            ),
            prediction_routing_selected_deviance_per_sample=(
                None
                if selected is None
                else float(selected["deviance"]) / float(len(scores))
            ),
            prediction_routing_active_deviance=active_deviance,
            prediction_routing_second_deviance=second_deviance,
            prediction_routing_margin=routing_margin,
            prediction_routing_gain_over_active=routing_gain,
        )

    def _routing_state_stats(self) -> dict[str, Any]:
        return {
            "routing_decisions": self.routing_decisions,
            "routing_active_selections": self.routing_active_selections,
            "routing_memory_selections": self.routing_memory_selections,
            "routing_source_fallback_selections": (
                self.routing_source_fallback_selections
            ),
            "routing_expert_evaluations": self.routing_expert_evaluations,
            "routing_score_samples": self.routing_score_samples,
            "routing_memory_selection_counts": [
                int(self.routing_memory_selection_counts.get(index, 0))
                for index in range(len(self.segment_memories))
            ],
            "routing_handoff_checks": self.routing_handoff_checks,
            "routing_handoff_confirmations": self.routing_handoff_confirmations,
            "routing_handoff_rejections": self.routing_handoff_rejections,
            "routing_handoff_fit_failures": self.routing_handoff_fit_failures,
            "routing_handoff_unimodal_batches": (
                self.routing_handoff_unimodal_batches
            ),
            "last_routing_expert": self.last_routing_expert,
            "last_routing_memory_index": self.last_routing_memory_index,
            "last_routing_candidate_count": self.last_routing_candidate_count,
            "last_routing_selected_deviance": self.last_routing_selected_deviance,
            "last_routing_active_deviance": self.last_routing_active_deviance,
            "last_routing_second_deviance": self.last_routing_second_deviance,
            "last_routing_margin": self.last_routing_margin,
            "last_routing_gain_over_active": self.last_routing_gain_over_active,
            "last_routing_handoff_memory_index": (
                self.last_routing_handoff_memory_index
            ),
            "last_routing_handoff_fixed_score": (
                self.last_routing_handoff_fixed_score
            ),
            "last_routing_handoff_new_bic": self.last_routing_handoff_new_bic,
            "last_routing_handoff_identity_penalty": (
                self.last_routing_handoff_identity_penalty
            ),
            "last_routing_handoff_gain": self.last_routing_handoff_gain,
            "routing_handoff_this_batch": self._routing_handoff_this_batch,
        }

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(self._routing_state_stats())
        return stats

    def _confirm_routing_handoff(
        self,
        scores: np.ndarray,
        memory_index: int,
    ) -> bool:
        self.routing_handoff_checks += 1
        self.last_routing_handoff_memory_index = memory_index
        self.last_routing_handoff_fixed_score = None
        self.last_routing_handoff_new_bic = None
        self.last_routing_handoff_identity_penalty = None
        self.last_routing_handoff_gain = None
        eligible_memories = sum(
            int(episode["mixture"]["components"]) >= 2
            for episode in self.segment_memories
        )
        if eligible_memories < 1:
            raise RuntimeError("ASCAL pre-route handoff has no eligible memory")

        try:
            new_mixture = fit_gmm_bic(
                scores,
                max_components=min(self.max_total_components, int(scores.size)),
                seed=0,
            )
        except (FloatingPointError, RuntimeError, ValueError):
            self.routing_handoff_fit_failures += 1
            self.routing_handoff_rejections += 1
            return False
        self.last_routing_handoff_new_bic = float(new_mixture["bic"])
        if int(new_mixture["components"]) < 2:
            self.routing_handoff_unimodal_batches += 1
            self.routing_handoff_rejections += 1
            return False

        identity_penalty = 2.0 * math.log(eligible_memories)
        fixed_score = _fixed_gmm_deviance(
            scores,
            self.segment_memories[memory_index]["mixture"],
        ) + identity_penalty
        gain = float(new_mixture["bic"]) - float(fixed_score)
        self.last_routing_handoff_fixed_score = float(fixed_score)
        self.last_routing_handoff_identity_penalty = float(identity_penalty)
        self.last_routing_handoff_gain = float(gain)
        if gain > 0.0:
            self.routing_handoff_confirmations += 1
            return True
        self.routing_handoff_rejections += 1
        return False

    def _start_routed_memory_visit(self, memory_index: int) -> None:
        old_samples = len(self.score_history)
        if self._mixture is not None and old_samples > 0:
            self._store_completed_segment(self._mixture, old_samples)

        episode = self.segment_memories[memory_index]
        self.score_history = []
        self.score_batches = []
        self._mixture = None
        self.boundary_history = []
        self.active_memory_index = memory_index
        self.recall_anchor_boundary = float(episode["boundary"])
        self.recall_start_batch = self.total_score_batches + 1
        self.last_recalled_memory_index = memory_index
        episode["recalls"] = int(episode["recalls"]) + 1
        self.memory_recall_events += 1
        self.segment_changes += 1
        self.segment_discarded_samples += old_samples
        self.last_change_suffix_batches = 1
        self._routing_handoff_this_batch = True

    def _pending_routing_state(self) -> tuple[dict[str, Any] | None, int]:
        if self._pending is None:
            return None, 0
        return (
            {
                key: self._pending.get(key)
                for key in self._ROUTING_PENDING_FIELDS
            },
            int(np.asarray(self._pending["scores"]).size),
        )

    def _apply_routing_learning_assignment(
        self,
        routing_state: dict[str, Any],
        scores: np.ndarray,
    ) -> None:
        if routing_state["prediction_routing_expert"] != "episodic_memory":
            return
        memory_index = routing_state["prediction_routing_memory_index"]
        if memory_index is None:
            raise RuntimeError("ASCAL pre-route selected memory without an index")
        memory_index = int(memory_index)
        if memory_index != self.active_memory_index and self._confirm_routing_handoff(
            scores,
            memory_index,
        ):
            self._start_routed_memory_visit(memory_index)

    def adapt(self, images: Any) -> AdaptationStats:
        routing_state, score_samples = self._pending_routing_state()

        self._routing_handoff_this_batch = False
        if (
            self.adaptation_mode == "full"
            and routing_state is not None
        ):
            self._apply_routing_learning_assignment(
                routing_state,
                np.asarray(self._pending["scores"], dtype=np.float64).reshape(-1),
            )

        stats = super().adapt(images)
        if self.adaptation_mode != "full" or routing_state is None:
            return stats

        candidate_count = int(
            routing_state["prediction_routing_candidate_count"] or 0
        )
        self.last_routing_expert = routing_state["prediction_routing_expert"]
        self.last_routing_memory_index = routing_state[
            "prediction_routing_memory_index"
        ]
        self.last_routing_candidate_count = candidate_count
        self.last_routing_selected_deviance = routing_state[
            "prediction_routing_selected_deviance"
        ]
        self.last_routing_active_deviance = routing_state[
            "prediction_routing_active_deviance"
        ]
        self.last_routing_second_deviance = routing_state[
            "prediction_routing_second_deviance"
        ]
        self.last_routing_margin = routing_state["prediction_routing_margin"]
        self.last_routing_gain_over_active = routing_state[
            "prediction_routing_gain_over_active"
        ]
        if candidate_count > 0:
            self.routing_decisions += 1
            self.routing_expert_evaluations += candidate_count
            self.routing_score_samples += score_samples
            if self.last_routing_expert == "episodic_memory":
                if self.last_routing_memory_index is None:
                    raise RuntimeError("ASCAL pre-route selected memory without an index")
                index = int(self.last_routing_memory_index)
                self.routing_memory_selections += 1
                self.routing_memory_selection_counts[index] = (
                    self.routing_memory_selection_counts.get(index, 0) + 1
                )
            elif self.last_routing_expert == "active_learning_state":
                self.routing_active_selections += 1
            else:
                self.routing_source_fallback_selections += 1
        if self._routing_handoff_this_batch:
            self._segment_changed = True
            self._memory_recalled_this_change = True
            self.last_change_batch = self.total_score_batches
        stats.extra.update(self._state_stats())
        return stats


class ASCALGMMSegmentedMemoryPosteriorMDLRoute(
    ASCALGMMSegmentedMemoryPosteriorPreRoute
):
    """Use one MDL-admitted expert assignment for prediction and adaptation."""

    _ROUTING_PENDING_FIELDS = (
        *ASCALGMMSegmentedMemoryPosteriorPreRoute._ROUTING_PENDING_FIELDS,
        "prediction_routing_proposed_expert",
        "prediction_routing_proposed_memory_index",
        "prediction_routing_proposed_deviance",
        "prediction_routing_proposed_margin",
        "prediction_routing_proposed_gain_over_active",
        "prediction_routing_admission_checked",
        "prediction_routing_admission_accepted",
        "prediction_routing_admission_reason",
        "prediction_routing_admission_memory_index",
        "prediction_routing_admission_fixed_score",
        "prediction_routing_admission_new_bic",
        "prediction_routing_admission_identity_penalty",
        "prediction_routing_admission_gain",
        "prediction_routing_admission_new_components",
        "prediction_routing_admission_fit_failure",
        "prediction_routing_admission_unimodal",
    )

    def _reset_state(self) -> None:
        super()._reset_state()
        self.routing_memory_proposals = 0
        self.routing_memory_admission_fallbacks = 0
        self.routing_active_memory_identity_reuses = 0
        self.last_routing_proposed_expert: str | None = None
        self.last_routing_proposed_memory_index: int | None = None
        self.last_routing_admission_reason: str | None = None

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "unlabeled_mdl_assigned_episodic_joint_density_"
                    "boundary_projection"
                ),
                "research_name": "ASCAL-JMP-MDLRoute",
                "research_version": "R10",
                "routing_rule": (
                    "minimum_fixed_deviance_proposes_a_memory_then_uniform_"
                    "identity_mdl_must_beat_a_temporary_current_batch_gmm_bic"
                ),
                "routing_readout": (
                    "only_the_mdl_admitted_expert_can_predict_the_current_batch"
                ),
                "routing_learning_separation": (
                    "prediction_computes_one_immutable_assignment_without_state_"
                    "mutation_and_adaptation_consumes_that_exact_assignment"
                ),
                "routing_adaptation_rule": (
                    "the_same_deployed_expert_receives_the_current_batch_or_the_"
                    "r01_active_novel_state_path_receives_it_when_reuse_is_rejected"
                ),
                "current_batch_gmm_role": (
                    "temporary_description_length_null_only_never_used_for_"
                    "classification_or_persisted"
                ),
                "assignment_consistency": "one_batch_one_expert_for_predict_and_adapt",
                "prediction_mutates_experts": False,
                "hyperparameter_rule": (
                    "no_routing_threshold_similarity_metric_fusion_weight_"
                    "memory_capacity_or_new_target_hyperparameter"
                ),
                "intentional_changes": [
                    "the R09 likelihood winner is only a proposal",
                    "a returning expert predicts only after parameter-free MDL admission",
                    "a rejected proposal falls back before prediction rather than after it",
                    "adaptation consumes the prediction-time assignment without refitting or rerouting",
                    "the temporary current-batch GMM is neither a classifier nor persistent state",
                    "all R01 expert construction and update rules remain unchanged",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_mdl_route"

    def _memory_admission_evidence(
        self,
        scores: np.ndarray,
        memory_index: int,
    ) -> dict[str, Any]:
        eligible_memories = sum(
            int(episode["mixture"]["components"]) >= 2
            for episode in self.segment_memories
        )
        if eligible_memories < 1:
            raise RuntimeError("ASCAL MDL route has no eligible memory")
        evidence: dict[str, Any] = {
            "checked": True,
            "accepted": False,
            "reason": "fit_failure",
            "memory_index": memory_index,
            "fixed_score": None,
            "new_bic": None,
            "identity_penalty": None,
            "gain": None,
            "new_components": None,
            "fit_failure": False,
            "unimodal": False,
        }
        try:
            new_mixture = fit_gmm_bic(
                scores,
                max_components=min(self.max_total_components, int(scores.size)),
                seed=0,
            )
        except (FloatingPointError, RuntimeError, ValueError):
            evidence["fit_failure"] = True
            return evidence

        components = int(new_mixture["components"])
        evidence["new_bic"] = float(new_mixture["bic"])
        evidence["new_components"] = components
        if components < 2:
            evidence["reason"] = "current_batch_unimodal"
            evidence["unimodal"] = True
            return evidence

        identity_penalty = 2.0 * math.log(eligible_memories)
        fixed_score = _fixed_gmm_deviance(
            scores,
            self.segment_memories[memory_index]["mixture"],
        ) + identity_penalty
        gain = float(new_mixture["bic"]) - float(fixed_score)
        accepted = gain > 0.0
        evidence.update(
            {
                "accepted": accepted,
                "reason": (
                    "memory_description_shorter"
                    if accepted
                    else "new_state_description_shorter_or_equal"
                ),
                "fixed_score": float(fixed_score),
                "identity_penalty": float(identity_penalty),
                "gain": float(gain),
            }
        )
        return evidence

    @staticmethod
    def _empty_admission(reason: str) -> dict[str, Any]:
        return {
            "checked": False,
            "accepted": False,
            "reason": reason,
            "memory_index": None,
            "fixed_score": None,
            "new_bic": None,
            "identity_penalty": None,
            "gain": None,
            "new_components": None,
            "fit_failure": False,
            "unimodal": False,
        }

    def predict(self, images: Any) -> PredictionBatch:
        scores = self._batch_scores(images)
        candidates: list[dict[str, Any]] = []
        proposal: dict[str, Any] | None = None
        selected: dict[str, Any] | None = None
        active: dict[str, Any] | None = None
        admission = self._empty_admission("no_routing_candidate")
        if self.adaptation_mode == "full":
            candidates = self._routing_candidates(scores)

        if candidates:
            proposal = min(candidates, key=lambda candidate: candidate["deviance"])
            active = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate["expert"] == "active_learning_state"
                ),
                None,
            )
            if proposal["expert"] == "episodic_memory":
                memory_index = int(proposal["memory_index"])
                if memory_index == self.active_memory_index:
                    selected = proposal
                    admission = self._empty_admission(
                        "already_active_memory_identity"
                    )
                    admission.update(
                        {"accepted": True, "memory_index": memory_index}
                    )
                else:
                    admission = self._memory_admission_evidence(
                        scores,
                        memory_index,
                    )
                    selected = proposal if admission["accepted"] else active
            else:
                selected = proposal
                admission = self._empty_admission("active_state_wins_deviance")

        ordered_deviances = sorted(float(item["deviance"]) for item in candidates)
        active_deviance = None if active is None else float(active["deviance"])
        proposal_margin = None
        proposal_gain = None
        if proposal is not None:
            if len(ordered_deviances) > 1:
                proposal_margin = (
                    float(ordered_deviances[1]) - float(proposal["deviance"])
                )
            if active_deviance is not None:
                proposal_gain = active_deviance - float(proposal["deviance"])

        second_deviance = None
        routing_margin = None
        routing_gain = None
        if selected is not None:
            alternatives = [
                float(candidate["deviance"])
                for candidate in candidates
                if candidate is not selected
            ]
            if alternatives:
                second_deviance = min(alternatives)
                routing_margin = second_deviance - float(selected["deviance"])
            if active_deviance is not None:
                routing_gain = active_deviance - float(selected["deviance"])
            boundary = float(selected["boundary"])
            probability = self._source_probability(scores - boundary)
            prediction_mode = self._prediction_mode_name
            candidate_boundary = float(selected["candidate_boundary"])
            real_components = int(selected["real_components"])
            fake_components = int(selected["fake_components"])
        else:
            boundary = 0.0
            probability = self._source_probability(scores)
            prediction_mode = "source_fallback"
            candidate_boundary = None
            real_components = 0
            fake_components = 0

        selected_expert = None if selected is None else str(selected["expert"])
        selected_memory_index = None if selected is None else selected["memory_index"]
        proposed_expert = None if proposal is None else str(proposal["expert"])
        proposed_memory_index = None if proposal is None else proposal["memory_index"]
        return self._prediction_batch(
            scores,
            probability,
            prediction_mode=prediction_mode,
            prediction_boundary=boundary,
            prediction_candidate_boundary=candidate_boundary,
            prediction_real_components=real_components,
            prediction_fake_components=fake_components,
            prediction_memory_index=self.active_memory_index,
            prediction_memory_recalled=self.recall_anchor_boundary is not None,
            prediction_memory_anchor_boundary=self.recall_anchor_boundary,
            prediction_memory_anchor_weight=self._recall_anchor_weight(),
            prediction_routing_expert=selected_expert,
            prediction_routing_memory_index=selected_memory_index,
            prediction_routing_candidate_count=len(candidates),
            prediction_routing_memory_candidate_count=sum(
                candidate["expert"] == "episodic_memory"
                for candidate in candidates
            ),
            prediction_routing_selected_deviance=(
                None if selected is None else float(selected["deviance"])
            ),
            prediction_routing_selected_deviance_per_sample=(
                None
                if selected is None
                else float(selected["deviance"]) / float(len(scores))
            ),
            prediction_routing_active_deviance=active_deviance,
            prediction_routing_second_deviance=second_deviance,
            prediction_routing_margin=routing_margin,
            prediction_routing_gain_over_active=routing_gain,
            prediction_routing_proposed_expert=proposed_expert,
            prediction_routing_proposed_memory_index=proposed_memory_index,
            prediction_routing_proposed_deviance=(
                None if proposal is None else float(proposal["deviance"])
            ),
            prediction_routing_proposed_margin=proposal_margin,
            prediction_routing_proposed_gain_over_active=proposal_gain,
            prediction_routing_admission_checked=bool(admission["checked"]),
            prediction_routing_admission_accepted=bool(admission["accepted"]),
            prediction_routing_admission_reason=str(admission["reason"]),
            prediction_routing_admission_memory_index=admission["memory_index"],
            prediction_routing_admission_fixed_score=admission["fixed_score"],
            prediction_routing_admission_new_bic=admission["new_bic"],
            prediction_routing_admission_identity_penalty=(
                admission["identity_penalty"]
            ),
            prediction_routing_admission_gain=admission["gain"],
            prediction_routing_admission_new_components=(
                admission["new_components"]
            ),
            prediction_routing_admission_fit_failure=bool(
                admission["fit_failure"]
            ),
            prediction_routing_admission_unimodal=bool(admission["unimodal"]),
        )

    def _apply_routing_learning_assignment(
        self,
        routing_state: dict[str, Any],
        scores: np.ndarray,
    ) -> None:
        del scores
        proposed_expert = routing_state["prediction_routing_proposed_expert"]
        proposed_memory_index = routing_state[
            "prediction_routing_proposed_memory_index"
        ]
        selected_expert = routing_state["prediction_routing_expert"]
        selected_memory_index = routing_state["prediction_routing_memory_index"]
        admission_checked = bool(
            routing_state["prediction_routing_admission_checked"]
        )
        admission_accepted = bool(
            routing_state["prediction_routing_admission_accepted"]
        )

        self.last_routing_proposed_expert = proposed_expert
        self.last_routing_proposed_memory_index = proposed_memory_index
        self.last_routing_admission_reason = routing_state[
            "prediction_routing_admission_reason"
        ]
        if proposed_expert == "episodic_memory":
            self.routing_memory_proposals += 1
            if selected_expert != "episodic_memory":
                self.routing_memory_admission_fallbacks += 1

        if admission_checked:
            memory_index = routing_state[
                "prediction_routing_admission_memory_index"
            ]
            if memory_index is None:
                raise RuntimeError("ASCAL MDL route checked memory without an index")
            self.routing_handoff_checks += 1
            self.last_routing_handoff_memory_index = int(memory_index)
            self.last_routing_handoff_fixed_score = routing_state[
                "prediction_routing_admission_fixed_score"
            ]
            self.last_routing_handoff_new_bic = routing_state[
                "prediction_routing_admission_new_bic"
            ]
            self.last_routing_handoff_identity_penalty = routing_state[
                "prediction_routing_admission_identity_penalty"
            ]
            self.last_routing_handoff_gain = routing_state[
                "prediction_routing_admission_gain"
            ]
            if routing_state["prediction_routing_admission_fit_failure"]:
                self.routing_handoff_fit_failures += 1
            if routing_state["prediction_routing_admission_unimodal"]:
                self.routing_handoff_unimodal_batches += 1
            if admission_accepted:
                self.routing_handoff_confirmations += 1
            else:
                self.routing_handoff_rejections += 1

        if selected_expert != "episodic_memory":
            return
        if selected_memory_index is None:
            raise RuntimeError("ASCAL MDL route selected memory without an index")
        selected_memory_index = int(selected_memory_index)
        if selected_memory_index == self.active_memory_index:
            if not admission_checked:
                self.routing_active_memory_identity_reuses += 1
            return
        if not admission_checked or not admission_accepted:
            raise RuntimeError(
                "ASCAL MDL route cannot adapt with an unconfirmed memory assignment"
            )
        self._start_routed_memory_visit(selected_memory_index)

    def _routing_state_stats(self) -> dict[str, Any]:
        stats = super()._routing_state_stats()
        stats.update(
            {
                "routing_memory_proposals": self.routing_memory_proposals,
                "routing_memory_admission_fallbacks": (
                    self.routing_memory_admission_fallbacks
                ),
                "routing_active_memory_identity_reuses": (
                    self.routing_active_memory_identity_reuses
                ),
                "last_routing_proposed_expert": self.last_routing_proposed_expert,
                "last_routing_proposed_memory_index": (
                    self.last_routing_proposed_memory_index
                ),
                "last_routing_admission_reason": self.last_routing_admission_reason,
            }
        )
        return stats


class ASCALGMMSegmentedMemoryPosteriorLiveRoute(
    ASCALGMMSegmentedMemoryPosteriorMDLRoute
):
    """Expose only one routable state for the currently active expert."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "unlabeled_unique_live_mdl_assigned_episodic_joint_"
                    "density_boundary_projection"
                ),
                "research_name": "ASCAL-JMP-LiveRoute",
                "research_version": "R11",
                "expert_identity_rule": "one_expert_one_routable_live_state",
                "active_snapshot_rule": (
                    "hide_the_active_experts_archived_snapshot_when_its_live_"
                    "gmm_is_eligible"
                ),
                "inactive_snapshot_rule": (
                    "retain_the_archived_snapshot_as_a_same_identity_fallback_"
                    "only_while_the_live_gmm_is_not_eligible"
                ),
                "routing_candidates": (
                    "one_active_live_state_plus_archived_non_active_experts"
                ),
                "intentional_changes": [
                    "R10 MDL proposal and admission equations are unchanged",
                    "R10 prediction and adaptation assignment consumption are unchanged",
                    "an eligible live expert suppresses only its own stale snapshot",
                    "all non-active archived experts remain routable",
                    "an ineligible live state may still read its own archived fallback",
                    "no expert is deleted and no memory update rule changes",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_live_route"

    def _routing_candidates(self, scores: np.ndarray) -> list[dict[str, Any]]:
        candidates = super()._routing_candidates(scores)
        if not self._mixture_active() or self.active_memory_index is None:
            return candidates
        return [
            candidate
            for candidate in candidates
            if not (
                candidate["expert"] == "episodic_memory"
                and candidate["memory_index"] == self.active_memory_index
            )
        ]


class ASCALGMMSegmentedMemoryPosteriorOrdinalRoute(
    ASCALGMMSegmentedMemoryPosteriorLiveRoute
):
    """Keep routed decisions while restoring a globally comparable rank."""

    _ORDINAL_PENDING_FIELDS = (
        "prediction_ordinal_applied",
        "prediction_ordinal_routed_fake_count",
        "prediction_ordinal_source_fake_count",
        "prediction_ordinal_decision_disagreements",
        "prediction_ordinal_label_mismatches",
    )

    def _reset_state(self) -> None:
        super()._reset_state()
        self.ordinal_batches = 0
        self.ordinal_samples = 0
        self.ordinal_decision_disagreements = 0
        self.ordinal_label_mismatches = 0

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "unlabeled_unique_live_mdl_routed_decision_with_immutable_"
                    "source_ordinal_readout"
                ),
                "research_name": "ASCAL-JMP-OrdinalRoute",
                "research_version": "R12",
                "routed_decision_rule": (
                    "exact_r11_mdl_admitted_expert_threshold_decision"
                ),
                "global_rank_coordinate": (
                    "immutable_frozen_source_probability_with_source_temperature"
                ),
                "ordinal_probability_rule": (
                    "real_decision_maps_to_one_half_times_source_probability_"
                    "fake_decision_maps_to_one_half_plus_one_half_times_source_"
                    "probability"
                ),
                "accuracy_invariance": (
                    "the_ordinal_readout_preserves_every_r11_hard_decision_exactly"
                ),
                "within_decision_order": (
                    "strictly_preserve_the_frozen_source_margin_order"
                ),
                "cross_batch_alignment": (
                    "expert_boundaries_choose_the_class_interval_but_never_"
                    "translate_the_within_interval_rank_coordinate"
                ),
                "source_fallback": "exact_r11_source_fallback_without_remapping",
                "new_target_hyperparameters": 0,
                "prediction_mutates_experts": False,
                "target_labels_used": False,
                "generator_boundaries_used": False,
                "semantic_features_used": False,
                "intentional_changes": [
                    "all R11 routing admission adaptation and memory rules remain unchanged",
                    "the admitted expert still supplies the current binary decision",
                    "the immutable source probability supplies only within-decision order",
                    "the final output remains one scalar probability with threshold zero point five",
                    "source fallback predictions remain byte-for-byte on the source probability path",
                    "no residual optimizer threshold temperature or fusion weight is introduced",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_ordinal_route"

    def predict(self, images: Any) -> PredictionBatch:
        routed = super().predict(images)
        if self._pending is None:
            raise RuntimeError("ASCAL ordinal route lost the routed prediction state")

        selected_expert = self._pending.get("prediction_routing_expert")
        if selected_expert is None:
            self._pending.update(
                {
                    "prediction_ordinal_applied": False,
                    "prediction_ordinal_routed_fake_count": int(
                        routed.pred_label.sum().item()
                    ),
                    "prediction_ordinal_source_fake_count": int(
                        routed.pred_label.sum().item()
                    ),
                    "prediction_ordinal_decision_disagreements": 0,
                    "prediction_ordinal_label_mismatches": 0,
                }
            )
            return routed

        scores = np.asarray(self._pending["scores"], dtype=np.float64).reshape(-1)
        source_probability = np.clip(
            self._source_probability(scores),
            1e-6,
            1.0 - 1e-6,
        )
        routed_fake = routed.pred_label.detach().cpu().numpy().astype(bool)
        source_fake = source_probability >= 0.5
        ordinal_probability = 0.5 * (
            source_probability + routed_fake.astype(np.float64)
        )
        pending_state = dict(self._pending)
        pending_state.pop("scores")
        pending_state.update(
            {
                "prediction_ordinal_applied": True,
                "prediction_ordinal_routed_fake_count": int(routed_fake.sum()),
                "prediction_ordinal_source_fake_count": int(source_fake.sum()),
                "prediction_ordinal_decision_disagreements": int(
                    np.count_nonzero(routed_fake != source_fake)
                ),
                "prediction_ordinal_label_mismatches": 0,
            }
        )
        ordinal = self._prediction_batch(
            scores,
            ordinal_probability,
            **pending_state,
        )
        label_mismatches = int(
            np.count_nonzero(
                ordinal.pred_label.detach().cpu().numpy()
                != routed.pred_label.detach().cpu().numpy()
            )
        )
        if label_mismatches:
            raise RuntimeError(
                "ASCAL ordinal readout changed an R11 routed hard decision"
            )
        return ordinal

    def adapt(self, images: Any) -> AdaptationStats:
        ordinal_state = None
        score_samples = 0
        if self._pending is not None:
            ordinal_state = {
                key: self._pending.get(key)
                for key in self._ORDINAL_PENDING_FIELDS
            }
            score_samples = int(np.asarray(self._pending["scores"]).size)

        stats = super().adapt(images)
        if ordinal_state is None:
            return stats

        if bool(ordinal_state["prediction_ordinal_applied"]):
            self.ordinal_batches += 1
            self.ordinal_samples += score_samples
        self.ordinal_decision_disagreements += int(
            ordinal_state["prediction_ordinal_decision_disagreements"] or 0
        )
        self.ordinal_label_mismatches += int(
            ordinal_state["prediction_ordinal_label_mismatches"] or 0
        )
        stats.extra.update(
            {
                **ordinal_state,
                "ordinal_batches": self.ordinal_batches,
                "ordinal_samples": self.ordinal_samples,
                "ordinal_decision_disagreements": (
                    self.ordinal_decision_disagreements
                ),
                "ordinal_label_mismatches": self.ordinal_label_mismatches,
            }
        )
        return stats


class ASCALGMMSegmentedMemoryPosteriorOrdinalRidge(
    ASCALGMMSegmentedMemoryPosteriorOrdinalRoute
):
    """Keep R12 decisions and learn one causal within-class ranker per expert."""

    _ORDINAL_RIDGE_MEMORY_KEY = "ordinal_ridge_state"

    def _reset_state(self) -> None:
        super()._reset_state()
        classifier = getattr(self.model, "classifier", None)
        weight = getattr(classifier, "weight", None)
        if weight is None or weight.ndim != 2 or int(weight.shape[0]) != 2:
            raise TypeError(
                "ASCAL ordinal ridge requires a two-class linear classifier"
            )
        direction = (
            weight[1].detach().float().cpu().numpy()
            - weight[0].detach().float().cpu().numpy()
        ).astype(np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if not math.isfinite(direction_norm) or direction_norm <= 0.0:
            raise ValueError(
                "ASCAL ordinal ridge requires a nonzero source score direction"
            )
        self._ordinal_ridge_source_direction = direction / direction_norm
        self.ordinal_ridge_feature_dim = int(direction.size)
        self._novel_ordinal_ridge_state = self._new_ordinal_ridge_state()
        self._ordinal_ridge_precomputed_scores: np.ndarray | None = None
        self._pending_ordinal_ridge_features: np.ndarray | None = None
        self._pending_ordinal_ridge_state: dict[str, Any] | None = None
        self._pending_ordinal_ridge_assignment: tuple[str, int | None] | None = None
        self._pending_ordinal_ridge_mixture: dict[str, Any] | None = None
        self.ordinal_ridge_updates = 0
        self.ordinal_ridge_candidate_samples = 0
        self.ordinal_ridge_batches = 0
        self.ordinal_ridge_samples = 0
        self.ordinal_ridge_solve_failures = 0
        self.ordinal_ridge_hard_label_mismatches = 0
        self.ordinal_ridge_last_effective_support = 0.0
        self.ordinal_ridge_last_reliability = 0.0
        self.ordinal_ridge_last_teacher_abs_mean = 0.0
        self.ordinal_ridge_last_source_abs_mean = 0.0
        self.ordinal_ridge_last_target_center = 0.0
        self.ordinal_ridge_last_target_abs_mean = 0.0
        self.ordinal_ridge_last_weight_norm = 0.0

    @property
    def trainable_parameters(self) -> int:
        if self.adaptation_mode == "static":
            return 0
        return len(self._all_ordinal_ridge_states()) * self.ordinal_ridge_feature_dim

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "r12_exact_hard_decisions_with_mdl_routed_causal_"
                    "within_class_online_ridge_ranking"
                ),
                "research_name": "ASCAL-JMP-OrdinalRidge",
                "research_version": "R15",
                "r12_protected_scope": (
                    "routing_admission_unique_live_state_hard_decision_"
                    "adaptation_assignment_segmentation_and_memory"
                ),
                "accuracy_invariance": (
                    "ordinal_half_interval_readout_preserves_every_r12_hard_"
                    "decision_exactly"
                ),
                "residual_scope": (
                    "one_zero_initialized_linear_rank_residual_per_r12_expert"
                ),
                "residual_input": (
                    "l2_normalized_frozen_features_orthogonal_to_the_source_"
                    "classifier_direction"
                ),
                "residual_teacher": (
                    "equal_prior_log_odds_of_the_prediction_time_selected_"
                    "r12_gmm"
                ),
                "residual_target_rule": (
                    "selected_gmm_teacher_logit_minus_immutable_source_logit_"
                    "then_reliability_weighted_batch_centering"
                ),
                "reliability_rule": (
                    "absolute_centered_selected_gmm_soft_posterior_no_threshold"
                ),
                "ridge_objective": (
                    "sum_reliability_times_squared_centered_logit_residual_"
                    "error_plus_unit_l2_weight_norm"
                ),
                "ridge_prior_precision": (
                    "fixed_identity_under_l2_normalized_residual_features"
                ),
                "ridge_update": (
                    "exact_recursive_least_squares_woodbury_after_prediction"
                ),
                "ridge_sufficient_statistics": (
                    "one_inverse_regularized_gram_matrix_and_one_weight_vector_"
                    "per_expert"
                ),
                "residual_intercept": "none",
                "prediction_residual_centering": (
                    "subtract_current_batch_mean_to_preserve_the_r12_global_"
                    "source_score_origin"
                ),
                "prediction_rule": (
                    "r12_expert_selects_real_or_fake_half_interval_while_source_"
                    "logit_plus_centered_selected_expert_ridge_orders_within_it"
                ),
                "source_fallback": "exact_r12_source_probability",
                "optimizer": "none_closed_form_recursive_ridge",
                "epoch": "none",
                "learning_rate": "none",
                "prediction_mutates_experts": False,
                "raw_images_stored": False,
                "raw_features_stored": False,
                "target_labels_used": False,
                "generator_boundaries_used": False,
                "semantic_features_used": False,
                "new_target_hyperparameters": 0,
                "hyperparameter_rule": (
                    "no_learning_rate_epoch_confidence_threshold_residual_weight_"
                    "routing_threshold_fusion_weight_or_memory_capacity"
                ),
                "intentional_changes": [
                    "all R12 routing decisions and learning assignments remain unchanged",
                    "the same R12 expert supplies the hard decision residual readout and update",
                    "the immutable source rank receives only a centered sample-specific correction",
                    (
                        "the R12 half-interval map prevents every residual from "
                        "crossing the decision threshold"
                    ),
                    (
                        "the selected pre-update GMM supplies a soft teacher and "
                        "continuous reliability"
                    ),
                    "only the prediction-time selected expert receives the batch after prediction",
                    (
                        "no image or per-sample feature is retained after its "
                        "sufficient-statistic update"
                    ),
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_ordinal_ridge"

    def _new_ordinal_ridge_state(self) -> dict[str, Any]:
        return {
            "inverse_gram": np.eye(
                self.ordinal_ridge_feature_dim,
                dtype=np.float64,
            ),
            "weights": np.zeros(self.ordinal_ridge_feature_dim, dtype=np.float64),
            "updates": 0,
            "candidate_samples": 0,
            "effective_support": 0.0,
            "weighted_target_square_sum": 0.0,
        }

    def _store_completed_segment(
        self,
        mixture: dict[str, Any],
        samples: int,
    ) -> int | None:
        was_novel = self.active_memory_index is None
        novel_state = self._novel_ordinal_ridge_state
        finalized_index = super()._store_completed_segment(mixture, samples)
        if (
            was_novel
            and finalized_index is not None
            and int(mixture["components"]) >= 2
        ):
            episode = self.segment_memories[finalized_index]
            episode.setdefault(self._ORDINAL_RIDGE_MEMORY_KEY, novel_state)
            self._novel_ordinal_ridge_state = self._new_ordinal_ridge_state()
        return finalized_index

    def _batch_scores_and_ordinal_ridge_features(
        self,
        images: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        if images.dim() == 5:
            batch, views = int(images.shape[0]), int(images.shape[1])
            flat = images.reshape(batch * views, *images.shape[2:])
        elif images.dim() == 4:
            batch, views = int(images.shape[0]), 1
            flat = images
        else:
            raise ValueError(
                "ASCAL ordinal ridge expects (B, C, H, W) or "
                "(B, V, C, H, W) images"
            )
        forward_features = getattr(self.model, "forward_features", None)
        classifier = getattr(self.model, "classifier", None)
        if not callable(forward_features) or not callable(classifier):
            raise TypeError(
                "ASCAL ordinal ridge requires forward_features and classifier"
            )
        with torch.no_grad():
            features = forward_features(flat.to(self.device, non_blocking=True))
            logits = classifier(features)
        scores = (
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
        if int(feature_values.shape[1]) != self.ordinal_ridge_feature_dim:
            raise ValueError(
                "ASCAL ordinal ridge feature dimension does not match the source head"
            )
        direction = self._ordinal_ridge_source_direction
        feature_values -= (feature_values @ direction)[:, None] * direction[None, :]
        norms = np.linalg.norm(feature_values, axis=1, keepdims=True)
        feature_values = np.divide(
            feature_values,
            norms,
            out=np.zeros_like(feature_values),
            where=norms > np.finfo(np.float64).eps,
        )
        return scores, feature_values

    def _batch_scores(self, images: Any) -> Any:
        if self._ordinal_ridge_precomputed_scores is not None:
            return self._ordinal_ridge_precomputed_scores.copy()
        return super()._batch_scores(images)

    def _peek_ordinal_ridge_state(
        self,
        assignment: tuple[str, int | None],
    ) -> dict[str, Any] | None:
        kind, memory_index = assignment
        if kind == "novel":
            return self._novel_ordinal_ridge_state
        if memory_index is None or not 0 <= memory_index < len(self.segment_memories):
            raise RuntimeError("ASCAL ordinal ridge memory index is out of range")
        state = self.segment_memories[memory_index].get(
            self._ORDINAL_RIDGE_MEMORY_KEY
        )
        return state if isinstance(state, dict) else None

    def _ensure_ordinal_ridge_state(
        self,
        assignment: tuple[str, int | None],
    ) -> dict[str, Any]:
        state = self._peek_ordinal_ridge_state(assignment)
        if state is not None:
            return state
        _, memory_index = assignment
        if memory_index is None:
            raise RuntimeError("ASCAL ordinal ridge lost its novel state")
        state = self._new_ordinal_ridge_state()
        self.segment_memories[memory_index][self._ORDINAL_RIDGE_MEMORY_KEY] = state
        return state

    def _all_ordinal_ridge_states(self) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        if int(self._novel_ordinal_ridge_state["candidate_samples"]) > 0:
            states.append(self._novel_ordinal_ridge_state)
        for episode in self.segment_memories:
            state = episode.get(self._ORDINAL_RIDGE_MEMORY_KEY)
            if isinstance(state, dict):
                states.append(state)
        return states

    @staticmethod
    def _ordinal_ridge_ready(state: dict[str, Any] | None) -> bool:
        if state is None or int(state["updates"]) <= 0:
            return False
        return float(np.linalg.norm(state["weights"])) > np.finfo(np.float64).eps

    def _ordinal_ridge_context(
        self,
    ) -> tuple[tuple[str, int | None], dict[str, Any]] | None:
        if self._pending is None:
            raise RuntimeError("ASCAL ordinal ridge lost its R12 prediction state")
        selected_expert = self._pending.get("prediction_routing_expert")
        if selected_expert is None:
            return None
        if selected_expert == "active_learning_state":
            if self._mixture is None or not self._mixture_active():
                raise RuntimeError("ASCAL ordinal ridge selected no eligible live GMM")
            memory_index = self.active_memory_index
            assignment = (
                "novel" if memory_index is None else "episodic_memory",
                memory_index,
            )
            return assignment, self._mixture
        if selected_expert != "episodic_memory":
            raise RuntimeError("ASCAL ordinal ridge received an unknown R12 expert")
        memory_index = self._pending.get("prediction_routing_memory_index")
        if memory_index is None:
            raise RuntimeError("ASCAL ordinal ridge selected memory without an index")
        memory_index = int(memory_index)
        if not 0 <= memory_index < len(self.segment_memories):
            raise RuntimeError("ASCAL ordinal ridge selected memory out of range")
        return ("episodic_memory", memory_index), self.segment_memories[
            memory_index
        ]["mixture"]

    def _ordinal_ridge_supervision(
        self,
        mixture: dict[str, Any],
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        source_logits = np.asarray(scores, dtype=np.float64) / self.temperature
        teacher_logits = np.clip(
            _joint_density_log_odds(scores, mixture),
            -60.0,
            60.0,
        )
        posterior = 1.0 / (1.0 + np.exp(-teacher_logits))
        reliability = np.abs(2.0 * posterior - 1.0)
        raw_targets = teacher_logits - source_logits
        effective_support = float(reliability.sum())
        target_center = 0.0
        if effective_support > np.finfo(np.float64).eps:
            target_center = float(
                np.sum(reliability * raw_targets) / effective_support
            )
        targets = raw_targets - target_center
        if not (
            np.all(np.isfinite(source_logits))
            and np.all(np.isfinite(targets))
        ):
            raise FloatingPointError(
                "ASCAL ordinal ridge produced non-finite supervision"
            )
        self.ordinal_ridge_last_effective_support = effective_support
        self.ordinal_ridge_last_reliability = float(np.mean(reliability))
        self.ordinal_ridge_last_teacher_abs_mean = float(
            np.mean(np.abs(teacher_logits))
        )
        self.ordinal_ridge_last_source_abs_mean = float(
            np.mean(np.abs(source_logits))
        )
        self.ordinal_ridge_last_target_center = target_center
        self.ordinal_ridge_last_target_abs_mean = float(np.mean(np.abs(targets)))
        return posterior, targets, reliability, target_center

    def _update_ordinal_ridge_state(
        self,
        state: dict[str, Any],
        mixture: dict[str, Any],
        scores: np.ndarray,
        features: np.ndarray,
    ) -> bool:
        _, targets, reliability, _ = self._ordinal_ridge_supervision(
            mixture,
            scores,
        )
        effective_support = float(reliability.sum())
        self.ordinal_ridge_candidate_samples += int(scores.size)
        state["candidate_samples"] = int(state["candidate_samples"]) + int(
            scores.size
        )
        if effective_support <= np.finfo(np.float64).eps:
            return False

        square_root_reliability = np.sqrt(reliability)
        design = square_root_reliability[:, None] * features
        response = square_root_reliability * targets
        inverse_gram = np.asarray(state["inverse_gram"], dtype=np.float64)
        weights = np.asarray(state["weights"], dtype=np.float64)
        inverse_times_design = inverse_gram @ design.T
        innovation_gram = (
            np.eye(int(scores.size), dtype=np.float64)
            + design @ inverse_times_design
        )
        innovation_gram = 0.5 * (innovation_gram + innovation_gram.T)
        try:
            gain = np.linalg.solve(
                innovation_gram,
                inverse_times_design.T,
            ).T
        except np.linalg.LinAlgError:
            self.ordinal_ridge_solve_failures += 1
            return False

        updated_weights = weights + gain @ (response - design @ weights)
        updated_inverse = inverse_gram - gain @ inverse_times_design.T
        updated_inverse = 0.5 * (updated_inverse + updated_inverse.T)
        if not (
            np.all(np.isfinite(updated_weights))
            and np.all(np.isfinite(updated_inverse))
        ):
            self.ordinal_ridge_solve_failures += 1
            return False

        state["weights"] = updated_weights
        state["inverse_gram"] = updated_inverse
        state["updates"] = int(state["updates"]) + 1
        state["effective_support"] = float(state["effective_support"]) + (
            effective_support
        )
        state["weighted_target_square_sum"] = float(
            state["weighted_target_square_sum"]
        ) + float(np.sum(reliability * targets**2))
        self.ordinal_ridge_updates += 1
        self.ordinal_ridge_last_weight_norm = float(
            np.linalg.norm(updated_weights)
        )
        return self._ordinal_ridge_ready(state)

    def _ordinal_ridge_state_stats(self) -> dict[str, Any]:
        states = self._all_ordinal_ridge_states()
        weight_norms = [float(np.linalg.norm(state["weights"])) for state in states]
        return {
            "ordinal_ridge_expert_count": len(states),
            "ordinal_ridge_ready_experts": sum(
                self._ordinal_ridge_ready(state) for state in states
            ),
            "ordinal_ridge_updates": self.ordinal_ridge_updates,
            "ordinal_ridge_candidate_samples": self.ordinal_ridge_candidate_samples,
            "ordinal_ridge_batches": self.ordinal_ridge_batches,
            "ordinal_ridge_samples": self.ordinal_ridge_samples,
            "ordinal_ridge_solve_failures": self.ordinal_ridge_solve_failures,
            "ordinal_ridge_hard_label_mismatches": (
                self.ordinal_ridge_hard_label_mismatches
            ),
            "ordinal_ridge_effective_support": sum(
                float(state["effective_support"]) for state in states
            ),
            "ordinal_ridge_last_effective_support": (
                self.ordinal_ridge_last_effective_support
            ),
            "ordinal_ridge_last_reliability": self.ordinal_ridge_last_reliability,
            "ordinal_ridge_last_teacher_abs_mean": (
                self.ordinal_ridge_last_teacher_abs_mean
            ),
            "ordinal_ridge_last_source_abs_mean": (
                self.ordinal_ridge_last_source_abs_mean
            ),
            "ordinal_ridge_last_target_center": (
                self.ordinal_ridge_last_target_center
            ),
            "ordinal_ridge_last_target_abs_mean": (
                self.ordinal_ridge_last_target_abs_mean
            ),
            "ordinal_ridge_max_weight_norm": max(weight_norms, default=0.0),
            "ordinal_ridge_mean_weight_norm": (
                float(np.mean(weight_norms)) if weight_norms else 0.0
            ),
            "ordinal_ridge_weight_parameters": (
                len(states) * self.ordinal_ridge_feature_dim
            ),
            "ordinal_ridge_inverse_gram_values": (
                len(states) * self.ordinal_ridge_feature_dim**2
            ),
            "ordinal_ridge_trainable_parameters": self.trainable_parameters,
        }

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(self._ordinal_ridge_state_stats())
        return stats

    def predict(self, images: Any) -> PredictionBatch:
        scores, features = self._batch_scores_and_ordinal_ridge_features(images)
        self._ordinal_ridge_precomputed_scores = scores
        try:
            ordinal = super().predict(images)
        finally:
            self._ordinal_ridge_precomputed_scores = None
        if self._pending is None:
            raise RuntimeError("ASCAL ordinal ridge lost the R12 pending state")

        context = self._ordinal_ridge_context()
        assignment = None if context is None else context[0]
        mixture = None if context is None else context[1]
        state = (
            None
            if assignment is None
            else self._peek_ordinal_ridge_state(assignment)
        )
        ready = self._ordinal_ridge_ready(state)
        if ready:
            if state is None:
                raise RuntimeError("ASCAL ordinal ridge lost its selected state")
            raw_residual = np.asarray(features @ state["weights"], dtype=np.float64)
        else:
            raw_residual = np.zeros(int(scores.size), dtype=np.float64)
        residual_center = float(np.mean(raw_residual))
        centered_residual = raw_residual - residual_center

        if context is None:
            probability = ordinal.prob_fake.detach().cpu().numpy().astype(np.float64)
        else:
            rank_probability = 1.0 / (
                1.0
                + np.exp(
                    -np.clip(
                        scores / self.temperature + centered_residual,
                        -60.0,
                        60.0,
                    )
                )
            )
            rank_probability = np.clip(rank_probability, 1e-6, 1.0 - 1e-6)
            routed_fake = (
                ordinal.pred_label.detach().cpu().numpy().astype(np.float64)
            )
            probability = 0.5 * (rank_probability + routed_fake)

        self._pending_ordinal_ridge_features = features
        self._pending_ordinal_ridge_state = state
        self._pending_ordinal_ridge_assignment = assignment
        self._pending_ordinal_ridge_mixture = (
            None if mixture is None else _copy_gmm(mixture)
        )
        pending_state = dict(self._pending)
        pending_state.pop("scores")
        pending_state.update(
            {
                "prediction_ordinal_ridge_applied": context is not None,
                "prediction_ordinal_ridge_ready": ready,
                "prediction_ordinal_ridge_residual_center": residual_center,
                "prediction_ordinal_ridge_residual_mean": float(
                    np.mean(centered_residual)
                ),
                "prediction_ordinal_ridge_residual_abs_mean": float(
                    np.mean(np.abs(centered_residual))
                ),
                "prediction_ordinal_ridge_residual_max_abs": float(
                    np.max(np.abs(centered_residual))
                ),
                "prediction_ordinal_ridge_hard_label_mismatches": 0,
            }
        )
        prediction = self._prediction_batch(scores, probability, **pending_state)
        mismatches = int(
            np.count_nonzero(
                prediction.pred_label.detach().cpu().numpy()
                != ordinal.pred_label.detach().cpu().numpy()
            )
        )
        if mismatches:
            raise RuntimeError("ASCAL ordinal ridge changed an R12 hard decision")
        return prediction

    def adapt(self, images: Any) -> AdaptationStats:
        if self._pending is None:
            return super().adapt(images)
        scores = np.asarray(self._pending["scores"], dtype=np.float64).reshape(-1)
        prediction_state = dict(self._pending)
        features = self._pending_ordinal_ridge_features
        state = self._pending_ordinal_ridge_state
        assignment = self._pending_ordinal_ridge_assignment
        mixture = self._pending_ordinal_ridge_mixture
        self._pending_ordinal_ridge_features = None
        self._pending_ordinal_ridge_state = None
        self._pending_ordinal_ridge_assignment = None
        self._pending_ordinal_ridge_mixture = None

        updated = False
        if self.adaptation_mode == "full" and assignment is not None:
            if features is None or int(features.shape[0]) != int(scores.size):
                raise RuntimeError(
                    "ASCAL ordinal ridge lost its matching prediction features"
                )
            if mixture is None:
                raise RuntimeError(
                    "ASCAL ordinal ridge lost its prediction-time selected GMM"
                )
            if state is None:
                state = self._ensure_ordinal_ridge_state(assignment)
            updated = self._update_ordinal_ridge_state(
                state,
                mixture,
                scores,
                features,
            )

        stats = super().adapt(images)
        if bool(prediction_state.get("prediction_ordinal_ridge_applied")):
            self.ordinal_ridge_batches += 1
            self.ordinal_ridge_samples += int(scores.size)
        mismatches = int(
            prediction_state.get(
                "prediction_ordinal_ridge_hard_label_mismatches",
                0,
            )
            or 0
        )
        self.ordinal_ridge_hard_label_mismatches += mismatches
        stats.extra.update(
            {
                **self._ordinal_ridge_state_stats(),
                "ordinal_ridge_updated": updated,
            }
        )
        return stats

    def discard_pending_prediction(self) -> None:
        self._ordinal_ridge_precomputed_scores = None
        self._pending_ordinal_ridge_features = None
        self._pending_ordinal_ridge_state = None
        self._pending_ordinal_ridge_assignment = None
        self._pending_ordinal_ridge_mixture = None
        super().discard_pending_prediction()


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

    def _residual_ready(self) -> bool:
        return self.residual_vector is not None

    def _residual_state_stats(self) -> dict[str, Any]:
        return {
            "residual_ready": self._residual_ready(),
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
            prediction_residual_ready=self._residual_ready(),
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


class ASCALGMMSegmentedMemoryPosteriorMixtureResidual(
    ASCALGMMSegmentedMemoryPosteriorGlobalResidual
):
    """Keep BIC-selected fake modes separate inside one causal residual."""

    def _reset_state(self) -> None:
        super()._reset_state()
        shape = (self.max_fake_components, self.residual_feature_dim)
        self.residual_fake_component_sums = np.zeros(shape, dtype=np.float64)
        self.residual_fake_component_supports = np.zeros(
            self.max_fake_components,
            dtype=np.float64,
        )
        self.residual_last_fake_component_supports = np.zeros(
            self.max_fake_components,
            dtype=np.float64,
        )
        self.residual_fake_components_observed = 0
        self.residual_multimodal_updates = 0

    @property
    def trainable_parameters(self) -> int:
        if self.adaptation_mode == "static":
            return 0
        return (1 + self.max_fake_components) * self.residual_feature_dim

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "immutable_source_score_gmm_with_one_causal_multimodal_"
                    "feature_rank_residual"
                ),
                "research_name": "ASCAL-JMP-MixtureResidual",
                "research_version": "R06",
                "residual_count": 1,
                "residual_scope": (
                    "one_shared_multimodal_readout_for_the_entire_continual_stream"
                ),
                "real_prototype_rule": (
                    "one_cumulative_soft_prototype_for_the_complete_real_gmm_block"
                ),
                "fake_prototype_rule": (
                    "one_cumulative_feature_prototype_per_ordered_bic_selected_"
                    "fake_gmm_component"
                ),
                "fake_component_count_rule": (
                    "target_bic_with_the_existing_source_anchor_complexity_cap"
                ),
                "component_identity_rule": (
                    "persistent_rank_inside_the_ordered_fake_score_block"
                ),
                "residual_update": (
                    "equal_prior_class_posterior_times_within_fake_block_"
                    "component_responsibility_no_optimizer"
                ),
                "residual_readout": (
                    "maximum_fake_component_cosine_minus_real_prototype_cosine"
                ),
                "prediction_rule": "r01_base_logit_plus_one_mixture_residual",
                "hyperparameter_rule": (
                    "no_component_count_residual_learning_rate_loss_weight_"
                    "confidence_threshold_memory_capacity_or_fusion_weight"
                ),
                "intentional_changes": [
                    "the R01 source-score GMM segmentation memory and boundary trajectory remain unchanged",
                    "only immutable source margins are appended to score history",
                    "the complete real score block still forms one stream-wide feature prototype",
                    "BIC-selected fake score components retain separate stream-wide feature prototypes",
                    "ordered fake-component rank gives deterministic identity without target labels or generator ids",
                    "the closest fake prototype supplies one bounded nonlinear rank residual",
                    "the residual remains a one-way student of immutable source-score density evidence",
                    "predictions use only GMM and residual state learned from earlier batches",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_mixture_residual"

    def _normalized_residual_prototypes(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        epsilon = np.finfo(np.float64).eps
        if self.residual_real_support <= epsilon:
            return None, np.empty((0, self.residual_feature_dim), dtype=np.float64)
        real_mean = self.residual_real_sum / self.residual_real_support
        real_norm = float(np.linalg.norm(real_mean))
        if real_norm <= epsilon:
            return None, np.empty((0, self.residual_feature_dim), dtype=np.float64)

        ready = self.residual_fake_component_supports > epsilon
        fake_means = np.divide(
            self.residual_fake_component_sums[ready],
            self.residual_fake_component_supports[ready, None],
        )
        if not len(fake_means):
            return real_mean / real_norm, fake_means
        fake_norms = np.linalg.norm(fake_means, axis=1)
        valid = fake_norms > epsilon
        return real_mean / real_norm, np.divide(
            fake_means[valid],
            fake_norms[valid, None],
        )

    def _residual_ready(self) -> bool:
        real_prototype, fake_prototypes = self._normalized_residual_prototypes()
        return real_prototype is not None and len(fake_prototypes) > 0

    def _residual_scores(self, features: np.ndarray) -> np.ndarray:
        real_prototype, fake_prototypes = self._normalized_residual_prototypes()
        if real_prototype is None or not len(fake_prototypes):
            return np.zeros(int(features.shape[0]), dtype=np.float64)
        fake_similarity = np.max(features @ fake_prototypes.T, axis=1)
        real_similarity = features @ real_prototype
        return np.asarray(fake_similarity - real_similarity, dtype=np.float64)

    def _residual_state_stats(self) -> dict[str, Any]:
        real_prototype, fake_prototypes = self._normalized_residual_prototypes()
        if real_prototype is None or not len(fake_prototypes):
            residual_norm = 0.0
        else:
            residual_norm = float(
                np.max(np.linalg.norm(fake_prototypes - real_prototype, axis=1))
            )
        return {
            "residual_ready": real_prototype is not None and len(fake_prototypes) > 0,
            "residual_count": 1,
            "residual_feature_dim": self.residual_feature_dim,
            "residual_updates": self.residual_updates,
            "residual_candidate_samples": self.residual_candidate_samples,
            "residual_real_support": self.residual_real_support,
            "residual_fake_support": self.residual_fake_support,
            "residual_norm": residual_norm,
            "residual_last_reliability": self.residual_last_reliability,
            "residual_last_real_support": self.residual_last_real_support,
            "residual_last_fake_support": self.residual_last_fake_support,
            "residual_fake_prototype_count": int(len(fake_prototypes)),
            "residual_fake_components_observed": self.residual_fake_components_observed,
            "residual_fake_component_supports": (
                self.residual_fake_component_supports.tolist()
            ),
            "residual_last_fake_component_supports": (
                self.residual_last_fake_component_supports.tolist()
            ),
            "residual_multimodal_updates": self.residual_multimodal_updates,
            "residual_readout_bound": 2.0,
        }

    def _update_global_residual(
        self,
        scores: np.ndarray,
        features: np.ndarray,
    ) -> bool:
        if self._mixture is None or not self._mixture_active():
            return False
        partition = dominant_gap_boundary(self._mixture)
        split = int(partition["real_components"])
        fake_components = int(partition["fake_components"])
        if fake_components > self.max_fake_components:
            raise RuntimeError("Target GMM exceeded the fixed fake-component cap")

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
        self.residual_last_fake_component_supports.fill(0.0)
        epsilon = np.finfo(np.float64).eps
        if fake_support <= epsilon or real_support <= epsilon:
            return False

        weights = np.asarray(self._mixture["weights"], dtype=np.float64)
        mus = np.asarray(self._mixture["mus"], dtype=np.float64)
        sigmas = np.asarray(self._mixture["sigmas"], dtype=np.float64)
        conditional = _block_component_responsibilities(
            scores,
            weights[split:],
            mus[split:],
            sigmas[split:],
        )
        component_weights = fake_weights[:, None] * conditional
        component_supports = component_weights.sum(axis=0)

        self.residual_real_sum += np.sum(real_weights[:, None] * features, axis=0)
        self.residual_real_support += real_support
        for index in range(fake_components):
            support = float(component_supports[index])
            self.residual_last_fake_component_supports[index] = support
            if support <= epsilon:
                continue
            self.residual_fake_component_sums[index] += np.sum(
                component_weights[:, index, None] * features,
                axis=0,
            )
            self.residual_fake_component_supports[index] += support
        self.residual_fake_support += fake_support
        self.residual_fake_components_observed = max(
            self.residual_fake_components_observed,
            fake_components,
        )
        if fake_components > 1:
            self.residual_multimodal_updates += 1
        if not self._residual_ready():
            return False
        self.residual_updates += 1
        return True


class ASCALGMMSegmentedMemoryPosteriorRoutedResidual(
    ASCALGMMSegmentedMemoryPosteriorProjection
):
    """Route one causal feature residual without routing the score boundary."""

    _RESIDUAL_MEMORY_KEY = "routed_residual_state"

    def _reset_state(self) -> None:
        super()._reset_state()
        classifier = getattr(self.model, "classifier", None)
        weight = getattr(classifier, "weight", None)
        if weight is None or weight.ndim != 2 or int(weight.shape[0]) != 2:
            raise TypeError(
                "ASCAL routed residual requires a two-class linear classifier"
            )
        direction = (
            weight[1].detach().float().cpu().numpy()
            - weight[0].detach().float().cpu().numpy()
        ).astype(np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if not math.isfinite(direction_norm) or direction_norm <= 0.0:
            raise ValueError(
                "ASCAL routed residual requires a nonzero source score direction"
            )
        self._source_feature_direction = direction / direction_norm
        self.residual_feature_dim = int(direction.size)
        self._novel_routed_residual_state = self._new_routed_residual_state()
        self._pending_routed_residual_features: np.ndarray | None = None
        self._pending_routed_residual_state: dict[str, Any] | None = None
        self._pending_routed_residual_assignment: tuple[str, int | None] | None = (
            None
        )
        self._pending_routed_residual_mixture: dict[str, Any] | None = None
        self.routed_residual_routing_decisions = 0
        self.routed_residual_active_selections = 0
        self.routed_residual_memory_selections = 0
        self.routed_residual_source_fallbacks = 0
        self.routed_residual_memory_proposals = 0
        self.routed_residual_admission_checks = 0
        self.routed_residual_admission_accepts = 0
        self.routed_residual_admission_rejects = 0
        self.routed_residual_admission_fit_failures = 0
        self.routed_residual_admission_unimodal = 0
        self.routed_residual_updates = 0
        self.routed_residual_candidate_samples = 0
        self.routed_residual_multimodal_updates = 0
        self.routed_residual_last_reliability = 0.0
        self.routed_residual_last_real_support = 0.0
        self.routed_residual_last_fake_support = 0.0
        self.routed_residual_last_fake_component_supports = np.zeros(
            self.max_fake_components,
            dtype=np.float64,
        )
        self.last_routed_residual_expert: str | None = None
        self.last_routed_residual_memory_index: int | None = None
        self.last_routed_residual_admission_reason: str | None = None

    @property
    def trainable_parameters(self) -> int:
        # The bank contains closed-form sufficient statistics, not optimizer
        # parameters or a trainable network.
        return 0

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "immutable_source_score_calibration_with_mdl_routed_"
                    "causal_multimodal_feature_residuals"
                ),
                "research_name": "ASCAL-JMP-RoutedResidual",
                "research_version": "R13",
                "immutable_history_coordinate": "frozen_source_logit_margin_only",
                "base_prediction_rule": (
                    "exact_r01_median_stabilized_posterior_boundary_projection"
                ),
                "score_boundary_routing": False,
                "residual_routing_coordinate": (
                    "current_batch_immutable_frozen_source_logit_margin"
                ),
                "residual_routing_candidates": (
                    "one_active_live_gmm_plus_archived_non_active_gmms"
                ),
                "residual_routing_rule": (
                    "minimum_fixed_deviance_with_parameter_free_mdl_admission_"
                    "for_non_active_memory"
                ),
                "residual_assignment_consistency": (
                    "the_same_selected_expert_reads_and_receives_the_current_batch"
                ),
                "residual_scope": (
                    "one_compact_real_plus_bic_fake_prototype_state_per_"
                    "discovered_score_expert"
                ),
                "residual_input": (
                    "l2_normalized_frozen_features_orthogonal_to_the_source_"
                    "classifier_direction"
                ),
                "residual_teacher": (
                    "equal_prior_posterior_of_the_prediction_time_selected_"
                    "immutable_source_score_gmm"
                ),
                "reliability_rule": "absolute_centered_soft_posterior_no_threshold",
                "fake_prototype_rule": (
                    "one_feature_prototype_per_ordered_bic_selected_fake_component"
                ),
                "residual_readout": (
                    "maximum_fake_component_cosine_minus_real_prototype_cosine"
                ),
                "prediction_rule": (
                    "r01_continuous_base_logit_plus_one_selected_expert_residual"
                ),
                "expert_network_count": 0,
                "optimizer": "none",
                "prediction_mutates_experts": False,
                "batch_transductive_prediction": True,
                "raw_images_stored": False,
                "raw_features_stored": False,
                "target_labels_used": False,
                "generator_boundaries_used": False,
                "semantic_features_used": False,
                "new_target_hyperparameters": 0,
                "hyperparameter_rule": (
                    "no_residual_learning_rate_loss_weight_confidence_threshold_"
                    "routing_threshold_fusion_weight_or_memory_capacity"
                ),
                "intentional_changes": [
                    "the R01 score GMM segmentation memory and boundary trajectory remain unchanged",
                    "the current batch selects a residual expert before its official prediction",
                    "a returning residual expert must pass the R11 parameter-free MDL admission rule",
                    "expert boundaries never translate the final score coordinate",
                    "only the selected compact prototype state receives the batch after prediction",
                    "pseudo labels come only from the selected pre-update source-score GMM",
                    "the residual bank stores sufficient statistics rather than images or raw features",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_routed_residual"

    def _new_routed_residual_state(self) -> dict[str, Any]:
        return {
            "real_sum": np.zeros(self.residual_feature_dim, dtype=np.float64),
            "real_support": 0.0,
            "fake_component_sums": np.zeros(
                (self.max_fake_components, self.residual_feature_dim),
                dtype=np.float64,
            ),
            "fake_component_supports": np.zeros(
                self.max_fake_components,
                dtype=np.float64,
            ),
            "updates": 0,
            "candidate_samples": 0,
            "multimodal_updates": 0,
        }

    def _store_completed_segment(
        self,
        mixture: dict[str, Any],
        samples: int,
    ) -> int | None:
        was_novel = self.active_memory_index is None
        novel_state = self._novel_routed_residual_state
        finalized_index = super()._store_completed_segment(mixture, samples)
        if (
            was_novel
            and finalized_index is not None
            and int(mixture["components"]) >= 2
        ):
            episode = self.segment_memories[finalized_index]
            episode.setdefault(self._RESIDUAL_MEMORY_KEY, novel_state)
            self._novel_routed_residual_state = self._new_routed_residual_state()
        return finalized_index

    def _batch_scores_and_routed_residual_features(
        self,
        images: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        if images.dim() == 5:
            batch, views = int(images.shape[0]), int(images.shape[1])
            flat = images.reshape(batch * views, *images.shape[2:])
        elif images.dim() == 4:
            batch, views = int(images.shape[0]), 1
            flat = images
        else:
            raise ValueError(
                "ASCAL routed residual expects (B, C, H, W) or "
                "(B, V, C, H, W) images"
            )
        forward_features = getattr(self.model, "forward_features", None)
        classifier = getattr(self.model, "classifier", None)
        if not callable(forward_features) or not callable(classifier):
            raise TypeError(
                "ASCAL routed residual requires forward_features and classifier"
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
                "ASCAL routed residual feature dimension does not match the source head"
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

    def _routed_residual_candidates(
        self,
        scores: np.ndarray,
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        active_eligible = self._mixture_active() and self._mixture is not None
        if active_eligible:
            candidates.append(
                {
                    "expert": "active_learning_state",
                    "memory_index": self.active_memory_index,
                    "assignment": (
                        "novel"
                        if self.active_memory_index is None
                        else "episodic_memory",
                        self.active_memory_index,
                    ),
                    "mixture": self._mixture,
                    "deviance": _fixed_gmm_deviance(scores, self._mixture),
                }
            )
        for index, episode in enumerate(self.segment_memories):
            mixture = episode["mixture"]
            if int(mixture["components"]) < 2:
                continue
            if active_eligible and index == self.active_memory_index:
                continue
            candidates.append(
                {
                    "expert": "episodic_memory",
                    "memory_index": index,
                    "assignment": ("episodic_memory", index),
                    "mixture": mixture,
                    "deviance": _fixed_gmm_deviance(scores, mixture),
                }
            )
        return candidates

    @staticmethod
    def _empty_routed_residual_admission(reason: str) -> dict[str, Any]:
        return {
            "checked": False,
            "accepted": False,
            "reason": reason,
            "memory_index": None,
            "fixed_score": None,
            "new_bic": None,
            "identity_penalty": None,
            "gain": None,
            "new_components": None,
            "fit_failure": False,
            "unimodal": False,
        }

    def _routed_residual_memory_admission(
        self,
        scores: np.ndarray,
        memory_index: int,
    ) -> dict[str, Any]:
        eligible_memories = sum(
            int(episode["mixture"]["components"]) >= 2
            for episode in self.segment_memories
        )
        if eligible_memories < 1:
            raise RuntimeError("ASCAL routed residual has no eligible memory")
        evidence = self._empty_routed_residual_admission("fit_failure")
        evidence.update({"checked": True, "memory_index": memory_index})
        try:
            new_mixture = fit_gmm_bic(
                scores,
                max_components=min(self.max_total_components, int(scores.size)),
                seed=0,
            )
        except (FloatingPointError, RuntimeError, ValueError):
            evidence["fit_failure"] = True
            return evidence

        components = int(new_mixture["components"])
        evidence["new_bic"] = float(new_mixture["bic"])
        evidence["new_components"] = components
        if components < 2:
            evidence["reason"] = "current_batch_unimodal"
            evidence["unimodal"] = True
            return evidence

        identity_penalty = 2.0 * math.log(eligible_memories)
        fixed_score = _fixed_gmm_deviance(
            scores,
            self.segment_memories[memory_index]["mixture"],
        ) + identity_penalty
        gain = float(new_mixture["bic"]) - float(fixed_score)
        accepted = gain > 0.0
        evidence.update(
            {
                "accepted": accepted,
                "reason": (
                    "memory_description_shorter"
                    if accepted
                    else "new_state_description_shorter_or_equal"
                ),
                "fixed_score": float(fixed_score),
                "identity_penalty": float(identity_penalty),
                "gain": float(gain),
            }
        )
        return evidence

    def _select_routed_residual_expert(
        self,
        scores: np.ndarray,
    ) -> tuple[
        list[dict[str, Any]],
        dict[str, Any] | None,
        dict[str, Any] | None,
        dict[str, Any],
    ]:
        candidates = self._routed_residual_candidates(scores)
        proposal = None
        selected = None
        admission = self._empty_routed_residual_admission("no_routing_candidate")
        if not candidates:
            return candidates, proposal, selected, admission

        proposal = min(candidates, key=lambda candidate: candidate["deviance"])
        active = next(
            (
                candidate
                for candidate in candidates
                if candidate["expert"] == "active_learning_state"
            ),
            None,
        )
        if proposal["expert"] == "episodic_memory":
            memory_index = int(proposal["memory_index"])
            if memory_index == self.active_memory_index:
                selected = proposal
                admission = self._empty_routed_residual_admission(
                    "already_active_memory_identity"
                )
                admission.update(
                    {"accepted": True, "memory_index": memory_index}
                )
            else:
                admission = self._routed_residual_memory_admission(
                    scores,
                    memory_index,
                )
                selected = proposal if admission["accepted"] else active
        else:
            selected = proposal
            admission = self._empty_routed_residual_admission(
                "active_state_wins_deviance"
            )
        return candidates, proposal, selected, admission

    def _peek_routed_residual_state(
        self,
        assignment: tuple[str, int | None],
    ) -> dict[str, Any] | None:
        kind, memory_index = assignment
        if kind == "novel":
            return self._novel_routed_residual_state
        if memory_index is None or not 0 <= memory_index < len(self.segment_memories):
            raise RuntimeError("ASCAL routed residual memory index is out of range")
        state = self.segment_memories[memory_index].get(self._RESIDUAL_MEMORY_KEY)
        return state if isinstance(state, dict) else None

    def _ensure_routed_residual_state(
        self,
        assignment: tuple[str, int | None],
    ) -> dict[str, Any]:
        state = self._peek_routed_residual_state(assignment)
        if state is not None:
            return state
        _, memory_index = assignment
        if memory_index is None:
            raise RuntimeError("ASCAL routed residual lost its novel state")
        state = self._new_routed_residual_state()
        self.segment_memories[memory_index][self._RESIDUAL_MEMORY_KEY] = state
        return state

    def _normalized_routed_residual_prototypes(
        self,
        state: dict[str, Any] | None,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        if state is None:
            return None, np.empty((0, self.residual_feature_dim), dtype=np.float64)
        epsilon = np.finfo(np.float64).eps
        real_support = float(state["real_support"])
        if real_support <= epsilon:
            return None, np.empty((0, self.residual_feature_dim), dtype=np.float64)
        real_mean = np.asarray(state["real_sum"], dtype=np.float64) / real_support
        real_norm = float(np.linalg.norm(real_mean))
        if real_norm <= epsilon:
            return None, np.empty((0, self.residual_feature_dim), dtype=np.float64)

        supports = np.asarray(
            state["fake_component_supports"],
            dtype=np.float64,
        )
        ready = supports > epsilon
        fake_means = np.divide(
            np.asarray(state["fake_component_sums"], dtype=np.float64)[ready],
            supports[ready, None],
        )
        if not len(fake_means):
            return real_mean / real_norm, fake_means
        fake_norms = np.linalg.norm(fake_means, axis=1)
        valid = fake_norms > epsilon
        return real_mean / real_norm, np.divide(
            fake_means[valid],
            fake_norms[valid, None],
        )

    def _routed_residual_scores(
        self,
        features: np.ndarray,
        state: dict[str, Any] | None,
    ) -> np.ndarray:
        real_prototype, fake_prototypes = (
            self._normalized_routed_residual_prototypes(state)
        )
        if real_prototype is None or not len(fake_prototypes):
            return np.zeros(int(features.shape[0]), dtype=np.float64)
        fake_similarity = np.max(features @ fake_prototypes.T, axis=1)
        real_similarity = features @ real_prototype
        return np.asarray(fake_similarity - real_similarity, dtype=np.float64)

    def _routed_residual_ready(self, state: dict[str, Any] | None) -> bool:
        real_prototype, fake_prototypes = (
            self._normalized_routed_residual_prototypes(state)
        )
        return real_prototype is not None and len(fake_prototypes) > 0

    def _routed_residual_readout_count(
        self,
        state: dict[str, Any],
    ) -> int:
        _, fake_prototypes = self._normalized_routed_residual_prototypes(state)
        return int(len(fake_prototypes))

    def _update_routed_residual_state(
        self,
        state: dict[str, Any],
        mixture: dict[str, Any],
        scores: np.ndarray,
        features: np.ndarray,
    ) -> bool:
        partition = dominant_gap_boundary(mixture)
        split = int(partition["real_components"])
        fake_components = int(partition["fake_components"])
        if fake_components > self.max_fake_components:
            raise RuntimeError("Target GMM exceeded the fixed fake-component cap")

        posterior = joint_density_fake_posterior(scores, mixture)
        reliability = np.abs(2.0 * posterior - 1.0)
        fake_weights = reliability * posterior
        real_weights = reliability * (1.0 - posterior)
        fake_support = float(fake_weights.sum())
        real_support = float(real_weights.sum())
        self.routed_residual_candidate_samples += int(scores.size)
        self.routed_residual_last_reliability = float(np.mean(reliability))
        self.routed_residual_last_real_support = real_support
        self.routed_residual_last_fake_support = fake_support
        self.routed_residual_last_fake_component_supports.fill(0.0)
        state["candidate_samples"] = int(state["candidate_samples"]) + int(
            scores.size
        )
        epsilon = np.finfo(np.float64).eps
        if fake_support <= epsilon or real_support <= epsilon:
            return False

        weights = np.asarray(mixture["weights"], dtype=np.float64)
        mus = np.asarray(mixture["mus"], dtype=np.float64)
        sigmas = np.asarray(mixture["sigmas"], dtype=np.float64)
        conditional = _block_component_responsibilities(
            scores,
            weights[split:],
            mus[split:],
            sigmas[split:],
        )
        component_weights = fake_weights[:, None] * conditional
        component_supports = component_weights.sum(axis=0)

        state["real_sum"] += np.sum(real_weights[:, None] * features, axis=0)
        state["real_support"] = float(state["real_support"]) + real_support
        for index in range(fake_components):
            support = float(component_supports[index])
            self.routed_residual_last_fake_component_supports[index] = support
            if support <= epsilon:
                continue
            state["fake_component_sums"][index] += np.sum(
                component_weights[:, index, None] * features,
                axis=0,
            )
            state["fake_component_supports"][index] += support
        if fake_components > 1:
            state["multimodal_updates"] = int(state["multimodal_updates"]) + 1
            self.routed_residual_multimodal_updates += 1
        if not self._routed_residual_ready(state):
            return False
        state["updates"] = int(state["updates"]) + 1
        self.routed_residual_updates += 1
        return True

    def _all_routed_residual_states(self) -> list[dict[str, Any]]:
        states: list[dict[str, Any]] = []
        if int(self._novel_routed_residual_state["candidate_samples"]) > 0:
            states.append(self._novel_routed_residual_state)
        for episode in self.segment_memories:
            state = episode.get(self._RESIDUAL_MEMORY_KEY)
            if isinstance(state, dict):
                states.append(state)
        return states

    def _routed_residual_state_stats(self) -> dict[str, Any]:
        states = self._all_routed_residual_states()
        ready = 0
        fake_prototypes = 0
        for state in states:
            if self._routed_residual_ready(state):
                ready += 1
                fake_prototypes += self._routed_residual_readout_count(state)
        return {
            "routed_residual_expert_count": len(states),
            "routed_residual_ready_experts": ready,
            "routed_residual_fake_prototype_count": fake_prototypes,
            "routed_residual_updates": self.routed_residual_updates,
            "routed_residual_candidate_samples": (
                self.routed_residual_candidate_samples
            ),
            "routed_residual_multimodal_updates": (
                self.routed_residual_multimodal_updates
            ),
            "routed_residual_routing_decisions": (
                self.routed_residual_routing_decisions
            ),
            "routed_residual_active_selections": (
                self.routed_residual_active_selections
            ),
            "routed_residual_memory_selections": (
                self.routed_residual_memory_selections
            ),
            "routed_residual_source_fallbacks": (
                self.routed_residual_source_fallbacks
            ),
            "routed_residual_memory_proposals": (
                self.routed_residual_memory_proposals
            ),
            "routed_residual_admission_checks": (
                self.routed_residual_admission_checks
            ),
            "routed_residual_admission_accepts": (
                self.routed_residual_admission_accepts
            ),
            "routed_residual_admission_rejects": (
                self.routed_residual_admission_rejects
            ),
            "routed_residual_admission_fit_failures": (
                self.routed_residual_admission_fit_failures
            ),
            "routed_residual_admission_unimodal": (
                self.routed_residual_admission_unimodal
            ),
            "routed_residual_last_reliability": (
                self.routed_residual_last_reliability
            ),
            "routed_residual_last_real_support": (
                self.routed_residual_last_real_support
            ),
            "routed_residual_last_fake_support": (
                self.routed_residual_last_fake_support
            ),
            "routed_residual_last_fake_component_supports": (
                self.routed_residual_last_fake_component_supports.tolist()
            ),
            "last_routed_residual_expert": self.last_routed_residual_expert,
            "last_routed_residual_memory_index": (
                self.last_routed_residual_memory_index
            ),
            "last_routed_residual_admission_reason": (
                self.last_routed_residual_admission_reason
            ),
            "routed_residual_readout_bound": 2.0,
            "routed_residual_trainable_parameters": 0,
        }

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(self._routed_residual_state_stats())
        return stats

    def predict(self, images: Any) -> PredictionBatch:
        scores, features = self._batch_scores_and_routed_residual_features(images)
        candidates: list[dict[str, Any]] = []
        proposal = None
        selected = None
        admission = self._empty_routed_residual_admission("adaptation_disabled")
        if self.adaptation_mode == "full":
            candidates, proposal, selected, admission = (
                self._select_routed_residual_expert(scores)
            )

        if self.adaptation_mode == "full" and self._mixture_active():
            partition = self._candidate_partition()
            candidate_boundary = float(partition["decision_boundary"])
            boundary = self._stabilized_boundary(candidate_boundary)
            base_logit = (scores - boundary) / self.temperature
            prediction_mode = self._prediction_mode_name
            real_components = int(partition["real_components"])
            fake_components = int(partition["fake_components"])
        else:
            candidate_boundary = None
            boundary = 0.0
            base_logit = scores / self.temperature
            prediction_mode = "source_fallback"
            real_components = 0
            fake_components = 0

        assignment = None if selected is None else selected["assignment"]
        residual_state = (
            None
            if assignment is None
            else self._peek_routed_residual_state(assignment)
        )
        residual_ready = self._routed_residual_ready(residual_state)
        residual_scores = self._routed_residual_scores(features, residual_state)
        final_logit = base_logit + residual_scores
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(final_logit, -60.0, 60.0))
        )
        self._pending_routed_residual_features = features
        self._pending_routed_residual_state = residual_state
        self._pending_routed_residual_assignment = assignment
        self._pending_routed_residual_mixture = (
            None if selected is None else _copy_gmm(selected["mixture"])
        )

        selected_expert = None if selected is None else str(selected["expert"])
        selected_memory_index = None if selected is None else selected["memory_index"]
        proposed_expert = None if proposal is None else str(proposal["expert"])
        proposed_memory_index = None if proposal is None else proposal["memory_index"]
        active_candidate = next(
            (
                item
                for item in candidates
                if item["expert"] == "active_learning_state"
            ),
            None,
        )
        return self._prediction_batch(
            scores,
            probability,
            prediction_mode=prediction_mode,
            prediction_boundary=boundary,
            prediction_candidate_boundary=candidate_boundary,
            prediction_real_components=real_components,
            prediction_fake_components=fake_components,
            prediction_memory_index=self.active_memory_index,
            prediction_memory_recalled=self.recall_anchor_boundary is not None,
            prediction_memory_anchor_boundary=self.recall_anchor_boundary,
            prediction_memory_anchor_weight=self._recall_anchor_weight(),
            prediction_residual_routing_expert=selected_expert,
            prediction_residual_routing_memory_index=selected_memory_index,
            prediction_residual_routing_candidate_count=len(candidates),
            prediction_residual_routing_memory_candidate_count=sum(
                item["expert"] == "episodic_memory" for item in candidates
            ),
            prediction_residual_routing_selected_deviance=(
                None if selected is None else float(selected["deviance"])
            ),
            prediction_residual_routing_active_deviance=(
                None
                if active_candidate is None
                else float(active_candidate["deviance"])
            ),
            prediction_residual_routing_proposed_expert=proposed_expert,
            prediction_residual_routing_proposed_memory_index=(
                proposed_memory_index
            ),
            prediction_residual_routing_admission_checked=bool(
                admission["checked"]
            ),
            prediction_residual_routing_admission_accepted=bool(
                admission["accepted"]
            ),
            prediction_residual_routing_admission_reason=str(admission["reason"]),
            prediction_residual_routing_admission_memory_index=(
                admission["memory_index"]
            ),
            prediction_residual_routing_admission_fixed_score=(
                admission["fixed_score"]
            ),
            prediction_residual_routing_admission_new_bic=admission["new_bic"],
            prediction_residual_routing_admission_identity_penalty=(
                admission["identity_penalty"]
            ),
            prediction_residual_routing_admission_gain=admission["gain"],
            prediction_residual_routing_admission_new_components=(
                admission["new_components"]
            ),
            prediction_residual_routing_admission_fit_failure=bool(
                admission["fit_failure"]
            ),
            prediction_residual_routing_admission_unimodal=bool(
                admission["unimodal"]
            ),
            prediction_routed_residual_ready=residual_ready,
            prediction_routed_residual_mean=float(np.mean(residual_scores)),
            prediction_routed_residual_abs_mean=float(
                np.mean(np.abs(residual_scores))
            ),
            prediction_routed_residual_max_abs=float(
                np.max(np.abs(residual_scores))
            ),
        )

    def adapt(self, images: Any) -> AdaptationStats:
        if self._pending is None:
            return super().adapt(images)
        scores = np.asarray(self._pending["scores"], dtype=np.float64).reshape(-1)
        prediction_state = dict(self._pending)
        features = self._pending_routed_residual_features
        assignment = self._pending_routed_residual_assignment
        mixture = self._pending_routed_residual_mixture
        residual_state = self._pending_routed_residual_state
        self._pending_routed_residual_features = None
        self._pending_routed_residual_state = None
        self._pending_routed_residual_assignment = None
        self._pending_routed_residual_mixture = None

        residual_updated = False
        if self.adaptation_mode == "full" and assignment is not None:
            if features is None or int(features.shape[0]) != int(scores.size):
                raise RuntimeError(
                    "ASCAL routed residual lost its matching prediction features"
                )
            if mixture is None:
                raise RuntimeError(
                    "ASCAL routed residual lost its prediction-time expert GMM"
                )
            if residual_state is None:
                residual_state = self._ensure_routed_residual_state(assignment)
            residual_updated = self._update_routed_residual_state(
                residual_state,
                mixture,
                scores,
                features,
            )

        stats = super().adapt(images)
        if self.adaptation_mode == "full":
            candidate_count = int(
                prediction_state.get(
                    "prediction_residual_routing_candidate_count",
                    0,
                )
                or 0
            )
            selected_expert = prediction_state.get(
                "prediction_residual_routing_expert"
            )
            selected_memory_index = prediction_state.get(
                "prediction_residual_routing_memory_index"
            )
            proposed_expert = prediction_state.get(
                "prediction_residual_routing_proposed_expert"
            )
            if candidate_count:
                self.routed_residual_routing_decisions += 1
            else:
                self.routed_residual_source_fallbacks += 1
            if selected_expert == "active_learning_state":
                self.routed_residual_active_selections += 1
            elif selected_expert == "episodic_memory":
                self.routed_residual_memory_selections += 1
            if proposed_expert == "episodic_memory":
                self.routed_residual_memory_proposals += 1
            if prediction_state.get(
                "prediction_residual_routing_admission_checked"
            ):
                self.routed_residual_admission_checks += 1
                if prediction_state.get(
                    "prediction_residual_routing_admission_accepted"
                ):
                    self.routed_residual_admission_accepts += 1
                else:
                    self.routed_residual_admission_rejects += 1
                if prediction_state.get(
                    "prediction_residual_routing_admission_fit_failure"
                ):
                    self.routed_residual_admission_fit_failures += 1
                if prediction_state.get(
                    "prediction_residual_routing_admission_unimodal"
                ):
                    self.routed_residual_admission_unimodal += 1
            self.last_routed_residual_expert = selected_expert
            self.last_routed_residual_memory_index = selected_memory_index
            self.last_routed_residual_admission_reason = prediction_state.get(
                "prediction_residual_routing_admission_reason"
            )
        stats.extra.update(
            {
                **self._routed_residual_state_stats(),
                "routed_residual_updated": residual_updated,
            }
        )
        return stats

    def discard_pending_prediction(self) -> None:
        super().discard_pending_prediction()
        self._pending_routed_residual_features = None
        self._pending_routed_residual_state = None
        self._pending_routed_residual_assignment = None
        self._pending_routed_residual_mixture = None


class ASCALGMMSegmentedMemoryPosteriorRoutedRidgeResidual(
    ASCALGMMSegmentedMemoryPosteriorRoutedResidual
):
    """Fit one exact online weighted-ridge residual per routed expert."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.routed_ridge_solve_failures = 0
        self.routed_ridge_last_effective_support = 0.0
        self.routed_ridge_last_target_abs_mean = 0.0
        self.routed_ridge_last_gain_norm = 0.0
        self.routed_ridge_last_weight_norm = 0.0

    @property
    def trainable_parameters(self) -> int:
        if self.adaptation_mode == "static":
            return 0
        return len(self._all_routed_residual_states()) * self.residual_feature_dim

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        for key in (
            "real_prototype_rule",
            "fake_prototype_rule",
            "fake_component_count_rule",
        ):
            metadata.pop(key, None)
        metadata.update(
            {
                "adaptive_role": (
                    "immutable_source_score_calibration_with_mdl_routed_"
                    "causal_online_linear_residual_heads"
                ),
                "research_name": "ASCAL-JMP-RoutedRidge",
                "research_version": "R14",
                "residual_scope": (
                    "one_zero_initialized_linear_head_per_discovered_score_expert"
                ),
                "expert_type": "online_weighted_ridge_linear_residual_head",
                "linear_head_per_expert": True,
                "residual_teacher": (
                    "signed_equal_prior_posterior_of_the_prediction_time_"
                    "selected_immutable_source_score_gmm"
                ),
                "soft_target_rule": "two_times_gmm_posterior_minus_one",
                "reliability_rule": "absolute_centered_soft_posterior_no_threshold",
                "ridge_objective": (
                    "sum_reliability_times_squared_linear_soft_target_error_"
                    "plus_unit_l2_weight_norm"
                ),
                "ridge_prior_precision": (
                    "fixed_identity_under_l2_normalized_residual_features"
                ),
                "ridge_update": (
                    "exact_recursive_least_squares_woodbury_update_after_prediction"
                ),
                "ridge_sufficient_statistics": (
                    "one_inverse_regularized_gram_matrix_and_one_weight_vector_"
                    "per_expert"
                ),
                "residual_intercept": "none_global_shift_is_handled_by_r01_base",
                "residual_readout_bound": "none",
                "residual_readout": "selected_expert_orthogonal_feature_dot_weight",
                "prediction_rule": (
                    "r01_continuous_base_logit_plus_selected_expert_linear_residual"
                ),
                "optimizer": "none_closed_form_recursive_ridge",
                "epoch": "none",
                "learning_rate": "none",
                "prediction_mutates_experts": False,
                "new_target_hyperparameters": 0,
                "hyperparameter_rule": (
                    "no_learning_rate_epoch_confidence_threshold_residual_weight_"
                    "routing_threshold_fusion_weight_or_memory_capacity"
                ),
                "intentional_changes": [
                    "all R13 routing admission base calibration and expert assignments remain unchanged",
                    "each R13 prototype state is replaced by one zero initialized linear residual head",
                    "the selected prediction time GMM supplies a bounded signed soft target and reliability",
                    "unit regularization is the fixed identity prior in normalized feature coordinates",
                    "Woodbury recursive least squares exactly accumulates all prior selected samples",
                    "only the prediction time selected expert receives the current batch after prediction",
                    "no image or per sample feature is retained after its sufficient statistic update",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_routed_ridge_residual"

    def _new_routed_residual_state(self) -> dict[str, Any]:
        return {
            "inverse_gram": np.eye(self.residual_feature_dim, dtype=np.float64),
            "weights": np.zeros(self.residual_feature_dim, dtype=np.float64),
            "updates": 0,
            "candidate_samples": 0,
            "effective_support": 0.0,
            "weighted_target_square_sum": 0.0,
        }

    def _routed_residual_ready(self, state: dict[str, Any] | None) -> bool:
        if state is None or int(state["updates"]) <= 0:
            return False
        return float(np.linalg.norm(state["weights"])) > np.finfo(np.float64).eps

    def _routed_residual_readout_count(
        self,
        state: dict[str, Any],
    ) -> int:
        return int(self._routed_residual_ready(state))

    def _routed_residual_scores(
        self,
        features: np.ndarray,
        state: dict[str, Any] | None,
    ) -> np.ndarray:
        if not self._routed_residual_ready(state):
            return np.zeros(int(features.shape[0]), dtype=np.float64)
        if state is None:
            raise RuntimeError("ASCAL routed ridge lost its selected expert state")
        return np.asarray(features @ state["weights"], dtype=np.float64)

    def _update_routed_residual_state(
        self,
        state: dict[str, Any],
        mixture: dict[str, Any],
        scores: np.ndarray,
        features: np.ndarray,
    ) -> bool:
        posterior = joint_density_fake_posterior(scores, mixture)
        targets = 2.0 * posterior - 1.0
        reliability = np.abs(targets)
        effective_support = float(reliability.sum())
        fake_support = float((reliability * posterior).sum())
        real_support = float((reliability * (1.0 - posterior)).sum())

        self.routed_residual_candidate_samples += int(scores.size)
        self.routed_residual_last_reliability = float(np.mean(reliability))
        self.routed_residual_last_real_support = real_support
        self.routed_residual_last_fake_support = fake_support
        self.routed_residual_last_fake_component_supports.fill(0.0)
        self.routed_ridge_last_effective_support = effective_support
        self.routed_ridge_last_target_abs_mean = float(np.mean(np.abs(targets)))
        self.routed_ridge_last_gain_norm = 0.0
        state["candidate_samples"] = int(state["candidate_samples"]) + int(
            scores.size
        )
        if effective_support <= np.finfo(np.float64).eps:
            return False

        square_root_reliability = np.sqrt(reliability)
        design = square_root_reliability[:, None] * features
        response = square_root_reliability * targets
        inverse_gram = np.asarray(state["inverse_gram"], dtype=np.float64)
        weights = np.asarray(state["weights"], dtype=np.float64)
        inverse_times_design = inverse_gram @ design.T
        innovation_gram = (
            np.eye(int(scores.size), dtype=np.float64)
            + design @ inverse_times_design
        )
        innovation_gram = 0.5 * (innovation_gram + innovation_gram.T)
        try:
            gain = np.linalg.solve(
                innovation_gram,
                inverse_times_design.T,
            ).T
        except np.linalg.LinAlgError:
            self.routed_ridge_solve_failures += 1
            return False

        residual_error = response - design @ weights
        updated_weights = weights + gain @ residual_error
        updated_inverse = inverse_gram - gain @ inverse_times_design.T
        updated_inverse = 0.5 * (updated_inverse + updated_inverse.T)
        if not (
            np.all(np.isfinite(updated_weights))
            and np.all(np.isfinite(updated_inverse))
        ):
            self.routed_ridge_solve_failures += 1
            return False

        state["weights"] = updated_weights
        state["inverse_gram"] = updated_inverse
        state["updates"] = int(state["updates"]) + 1
        state["effective_support"] = float(state["effective_support"]) + (
            effective_support
        )
        state["weighted_target_square_sum"] = float(
            state["weighted_target_square_sum"]
        ) + float(np.sum(reliability * targets**2))
        self.routed_residual_updates += 1
        self.routed_ridge_last_gain_norm = float(np.linalg.norm(gain))
        self.routed_ridge_last_weight_norm = float(np.linalg.norm(updated_weights))
        return self._routed_residual_ready(state)

    def _routed_residual_state_stats(self) -> dict[str, Any]:
        stats = super()._routed_residual_state_stats()
        for key in (
            "routed_residual_fake_prototype_count",
            "routed_residual_last_fake_component_supports",
            "routed_residual_multimodal_updates",
            "routed_residual_readout_bound",
        ):
            stats.pop(key, None)
        states = self._all_routed_residual_states()
        weight_norms = [float(np.linalg.norm(state["weights"])) for state in states]
        effective_support = sum(
            float(state["effective_support"]) for state in states
        )
        stats.update(
            {
                "routed_ridge_head_count": len(states),
                "routed_ridge_ready_heads": sum(
                    self._routed_residual_ready(state) for state in states
                ),
                "routed_ridge_weight_parameters": (
                    len(states) * self.residual_feature_dim
                ),
                "routed_ridge_inverse_gram_values": (
                    len(states) * self.residual_feature_dim**2
                ),
                "routed_ridge_effective_support": effective_support,
                "routed_ridge_max_weight_norm": max(weight_norms, default=0.0),
                "routed_ridge_mean_weight_norm": (
                    float(np.mean(weight_norms)) if weight_norms else 0.0
                ),
                "routed_ridge_solve_failures": self.routed_ridge_solve_failures,
                "routed_ridge_last_effective_support": (
                    self.routed_ridge_last_effective_support
                ),
                "routed_ridge_last_target_abs_mean": (
                    self.routed_ridge_last_target_abs_mean
                ),
                "routed_ridge_last_gain_norm": self.routed_ridge_last_gain_norm,
                "routed_ridge_last_weight_norm": (
                    self.routed_ridge_last_weight_norm
                ),
                "routed_residual_trainable_parameters": (
                    self.trainable_parameters
                ),
            }
        )
        return stats


class ASCALGMMSegmentedMemoryPosteriorRealDeviationResidual(
    ASCALGMMSegmentedMemoryPosteriorGlobalResidual
):
    """Rank samples by centered deviation from one causal real anchor."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "immutable_source_score_gmm_with_one_causal_centered_"
                    "real_manifold_deviation_residual"
                ),
                "research_name": "ASCAL-JMP-RealDeviation",
                "research_version": "R07",
                "residual_count": 1,
                "residual_scope": (
                    "one_shared_real_anchor_for_the_entire_continual_stream"
                ),
                "real_prototype_rule": (
                    "one_cumulative_soft_prototype_for_the_complete_real_gmm_block"
                ),
                "fake_prototype_rule": "none_fake_is_treated_as_open_heterogeneity",
                "residual_update": (
                    "cumulative_soft_real_feature_mean_no_optimizer_or_fake_teacher"
                ),
                "residual_readout": (
                    "half_centered_squared_distance_to_the_soft_real_mean_"
                    "equal_to_R_times_R_minus_cosine"
                ),
                "residual_expectation": "zero_under_the_accumulated_soft_real_measure",
                "residual_range": "closed_interval_minus_0p25_to_2",
                "prediction_rule": "r01_base_logit_plus_one_real_deviation_residual",
                "hyperparameter_rule": (
                    "no_fake_component_count_residual_learning_rate_loss_weight_"
                    "confidence_threshold_memory_capacity_or_fusion_weight"
                ),
                "intentional_changes": [
                    "the R01 source-score GMM segmentation memory and boundary trajectory remain unchanged",
                    "only immutable source margins are appended to score history",
                    "soft source-score evidence updates one stream-wide real feature mean",
                    "fake feature prototypes are removed rather than forced to summarize heterogeneous generators",
                    "the centered distance has zero soft-real expectation and an analytical fixed bound",
                    "the residual remains a one-way student of immutable source-score density evidence",
                    "predictions use only GMM and residual state learned from earlier batches",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_real_deviation_residual"

    def _normalized_real_prototype(self) -> tuple[np.ndarray | None, float]:
        epsilon = np.finfo(np.float64).eps
        if self.residual_real_support <= epsilon:
            return None, 0.0
        real_mean = self.residual_real_sum / self.residual_real_support
        resultant = float(np.linalg.norm(real_mean))
        if resultant <= epsilon:
            return None, 0.0
        return real_mean / resultant, min(resultant, 1.0)

    def _residual_ready(self) -> bool:
        prototype, _ = self._normalized_real_prototype()
        return prototype is not None

    def _residual_scores(self, features: np.ndarray) -> np.ndarray:
        prototype, resultant = self._normalized_real_prototype()
        if prototype is None:
            return np.zeros(int(features.shape[0]), dtype=np.float64)
        cosine = features @ prototype
        return np.asarray(resultant * (resultant - cosine), dtype=np.float64)

    def _residual_state_stats(self) -> dict[str, Any]:
        stats = super()._residual_state_stats()
        _, resultant = self._normalized_real_prototype()
        stats.update(
            {
                "residual_norm": resultant,
                "residual_real_resultant_length": resultant,
                "residual_fake_prototype_count": 0,
                "residual_readout_lower_bound": -0.25,
                "residual_readout_upper_bound": 2.0,
            }
        )
        return stats

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
        if real_support <= epsilon:
            return False

        self.residual_real_sum += np.sum(real_weights[:, None] * features, axis=0)
        self.residual_real_support += real_support
        self.residual_fake_support += fake_support
        if not self._residual_ready():
            return False
        self.residual_updates += 1
        return True


class ASCALGMMSegmentedMemoryPosteriorConditionalResidual(
    ASCALGMMSegmentedMemoryPosteriorGlobalResidual
):
    """Add the source-conditioned innovation of the global prototype score."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.conditional_moment_samples = 0
        self.conditional_source_mean = 0.0
        self.conditional_raw_mean = 0.0
        self.conditional_source_m2 = 0.0
        self.conditional_raw_m2 = 0.0
        self.conditional_cross_moment = 0.0
        self.conditional_moment_updates = 0
        self._pending_conditional_scores: np.ndarray | None = None
        self._pending_raw_residual_scores: np.ndarray | None = None
        self._pending_raw_residual_ready = False

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "immutable_source_score_gmm_with_one_causal_global_"
                    "prototype_residual_and_source_conditioned_innovation"
                ),
                "research_name": "ASCAL-JMP-ConditionalResidual",
                "research_version": "R08",
                "residual_count": 1,
                "residual_scope": (
                    "one_global_prototype_score_plus_its_stream_wide_"
                    "source_conditioned_innovation"
                ),
                "conditional_moment_input": (
                    "immutable_source_margin_and_causal_global_prototype_score"
                ),
                "conditional_moment_update": (
                    "population_second_moments_of_preupdate_predictions_"
                    "merged_by_exact_parallel_welford_updates"
                ),
                "innovation_rule": (
                    "raw_prototype_score_minus_its_online_linear_projection_"
                    "on_the_immutable_source_margin"
                ),
                "innovation_trust": (
                    "positive_online_correlation_between_source_margin_and_"
                    "raw_prototype_score"
                ),
                "innovation_scale": (
                    "positive_correlation_times_source_standard_deviation_"
                    "over_raw_residual_standard_deviation_and_temperature"
                ),
                "innovation_rms_bound": (
                    "at_most_one_half_of_the_source_margin_standard_deviation_"
                    "over_temperature_under_the_accumulated_measure"
                ),
                "prediction_rule": (
                    "r01_base_logit_plus_global_prototype_score_plus_"
                    "source_conditioned_innovation"
                ),
                "hyperparameter_rule": (
                    "no_conditional_window_learning_rate_threshold_fusion_"
                    "weight_memory_capacity_or_shrinkage_parameter"
                ),
                "intentional_changes": [
                    "the R01 source-score GMM segmentation memory and boundary trajectory remain unchanged",
                    "the R05 global real and fake prototype score remains the sole feature residual",
                    "preupdate source margins and prototype scores update exact causal second moments",
                    "the adaptive addition retains only prototype evidence not linearly explained by source score",
                    "nonpositive agreement with the immutable source score gives zero innovation trust",
                    "variance matching and trust are derived from history rather than target-tuned constants",
                    "predictions use only prototype and moment state learned from earlier batches",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_conditional_residual"

    def _conditional_moment_state(self) -> dict[str, Any]:
        samples = self.conditional_moment_samples
        epsilon = np.finfo(np.float64).eps
        if samples <= 0:
            source_variance = 0.0
            raw_variance = 0.0
            covariance = 0.0
        else:
            source_variance = max(self.conditional_source_m2 / samples, 0.0)
            raw_variance = max(self.conditional_raw_m2 / samples, 0.0)
            covariance = self.conditional_cross_moment / samples

        moments_ready = (
            samples > 1
            and source_variance > epsilon
            and raw_variance > epsilon
        )
        if moments_ready:
            correlation = covariance / math.sqrt(source_variance * raw_variance)
            correlation = float(np.clip(correlation, -1.0, 1.0))
            slope = covariance / source_variance
        else:
            correlation = 0.0
            slope = 0.0
        trust = max(correlation, 0.0)
        innovation_ready = moments_ready and trust > epsilon
        if innovation_ready:
            scale = (
                trust
                * math.sqrt(source_variance / raw_variance)
                / self.temperature
            )
            rms_ratio = trust * math.sqrt(max(1.0 - correlation**2, 0.0))
        else:
            scale = 0.0
            rms_ratio = 0.0
        return {
            "conditional_moment_samples": samples,
            "conditional_moment_updates": self.conditional_moment_updates,
            "conditional_source_mean": self.conditional_source_mean,
            "conditional_raw_residual_mean": self.conditional_raw_mean,
            "conditional_source_variance": source_variance,
            "conditional_raw_residual_variance": raw_variance,
            "conditional_covariance": covariance,
            "conditional_correlation": correlation,
            "conditional_projection_slope": slope,
            "conditional_innovation_trust": trust,
            "conditional_innovation_scale": scale,
            "conditional_innovation_rms_source_ratio": rms_ratio,
            "conditional_moments_ready": moments_ready,
            "conditional_innovation_ready": innovation_ready,
        }

    def _conditional_innovation(
        self,
        scores: np.ndarray,
        raw_residual: np.ndarray,
    ) -> np.ndarray:
        state = self._conditional_moment_state()
        if not state["conditional_innovation_ready"]:
            return np.zeros(int(scores.size), dtype=np.float64)
        novel = (
            raw_residual
            - float(state["conditional_raw_residual_mean"])
            - float(state["conditional_projection_slope"])
            * (scores - float(state["conditional_source_mean"]))
        )
        return np.asarray(
            float(state["conditional_innovation_scale"]) * novel,
            dtype=np.float64,
        )

    def _residual_state_stats(self) -> dict[str, Any]:
        stats = super()._residual_state_stats()
        stats.update(self._conditional_moment_state())
        return stats

    def _update_conditional_moments(
        self,
        scores: np.ndarray,
        raw_residual: np.ndarray,
    ) -> bool:
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        raw_residual = np.asarray(raw_residual, dtype=np.float64).reshape(-1)
        if scores.size != raw_residual.size:
            raise ValueError("Conditional residual moments require matching samples")
        if not scores.size:
            return False
        if not np.all(np.isfinite(scores)) or not np.all(np.isfinite(raw_residual)):
            raise ValueError("Conditional residual moments require finite values")

        batch_samples = int(scores.size)
        batch_source_mean = float(np.mean(scores))
        batch_raw_mean = float(np.mean(raw_residual))
        source_centered = scores - batch_source_mean
        raw_centered = raw_residual - batch_raw_mean
        batch_source_m2 = float(source_centered @ source_centered)
        batch_raw_m2 = float(raw_centered @ raw_centered)
        batch_cross = float(source_centered @ raw_centered)

        previous_samples = self.conditional_moment_samples
        total_samples = previous_samples + batch_samples
        source_delta = batch_source_mean - self.conditional_source_mean
        raw_delta = batch_raw_mean - self.conditional_raw_mean
        merge_weight = previous_samples * batch_samples / total_samples
        self.conditional_source_m2 += (
            batch_source_m2 + source_delta**2 * merge_weight
        )
        self.conditional_raw_m2 += batch_raw_m2 + raw_delta**2 * merge_weight
        self.conditional_cross_moment += (
            batch_cross + source_delta * raw_delta * merge_weight
        )
        self.conditional_source_mean += source_delta * batch_samples / total_samples
        self.conditional_raw_mean += raw_delta * batch_samples / total_samples
        self.conditional_moment_samples = total_samples
        self.conditional_moment_updates += 1
        return True

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

        raw_ready = self._residual_ready()
        raw_residual = self._residual_scores(features)
        innovation = self._conditional_innovation(scores, raw_residual)
        residual_scores = raw_residual + innovation
        final_logit = base_logit + residual_scores
        probability = 1.0 / (
            1.0 + np.exp(-np.clip(final_logit, -60.0, 60.0))
        )
        conditional_state = self._conditional_moment_state()
        self._pending_residual_features = features
        self._pending_conditional_scores = scores.copy()
        self._pending_raw_residual_scores = raw_residual.copy()
        self._pending_raw_residual_ready = raw_ready
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
            prediction_residual_ready=raw_ready,
            prediction_residual_mean=float(np.mean(residual_scores)),
            prediction_residual_abs_mean=float(np.mean(np.abs(residual_scores))),
            prediction_residual_max_abs=float(np.max(np.abs(residual_scores))),
            prediction_raw_residual_mean=float(np.mean(raw_residual)),
            prediction_raw_residual_abs_mean=float(np.mean(np.abs(raw_residual))),
            prediction_raw_residual_max_abs=float(np.max(np.abs(raw_residual))),
            prediction_conditional_innovation_mean=float(np.mean(innovation)),
            prediction_conditional_innovation_abs_mean=float(
                np.mean(np.abs(innovation))
            ),
            prediction_conditional_innovation_max_abs=float(
                np.max(np.abs(innovation))
            ),
            prediction_conditional_moments_ready=conditional_state[
                "conditional_moments_ready"
            ],
            prediction_conditional_innovation_ready=conditional_state[
                "conditional_innovation_ready"
            ],
            prediction_conditional_correlation=conditional_state[
                "conditional_correlation"
            ],
            prediction_conditional_innovation_trust=conditional_state[
                "conditional_innovation_trust"
            ],
            prediction_conditional_innovation_scale=conditional_state[
                "conditional_innovation_scale"
            ],
        )

    def adapt(self, images: Any) -> AdaptationStats:
        if self._pending is None:
            return super().adapt(images)
        scores = self._pending_conditional_scores
        raw_residual = self._pending_raw_residual_scores
        raw_ready = self._pending_raw_residual_ready
        self._pending_conditional_scores = None
        self._pending_raw_residual_scores = None
        self._pending_raw_residual_ready = False
        stats = super().adapt(images)
        moment_updated = False
        if self.adaptation_mode == "full" and raw_ready:
            if scores is None or raw_residual is None:
                raise RuntimeError("ASCAL conditional residual lost prediction moments")
            moment_updated = self._update_conditional_moments(scores, raw_residual)
        stats.extra.update(self._conditional_moment_state())
        stats.extra["conditional_moment_updated"] = moment_updated
        return stats

    def discard_pending_prediction(self) -> None:
        super().discard_pending_prediction()
        self._pending_conditional_scores = None
        self._pending_raw_residual_scores = None
        self._pending_raw_residual_ready = False


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
