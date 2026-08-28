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
Its joint-ridge variant instead adds an uncentered feature residual and bias to
the full ordinal-route log odds, so a zero state is exact R12 while learned
expert evidence may improve both ranking and the binary decision boundary.
Its pairwise-ridge variant keeps one bias-free stream-wide feature ranker. R12
supplies the monotone pseudo-class side, the selected GMM supplies only bounded
soft reliability, and an exact pairwise Ridge update learns unit fake-over-real
feature differences without inheriting an expert-specific posterior scale.
Its RMS-ridge-expert variant gives every routed R12 expert a direct two-output
online Ridge classifier over frozen normalized CLIP features. R12 supplies the
hard pseudo-class, the selected GMM supplies continuous reliability, and
per-expert historical RMS margins put the old R12 score and analytic classifier
on a comparable scale before their evidence is added.
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

from src.models.analytic_ridge import source_analytic_ridge_arrays
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
        forward_classifier_features = getattr(
            self.model, "forward_classifier_features", None
        )
        classifier = getattr(self.model, "classifier", None)
        if not callable(forward_features) or not callable(classifier):
            raise TypeError(
                "ASCAL ordinal ridge requires forward_features and classifier"
            )
        with torch.no_grad():
            features = forward_features(flat.to(self.device, non_blocking=True))
            classifier_features = (
                forward_classifier_features(features)
                if callable(forward_classifier_features)
                else features
            )
            logits = classifier(classifier_features)
        scores = (
            binary_score(logits)
            .view(batch, views)
            .mean(dim=1)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        feature_values = (
            classifier_features.detach()
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


class ASCALGMMSegmentedMemoryPosteriorJointRidge(
    ASCALGMMSegmentedMemoryPosteriorOrdinalRidge
):
    """Add one unlocked causal feature log-odds residual to each R12 expert."""

    _ORDINAL_RIDGE_MEMORY_KEY = "joint_ridge_state"

    def _reset_state(self) -> None:
        super()._reset_state()
        self.joint_ridge_backbone_feature_dim = self.ordinal_ridge_feature_dim
        self.ordinal_ridge_feature_dim += 1
        self._novel_ordinal_ridge_state = self._new_ordinal_ridge_state()
        self._pending_joint_ridge_base_logits: np.ndarray | None = None
        self.joint_ridge_batches = 0
        self.joint_ridge_samples = 0
        self.joint_ridge_ready_batches = 0
        self.joint_ridge_label_changes = 0
        self.joint_ridge_real_to_fake = 0
        self.joint_ridge_fake_to_real = 0
        self.joint_ridge_last_base_abs_mean = 0.0
        self.joint_ridge_last_bias_abs = 0.0

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "r12_initialized_mdl_routed_causal_joint_log_odds_"
                    "online_ridge_adaptation"
                ),
                "research_name": "ASCAL-JMP-JointRidge",
                "research_version": "R16",
                "r12_protected_scope": (
                    "routing_admission_unique_live_state_initial_prediction_"
                    "adaptation_assignment_segmentation_and_memory"
                ),
                "r12_initialization": (
                    "every_new_or_untrained_expert_predicts_exactly_as_r12"
                ),
                "accuracy_invariance": (
                    "none_the_learned_feature_residual_and_bias_may_cross_"
                    "the_r12_decision_boundary"
                ),
                "accuracy_behavior": (
                    "joint_ridge_can_change_both_sample_decisions_and_the_"
                    "expert_specific_effective_boundary"
                ),
                "residual_scope": (
                    "one_zero_initialized_linear_joint_log_odds_residual_"
                    "per_r12_expert"
                ),
                "residual_input": (
                    "l2_normalized_frozen_features_orthogonal_to_the_source_"
                    "classifier_direction_plus_one_constant_bias_coordinate"
                ),
                "residual_teacher": (
                    "equal_prior_log_odds_of_the_prediction_time_selected_"
                    "r12_gmm"
                ),
                "residual_target_rule": (
                    "selected_gmm_teacher_logit_minus_prediction_time_r12_"
                    "base_logit_without_centering"
                ),
                "reliability_rule": (
                    "absolute_centered_selected_gmm_soft_posterior_no_threshold"
                ),
                "joint_odds_rule": (
                    "final_odds_equal_r12_odds_times_exp_of_expert_ridge_"
                    "feature_evidence"
                ),
                "ridge_objective": (
                    "sum_reliability_times_squared_uncentered_logit_residual_"
                    "error_plus_unit_l2_weight_norm"
                ),
                "ridge_prior_precision": (
                    "fixed_identity_for_unit_feature_and_unit_bias_coordinates"
                ),
                "ridge_update": (
                    "exact_recursive_least_squares_woodbury_after_prediction"
                ),
                "ridge_sufficient_statistics": (
                    "one_inverse_regularized_gram_matrix_and_one_weight_vector_"
                    "per_expert"
                ),
                "residual_intercept": "one_learned_bias_coordinate_per_expert",
                "target_residual_centering": "none",
                "prediction_residual_centering": "none",
                "prediction_rule": (
                    "sigmoid_of_r12_base_logit_plus_selected_expert_feature_"
                    "residual_and_bias_without_half_interval_locking"
                ),
                "routing_score_coordinate": (
                    "immutable_source_score_never_the_joint_ridge_output"
                ),
                "gmm_update_score_coordinate": (
                    "immutable_source_score_never_the_joint_ridge_output"
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
                    "all R12 routing and continual learning assignments remain unchanged",
                    "zero initialized experts reproduce the complete R12 prediction",
                    "the selected old GMM supplies a soft teacher and continuous reliability",
                    "one constant Ridge coordinate learns an expert-specific boundary shift",
                    "the feature residual may reorder samples and cross the R12 threshold",
                    "neither target nor prediction residuals are batch centered",
                    "joint outputs never rewrite the stable score coordinate used by GMM routing",
                    "only the prediction-time selected expert updates after prediction",
                    "no image or per-sample feature remains after the sufficient-statistic update",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_joint_ridge"

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
                "ASCAL joint ridge expects (B, C, H, W) or "
                "(B, V, C, H, W) images"
            )
        forward_features = getattr(self.model, "forward_features", None)
        classifier = getattr(self.model, "classifier", None)
        if not callable(forward_features) or not callable(classifier):
            raise TypeError(
                "ASCAL joint ridge requires forward_features and classifier"
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
        if int(feature_values.shape[1]) != self.joint_ridge_backbone_feature_dim:
            raise ValueError(
                "ASCAL joint ridge feature dimension does not match the source head"
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
        feature_values = np.concatenate(
            [feature_values, np.ones((batch, 1), dtype=np.float64)],
            axis=1,
        )
        if int(feature_values.shape[1]) != self.ordinal_ridge_feature_dim:
            raise RuntimeError("ASCAL joint ridge failed to append its bias feature")
        return scores, feature_values

    def _joint_ridge_supervision(
        self,
        mixture: dict[str, Any],
        scores: np.ndarray,
        base_logits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        teacher_logits = np.clip(
            _joint_density_log_odds(scores, mixture),
            -60.0,
            60.0,
        )
        base_logits = np.asarray(base_logits, dtype=np.float64).reshape(-1)
        if base_logits.shape != teacher_logits.shape:
            raise RuntimeError(
                "ASCAL joint ridge base logit shape does not match its teacher"
            )
        posterior = 1.0 / (1.0 + np.exp(-teacher_logits))
        reliability = np.abs(2.0 * posterior - 1.0)
        targets = teacher_logits - base_logits
        if not (
            np.all(np.isfinite(base_logits))
            and np.all(np.isfinite(targets))
            and np.all(np.isfinite(reliability))
        ):
            raise FloatingPointError(
                "ASCAL joint ridge produced non-finite supervision"
            )
        self.ordinal_ridge_last_effective_support = float(reliability.sum())
        self.ordinal_ridge_last_reliability = float(np.mean(reliability))
        self.ordinal_ridge_last_teacher_abs_mean = float(
            np.mean(np.abs(teacher_logits))
        )
        self.ordinal_ridge_last_source_abs_mean = float(
            np.mean(np.abs(base_logits))
        )
        self.ordinal_ridge_last_target_center = 0.0
        self.ordinal_ridge_last_target_abs_mean = float(np.mean(np.abs(targets)))
        self.joint_ridge_last_base_abs_mean = float(np.mean(np.abs(base_logits)))
        return posterior, targets, reliability

    def _update_ordinal_ridge_state(
        self,
        state: dict[str, Any],
        mixture: dict[str, Any],
        scores: np.ndarray,
        features: np.ndarray,
    ) -> bool:
        base_logits = self._pending_joint_ridge_base_logits
        if base_logits is None:
            raise RuntimeError(
                "ASCAL joint ridge lost its prediction-time R12 base logits"
            )
        _, targets, reliability = self._joint_ridge_supervision(
            mixture,
            scores,
            base_logits,
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
        self.joint_ridge_last_bias_abs = float(abs(updated_weights[-1]))
        return self._ordinal_ridge_ready(state)

    def _joint_ridge_state_stats(self) -> dict[str, Any]:
        states = self._all_ordinal_ridge_states()
        bias_values = [float(state["weights"][-1]) for state in states]
        feature_norms = [
            float(np.linalg.norm(state["weights"][:-1])) for state in states
        ]
        return {
            "joint_ridge_expert_count": len(states),
            "joint_ridge_ready_experts": sum(
                self._ordinal_ridge_ready(state) for state in states
            ),
            "joint_ridge_updates": self.ordinal_ridge_updates,
            "joint_ridge_candidate_samples": self.ordinal_ridge_candidate_samples,
            "joint_ridge_batches": self.joint_ridge_batches,
            "joint_ridge_samples": self.joint_ridge_samples,
            "joint_ridge_ready_batches": self.joint_ridge_ready_batches,
            "joint_ridge_solve_failures": self.ordinal_ridge_solve_failures,
            "joint_ridge_label_changes": self.joint_ridge_label_changes,
            "joint_ridge_real_to_fake": self.joint_ridge_real_to_fake,
            "joint_ridge_fake_to_real": self.joint_ridge_fake_to_real,
            "joint_ridge_effective_support": sum(
                float(state["effective_support"]) for state in states
            ),
            "joint_ridge_last_effective_support": (
                self.ordinal_ridge_last_effective_support
            ),
            "joint_ridge_last_reliability": self.ordinal_ridge_last_reliability,
            "joint_ridge_last_teacher_abs_mean": (
                self.ordinal_ridge_last_teacher_abs_mean
            ),
            "joint_ridge_last_base_abs_mean": self.joint_ridge_last_base_abs_mean,
            "joint_ridge_last_target_abs_mean": (
                self.ordinal_ridge_last_target_abs_mean
            ),
            "joint_ridge_last_bias_abs": self.joint_ridge_last_bias_abs,
            "joint_ridge_max_bias_abs": max(
                (abs(value) for value in bias_values),
                default=0.0,
            ),
            "joint_ridge_mean_bias": (
                float(np.mean(bias_values)) if bias_values else 0.0
            ),
            "joint_ridge_max_feature_weight_norm": max(
                feature_norms,
                default=0.0,
            ),
            "joint_ridge_weight_parameters": (
                len(states) * self.ordinal_ridge_feature_dim
            ),
            "joint_ridge_inverse_gram_values": (
                len(states) * self.ordinal_ridge_feature_dim**2
            ),
            "joint_ridge_trainable_parameters": self.trainable_parameters,
        }

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(self._joint_ridge_state_stats())
        return stats

    def predict(self, images: Any) -> PredictionBatch:
        scores, features = self._batch_scores_and_ordinal_ridge_features(images)
        self._ordinal_ridge_precomputed_scores = scores
        try:
            ordinal = ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.predict(
                self,
                images,
            )
        finally:
            self._ordinal_ridge_precomputed_scores = None
        if self._pending is None:
            raise RuntimeError("ASCAL joint ridge lost the R12 pending state")

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
                raise RuntimeError("ASCAL joint ridge lost its selected state")
            residual = np.asarray(features @ state["weights"], dtype=np.float64)
        else:
            residual = np.zeros(int(scores.size), dtype=np.float64)

        base_probability = (
            ordinal.prob_fake.detach().cpu().numpy().astype(np.float64)
        )
        base_probability = np.clip(base_probability, 1e-6, 1.0 - 1e-6)
        base_logits = np.log(base_probability / (1.0 - base_probability))
        if context is None or not ready:
            probability = base_probability
        else:
            joint_logits = np.clip(base_logits + residual, -60.0, 60.0)
            probability = 1.0 / (1.0 + np.exp(-joint_logits))

        base_labels = ordinal.pred_label.detach().cpu().numpy().astype(np.int64)
        joint_labels = (probability >= 0.5).astype(np.int64)
        label_changes = int(np.count_nonzero(joint_labels != base_labels))
        real_to_fake = int(
            np.count_nonzero((base_labels == 0) & (joint_labels == 1))
        )
        fake_to_real = int(
            np.count_nonzero((base_labels == 1) & (joint_labels == 0))
        )

        self._pending_joint_ridge_base_logits = base_logits.copy()
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
                "prediction_ordinal_ridge_residual_center": 0.0,
                "prediction_ordinal_ridge_residual_mean": float(
                    np.mean(residual)
                ),
                "prediction_ordinal_ridge_residual_abs_mean": float(
                    np.mean(np.abs(residual))
                ),
                "prediction_ordinal_ridge_residual_max_abs": float(
                    np.max(np.abs(residual))
                ),
                "prediction_ordinal_ridge_hard_label_mismatches": label_changes,
                "prediction_joint_ridge_applied": context is not None,
                "prediction_joint_ridge_ready": ready,
                "prediction_joint_ridge_residual_mean": float(
                    np.mean(residual)
                ),
                "prediction_joint_ridge_residual_abs_mean": float(
                    np.mean(np.abs(residual))
                ),
                "prediction_joint_ridge_residual_max_abs": float(
                    np.max(np.abs(residual))
                ),
                "prediction_joint_ridge_bias": (
                    0.0 if state is None else float(state["weights"][-1])
                ),
                "prediction_joint_ridge_label_changes": label_changes,
                "prediction_joint_ridge_real_to_fake": real_to_fake,
                "prediction_joint_ridge_fake_to_real": fake_to_real,
            }
        )
        return self._prediction_batch(scores, probability, **pending_state)

    def adapt(self, images: Any) -> AdaptationStats:
        prediction_state = None if self._pending is None else dict(self._pending)
        try:
            stats = super().adapt(images)
        finally:
            self._pending_joint_ridge_base_logits = None
        if prediction_state is None:
            return stats

        samples = int(np.asarray(prediction_state["scores"]).size)
        if bool(prediction_state.get("prediction_joint_ridge_applied")):
            self.joint_ridge_batches += 1
            self.joint_ridge_samples += samples
        if bool(prediction_state.get("prediction_joint_ridge_ready")):
            self.joint_ridge_ready_batches += 1
        self.joint_ridge_label_changes += int(
            prediction_state.get("prediction_joint_ridge_label_changes", 0) or 0
        )
        self.joint_ridge_real_to_fake += int(
            prediction_state.get("prediction_joint_ridge_real_to_fake", 0) or 0
        )
        self.joint_ridge_fake_to_real += int(
            prediction_state.get("prediction_joint_ridge_fake_to_real", 0) or 0
        )
        stats.extra.update(
            {
                **self._joint_ridge_state_stats(),
                "joint_ridge_updated": bool(
                    stats.extra.get("ordinal_ridge_updated", False)
                ),
            }
        )
        return stats

    def discard_pending_prediction(self) -> None:
        self._pending_joint_ridge_base_logits = None
        super().discard_pending_prediction()


class ASCALGMMSegmentedMemoryPosteriorPairwiseRidge(
    ASCALGMMSegmentedMemoryPosteriorOrdinalRoute
):
    """Learn one causal bias-free global rank residual from soft pair weights."""

    def _reset_state(self) -> None:
        super()._reset_state()
        classifier = getattr(self.model, "classifier", None)
        weight = getattr(classifier, "weight", None)
        if weight is None or weight.ndim != 2 or int(weight.shape[0]) != 2:
            raise TypeError(
                "ASCAL pairwise ridge requires a two-class linear classifier"
            )
        direction = (
            weight[1].detach().float().cpu().numpy()
            - weight[0].detach().float().cpu().numpy()
        ).astype(np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if not math.isfinite(direction_norm) or direction_norm <= 0.0:
            raise ValueError(
                "ASCAL pairwise ridge requires a nonzero source score direction"
            )
        self._pairwise_ridge_source_direction = direction / direction_norm
        self.pairwise_ridge_feature_dim = int(direction.size)
        self._pairwise_ridge_state = self._new_pairwise_ridge_state()
        self._pairwise_ridge_precomputed_scores: np.ndarray | None = None
        self._pending_pairwise_ridge_features: np.ndarray | None = None
        self._pending_pairwise_ridge_mixture: dict[str, Any] | None = None
        self._pending_pairwise_ridge_labels: np.ndarray | None = None
        self.pairwise_ridge_batches = 0
        self.pairwise_ridge_samples = 0
        self.pairwise_ridge_ready_batches = 0
        self.pairwise_ridge_updates = 0
        self.pairwise_ridge_candidate_samples = 0
        self.pairwise_ridge_candidate_pairs = 0
        self.pairwise_ridge_solve_failures = 0
        self.pairwise_ridge_label_changes = 0
        self.pairwise_ridge_real_to_fake = 0
        self.pairwise_ridge_fake_to_real = 0
        self.pairwise_ridge_last_effective_pair_mass = 0.0
        self.pairwise_ridge_last_reliability = 0.0
        self.pairwise_ridge_last_fake_mass = 0.0
        self.pairwise_ridge_last_real_mass = 0.0
        self.pairwise_ridge_last_pair_rank = 0
        self.pairwise_ridge_last_posterior_conflicts = 0
        self.pairwise_ridge_last_weight_norm = 0.0

    @property
    def trainable_parameters(self) -> int:
        if self.adaptation_mode == "static":
            return 0
        return self.pairwise_ridge_feature_dim

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "r12_initialized_global_soft_pairwise_feature_rank_"
                    "online_ridge_adaptation"
                ),
                "research_name": "ASCAL-JMP-PairwiseRidge",
                "research_version": "R17",
                "r12_protected_scope": (
                    "routing_admission_unique_live_state_adaptation_assignment_"
                    "segmentation_and_memory"
                ),
                "r12_initialization": (
                    "an_untrained_global_ranker_predicts_exactly_as_r12"
                ),
                "accuracy_invariance": (
                    "none_the_sample_specific_pairwise_residual_may_cross_"
                    "the_r12_decision_boundary"
                ),
                "residual_scope": (
                    "one_zero_initialized_bias_free_stream_wide_linear_ranker"
                ),
                "residual_input": (
                    "l2_normalized_frozen_features_orthogonal_to_the_source_"
                    "classifier_direction_without_a_bias_coordinate"
                ),
                "pair_class_side": (
                    "the_r12_monotone_routed_decision_never_the_unconstrained_"
                    "gmm_density_tail"
                ),
                "pair_soft_assignment": (
                    "project_the_selected_gmm_posterior_onto_its_r12_decision_"
                    "half_interval"
                ),
                "pair_reliability": (
                    "absolute_centered_projected_gmm_posterior_no_threshold"
                ),
                "posterior_conflict_rule": (
                    "a_gmm_posterior_on_the_opposite_side_of_the_r12_decision_"
                    "projects_to_one_half_and_contributes_zero_pair_weight"
                ),
                "pair_objective": (
                    "soft_expected_fake_real_feature_difference_equals_one_"
                    "plus_unit_l2_weight_norm"
                ),
                "pair_target": "one_unit_fake_over_real_rank_difference",
                "pair_intercept": "none_pair_differences_cancel_any_constant",
                "pair_compression": (
                    "exact_current_batch_soft_pair_laplacian_eigendecomposition_"
                    "with_rank_at_most_batch_size_minus_one"
                ),
                "ridge_prior_precision": "fixed_identity_on_unit_features",
                "ridge_update": (
                    "exact_recursive_least_squares_woodbury_after_prediction"
                ),
                "ridge_sufficient_statistics": (
                    "one_stream_wide_inverse_regularized_gram_matrix_and_one_"
                    "weight_vector"
                ),
                "prediction_rule": (
                    "sigmoid_of_r12_base_logit_plus_one_global_bias_free_"
                    "feature_rank_residual"
                ),
                "global_score_rule": (
                    "one_shared_feature_rank_coordinate_for_every_routed_expert"
                ),
                "routing_score_coordinate": (
                    "immutable_source_score_never_the_pairwise_ridge_output"
                ),
                "gmm_update_score_coordinate": (
                    "immutable_source_score_never_the_pairwise_ridge_output"
                ),
                "source_fallback": "exact_r12_source_probability",
                "optimizer": "none_closed_form_recursive_ridge",
                "epoch": "none",
                "learning_rate": "none",
                "prediction_mutates_ranker": False,
                "raw_images_stored": False,
                "raw_features_stored": False,
                "raw_pairs_stored": False,
                "target_labels_used": False,
                "generator_boundaries_used": False,
                "semantic_features_used": False,
                "new_target_hyperparameters": 0,
                "hyperparameter_rule": (
                    "no_learning_rate_epoch_confidence_threshold_residual_weight_"
                    "routing_threshold_fusion_weight_pair_margin_or_memory_capacity"
                ),
                "intentional_changes": [
                    "all R12 routing and continual learning assignments remain unchanged",
                    "the selected R12 GMM supplies reliability but never a logit target",
                    "R12 supplies the monotone pseudo-class side for every soft pair",
                    "posterior evidence contradicting that side receives zero weight",
                    "one global head avoids expert-specific feature score scales",
                    "pair differences remove the residual intercept by construction",
                    "the bounded unit rank target cannot inherit extreme GMM log odds",
                    "predictions use only the global state learned after earlier batches",
                    "no image feature or pair remains after the sufficient-statistic update",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_pairwise_ridge"

    def _new_pairwise_ridge_state(self) -> dict[str, Any]:
        return {
            "inverse_gram": np.eye(
                self.pairwise_ridge_feature_dim,
                dtype=np.float64,
            ),
            "weights": np.zeros(
                self.pairwise_ridge_feature_dim,
                dtype=np.float64,
            ),
            "updates": 0,
            "candidate_samples": 0,
            "candidate_pairs": 0,
            "effective_pair_mass": 0.0,
            "compressed_rank_sum": 0,
        }

    @staticmethod
    def _pairwise_ridge_ready(state: dict[str, Any]) -> bool:
        if int(state["updates"]) <= 0:
            return False
        return float(np.linalg.norm(state["weights"])) > np.finfo(np.float64).eps

    def _batch_scores_and_pairwise_ridge_features(
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
                "ASCAL pairwise ridge expects (B, C, H, W) or "
                "(B, V, C, H, W) images"
            )
        forward_features = getattr(self.model, "forward_features", None)
        classifier = getattr(self.model, "classifier", None)
        if not callable(forward_features) or not callable(classifier):
            raise TypeError(
                "ASCAL pairwise ridge requires forward_features and classifier"
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
        if int(feature_values.shape[1]) != self.pairwise_ridge_feature_dim:
            raise ValueError(
                "ASCAL pairwise ridge feature dimension does not match the source head"
            )
        direction = self._pairwise_ridge_source_direction
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
        if self._pairwise_ridge_precomputed_scores is not None:
            return self._pairwise_ridge_precomputed_scores.copy()
        return super()._batch_scores(images)

    def _pairwise_ridge_context(self) -> dict[str, Any] | None:
        if self._pending is None:
            raise RuntimeError("ASCAL pairwise ridge lost its R12 prediction state")
        selected_expert = self._pending.get("prediction_routing_expert")
        if selected_expert is None:
            return None
        if selected_expert == "active_learning_state":
            if self._mixture is None or not self._mixture_active():
                raise RuntimeError(
                    "ASCAL pairwise ridge selected no eligible live GMM"
                )
            return self._mixture
        if selected_expert != "episodic_memory":
            raise RuntimeError("ASCAL pairwise ridge received an unknown R12 expert")
        memory_index = self._pending.get("prediction_routing_memory_index")
        if memory_index is None:
            raise RuntimeError(
                "ASCAL pairwise ridge selected memory without an index"
            )
        memory_index = int(memory_index)
        if not 0 <= memory_index < len(self.segment_memories):
            raise RuntimeError("ASCAL pairwise ridge selected memory out of range")
        return self.segment_memories[memory_index]["mixture"]

    @staticmethod
    def _pairwise_ridge_batch_system(
        features: np.ndarray,
        fake_mass: np.ndarray,
        real_mass: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        features = np.asarray(features, dtype=np.float64)
        fake_mass = np.asarray(fake_mass, dtype=np.float64).reshape(-1)
        real_mass = np.asarray(real_mass, dtype=np.float64).reshape(-1)
        if features.ndim != 2 or int(features.shape[0]) != int(fake_mass.size):
            raise ValueError("Pairwise Ridge features and soft masses must align")
        if real_mass.shape != fake_mass.shape:
            raise ValueError("Pairwise Ridge fake and real masses must align")
        if np.any(fake_mass < 0.0) or np.any(real_mass < 0.0):
            raise ValueError("Pairwise Ridge soft masses must be nonnegative")

        fake_total = float(fake_mass.sum())
        real_total = float(real_mass.sum())
        effective_pair_mass = float(
            fake_total * real_total - np.dot(fake_mass, real_mass)
        )
        dimension = int(features.shape[1])
        if effective_pair_mass <= np.finfo(np.float64).eps:
            return (
                np.zeros((0, dimension), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                max(effective_pair_mass, 0.0),
                0,
            )

        laplacian = np.diag(
            real_total * fake_mass + fake_total * real_mass
        ) - np.outer(fake_mass, real_mass) - np.outer(real_mass, fake_mass)
        laplacian = 0.5 * (laplacian + laplacian.T)
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        eigenvalue_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
        tolerance = (
            np.finfo(np.float64).eps
            * max(int(laplacian.shape[0]), 1)
            * eigenvalue_scale
        )
        positive = eigenvalues > tolerance
        if not np.any(positive):
            return (
                np.zeros((0, dimension), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                effective_pair_mass,
                0,
            )

        values = eigenvalues[positive]
        vectors = eigenvectors[:, positive]
        roots = np.sqrt(values)
        design = roots[:, None] * (vectors.T @ features)
        coefficient = real_total * fake_mass - fake_total * real_mass
        response = (vectors.T @ coefficient) / roots
        if not (
            np.all(np.isfinite(design))
            and np.all(np.isfinite(response))
        ):
            raise FloatingPointError(
                "ASCAL pairwise ridge produced a non-finite compressed pair system"
            )
        return design, response, effective_pair_mass, int(values.size)

    def _pairwise_ridge_supervision(
        self,
        mixture: dict[str, Any],
        scores: np.ndarray,
        routed_labels: np.ndarray,
        features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, int]:
        posterior = np.asarray(
            joint_density_fake_posterior(scores, mixture),
            dtype=np.float64,
        ).reshape(-1)
        routed_fake = np.asarray(routed_labels, dtype=np.int64).reshape(-1).astype(bool)
        if posterior.shape != routed_fake.shape:
            raise RuntimeError(
                "ASCAL pairwise ridge posterior and R12 labels do not align"
            )
        projected = np.where(
            routed_fake,
            np.maximum(posterior, 0.5),
            np.minimum(posterior, 0.5),
        )
        reliability = np.abs(2.0 * projected - 1.0)
        fake_mass = reliability * projected
        real_mass = reliability * (1.0 - projected)
        conflicts = int(
            np.count_nonzero(
                (posterior - 0.5)
                * np.where(routed_fake, 1.0, -1.0)
                < 0.0
            )
        )
        design, response, pair_mass, pair_rank = self._pairwise_ridge_batch_system(
            features,
            fake_mass,
            real_mass,
        )
        self.pairwise_ridge_last_effective_pair_mass = pair_mass
        self.pairwise_ridge_last_reliability = float(np.mean(reliability))
        self.pairwise_ridge_last_fake_mass = float(fake_mass.sum())
        self.pairwise_ridge_last_real_mass = float(real_mass.sum())
        self.pairwise_ridge_last_pair_rank = pair_rank
        self.pairwise_ridge_last_posterior_conflicts = conflicts
        return design, response, pair_mass, pair_rank

    def _update_pairwise_ridge_state(
        self,
        mixture: dict[str, Any],
        scores: np.ndarray,
        routed_labels: np.ndarray,
        features: np.ndarray,
    ) -> bool:
        state = self._pairwise_ridge_state
        samples = int(scores.size)
        candidate_pairs = samples * max(samples - 1, 0)
        self.pairwise_ridge_candidate_samples += samples
        self.pairwise_ridge_candidate_pairs += candidate_pairs
        state["candidate_samples"] = int(state["candidate_samples"]) + samples
        state["candidate_pairs"] = int(state["candidate_pairs"]) + candidate_pairs
        design, response, pair_mass, pair_rank = self._pairwise_ridge_supervision(
            mixture,
            scores,
            routed_labels,
            features,
        )
        if pair_rank <= 0 or pair_mass <= np.finfo(np.float64).eps:
            return False

        inverse_gram = np.asarray(state["inverse_gram"], dtype=np.float64)
        weights = np.asarray(state["weights"], dtype=np.float64)
        inverse_times_design = inverse_gram @ design.T
        innovation_gram = (
            np.eye(int(design.shape[0]), dtype=np.float64)
            + design @ inverse_times_design
        )
        innovation_gram = 0.5 * (innovation_gram + innovation_gram.T)
        try:
            gain = np.linalg.solve(
                innovation_gram,
                inverse_times_design.T,
            ).T
        except np.linalg.LinAlgError:
            self.pairwise_ridge_solve_failures += 1
            return False

        updated_weights = weights + gain @ (response - design @ weights)
        updated_inverse = inverse_gram - gain @ inverse_times_design.T
        updated_inverse = 0.5 * (updated_inverse + updated_inverse.T)
        if not (
            np.all(np.isfinite(updated_weights))
            and np.all(np.isfinite(updated_inverse))
        ):
            self.pairwise_ridge_solve_failures += 1
            return False

        state["weights"] = updated_weights
        state["inverse_gram"] = updated_inverse
        state["updates"] = int(state["updates"]) + 1
        state["effective_pair_mass"] = float(state["effective_pair_mass"]) + pair_mass
        state["compressed_rank_sum"] = int(state["compressed_rank_sum"]) + pair_rank
        self.pairwise_ridge_updates += 1
        self.pairwise_ridge_last_weight_norm = float(
            np.linalg.norm(updated_weights)
        )
        return self._pairwise_ridge_ready(state)

    def _pairwise_ridge_state_stats(self) -> dict[str, Any]:
        state = self._pairwise_ridge_state
        state_active = int(state["candidate_samples"]) > 0
        return {
            "pairwise_ridge_global_state_count": int(state_active),
            "pairwise_ridge_ready": self._pairwise_ridge_ready(state),
            "pairwise_ridge_updates": self.pairwise_ridge_updates,
            "pairwise_ridge_candidate_samples": (
                self.pairwise_ridge_candidate_samples
            ),
            "pairwise_ridge_candidate_pairs": self.pairwise_ridge_candidate_pairs,
            "pairwise_ridge_effective_pair_mass": float(
                state["effective_pair_mass"]
            ),
            "pairwise_ridge_compressed_rank_sum": int(
                state["compressed_rank_sum"]
            ),
            "pairwise_ridge_batches": self.pairwise_ridge_batches,
            "pairwise_ridge_samples": self.pairwise_ridge_samples,
            "pairwise_ridge_ready_batches": self.pairwise_ridge_ready_batches,
            "pairwise_ridge_solve_failures": self.pairwise_ridge_solve_failures,
            "pairwise_ridge_label_changes": self.pairwise_ridge_label_changes,
            "pairwise_ridge_real_to_fake": self.pairwise_ridge_real_to_fake,
            "pairwise_ridge_fake_to_real": self.pairwise_ridge_fake_to_real,
            "pairwise_ridge_last_effective_pair_mass": (
                self.pairwise_ridge_last_effective_pair_mass
            ),
            "pairwise_ridge_last_reliability": (
                self.pairwise_ridge_last_reliability
            ),
            "pairwise_ridge_last_fake_mass": self.pairwise_ridge_last_fake_mass,
            "pairwise_ridge_last_real_mass": self.pairwise_ridge_last_real_mass,
            "pairwise_ridge_last_pair_rank": self.pairwise_ridge_last_pair_rank,
            "pairwise_ridge_last_posterior_conflicts": (
                self.pairwise_ridge_last_posterior_conflicts
            ),
            "pairwise_ridge_weight_norm": float(np.linalg.norm(state["weights"])),
            "pairwise_ridge_weight_parameters": (
                self.pairwise_ridge_feature_dim if state_active else 0
            ),
            "pairwise_ridge_inverse_gram_values": (
                self.pairwise_ridge_feature_dim**2 if state_active else 0
            ),
            "pairwise_ridge_trainable_parameters": self.trainable_parameters,
        }

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(self._pairwise_ridge_state_stats())
        return stats

    def predict(self, images: Any) -> PredictionBatch:
        scores, features = self._batch_scores_and_pairwise_ridge_features(images)
        self._pairwise_ridge_precomputed_scores = scores
        try:
            ordinal = super().predict(images)
        finally:
            self._pairwise_ridge_precomputed_scores = None
        if self._pending is None:
            raise RuntimeError("ASCAL pairwise ridge lost the R12 pending state")

        mixture = self._pairwise_ridge_context()
        ready = mixture is not None and self._pairwise_ridge_ready(
            self._pairwise_ridge_state
        )
        if ready:
            residual = np.asarray(
                features @ self._pairwise_ridge_state["weights"],
                dtype=np.float64,
            )
        else:
            residual = np.zeros(int(scores.size), dtype=np.float64)

        base_probability = (
            ordinal.prob_fake.detach().cpu().numpy().astype(np.float64)
        )
        base_probability = np.clip(base_probability, 1e-6, 1.0 - 1e-6)
        if not ready:
            probability = base_probability
        else:
            base_logits = np.log(base_probability / (1.0 - base_probability))
            final_logits = np.clip(base_logits + residual, -60.0, 60.0)
            probability = 1.0 / (1.0 + np.exp(-final_logits))

        base_labels = ordinal.pred_label.detach().cpu().numpy().astype(np.int64)
        final_labels = (probability >= 0.5).astype(np.int64)
        label_changes = int(np.count_nonzero(final_labels != base_labels))
        real_to_fake = int(
            np.count_nonzero((base_labels == 0) & (final_labels == 1))
        )
        fake_to_real = int(
            np.count_nonzero((base_labels == 1) & (final_labels == 0))
        )

        self._pending_pairwise_ridge_features = features
        self._pending_pairwise_ridge_mixture = (
            None if mixture is None else _copy_gmm(mixture)
        )
        self._pending_pairwise_ridge_labels = base_labels.copy()
        pending_state = dict(self._pending)
        pending_state.pop("scores")
        pending_state.update(
            {
                "prediction_pairwise_ridge_applied": mixture is not None,
                "prediction_pairwise_ridge_ready": ready,
                "prediction_pairwise_ridge_residual_mean": float(
                    np.mean(residual)
                ),
                "prediction_pairwise_ridge_residual_abs_mean": float(
                    np.mean(np.abs(residual))
                ),
                "prediction_pairwise_ridge_residual_max_abs": float(
                    np.max(np.abs(residual))
                ),
                "prediction_pairwise_ridge_label_changes": label_changes,
                "prediction_pairwise_ridge_real_to_fake": real_to_fake,
                "prediction_pairwise_ridge_fake_to_real": fake_to_real,
                "prediction_pairwise_ridge_global_updates": int(
                    self._pairwise_ridge_state["updates"]
                ),
            }
        )
        return self._prediction_batch(scores, probability, **pending_state)

    def adapt(self, images: Any) -> AdaptationStats:
        if self._pending is None:
            return super().adapt(images)
        scores = np.asarray(self._pending["scores"], dtype=np.float64).reshape(-1)
        prediction_state = dict(self._pending)
        features = self._pending_pairwise_ridge_features
        mixture = self._pending_pairwise_ridge_mixture
        routed_labels = self._pending_pairwise_ridge_labels
        self._pending_pairwise_ridge_features = None
        self._pending_pairwise_ridge_mixture = None
        self._pending_pairwise_ridge_labels = None

        updated = False
        if self.adaptation_mode == "full" and mixture is not None:
            if features is None or int(features.shape[0]) != int(scores.size):
                raise RuntimeError(
                    "ASCAL pairwise ridge lost its matching prediction features"
                )
            if routed_labels is None or int(routed_labels.size) != int(scores.size):
                raise RuntimeError(
                    "ASCAL pairwise ridge lost its matching R12 decisions"
                )
            updated = self._update_pairwise_ridge_state(
                mixture,
                scores,
                routed_labels,
                features,
            )

        stats = super().adapt(images)
        if bool(prediction_state.get("prediction_pairwise_ridge_applied")):
            self.pairwise_ridge_batches += 1
            self.pairwise_ridge_samples += int(scores.size)
        if bool(prediction_state.get("prediction_pairwise_ridge_ready")):
            self.pairwise_ridge_ready_batches += 1
        self.pairwise_ridge_label_changes += int(
            prediction_state.get("prediction_pairwise_ridge_label_changes", 0) or 0
        )
        self.pairwise_ridge_real_to_fake += int(
            prediction_state.get("prediction_pairwise_ridge_real_to_fake", 0) or 0
        )
        self.pairwise_ridge_fake_to_real += int(
            prediction_state.get("prediction_pairwise_ridge_fake_to_real", 0) or 0
        )
        stats.extra.update(
            {
                **self._pairwise_ridge_state_stats(),
                "pairwise_ridge_updated": updated,
            }
        )
        return stats

    def discard_pending_prediction(self) -> None:
        self._pairwise_ridge_precomputed_scores = None
        self._pending_pairwise_ridge_features = None
        self._pending_pairwise_ridge_mixture = None
        self._pending_pairwise_ridge_labels = None
        super().discard_pending_prediction()


class ASCALGMMSegmentedMemoryPosteriorAnalyticExpert(
    ASCALGMMSegmentedMemoryPosteriorOrdinalRidge
):
    """Give each R12 expert one causal bounded soft-label Ridge classifier."""

    _ORDINAL_RIDGE_MEMORY_KEY = "analytic_expert_state"

    def _reset_state(self) -> None:
        super()._reset_state()
        self.analytic_expert_backbone_feature_dim = self.ordinal_ridge_feature_dim
        self.ordinal_ridge_feature_dim += 2
        self._novel_ordinal_ridge_state = self._new_ordinal_ridge_state()
        self._pending_analytic_expert_labels: np.ndarray | None = None
        self.analytic_expert_updates = 0
        self.analytic_expert_candidate_samples = 0
        self.analytic_expert_batches = 0
        self.analytic_expert_samples = 0
        self.analytic_expert_ready_batches = 0
        self.analytic_expert_solve_failures = 0
        self.analytic_expert_posterior_conflicts = 0
        self.analytic_expert_label_changes = 0
        self.analytic_expert_real_to_fake = 0
        self.analytic_expert_fake_to_real = 0
        self.analytic_expert_clipped_outputs = 0
        self.analytic_expert_last_effective_support = 0.0
        self.analytic_expert_last_reliability = 0.0
        self.analytic_expert_last_target_abs_mean = 0.0
        self.analytic_expert_last_posterior_conflicts = 0
        self.analytic_expert_last_anchor = 1.0
        self.analytic_expert_last_feature_weight_norm = 0.0
        self.analytic_expert_last_bias = 0.0

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        getter = ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.reproduction_metadata.fget
        if getter is None:
            raise RuntimeError("ASCAL analytic expert lost the R12 metadata getter")
        metadata = getter(self)
        metadata.update(
            {
                "adaptive_role": (
                    "r12_routed_per_expert_causal_bounded_soft_label_analytic_"
                    "ridge_adaptation"
                ),
                "research_name": "ASCAL-JMP-AnalyticExpert",
                "research_version": "R18",
                "r12_protected_scope": (
                    "immutable_source_score_unique_live_mdl_routing_gmm_"
                    "segmentation_memory_handoff_and_adaptation_assignment"
                ),
                "protected_initialization": (
                    "every_new_or_untrained_expert_reproduces_the_exact_r12_"
                    "probability"
                ),
                "accuracy_invariance": (
                    "exact_r12_until_the_selected_expert_has_past_soft_"
                    "supervision_then_sample_specific_changes_are_allowed"
                ),
                "expert_scope": (
                    "one_online_analytic_ridge_state_per_r12_gmm_expert"
                ),
                "expert_input": (
                    "r12_signed_probability_anchor_plus_l2_normalized_frozen_"
                    "features_orthogonal_to_the_source_classifier_direction_"
                    "plus_one_constant_bias"
                ),
                "expert_decomposition": (
                    "learned_anchor_scale_plus_feature_ranking_evidence_plus_"
                    "expert_bias"
                ),
                "soft_teacher": (
                    "equal_prior_prediction_time_selected_gmm_posterior_"
                    "projected_onto_the_r12_routed_class_side"
                ),
                "soft_target": "two_times_projected_posterior_minus_one",
                "reliability_rule": (
                    "absolute_signed_projected_posterior_without_a_hard_threshold"
                ),
                "posterior_conflict_rule": (
                    "contradictory_gmm_side_projects_to_half_and_zero_weight"
                ),
                "ridge_objective": (
                    "sum_reliability_times_squared_bounded_soft_class_error_"
                    "plus_squared_distance_from_the_r12_anchor_prior"
                ),
                "ridge_prior_mean": (
                    "unit_r12_anchor_scale_zero_feature_weights_and_zero_bias"
                ),
                "ridge_prior_precision": (
                    "fixed_identity_for_bounded_or_l2_normalized_coordinates"
                ),
                "ridge_update": (
                    "exact_recursive_least_squares_woodbury_after_prediction"
                ),
                "ridge_sufficient_statistics": (
                    "one_inverse_regularized_gram_matrix_and_one_weight_vector_"
                    "per_expert"
                ),
                "prediction_rule": (
                    "one_half_times_one_plus_the_selected_old_expert_signed_"
                    "ridge_score_clipped_only_to_the_probability_range"
                ),
                "routing_score_coordinate": (
                    "immutable_source_score_never_the_analytic_expert_output"
                ),
                "gmm_update_score_coordinate": (
                    "immutable_source_score_never_the_analytic_expert_output"
                ),
                "predict_then_adapt_order": (
                    "route_and_predict_with_old_expert_then_update_that_same_"
                    "expert_for_the_next_batch"
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
                    "all R12 routing and continual expert assignments remain unchanged",
                    "each R12 expert now restores its own analytic Ridge state",
                    "the centered prior reproduces the complete R12 probability before learning",
                    "the selected old GMM supplies a bounded soft label and continuous reliability",
                    "posterior evidence contradicting the R12 class side receives zero weight",
                    (
                        "feature weights can change sample order while the bias "
                        "can change the boundary"
                    ),
                    (
                        "analytic outputs never rewrite the source coordinate "
                        "used by routing or GMM fitting"
                    ),
                    "only the prediction-time selected expert updates after the batch prediction",
                    "no image or per-sample feature remains after the sufficient-statistic update",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_analytic_expert"

    def _new_ordinal_ridge_state(self) -> dict[str, Any]:
        weights = np.zeros(self.ordinal_ridge_feature_dim, dtype=np.float64)
        weights[0] = 1.0
        return {
            "inverse_gram": np.eye(
                self.ordinal_ridge_feature_dim,
                dtype=np.float64,
            ),
            "weights": weights,
            "updates": 0,
            "candidate_samples": 0,
            "effective_support": 0.0,
            "weighted_target_square_sum": 0.0,
            "posterior_conflicts": 0,
        }

    def _batch_scores_and_analytic_expert_features(
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
                "ASCAL analytic expert expects (B, C, H, W) or "
                "(B, V, C, H, W) images"
            )
        forward_features = getattr(self.model, "forward_features", None)
        classifier = getattr(self.model, "classifier", None)
        if not callable(forward_features) or not callable(classifier):
            raise TypeError(
                "ASCAL analytic expert requires forward_features and classifier"
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
        if int(feature_values.shape[1]) != self.analytic_expert_backbone_feature_dim:
            raise ValueError(
                "ASCAL analytic expert feature dimension does not match the source head"
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

    def _analytic_expert_design(
        self,
        base_probability: np.ndarray,
        features: np.ndarray,
    ) -> np.ndarray:
        base_probability = np.asarray(base_probability, dtype=np.float64).reshape(-1)
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or int(features.shape[0]) != int(base_probability.size):
            raise ValueError(
                "ASCAL analytic expert probabilities and features must align"
            )
        signed_anchor = 2.0 * base_probability - 1.0
        design = np.concatenate(
            [
                signed_anchor[:, None],
                features,
                np.ones((base_probability.size, 1), dtype=np.float64),
            ],
            axis=1,
        )
        if int(design.shape[1]) != self.ordinal_ridge_feature_dim:
            raise RuntimeError("ASCAL analytic expert built the wrong Ridge dimension")
        return design

    def _analytic_expert_supervision(
        self,
        mixture: dict[str, Any],
        scores: np.ndarray,
        routed_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        posterior = np.asarray(
            joint_density_fake_posterior(scores, mixture),
            dtype=np.float64,
        ).reshape(-1)
        routed_fake = np.asarray(routed_labels, dtype=np.int64).reshape(-1).astype(bool)
        if posterior.shape != routed_fake.shape:
            raise RuntimeError(
                "ASCAL analytic expert posterior and R12 labels do not align"
            )
        conflicts = (posterior - 0.5) * np.where(routed_fake, 1.0, -1.0) < 0.0
        projected = np.where(
            routed_fake,
            np.maximum(posterior, 0.5),
            np.minimum(posterior, 0.5),
        )
        targets = 2.0 * projected - 1.0
        reliability = np.abs(targets)
        if not (
            np.all(np.isfinite(projected))
            and np.all(np.isfinite(targets))
            and np.all(np.isfinite(reliability))
        ):
            raise FloatingPointError(
                "ASCAL analytic expert produced non-finite supervision"
            )
        conflict_count = int(np.count_nonzero(conflicts))
        self.analytic_expert_last_effective_support = float(reliability.sum())
        self.analytic_expert_last_reliability = float(np.mean(reliability))
        self.analytic_expert_last_target_abs_mean = float(np.mean(np.abs(targets)))
        self.analytic_expert_last_posterior_conflicts = conflict_count
        return projected, targets, reliability, conflict_count

    def _update_analytic_expert_state(
        self,
        state: dict[str, Any],
        mixture: dict[str, Any],
        scores: np.ndarray,
        routed_labels: np.ndarray,
        features: np.ndarray,
    ) -> bool:
        _, targets, reliability, conflicts = self._analytic_expert_supervision(
            mixture,
            scores,
            routed_labels,
        )
        samples = int(scores.size)
        effective_support = float(reliability.sum())
        self.analytic_expert_candidate_samples += samples
        self.analytic_expert_posterior_conflicts += conflicts
        state["candidate_samples"] = int(state["candidate_samples"]) + samples
        state["posterior_conflicts"] = int(state["posterior_conflicts"]) + conflicts
        if effective_support <= np.finfo(np.float64).eps:
            return False

        square_root_reliability = np.sqrt(reliability)
        design = square_root_reliability[:, None] * features
        response = square_root_reliability * targets
        inverse_gram = np.asarray(state["inverse_gram"], dtype=np.float64)
        weights = np.asarray(state["weights"], dtype=np.float64)
        inverse_times_design = inverse_gram @ design.T
        innovation_gram = (
            np.eye(int(design.shape[0]), dtype=np.float64)
            + design @ inverse_times_design
        )
        innovation_gram = 0.5 * (innovation_gram + innovation_gram.T)
        try:
            gain = np.linalg.solve(
                innovation_gram,
                inverse_times_design.T,
            ).T
        except np.linalg.LinAlgError:
            self.analytic_expert_solve_failures += 1
            return False

        updated_weights = weights + gain @ (response - design @ weights)
        updated_inverse = inverse_gram - gain @ inverse_times_design.T
        updated_inverse = 0.5 * (updated_inverse + updated_inverse.T)
        if not (
            np.all(np.isfinite(updated_weights))
            and np.all(np.isfinite(updated_inverse))
        ):
            self.analytic_expert_solve_failures += 1
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
        self.analytic_expert_updates += 1
        self.analytic_expert_last_anchor = float(updated_weights[0])
        self.analytic_expert_last_feature_weight_norm = float(
            np.linalg.norm(updated_weights[1:-1])
        )
        self.analytic_expert_last_bias = float(updated_weights[-1])
        return self._ordinal_ridge_ready(state)

    def _analytic_expert_state_stats(self) -> dict[str, Any]:
        states = self._all_ordinal_ridge_states()
        anchors = [float(state["weights"][0]) for state in states]
        feature_norms = [
            float(np.linalg.norm(state["weights"][1:-1])) for state in states
        ]
        biases = [float(state["weights"][-1]) for state in states]
        return {
            "analytic_expert_count": len(states),
            "analytic_expert_ready_experts": sum(
                self._ordinal_ridge_ready(state) for state in states
            ),
            "analytic_expert_updates": self.analytic_expert_updates,
            "analytic_expert_candidate_samples": (
                self.analytic_expert_candidate_samples
            ),
            "analytic_expert_batches": self.analytic_expert_batches,
            "analytic_expert_samples": self.analytic_expert_samples,
            "analytic_expert_ready_batches": self.analytic_expert_ready_batches,
            "analytic_expert_solve_failures": self.analytic_expert_solve_failures,
            "analytic_expert_posterior_conflicts": (
                self.analytic_expert_posterior_conflicts
            ),
            "analytic_expert_label_changes": self.analytic_expert_label_changes,
            "analytic_expert_real_to_fake": self.analytic_expert_real_to_fake,
            "analytic_expert_fake_to_real": self.analytic_expert_fake_to_real,
            "analytic_expert_clipped_outputs": self.analytic_expert_clipped_outputs,
            "analytic_expert_effective_support": sum(
                float(state["effective_support"]) for state in states
            ),
            "analytic_expert_last_effective_support": (
                self.analytic_expert_last_effective_support
            ),
            "analytic_expert_last_reliability": (
                self.analytic_expert_last_reliability
            ),
            "analytic_expert_last_target_abs_mean": (
                self.analytic_expert_last_target_abs_mean
            ),
            "analytic_expert_last_posterior_conflicts": (
                self.analytic_expert_last_posterior_conflicts
            ),
            "analytic_expert_mean_anchor": (
                float(np.mean(anchors)) if anchors else 1.0
            ),
            "analytic_expert_max_anchor_deviation": max(
                (abs(value - 1.0) for value in anchors),
                default=0.0,
            ),
            "analytic_expert_max_feature_weight_norm": max(
                feature_norms,
                default=0.0,
            ),
            "analytic_expert_max_bias_abs": max(
                (abs(value) for value in biases),
                default=0.0,
            ),
            "analytic_expert_weight_parameters": (
                len(states) * self.ordinal_ridge_feature_dim
            ),
            "analytic_expert_inverse_gram_values": (
                len(states) * self.ordinal_ridge_feature_dim**2
            ),
            "analytic_expert_trainable_parameters": self.trainable_parameters,
        }

    def _state_stats(self) -> dict[str, Any]:
        stats = ASCALGMMSegmentedMemoryPosteriorOrdinalRoute._state_stats(self)
        stats.update(self._analytic_expert_state_stats())
        return stats

    def predict(self, images: Any) -> PredictionBatch:
        scores, backbone_features = self._batch_scores_and_analytic_expert_features(
            images
        )
        self._ordinal_ridge_precomputed_scores = scores
        try:
            ordinal = ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.predict(
                self,
                images,
            )
        finally:
            self._ordinal_ridge_precomputed_scores = None
        if self._pending is None:
            raise RuntimeError("ASCAL analytic expert lost the R12 pending state")

        context = self._ordinal_ridge_context()
        assignment = None if context is None else context[0]
        mixture = None if context is None else context[1]
        state = (
            None
            if assignment is None
            else self._peek_ordinal_ridge_state(assignment)
        )
        ready = self._ordinal_ridge_ready(state)
        base_probability = (
            ordinal.prob_fake.detach().cpu().numpy().astype(np.float64)
        )
        features = self._analytic_expert_design(
            base_probability,
            backbone_features,
        )
        base_coordinate = features[:, 0]
        if not ready:
            final_coordinate = base_coordinate.copy()
            probability = base_probability.copy()
        else:
            if state is None:
                raise RuntimeError("ASCAL analytic expert lost its selected state")
            final_coordinate = np.asarray(
                features @ state["weights"],
                dtype=np.float64,
            )
            raw_probability = 0.5 * (final_coordinate + 1.0)
            probability = np.clip(raw_probability, 1e-6, 1.0 - 1e-6)
            self.analytic_expert_clipped_outputs += int(
                np.count_nonzero(probability != raw_probability)
            )

        base_labels = ordinal.pred_label.detach().cpu().numpy().astype(np.int64)
        final_labels = (probability >= 0.5).astype(np.int64)
        label_changes = int(np.count_nonzero(final_labels != base_labels))
        real_to_fake = int(
            np.count_nonzero((base_labels == 0) & (final_labels == 1))
        )
        fake_to_real = int(
            np.count_nonzero((base_labels == 1) & (final_labels == 0))
        )
        correction = final_coordinate - base_coordinate

        self._pending_ordinal_ridge_features = features
        self._pending_ordinal_ridge_state = state
        self._pending_ordinal_ridge_assignment = assignment
        self._pending_ordinal_ridge_mixture = (
            None if mixture is None else _copy_gmm(mixture)
        )
        self._pending_analytic_expert_labels = base_labels.copy()
        pending_state = dict(self._pending)
        pending_state.pop("scores")
        pending_state.update(
            {
                "prediction_analytic_expert_applied": context is not None,
                "prediction_analytic_expert_ready": ready,
                "prediction_analytic_expert_correction_mean": float(
                    np.mean(correction)
                ),
                "prediction_analytic_expert_correction_abs_mean": float(
                    np.mean(np.abs(correction))
                ),
                "prediction_analytic_expert_correction_max_abs": float(
                    np.max(np.abs(correction))
                ),
                "prediction_analytic_expert_anchor": (
                    1.0 if state is None else float(state["weights"][0])
                ),
                "prediction_analytic_expert_feature_weight_norm": (
                    0.0
                    if state is None
                    else float(np.linalg.norm(state["weights"][1:-1]))
                ),
                "prediction_analytic_expert_bias": (
                    0.0 if state is None else float(state["weights"][-1])
                ),
                "prediction_analytic_expert_label_changes": label_changes,
                "prediction_analytic_expert_real_to_fake": real_to_fake,
                "prediction_analytic_expert_fake_to_real": fake_to_real,
            }
        )
        return self._prediction_batch(scores, probability, **pending_state)

    def adapt(self, images: Any) -> AdaptationStats:
        if self._pending is None:
            return ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.adapt(self, images)
        scores = np.asarray(self._pending["scores"], dtype=np.float64).reshape(-1)
        prediction_state = dict(self._pending)
        features = self._pending_ordinal_ridge_features
        state = self._pending_ordinal_ridge_state
        assignment = self._pending_ordinal_ridge_assignment
        mixture = self._pending_ordinal_ridge_mixture
        routed_labels = self._pending_analytic_expert_labels
        self._pending_ordinal_ridge_features = None
        self._pending_ordinal_ridge_state = None
        self._pending_ordinal_ridge_assignment = None
        self._pending_ordinal_ridge_mixture = None
        self._pending_analytic_expert_labels = None

        updated = False
        if self.adaptation_mode == "full" and assignment is not None:
            if features is None or int(features.shape[0]) != int(scores.size):
                raise RuntimeError(
                    "ASCAL analytic expert lost its matching prediction features"
                )
            if mixture is None:
                raise RuntimeError(
                    "ASCAL analytic expert lost its prediction-time selected GMM"
                )
            if routed_labels is None or int(routed_labels.size) != int(scores.size):
                raise RuntimeError(
                    "ASCAL analytic expert lost its matching R12 decisions"
                )
            if state is None:
                state = self._ensure_ordinal_ridge_state(assignment)
            updated = self._update_analytic_expert_state(
                state,
                mixture,
                scores,
                routed_labels,
                features,
            )

        stats = ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.adapt(self, images)
        if bool(prediction_state.get("prediction_analytic_expert_applied")):
            self.analytic_expert_batches += 1
            self.analytic_expert_samples += int(scores.size)
        if bool(prediction_state.get("prediction_analytic_expert_ready")):
            self.analytic_expert_ready_batches += 1
        self.analytic_expert_label_changes += int(
            prediction_state.get("prediction_analytic_expert_label_changes", 0) or 0
        )
        self.analytic_expert_real_to_fake += int(
            prediction_state.get("prediction_analytic_expert_real_to_fake", 0) or 0
        )
        self.analytic_expert_fake_to_real += int(
            prediction_state.get("prediction_analytic_expert_fake_to_real", 0) or 0
        )
        stats.extra.update(
            {
                **self._analytic_expert_state_stats(),
                "analytic_expert_updated": updated,
            }
        )
        return stats

    def discard_pending_prediction(self) -> None:
        self._ordinal_ridge_precomputed_scores = None
        self._pending_ordinal_ridge_features = None
        self._pending_ordinal_ridge_state = None
        self._pending_ordinal_ridge_assignment = None
        self._pending_ordinal_ridge_mixture = None
        self._pending_analytic_expert_labels = None
        ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.discard_pending_prediction(self)


class ASCALGMMSegmentedMemoryPosteriorRMSRidgeExpert(
    ASCALGMMSegmentedMemoryPosteriorOrdinalRidge
):
    """Fuse R12 with one confidence-weighted binary Ridge per routed expert."""

    _ORDINAL_RIDGE_MEMORY_KEY = "rms_ridge_expert_state"

    def _reset_state(self) -> None:
        super()._reset_state()
        self.rms_ridge_expert_backbone_feature_dim = self.ordinal_ridge_feature_dim
        self.ordinal_ridge_feature_dim += 1
        self._novel_ordinal_ridge_state = self._new_ordinal_ridge_state()
        self._pending_rms_ridge_expert_labels: np.ndarray | None = None
        self._pending_rms_ridge_expert_base_margins: np.ndarray | None = None
        self.rms_ridge_expert_updates = 0
        self.rms_ridge_expert_candidate_samples = 0
        self.rms_ridge_expert_routed_batches = 0
        self.rms_ridge_expert_routed_samples = 0
        self.rms_ridge_expert_applied_batches = 0
        self.rms_ridge_expert_applied_samples = 0
        self.rms_ridge_expert_cold_start_batches = 0
        self.rms_ridge_expert_solve_failures = 0
        self.rms_ridge_expert_posterior_conflicts = 0
        self.rms_ridge_expert_label_changes = 0
        self.rms_ridge_expert_real_to_fake = 0
        self.rms_ridge_expert_fake_to_real = 0
        self.rms_ridge_expert_last_effective_support = 0.0
        self.rms_ridge_expert_last_reliability = 0.0
        self.rms_ridge_expert_last_real_mass = 0.0
        self.rms_ridge_expert_last_fake_mass = 0.0
        self.rms_ridge_expert_last_posterior_conflicts = 0

    @property
    def trainable_parameters(self) -> int:
        if self.adaptation_mode == "static":
            return 0
        return (
            len(self._all_ordinal_ridge_states())
            * self.ordinal_ridge_feature_dim
            * 2
        )

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        getter = ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.reproduction_metadata.fget
        if getter is None:
            raise RuntimeError("ASCAL RMS Ridge expert lost the R12 metadata getter")
        metadata = getter(self)
        metadata.update(
            {
                "adaptive_role": (
                    "r12_routed_per_expert_causal_direct_binary_analytic_ridge_"
                    "with_historical_rms_evidence_alignment"
                ),
                "research_name": "ASCAL-JMP-RMSRidgeExpert",
                "research_version": "R19",
                "r12_protected_scope": (
                    "immutable_source_score_unique_live_mdl_routing_gmm_"
                    "segmentation_memory_handoff_and_adaptation_assignment"
                ),
                "protected_initialization": (
                    "exact_r12_probability_until_the_selected_expert_has_"
                    "reliable_mass_for_both_pseudo_classes_and_valid_rms"
                ),
                "expert_scope": (
                    "one_online_two_output_analytic_ridge_state_per_r12_gmm_expert"
                ),
                "expert_input": (
                    "l2_normalized_frozen_clip_feature_plus_one_constant_bias"
                ),
                "pseudo_label": "exact_r12_routed_hard_class_after_prediction",
                "reliability_rule": (
                    "absolute_selected_gmm_signed_posterior_with_no_hard_threshold"
                ),
                "posterior_conflict_rule": (
                    "gmm_posterior_on_the_opposite_r12_class_side_gets_zero_weight"
                ),
                "ridge_target": "real_or_fake_one_hot_vector",
                "ridge_objective": (
                    "sum_reliability_times_two_output_one_hot_squared_error_"
                    "plus_unit_frobenius_weight_norm"
                ),
                "ridge_prior_precision": (
                    "fixed_unit_identity_under_l2_normalized_clip_features"
                ),
                "ridge_update": (
                    "exact_recursive_least_squares_woodbury_after_prediction"
                ),
                "ridge_sufficient_statistics": (
                    "inverse_regularized_gram_cross_covariance_two_output_"
                    "weights_class_masses_and_r12_margin_square_sum_per_expert"
                ),
                "ridge_margin": "fake_output_score_minus_real_output_score",
                "scale_alignment": (
                    "per_expert_historical_reliability_weighted_rms_for_r12_logit_"
                    "and_current_ridge_margin"
                ),
                "ridge_rms_identity": (
                    "d_transpose_c_d_equals_d_transpose_q_minus_squared_d_norm"
                ),
                "prediction_rule": (
                    "sigmoid_of_r12_logit_over_past_r12_rms_plus_selected_"
                    "ridge_margin_over_its_past_feature_rms"
                ),
                "cold_start_rule": (
                    "return_exact_r12_probability_without_a_minimum_sample_knob"
                ),
                "final_probability_semantics": (
                    "monotone_evaluator_compatible_score_not_a_calibrated_posterior"
                ),
                "routing_score_coordinate": (
                    "immutable_source_score_never_the_ridge_expert_output"
                ),
                "gmm_update_score_coordinate": (
                    "immutable_source_score_never_the_ridge_expert_output"
                ),
                "predict_then_adapt_order": (
                    "route_and_predict_with_old_expert_then_update_that_same_"
                    "expert_for_the_next_batch"
                ),
                "fusion_weight": "none_equal_unit_rms_evidence",
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
                    "no_learning_rate_epoch_confidence_threshold_temperature_"
                    "fusion_coefficient_routing_threshold_or_memory_capacity"
                ),
                "intentional_changes": [
                    "all R12 routing and continual expert assignments remain unchanged",
                    "each R12 expert restores its own direct binary Ridge classifier",
                    "R12 supplies only the pseudo-class and never a regression target",
                    "the prediction-time selected GMM supplies only continuous reliability",
                    "the complete frozen CLIP feature is normalized and retained as input",
                    (
                        "historical sufficient statistics align both evidence "
                        "scales without target tuning"
                    ),
                    "only experts with reliable mass on both pseudo-classes may change R12",
                    "only the prediction-time selected expert updates after prediction",
                    "no image or per-sample feature remains after the analytic update",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_rms_ridge_expert"

    def _new_ordinal_ridge_state(self) -> dict[str, Any]:
        return {
            "inverse_gram": np.eye(
                self.ordinal_ridge_feature_dim,
                dtype=np.float64,
            ),
            "cross_covariance": np.zeros(
                (self.ordinal_ridge_feature_dim, 2),
                dtype=np.float64,
            ),
            "weights": np.zeros(
                (self.ordinal_ridge_feature_dim, 2),
                dtype=np.float64,
            ),
            "updates": 0,
            "candidate_samples": 0,
            "effective_support": 0.0,
            "class_mass": np.zeros(2, dtype=np.float64),
            "base_margin_square_sum": 0.0,
            "posterior_conflicts": 0,
        }

    def _batch_scores_and_rms_ridge_expert_features(
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
                "ASCAL RMS Ridge expert expects (B, C, H, W) or "
                "(B, V, C, H, W) images"
            )
        forward_features = getattr(self.model, "forward_features", None)
        forward_classifier_features = getattr(
            self.model, "forward_classifier_features", None
        )
        classifier = getattr(self.model, "classifier", None)
        if not callable(forward_features) or not callable(classifier):
            raise TypeError(
                "ASCAL RMS Ridge expert requires forward_features and classifier"
            )
        with torch.no_grad():
            features = forward_features(flat.to(self.device, non_blocking=True))
            classifier_features = (
                forward_classifier_features(features)
                if callable(forward_classifier_features)
                else features
            )
            logits = classifier(classifier_features)
        scores = (
            binary_score(logits)
            .view(batch, views)
            .mean(dim=1)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        feature_values = (
            classifier_features.detach()
            .float()
            .view(batch, views, -1)
            .mean(dim=1)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        if int(feature_values.shape[1]) != self.rms_ridge_expert_backbone_feature_dim:
            raise ValueError(
                "ASCAL RMS Ridge expert feature dimension does not match the source head"
            )
        norms = np.linalg.norm(feature_values, axis=1, keepdims=True)
        feature_values = np.divide(
            feature_values,
            norms,
            out=np.zeros_like(feature_values),
            where=norms > np.finfo(np.float64).eps,
        )
        design = np.concatenate(
            [
                feature_values,
                np.ones((batch, 1), dtype=np.float64),
            ],
            axis=1,
        )
        if int(design.shape[1]) != self.ordinal_ridge_feature_dim:
            raise RuntimeError("ASCAL RMS Ridge expert built the wrong dimension")
        return scores, design

    def _rms_ridge_expert_supervision(
        self,
        mixture: dict[str, Any],
        scores: np.ndarray,
        routed_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        posterior = np.asarray(
            joint_density_fake_posterior(scores, mixture),
            dtype=np.float64,
        ).reshape(-1)
        labels = np.asarray(routed_labels, dtype=np.int64).reshape(-1)
        if posterior.shape != labels.shape or np.any((labels < 0) | (labels > 1)):
            raise RuntimeError(
                "ASCAL RMS Ridge expert posterior and R12 labels do not align"
            )
        signed_posterior = 2.0 * posterior - 1.0
        conflicts = signed_posterior * np.where(labels == 1, 1.0, -1.0) < 0.0
        reliability = np.where(conflicts, 0.0, np.abs(signed_posterior))
        targets = np.zeros((labels.size, 2), dtype=np.float64)
        targets[np.arange(labels.size), labels] = 1.0
        if not (
            np.all(np.isfinite(posterior))
            and np.all(np.isfinite(reliability))
        ):
            raise FloatingPointError(
                "ASCAL RMS Ridge expert produced non-finite supervision"
            )
        conflict_count = int(np.count_nonzero(conflicts))
        class_mass = np.bincount(labels, weights=reliability, minlength=2)
        self.rms_ridge_expert_last_effective_support = float(reliability.sum())
        self.rms_ridge_expert_last_reliability = float(np.mean(reliability))
        self.rms_ridge_expert_last_real_mass = float(class_mass[0])
        self.rms_ridge_expert_last_fake_mass = float(class_mass[1])
        self.rms_ridge_expert_last_posterior_conflicts = conflict_count
        return targets, reliability, posterior, conflict_count

    def _update_rms_ridge_expert_state(
        self,
        state: dict[str, Any],
        mixture: dict[str, Any],
        scores: np.ndarray,
        routed_labels: np.ndarray,
        features: np.ndarray,
        base_margins: np.ndarray,
    ) -> bool:
        targets, reliability, _, conflicts = self._rms_ridge_expert_supervision(
            mixture,
            scores,
            routed_labels,
        )
        features = np.asarray(features, dtype=np.float64)
        base_margins = np.asarray(base_margins, dtype=np.float64).reshape(-1)
        samples = int(scores.size)
        if (
            features.shape != (samples, self.ordinal_ridge_feature_dim)
            or base_margins.shape != (samples,)
            or not np.all(np.isfinite(features))
            or not np.all(np.isfinite(base_margins))
        ):
            raise RuntimeError(
                "ASCAL RMS Ridge expert received misaligned prediction statistics"
            )
        effective_support = float(reliability.sum())
        self.rms_ridge_expert_candidate_samples += samples
        self.rms_ridge_expert_posterior_conflicts += conflicts
        state["candidate_samples"] = int(state["candidate_samples"]) + samples
        state["posterior_conflicts"] = int(state["posterior_conflicts"]) + conflicts
        if effective_support <= np.finfo(np.float64).eps:
            return False

        square_root_reliability = np.sqrt(reliability)
        design = square_root_reliability[:, None] * features
        response = square_root_reliability[:, None] * targets
        inverse_gram = np.asarray(state["inverse_gram"], dtype=np.float64)
        weights = np.asarray(state["weights"], dtype=np.float64)
        inverse_times_design = inverse_gram @ design.T
        innovation_gram = (
            np.eye(samples, dtype=np.float64) + design @ inverse_times_design
        )
        innovation_gram = 0.5 * (innovation_gram + innovation_gram.T)
        try:
            gain = np.linalg.solve(
                innovation_gram,
                inverse_times_design.T,
            ).T
        except np.linalg.LinAlgError:
            self.rms_ridge_expert_solve_failures += 1
            return False

        updated_weights = weights + gain @ (response - design @ weights)
        updated_inverse = inverse_gram - gain @ inverse_times_design.T
        updated_inverse = 0.5 * (updated_inverse + updated_inverse.T)
        updated_cross_covariance = np.asarray(
            state["cross_covariance"],
            dtype=np.float64,
        ) + features.T @ (reliability[:, None] * targets)
        if not (
            np.all(np.isfinite(updated_weights))
            and np.all(np.isfinite(updated_inverse))
            and np.all(np.isfinite(updated_cross_covariance))
        ):
            self.rms_ridge_expert_solve_failures += 1
            return False

        labels = np.argmax(targets, axis=1).astype(np.int64, copy=False)
        class_mass = np.bincount(labels, weights=reliability, minlength=2)
        state["weights"] = updated_weights
        state["inverse_gram"] = updated_inverse
        state["cross_covariance"] = updated_cross_covariance
        state["updates"] = int(state["updates"]) + 1
        state["effective_support"] = float(state["effective_support"]) + (
            effective_support
        )
        state["class_mass"] = np.asarray(
            state["class_mass"],
            dtype=np.float64,
        ) + class_mass
        state["base_margin_square_sum"] = float(
            state["base_margin_square_sum"]
        ) + float(np.sum(reliability * base_margins**2))
        self.rms_ridge_expert_updates += 1
        return self._ordinal_ridge_ready(state)

    def _rms_ridge_expert_scales(
        self,
        state: dict[str, Any] | None,
    ) -> tuple[float, float, float] | None:
        if state is None:
            return None
        support = float(state["effective_support"])
        if not math.isfinite(support) or support <= np.finfo(np.float64).eps:
            return None
        base_energy = float(state["base_margin_square_sum"])
        weights = np.asarray(state["weights"], dtype=np.float64)
        cross_covariance = np.asarray(
            state["cross_covariance"],
            dtype=np.float64,
        )
        if (
            weights.shape != (self.ordinal_ridge_feature_dim, 2)
            or cross_covariance.shape != weights.shape
            or not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(cross_covariance))
        ):
            return None
        direction = weights[:, 1] - weights[:, 0]
        cross_direction = cross_covariance[:, 1] - cross_covariance[:, 0]
        ridge_energy = float(
            direction @ cross_direction - direction @ direction
        )
        ridge_energy = max(0.0, ridge_energy)
        if not (
            math.isfinite(base_energy)
            and math.isfinite(ridge_energy)
            and base_energy > 0.0
            and ridge_energy > 0.0
        ):
            return None
        base_rms = math.sqrt(base_energy / support)
        ridge_rms = math.sqrt(ridge_energy / support)
        if not (
            math.isfinite(base_rms)
            and math.isfinite(ridge_rms)
            and base_rms > np.finfo(np.float64).eps
            and ridge_rms > np.finfo(np.float64).eps
        ):
            return None
        return base_rms, ridge_rms, ridge_energy

    def _ordinal_ridge_ready(self, state: dict[str, Any] | None) -> bool:
        if state is None or int(state["updates"]) <= 0:
            return False
        class_mass = np.asarray(state["class_mass"], dtype=np.float64).reshape(-1)
        if (
            class_mass.shape != (2,)
            or not np.all(np.isfinite(class_mass))
            or np.any(class_mass <= np.finfo(np.float64).eps)
        ):
            return False
        return self._rms_ridge_expert_scales(state) is not None

    @staticmethod
    def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        probability = np.empty_like(values)
        positive = values >= 0.0
        probability[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
        negative_exp = np.exp(values[~positive])
        probability[~positive] = negative_exp / (1.0 + negative_exp)
        return probability

    def _rms_ridge_expert_probability(
        self,
        base_probability: np.ndarray,
        features: np.ndarray,
        state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        scales = self._rms_ridge_expert_scales(state)
        if scales is None:
            raise RuntimeError("ASCAL RMS Ridge expert has no valid historical scale")
        base_rms, ridge_rms, _ = scales
        base_probability = np.asarray(base_probability, dtype=np.float64).reshape(-1)
        epsilon = np.finfo(np.float64).eps
        bounded_probability = np.clip(base_probability, epsilon, 1.0 - epsilon)
        base_margin = np.log(bounded_probability / (1.0 - bounded_probability))
        weights = np.asarray(state["weights"], dtype=np.float64)
        direction = weights[:, 1] - weights[:, 0]
        ridge_margin = np.asarray(features @ direction, dtype=np.float64)
        fused_margin = base_margin / base_rms + ridge_margin / ridge_rms
        probability = self._stable_sigmoid(fused_margin)
        if not (
            np.all(np.isfinite(base_margin))
            and np.all(np.isfinite(ridge_margin))
            and np.all(np.isfinite(probability))
        ):
            raise FloatingPointError(
                "ASCAL RMS Ridge expert produced a non-finite prediction"
            )
        return probability, base_margin, ridge_margin, base_rms, ridge_rms

    def _rms_ridge_expert_state_stats(self) -> dict[str, Any]:
        states = self._all_ordinal_ridge_states()
        ready_states = [state for state in states if self._ordinal_ridge_ready(state)]
        scales = [
            self._rms_ridge_expert_scales(state) for state in ready_states
        ]
        valid_scales = [scale for scale in scales if scale is not None]
        class_masses = [
            np.asarray(state["class_mass"], dtype=np.float64) for state in states
        ]
        weight_norms = [
            float(np.linalg.norm(np.asarray(state["weights"], dtype=np.float64)))
            for state in states
        ]
        return {
            "rms_ridge_expert_count": len(states),
            "rms_ridge_expert_ready_experts": len(ready_states),
            "rms_ridge_expert_updates": self.rms_ridge_expert_updates,
            "rms_ridge_expert_candidate_samples": (
                self.rms_ridge_expert_candidate_samples
            ),
            "rms_ridge_expert_routed_batches": self.rms_ridge_expert_routed_batches,
            "rms_ridge_expert_routed_samples": self.rms_ridge_expert_routed_samples,
            "rms_ridge_expert_applied_batches": (
                self.rms_ridge_expert_applied_batches
            ),
            "rms_ridge_expert_applied_samples": (
                self.rms_ridge_expert_applied_samples
            ),
            "rms_ridge_expert_cold_start_batches": (
                self.rms_ridge_expert_cold_start_batches
            ),
            "rms_ridge_expert_solve_failures": (
                self.rms_ridge_expert_solve_failures
            ),
            "rms_ridge_expert_posterior_conflicts": (
                self.rms_ridge_expert_posterior_conflicts
            ),
            "rms_ridge_expert_label_changes": self.rms_ridge_expert_label_changes,
            "rms_ridge_expert_real_to_fake": self.rms_ridge_expert_real_to_fake,
            "rms_ridge_expert_fake_to_real": self.rms_ridge_expert_fake_to_real,
            "rms_ridge_expert_effective_support": sum(
                float(state["effective_support"]) for state in states
            ),
            "rms_ridge_expert_real_mass": sum(
                float(mass[0]) for mass in class_masses
            ),
            "rms_ridge_expert_fake_mass": sum(
                float(mass[1]) for mass in class_masses
            ),
            "rms_ridge_expert_base_margin_square_sum": sum(
                float(state["base_margin_square_sum"]) for state in states
            ),
            "rms_ridge_expert_last_effective_support": (
                self.rms_ridge_expert_last_effective_support
            ),
            "rms_ridge_expert_last_reliability": (
                self.rms_ridge_expert_last_reliability
            ),
            "rms_ridge_expert_last_real_mass": (
                self.rms_ridge_expert_last_real_mass
            ),
            "rms_ridge_expert_last_fake_mass": (
                self.rms_ridge_expert_last_fake_mass
            ),
            "rms_ridge_expert_last_posterior_conflicts": (
                self.rms_ridge_expert_last_posterior_conflicts
            ),
            "rms_ridge_expert_mean_base_rms": (
                float(np.mean([scale[0] for scale in valid_scales]))
                if valid_scales
                else 0.0
            ),
            "rms_ridge_expert_mean_ridge_rms": (
                float(np.mean([scale[1] for scale in valid_scales]))
                if valid_scales
                else 0.0
            ),
            "rms_ridge_expert_max_weight_norm": max(weight_norms, default=0.0),
            "rms_ridge_expert_weight_parameters": (
                len(states) * self.ordinal_ridge_feature_dim * 2
            ),
            "rms_ridge_expert_inverse_gram_values": (
                len(states) * self.ordinal_ridge_feature_dim**2
            ),
            "rms_ridge_expert_cross_covariance_values": (
                len(states) * self.ordinal_ridge_feature_dim * 2
            ),
            "rms_ridge_expert_trainable_parameters": self.trainable_parameters,
        }

    def _state_stats(self) -> dict[str, Any]:
        stats = ASCALGMMSegmentedMemoryPosteriorOrdinalRoute._state_stats(self)
        stats.update(self._rms_ridge_expert_state_stats())
        return stats

    def predict(self, images: Any) -> PredictionBatch:
        scores, features = self._batch_scores_and_rms_ridge_expert_features(images)
        self._ordinal_ridge_precomputed_scores = scores
        try:
            ordinal = ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.predict(
                self,
                images,
            )
        finally:
            self._ordinal_ridge_precomputed_scores = None
        if self._pending is None:
            raise RuntimeError("ASCAL RMS Ridge expert lost the R12 pending state")

        context = self._ordinal_ridge_context()
        assignment = None if context is None else context[0]
        mixture = None if context is None else context[1]
        state = (
            None
            if assignment is None
            else self._peek_ordinal_ridge_state(assignment)
        )
        ready = self._ordinal_ridge_ready(state)
        base_probability = (
            ordinal.prob_fake.detach().cpu().numpy().astype(np.float64)
        )
        epsilon = np.finfo(np.float64).eps
        bounded_base = np.clip(base_probability, epsilon, 1.0 - epsilon)
        base_margin = np.log(bounded_base / (1.0 - bounded_base))
        ridge_margin = np.zeros_like(base_margin)
        base_rms = 0.0
        ridge_rms = 0.0
        if ready:
            if state is None:
                raise RuntimeError("ASCAL RMS Ridge expert lost its selected state")
            (
                probability,
                base_margin,
                ridge_margin,
                base_rms,
                ridge_rms,
            ) = self._rms_ridge_expert_probability(
                base_probability,
                features,
                state,
            )
        else:
            probability = base_probability.copy()

        base_labels = ordinal.pred_label.detach().cpu().numpy().astype(np.int64)
        final_labels = (probability >= 0.5).astype(np.int64)
        label_changes = int(np.count_nonzero(final_labels != base_labels))
        real_to_fake = int(
            np.count_nonzero((base_labels == 0) & (final_labels == 1))
        )
        fake_to_real = int(
            np.count_nonzero((base_labels == 1) & (final_labels == 0))
        )

        self._pending_ordinal_ridge_features = features
        self._pending_ordinal_ridge_state = state
        self._pending_ordinal_ridge_assignment = assignment
        self._pending_ordinal_ridge_mixture = (
            None if mixture is None else _copy_gmm(mixture)
        )
        self._pending_rms_ridge_expert_labels = base_labels.copy()
        self._pending_rms_ridge_expert_base_margins = base_margin.copy()
        pending_state = dict(self._pending)
        pending_state.pop("scores")
        pending_state.update(
            {
                "prediction_rms_ridge_expert_routed": context is not None,
                "prediction_rms_ridge_expert_ready": ready,
                "prediction_rms_ridge_expert_applied": ready,
                "prediction_rms_ridge_expert_base_rms": base_rms,
                "prediction_rms_ridge_expert_ridge_rms": ridge_rms,
                "prediction_rms_ridge_expert_ridge_margin_mean": float(
                    np.mean(ridge_margin)
                ),
                "prediction_rms_ridge_expert_ridge_margin_abs_mean": float(
                    np.mean(np.abs(ridge_margin))
                ),
                "prediction_rms_ridge_expert_ridge_margin_max_abs": float(
                    np.max(np.abs(ridge_margin))
                ),
                "prediction_rms_ridge_expert_label_changes": label_changes,
                "prediction_rms_ridge_expert_real_to_fake": real_to_fake,
                "prediction_rms_ridge_expert_fake_to_real": fake_to_real,
            }
        )
        return self._prediction_batch(scores, probability, **pending_state)

    def adapt(self, images: Any) -> AdaptationStats:
        if self._pending is None:
            return ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.adapt(self, images)
        scores = np.asarray(self._pending["scores"], dtype=np.float64).reshape(-1)
        prediction_state = dict(self._pending)
        features = self._pending_ordinal_ridge_features
        state = self._pending_ordinal_ridge_state
        assignment = self._pending_ordinal_ridge_assignment
        mixture = self._pending_ordinal_ridge_mixture
        routed_labels = self._pending_rms_ridge_expert_labels
        base_margins = self._pending_rms_ridge_expert_base_margins
        self._pending_ordinal_ridge_features = None
        self._pending_ordinal_ridge_state = None
        self._pending_ordinal_ridge_assignment = None
        self._pending_ordinal_ridge_mixture = None
        self._pending_rms_ridge_expert_labels = None
        self._pending_rms_ridge_expert_base_margins = None

        updated = False
        if self.adaptation_mode == "full" and assignment is not None:
            if features is None or int(features.shape[0]) != int(scores.size):
                raise RuntimeError(
                    "ASCAL RMS Ridge expert lost its prediction features"
                )
            if mixture is None:
                raise RuntimeError(
                    "ASCAL RMS Ridge expert lost its prediction-time selected GMM"
                )
            if routed_labels is None or int(routed_labels.size) != int(scores.size):
                raise RuntimeError("ASCAL RMS Ridge expert lost its R12 decisions")
            if base_margins is None or int(base_margins.size) != int(scores.size):
                raise RuntimeError("ASCAL RMS Ridge expert lost its R12 margins")
            if state is None:
                state = self._ensure_ordinal_ridge_state(assignment)
            updated = self._update_rms_ridge_expert_state(
                state,
                mixture,
                scores,
                routed_labels,
                features,
                base_margins,
            )

        stats = ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.adapt(self, images)
        routed = bool(prediction_state.get("prediction_rms_ridge_expert_routed"))
        applied = bool(prediction_state.get("prediction_rms_ridge_expert_applied"))
        if routed:
            self.rms_ridge_expert_routed_batches += 1
            self.rms_ridge_expert_routed_samples += int(scores.size)
        if applied:
            self.rms_ridge_expert_applied_batches += 1
            self.rms_ridge_expert_applied_samples += int(scores.size)
        elif routed:
            self.rms_ridge_expert_cold_start_batches += 1
        self.rms_ridge_expert_label_changes += int(
            prediction_state.get("prediction_rms_ridge_expert_label_changes", 0)
            or 0
        )
        self.rms_ridge_expert_real_to_fake += int(
            prediction_state.get("prediction_rms_ridge_expert_real_to_fake", 0)
            or 0
        )
        self.rms_ridge_expert_fake_to_real += int(
            prediction_state.get("prediction_rms_ridge_expert_fake_to_real", 0)
            or 0
        )
        stats.extra.update(
            {
                **self._rms_ridge_expert_state_stats(),
                "rms_ridge_expert_updated": updated,
            }
        )
        return stats

    def discard_pending_prediction(self) -> None:
        self._ordinal_ridge_precomputed_scores = None
        self._pending_ordinal_ridge_features = None
        self._pending_ordinal_ridge_state = None
        self._pending_ordinal_ridge_assignment = None
        self._pending_ordinal_ridge_mixture = None
        self._pending_rms_ridge_expert_labels = None
        self._pending_rms_ridge_expert_base_margins = None
        ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.discard_pending_prediction(self)


class ASCALGMMSegmentedMemoryPosteriorEqualPriorRidgeExpert(
    ASCALGMMSegmentedMemoryPosteriorRMSRidgeExpert
):
    """Center each R19 expert at its equal-prior class-centroid midpoint."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "r19_direct_binary_ridge_with_parameter_free_equal_prior_"
                    "expert_readout"
                ),
                "research_name": "ASCAL-JMP-EqualPriorRidge",
                "research_version": "R20",
                "r19_protected_scope": (
                    "r12_routing_gmm_segmentation_memory_handoff_pseudo_labels_"
                    "reliability_rls_updates_and_unit_rms_fusion"
                ),
                "diagnosed_failure": (
                    "reliability_weighted_r12_pseudo_class_mass_can_encode_a_"
                    "spurious_target_class_prior_in_the_ridge_bias"
                ),
                "equal_prior_statistics": (
                    "two_historical_reliability_weighted_ridge_margin_class_"
                    "centroids_derived_from_existing_cross_covariance_and_class_mass"
                ),
                "equal_prior_center": (
                    "one_half_times_real_margin_centroid_plus_fake_margin_centroid"
                ),
                "centered_ridge_margin": (
                    "raw_ridge_margin_minus_historical_equal_prior_center"
                ),
                "scale_alignment": (
                    "per_expert_historical_reliability_weighted_rms_for_r12_"
                    "logit_and_equal_prior_centered_ridge_margin"
                ),
                "centered_ridge_energy": (
                    "exact_sufficient_statistic_identity_without_replaying_samples"
                ),
                "prediction_rule": (
                    "sigmoid_of_r12_logit_over_past_r12_rms_plus_equal_prior_"
                    "centered_ridge_margin_over_its_past_centered_rms"
                ),
                "class_prior_assumption": (
                    "equal_real_fake_prior_matching_balanced_binary_evaluation"
                ),
                "new_persistent_state": "none_reuse_r19_sufficient_statistics",
                "new_target_hyperparameters": 0,
                "intentional_changes": [
                    "all R19 routing learning and expert ownership remain unchanged",
                    "R19 direct one-hot Ridge weights remain byte-for-byte unchanged",
                    (
                        "each old expert removes only the midpoint of its two "
                        "historical pseudo-class margin centroids"
                    ),
                    (
                        "the centered Ridge RMS is derived exactly from existing "
                        "sufficient statistics"
                    ),
                    "cold or invalid experts still return the exact R12 probability",
                    "no target sample label image feature or new tuning knob is retained",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_equal_prior_ridge_expert"

    def _equal_prior_ridge_moments(
        self,
        state: dict[str, Any] | None,
    ) -> tuple[float, float, float] | None:
        if state is None:
            return None
        support = float(state["effective_support"])
        class_mass = np.asarray(state["class_mass"], dtype=np.float64).reshape(-1)
        weights = np.asarray(state["weights"], dtype=np.float64)
        cross_covariance = np.asarray(
            state["cross_covariance"],
            dtype=np.float64,
        )
        if (
            not math.isfinite(support)
            or support <= np.finfo(np.float64).eps
            or class_mass.shape != (2,)
            or np.any(class_mass <= np.finfo(np.float64).eps)
            or not np.all(np.isfinite(class_mass))
            or weights.shape != (self.ordinal_ridge_feature_dim, 2)
            or cross_covariance.shape != weights.shape
            or not np.all(np.isfinite(weights))
            or not np.all(np.isfinite(cross_covariance))
        ):
            return None

        direction = weights[:, 1] - weights[:, 0]
        real_feature_sum = cross_covariance[:, 0]
        fake_feature_sum = cross_covariance[:, 1]
        real_centroid = float(direction @ real_feature_sum / class_mass[0])
        fake_centroid = float(direction @ fake_feature_sum / class_mass[1])
        center = 0.5 * (real_centroid + fake_centroid)
        separation = fake_centroid - real_centroid

        cross_direction = fake_feature_sum - real_feature_sum
        raw_energy = float(direction @ cross_direction - direction @ direction)
        total_feature_sum = real_feature_sum + fake_feature_sum
        centered_energy = float(
            raw_energy
            - 2.0 * center * float(direction @ total_feature_sum)
            + center**2 * support
        )
        centered_energy = max(0.0, centered_energy)
        if not all(
            math.isfinite(value)
            for value in (
                real_centroid,
                fake_centroid,
                center,
                separation,
                centered_energy,
            )
        ):
            return None
        return center, separation, centered_energy

    def _rms_ridge_expert_scales(
        self,
        state: dict[str, Any] | None,
    ) -> tuple[float, float, float] | None:
        if state is None:
            return None
        support = float(state["effective_support"])
        base_energy = float(state["base_margin_square_sum"])
        moments = self._equal_prior_ridge_moments(state)
        if moments is None:
            return None
        _, _, centered_energy = moments
        if not (
            math.isfinite(base_energy)
            and base_energy > 0.0
            and centered_energy > 0.0
        ):
            return None
        base_rms = math.sqrt(base_energy / support)
        ridge_rms = math.sqrt(centered_energy / support)
        if not (
            math.isfinite(base_rms)
            and math.isfinite(ridge_rms)
            and base_rms > np.finfo(np.float64).eps
            and ridge_rms > np.finfo(np.float64).eps
        ):
            return None
        return base_rms, ridge_rms, centered_energy

    def _rms_ridge_expert_probability(
        self,
        base_probability: np.ndarray,
        features: np.ndarray,
        state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        scales = self._rms_ridge_expert_scales(state)
        moments = self._equal_prior_ridge_moments(state)
        if scales is None or moments is None:
            raise RuntimeError(
                "ASCAL equal-prior Ridge expert has no valid historical scale"
            )
        base_rms, ridge_rms, _ = scales
        center, _, _ = moments
        base_probability = np.asarray(base_probability, dtype=np.float64).reshape(-1)
        epsilon = np.finfo(np.float64).eps
        bounded_probability = np.clip(base_probability, epsilon, 1.0 - epsilon)
        base_margin = np.log(bounded_probability / (1.0 - bounded_probability))
        weights = np.asarray(state["weights"], dtype=np.float64)
        direction = weights[:, 1] - weights[:, 0]
        ridge_margin = np.asarray(features @ direction - center, dtype=np.float64)
        fused_margin = base_margin / base_rms + ridge_margin / ridge_rms
        probability = self._stable_sigmoid(fused_margin)
        if not (
            np.all(np.isfinite(base_margin))
            and np.all(np.isfinite(ridge_margin))
            and np.all(np.isfinite(probability))
        ):
            raise FloatingPointError(
                "ASCAL equal-prior Ridge expert produced a non-finite prediction"
            )
        return probability, base_margin, ridge_margin, base_rms, ridge_rms

    def _rms_ridge_expert_state_stats(self) -> dict[str, Any]:
        stats = super()._rms_ridge_expert_state_stats()
        moments = [
            self._equal_prior_ridge_moments(state)
            for state in self._all_ordinal_ridge_states()
            if self._ordinal_ridge_ready(state)
        ]
        valid = [moment for moment in moments if moment is not None]
        centers = [moment[0] for moment in valid]
        separations = [moment[1] for moment in valid]
        stats.update(
            {
                "equal_prior_ridge_ready_experts": len(valid),
                "equal_prior_ridge_mean_center": (
                    float(np.mean(centers)) if centers else 0.0
                ),
                "equal_prior_ridge_mean_abs_center": (
                    float(np.mean(np.abs(centers))) if centers else 0.0
                ),
                "equal_prior_ridge_mean_class_separation": (
                    float(np.mean(separations)) if separations else 0.0
                ),
                "equal_prior_ridge_min_class_separation": (
                    min(separations, default=0.0)
                ),
            }
        )
        return stats

    def predict(self, images: Any) -> PredictionBatch:
        prediction = super().predict(images)
        if self._pending is None:
            raise RuntimeError("ASCAL equal-prior Ridge lost its pending state")
        center = 0.0
        separation = 0.0
        state = self._pending_ordinal_ridge_state
        if bool(self._pending.get("prediction_rms_ridge_expert_ready")):
            moments = self._equal_prior_ridge_moments(state)
            if moments is None:
                raise RuntimeError(
                    "ASCAL equal-prior Ridge lost its prediction moments"
                )
            center, separation, _ = moments
        self._pending.update(
            {
                "prediction_equal_prior_ridge_applied": bool(
                    self._pending.get("prediction_rms_ridge_expert_applied")
                ),
                "prediction_equal_prior_ridge_center": center,
                "prediction_equal_prior_ridge_class_separation": separation,
            }
        )
        return prediction


class ASCALGMMSegmentedMemoryPosteriorEvidenceGatedRidgeExpert(
    ASCALGMMSegmentedMemoryPosteriorEqualPriorRidgeExpert
):
    """Let an R20 expert replace R12 only where its inverse Gram has evidence."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "r20_equal_prior_direct_ridge_with_parameter_free_"
                    "sample_level_inverse_gram_evidence_gating"
                ),
                "research_name": "ASCAL-JMP-EvidenceGatedRidge",
                "research_version": "R21",
                "r20_protected_scope": (
                    "routing_gmm_segmentation_memory_handoff_pseudo_labels_"
                    "reliability_rls_state_equal_prior_center_and_scale"
                ),
                "diagnosed_failure": (
                    "r20_forces_a_new_direct_classifier_into_every_ready_"
                    "expert_prediction_without_sample_direction_evidence"
                ),
                "evidence_statistic": (
                    "one_minus_posterior_to_prior_feature_direction_variance_"
                    "ratio_from_the_prediction_time_inverse_regularized_gram"
                ),
                "evidence_query": (
                    "l2_normalized_clip_feature_with_constant_bias_coordinate_"
                    "zeroed_to_exclude_intercept_only_evidence"
                ),
                "evidence_range": "analytic_unit_interval",
                "ridge_scale_alignment": (
                    "equal_prior_centered_ridge_margin_times_r12_rms_over_"
                    "centered_ridge_rms"
                ),
                "prediction_rule": (
                    "convex_interpolation_in_r12_logit_units_from_r12_to_"
                    "aligned_direct_ridge_using_sample_level_evidence"
                ),
                "zero_evidence_rule": "return_exact_r12_probability",
                "full_evidence_rule": "use_the_aligned_direct_ridge_classifier",
                "gate_timing": (
                    "prediction_time_old_expert_inverse_gram_before_current_"
                    "batch_update"
                ),
                "new_persistent_state": "none_reuse_r20_inverse_gram",
                "new_target_hyperparameters": 0,
                "intentional_changes": [
                    "all R20 routing supervision and analytic updates remain unchanged",
                    "R20 equal-prior center and historical RMS remain unchanged",
                    (
                        "the old inverse Gram supplies one parameter-free evidence "
                        "value for each current sample direction"
                    ),
                    (
                        "the direct Ridge replaces rather than duplicates R12 as "
                        "its evidence grows"
                    ),
                    "the constant bias cannot by itself grant sample-level trust",
                    "no target sample label feature or additional knob is retained",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_evidence_gated_ridge_expert"

    def _inverse_gram_feature_evidence(
        self,
        features: np.ndarray,
        state: dict[str, Any],
    ) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        inverse_gram = np.asarray(state["inverse_gram"], dtype=np.float64)
        if (
            features.ndim != 2
            or int(features.shape[1]) != self.ordinal_ridge_feature_dim
            or inverse_gram.shape
            != (self.ordinal_ridge_feature_dim, self.ordinal_ridge_feature_dim)
            or not np.all(np.isfinite(features))
            or not np.all(np.isfinite(inverse_gram))
        ):
            raise RuntimeError(
                "ASCAL evidence-gated Ridge received an invalid inverse Gram query"
            )

        feature_query = features.copy()
        feature_query[:, -1] = 0.0
        prior_variance = np.einsum(
            "ij,ij->i",
            feature_query,
            feature_query,
        )
        posterior_projection = feature_query @ inverse_gram
        variance_reduction = np.einsum(
            "ij,ij->i",
            feature_query,
            feature_query - posterior_projection,
        )
        evidence = np.divide(
            variance_reduction,
            prior_variance,
            out=np.zeros_like(prior_variance),
            where=prior_variance > np.finfo(np.float64).eps,
        )
        evidence = np.clip(evidence, 0.0, 1.0)
        if not np.all(np.isfinite(evidence)):
            raise FloatingPointError(
                "ASCAL evidence-gated Ridge produced non-finite evidence"
            )
        return evidence

    def _rms_ridge_expert_probability(
        self,
        base_probability: np.ndarray,
        features: np.ndarray,
        state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        (
            _,
            base_margin,
            ridge_margin,
            base_rms,
            ridge_rms,
        ) = super()._rms_ridge_expert_probability(
            base_probability,
            features,
            state,
        )
        evidence = self._inverse_gram_feature_evidence(features, state)
        aligned_ridge_margin = ridge_margin * (base_rms / ridge_rms)
        fused_margin = base_margin + evidence * (
            aligned_ridge_margin - base_margin
        )
        probability = self._stable_sigmoid(fused_margin)
        base_probability = np.asarray(base_probability, dtype=np.float64).reshape(-1)
        probability = np.where(evidence == 0.0, base_probability, probability)
        if not (
            np.all(np.isfinite(aligned_ridge_margin))
            and np.all(np.isfinite(probability))
        ):
            raise FloatingPointError(
                "ASCAL evidence-gated Ridge produced a non-finite prediction"
            )
        return probability, base_margin, ridge_margin, base_rms, ridge_rms

    def predict(self, images: Any) -> PredictionBatch:
        prediction = super().predict(images)
        if self._pending is None:
            raise RuntimeError("ASCAL evidence-gated Ridge lost its pending state")
        applied = bool(self._pending.get("prediction_rms_ridge_expert_applied"))
        evidence = np.zeros(0, dtype=np.float64)
        if applied:
            features = self._pending_ordinal_ridge_features
            state = self._pending_ordinal_ridge_state
            if features is None or state is None:
                raise RuntimeError(
                    "ASCAL evidence-gated Ridge lost its prediction-time expert"
                )
            evidence = self._inverse_gram_feature_evidence(features, state)
        self._pending.update(
            {
                "prediction_evidence_gated_ridge_applied": applied,
                "prediction_evidence_gated_ridge_mean": (
                    float(np.mean(evidence)) if evidence.size else 0.0
                ),
                "prediction_evidence_gated_ridge_min": (
                    float(np.min(evidence)) if evidence.size else 0.0
                ),
                "prediction_evidence_gated_ridge_max": (
                    float(np.max(evidence)) if evidence.size else 0.0
                ),
            }
        )
        return prediction


class ASCALGMMSegmentedMemoryPosteriorFeatureRoutedTrustedRidge(
    ASCALGMMSegmentedMemoryPosteriorRMSRidgeExpert
):
    """Route by frozen CLIP features and train direct Ridge with GMM trust."""

    def _reset_state(self) -> None:
        self._feature_route_query: np.ndarray | None = None
        super()._reset_state()

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_feature_routed_per_expert_gmm_trusted_"
                    "direct_binary_analytic_ridge"
                ),
                "research_name": "ASCAL-JMP-FeatureRoutedTrustedRidge",
                "research_version": "R22",
                "routing_coordinate": (
                    "l2_normalized_frozen_clip_feature_orthogonal_to_"
                    "source_binary_head"
                ),
                "routing_class_direction": (
                    "remove_the_frozen_source_binary_head_direction_before_"
                    "normalization"
                ),
                "routing_rule": (
                    "maximum_mean_cosine_similarity_to_historical_expert_"
                    "prototype_with_active_tie_priority"
                ),
                "routing_granularity": "one_current_unlabeled_stream_batch",
                "routing_score_used": False,
                "routing_admission": (
                    "feature_winner_is_admitted_without_source_score_mdl"
                ),
                "gmm_role": (
                    "prediction_time_old_expert_equal_prior_pseudo_label_"
                    "and_continuous_reliability_only"
                ),
                "gmm_in_final_prediction": False,
                "pseudo_label": "selected_gmm_equal_prior_posterior_argmax",
                "reliability_rule": (
                    "absolute_selected_gmm_signed_posterior_without_threshold"
                ),
                "ridge_target": "real_or_fake_one_hot_vector",
                "ridge_update": (
                    "exact_recursive_least_squares_woodbury_after_prediction"
                ),
                "prediction_rule": (
                    "equal_average_of_frozen_source_probability_and_selected_"
                    "direct_ridge_softmax_probability"
                ),
                "cold_start_rule": (
                    "zero_ridge_is_neutral_probability_one_half_and_preserves_"
                    "source_accuracy_and_auc"
                ),
                "gmm_update_score_coordinate": "immutable_frozen_source_score",
                "ridge_output_updates_gmm": False,
                "prediction_mutates_experts": False,
                "raw_images_stored": False,
                "raw_features_stored": False,
                "target_labels_used": False,
                "generator_boundaries_used": False,
                "semantic_features_used": True,
                "new_target_hyperparameters": 0,
                "intentional_changes": [
                    (
                        "historical expert selection uses frozen CLIP features "
                        "orthogonal to the source real-fake direction"
                    ),
                    "the selected old GMM supplies pseudo-classes and reliability only",
                    "Ridge learns direct one-hot binary targets rather than a GMM score",
                    "the current final score contains only frozen Base and direct Ridge",
                    "the selected expert receives the batch only after prediction",
                    "no image or per-sample feature survives its analytic update",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_feature_routed_gmm_trusted_ridge"

    def _new_ordinal_ridge_state(self) -> dict[str, Any]:
        state = super()._new_ordinal_ridge_state()
        weight_rows = int(np.asarray(state["weights"]).shape[0])
        state.update(
            {
                "route_feature_sum": np.zeros(
                    max(1, weight_rows - 1),
                    dtype=np.float64,
                ),
                "route_feature_mass": 0.0,
            }
        )
        return state

    def _batch_scores_and_rms_ridge_expert_features(
        self,
        images: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        scores, features = super()._batch_scores_and_rms_ridge_expert_features(
            images
        )
        self._feature_route_query = self._feature_route_coordinates(
            features[:, :-1]
        )
        return scores, features

    def _feature_route_coordinates(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != (
            self.rms_ridge_expert_backbone_feature_dim
        ):
            raise ValueError("ASCAL feature route received invalid CLIP features")
        direction = np.asarray(
            self._ordinal_ridge_source_direction,
            dtype=np.float64,
        ).reshape(-1)
        projected = values - (values @ direction)[:, None] * direction[None, :]
        norms = np.linalg.norm(projected, axis=1, keepdims=True)
        return np.divide(
            projected,
            norms,
            out=np.zeros_like(projected),
            where=norms > np.finfo(np.float64).eps,
        )

    def _route_prototype(
        self,
        state: dict[str, Any] | None,
    ) -> np.ndarray | None:
        if state is None:
            return None
        feature_sum = np.asarray(
            state.get("route_feature_sum", []),
            dtype=np.float64,
        ).reshape(-1)
        mass = float(state.get("route_feature_mass", 0.0))
        if (
            feature_sum.size != self.rms_ridge_expert_backbone_feature_dim
            or not np.all(np.isfinite(feature_sum))
            or not math.isfinite(mass)
            or mass <= np.finfo(np.float64).eps
        ):
            return None
        norm = float(np.linalg.norm(feature_sum))
        if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
            return None
        return feature_sum / norm

    def _routing_candidates(self, scores: np.ndarray) -> list[dict[str, Any]]:
        candidates = super()._routing_candidates(scores)
        query = self._feature_route_query
        if query is None or query.shape != (
            int(np.asarray(scores).size),
            self.rms_ridge_expert_backbone_feature_dim,
        ):
            raise RuntimeError("ASCAL feature route lost the current CLIP features")

        routed_candidates: list[dict[str, Any]] = []
        for candidate in candidates:
            memory_index = candidate["memory_index"]
            if candidate["expert"] == "active_learning_state":
                assignment = (
                    "novel" if self.active_memory_index is None else "episodic_memory",
                    self.active_memory_index,
                )
            else:
                assignment = ("episodic_memory", int(memory_index))
            state = self._peek_ordinal_ridge_state(assignment)
            prototype = self._route_prototype(state)
            if prototype is None:
                if candidate["expert"] != "active_learning_state":
                    continue
                similarity = 1.0
            else:
                similarity = float(np.mean(query @ prototype))
            if not math.isfinite(similarity):
                continue
            routed = dict(candidate)
            routed["source_score_deviance"] = float(candidate["deviance"])
            routed["feature_similarity"] = similarity
            routed["deviance"] = float(scores.size) * (1.0 - similarity)
            routed_candidates.append(routed)
        return routed_candidates

    def _memory_admission_evidence(
        self,
        scores: np.ndarray,
        memory_index: int,
    ) -> dict[str, Any]:
        del scores
        evidence = self._empty_admission("feature_similarity_winner")
        evidence.update(
            {
                "checked": True,
                "accepted": True,
                "memory_index": int(memory_index),
            }
        )
        return evidence

    def _rms_ridge_expert_supervision(
        self,
        mixture: dict[str, Any],
        scores: np.ndarray,
        routed_labels: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        del routed_labels
        posterior = np.asarray(
            joint_density_fake_posterior(scores, mixture),
            dtype=np.float64,
        ).reshape(-1)
        labels = (posterior >= 0.5).astype(np.int64)
        reliability = np.abs(2.0 * posterior - 1.0)
        targets = np.zeros((labels.size, 2), dtype=np.float64)
        targets[np.arange(labels.size), labels] = 1.0
        if not (
            np.all(np.isfinite(posterior))
            and np.all(np.isfinite(reliability))
        ):
            raise FloatingPointError(
                "ASCAL feature-routed trusted Ridge produced non-finite supervision"
            )
        class_mass = np.bincount(labels, weights=reliability, minlength=2)
        self.rms_ridge_expert_last_effective_support = float(reliability.sum())
        self.rms_ridge_expert_last_reliability = float(np.mean(reliability))
        self.rms_ridge_expert_last_real_mass = float(class_mass[0])
        self.rms_ridge_expert_last_fake_mass = float(class_mass[1])
        self.rms_ridge_expert_last_posterior_conflicts = 0
        return targets, reliability, posterior, 0

    def _update_rms_ridge_expert_state(
        self,
        state: dict[str, Any],
        mixture: dict[str, Any],
        scores: np.ndarray,
        routed_labels: np.ndarray,
        features: np.ndarray,
        base_margins: np.ndarray,
    ) -> bool:
        route_features = self._feature_route_coordinates(
            np.asarray(features, dtype=np.float64)[:, :-1]
        )
        state["route_feature_sum"] = np.asarray(
            state["route_feature_sum"],
            dtype=np.float64,
        ) + np.sum(route_features, axis=0)
        state["route_feature_mass"] = float(state["route_feature_mass"]) + float(
            route_features.shape[0]
        )
        updated = super()._update_rms_ridge_expert_state(
            state,
            mixture,
            scores,
            routed_labels,
            features,
            base_margins,
        )
        return updated

    @staticmethod
    def _ordinal_ridge_ready(state: dict[str, Any] | None) -> bool:
        return state is not None

    def _rms_ridge_expert_probability(
        self,
        base_probability: np.ndarray,
        features: np.ndarray,
        state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        if self._pending is None:
            raise RuntimeError("ASCAL feature-routed trusted Ridge lost Base scores")
        scores = np.asarray(self._pending["scores"], dtype=np.float64).reshape(-1)
        base_probability = np.asarray(base_probability, dtype=np.float64).reshape(-1)
        if base_probability.shape != scores.shape:
            raise RuntimeError(
                "ASCAL feature-routed trusted Ridge received misaligned Base scores"
            )
        source_probability = np.asarray(
            self._source_probability(scores),
            dtype=np.float64,
        ).reshape(-1)
        epsilon = np.finfo(np.float64).eps
        bounded_source = np.clip(source_probability, epsilon, 1.0 - epsilon)
        base_margin = np.log(bounded_source / (1.0 - bounded_source))
        weights = np.asarray(state["weights"], dtype=np.float64)
        direction = weights[:, 1] - weights[:, 0]
        ridge_margin = np.asarray(features @ direction, dtype=np.float64)
        ridge_probability = self._stable_sigmoid(ridge_margin)
        probability = 0.5 * (source_probability + ridge_probability)
        if not (
            np.all(np.isfinite(base_margin))
            and np.all(np.isfinite(ridge_margin))
            and np.all(np.isfinite(probability))
        ):
            raise FloatingPointError(
                "ASCAL feature-routed trusted Ridge produced a non-finite prediction"
            )
        return probability, base_margin, ridge_margin, 1.0, 1.0

    def predict(self, images: Any) -> PredictionBatch:
        self._feature_route_query = None
        try:
            return super().predict(images)
        finally:
            self._feature_route_query = None


class ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidge(
    ASCALGMMSegmentedMemoryPosteriorFeatureRoutedTrustedRidge
):
    """Clone complete source Ridge statistics into every routed expert."""

    def _reset_state(self) -> None:
        classifier = getattr(self.model, "classifier", None)
        feature_dim = int(getattr(classifier, "in_features", 0))
        self._source_analytic_ridge = source_analytic_ridge_arrays(
            self.config.get("source_analytic_ridge"),
            expected_feature_dim=feature_dim,
        )
        if getattr(self.model, "classifier_feature_normalization", None) != "l2":
            raise ValueError(
                "ASCAL source-Ridge inheritance requires l2 classifier features"
            )
        super()._reset_state()

        weights = np.asarray(
            self._source_analytic_ridge["weights"], dtype=np.float64
        )
        model_weights = np.concatenate(
            (
                classifier.weight.detach().float().cpu().numpy().T,
                classifier.bias.detach().float().cpu().numpy()[None, :],
            ),
            axis=0,
        ).astype(np.float64)
        if not np.allclose(model_weights, weights, rtol=1e-5, atol=1e-7):
            raise ValueError(
                "ASCAL source-Ridge checkpoint head does not match its statistics"
            )

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_feature_routed_gmm_trusted_source_initialized_"
                    "analytic_ridge_experts"
                ),
                "research_name": "ASCAL-JMP-SourceRidgeInheritance",
                "research_version": "R23",
                "source_classifier": (
                    "two_output_ridge_fit_on_labeled_source_l2_clip_lora_"
                    "features_plus_bias"
                ),
                "source_state_inheritance": (
                    "every_new_expert_clones_source_inverse_gram_cross_"
                    "covariance_weights_class_mass_and_support"
                ),
                "expert_update": (
                    "gmm_reliability_weighted_target_pseudo_samples_are_added_"
                    "to_the_complete_source_sufficient_statistics"
                ),
                "prediction_rule": (
                    "selected_expert_ridge_margin_with_frozen_source_temperature_only"
                ),
                "base_probability_in_final_prediction": False,
                "cold_start_rule": "exact_source_ridge_classifier",
                "feature_coordinate": (
                    "same_l2_normalized_clip_lora_feature_plus_bias_for_source_"
                    "and_every_expert"
                ),
                "source_ridge_profile": self._source_analytic_ridge["profile"],
                "source_ridge_statistics_sha256": self._source_analytic_ridge[
                    "statistics_sha256"
                ],
                "source_ridge_samples": self._source_analytic_ridge["samples"],
                "new_target_hyperparameters": 0,
                "intentional_changes": [
                    "the deployed source classifier is itself an analytic Ridge",
                    "every expert starts from the complete labeled-source statistics",
                    "source and target updates share exactly one feature coordinate",
                    "the selected expert Ridge is the only final classifier",
                    "the immutable source-Ridge score remains the GMM coordinate",
                    "no target label image or per-sample feature is retained",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_feature_routed_source_ridge_inheritance"

    def _new_ordinal_ridge_state(self) -> dict[str, Any]:
        state = super()._new_ordinal_ridge_state()
        source = getattr(self, "_source_analytic_ridge", None)
        if source is None:
            return state
        weights = np.asarray(state["weights"], dtype=np.float64)
        if weights.shape != np.asarray(source["weights"]).shape:
            return state
        source_class_mass = np.asarray(source["class_mass"], dtype=np.float64)
        state.update(
            {
                "inverse_gram": np.asarray(
                    source["inverse_gram"], dtype=np.float64
                ).copy(),
                "cross_covariance": np.asarray(
                    source["cross_covariance"], dtype=np.float64
                ).copy(),
                "weights": np.asarray(source["weights"], dtype=np.float64).copy(),
                "updates": 0,
                "candidate_samples": 0,
                "effective_support": float(source["samples"]),
                "class_mass": source_class_mass.copy(),
                "base_margin_square_sum": 0.0,
                "posterior_conflicts": 0,
                "source_prior_samples": int(source["samples"]),
                "source_prior_class_mass": source_class_mass.copy(),
                "target_effective_support": 0.0,
                "target_class_mass": np.zeros(2, dtype=np.float64),
            }
        )
        return state

    def _update_rms_ridge_expert_state(
        self,
        state: dict[str, Any],
        mixture: dict[str, Any],
        scores: np.ndarray,
        routed_labels: np.ndarray,
        features: np.ndarray,
        base_margins: np.ndarray,
    ) -> bool:
        updated = super()._update_rms_ridge_expert_state(
            state,
            mixture,
            scores,
            routed_labels,
            features,
            base_margins,
        )
        if updated:
            state["target_effective_support"] = float(
                state["target_effective_support"]
            ) + float(self.rms_ridge_expert_last_effective_support)
            state["target_class_mass"] = np.asarray(
                state["target_class_mass"], dtype=np.float64
            ) + np.asarray(
                [
                    self.rms_ridge_expert_last_real_mass,
                    self.rms_ridge_expert_last_fake_mass,
                ],
                dtype=np.float64,
            )
        return updated

    def _rms_ridge_expert_probability(
        self,
        base_probability: np.ndarray,
        features: np.ndarray,
        state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        del base_probability
        if self._pending is None:
            raise RuntimeError("ASCAL source-Ridge inheritance lost Base scores")
        scores = np.asarray(self._pending["scores"], dtype=np.float64).reshape(-1)
        source_probability = np.asarray(
            self._source_probability(scores), dtype=np.float64
        )
        epsilon = np.finfo(np.float64).eps
        bounded_source = np.clip(source_probability, epsilon, 1.0 - epsilon)
        base_margin = np.log(bounded_source / (1.0 - bounded_source))
        weights = np.asarray(state["weights"], dtype=np.float64)
        direction = weights[:, 1] - weights[:, 0]
        ridge_margin = np.asarray(features @ direction, dtype=np.float64)
        probability = self._stable_sigmoid(ridge_margin / self.temperature)
        if not (
            np.all(np.isfinite(base_margin))
            and np.all(np.isfinite(ridge_margin))
            and np.all(np.isfinite(probability))
        ):
            raise FloatingPointError(
                "ASCAL source-Ridge inheritance produced a non-finite prediction"
            )
        return probability, base_margin, ridge_margin, 1.0, 1.0

    def _rms_ridge_expert_state_stats(self) -> dict[str, Any]:
        stats = super()._rms_ridge_expert_state_stats()
        states = self._all_ordinal_ridge_states()
        stats.update(
            {
                "source_ridge_prior_samples": int(
                    self._source_analytic_ridge["samples"]
                ),
                "source_ridge_statistics_sha256": self._source_analytic_ridge[
                    "statistics_sha256"
                ],
                "source_ridge_inherited_experts": len(states),
                "source_ridge_target_effective_support": float(
                    sum(
                        float(state.get("target_effective_support", 0.0))
                        for state in states
                    )
                ),
            }
        )
        return stats


class ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidgeGMMReadout(
    ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceRidge
):
    """Diagnose the routed GMM decision while updating Ridge only in shadow."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_feature_routed_gmm_expert_with_r12_ordinal_"
                    "readout_and_shadow_source_initialized_ridge"
                ),
                "research_name": "ASCAL-JMP-SourceRidgeGMMReadout",
                "research_version": "R24",
                "diagnostic_question": (
                    "classification_ability_of_the_current_feature_routed_"
                    "expert_gmms_without_expert_ridge_logits"
                ),
                "protected_scope": (
                    "r23_source_checkpoint_feature_route_gmm_segmentation_"
                    "memory_handoff_pseudo_supervision_and_state_updates"
                ),
                "prediction_rule": (
                    "selected_expert_gmm_bayes_decision_with_r12_immutable_"
                    "source_ridge_probability_as_within_class_ordinal_rank"
                ),
                "gmm_in_final_prediction": True,
                "gmm_role": "selected_old_expert_binary_decision",
                "expert_ridge_in_final_prediction": False,
                "base_probability_in_final_prediction": True,
                "ridge_state_role": (
                    "shadow_update_only_to_preserve_the_exact_r23_expert_and_"
                    "feature_route_state_trajectory"
                ),
                "accuracy_invariance": (
                    "final_threshold_decision_equals_the_selected_gmm_routed_"
                    "decision_exactly"
                ),
                "within_decision_order": (
                    "immutable_source_ridge_probability_preserves_global_"
                    "sample_order_inside_each_gmm_decision_side"
                ),
                "new_target_hyperparameters": 0,
                "intentional_changes": [
                    "the R23 expert Ridge logit is excluded from final prediction",
                    "the selected historical GMM supplies the final binary decision",
                    (
                        "the immutable source-Ridge probability is used only for "
                        "R12 within-decision ordering"
                    ),
                    (
                        "shadow Ridge and route-prototype updates remain exact so "
                        "the R23 state trajectory is unchanged"
                    ),
                    "no target label threshold optimizer or new parameter is added",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_feature_routed_source_ridge_gmm_readout"

    def _rms_ridge_expert_probability(
        self,
        base_probability: np.ndarray,
        features: np.ndarray,
        state: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
        (
            _,
            base_margin,
            ridge_margin,
            base_rms,
            ridge_rms,
        ) = super()._rms_ridge_expert_probability(
            base_probability,
            features,
            state,
        )
        probability = np.asarray(base_probability, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(probability)):
            raise FloatingPointError(
                "ASCAL source-Ridge GMM readout produced a non-finite prediction"
            )
        return probability.copy(), base_margin, ridge_margin, base_rms, ridge_rms


class ASCALGMMSegmentedMemoryPosteriorFeatureRoutedGaussianReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorFeatureRoutedTrustedRidge
):
    """Learn one Base-anchored MLP from each expert's Gaussian feature replay."""

    _ORDINAL_RIDGE_MEMORY_KEY = "gaussian_replay_mlp_state"

    def _reset_state(self) -> None:
        super()._reset_state()
        classifier = getattr(self.model, "classifier", None)
        weight = getattr(classifier, "weight", None)
        if weight is None or weight.ndim != 2 or int(weight.shape[0]) != 2:
            raise TypeError(
                "ASCAL Gaussian replay MLP requires a two-class linear classifier"
            )
        bias = getattr(classifier, "bias", None)
        self.gaussian_replay_feature_dim = int(weight.shape[1])
        self.gaussian_replay_hidden_dim = int(
            self.config.get("feature_replay_hidden_dim", 64)
        )
        self.gaussian_replay_learning_rate = float(
            self.config.get("feature_replay_learning_rate", 1e-3)
        )
        self.gaussian_replay_variance_floor = float(
            self.config.get("feature_replay_variance_floor", 1e-6)
        )
        self.gaussian_replay_seed = int(
            self.config.get("feature_replay_seed", 0)
        )
        if self.gaussian_replay_hidden_dim < 1:
            raise ValueError("ASCAL Gaussian replay MLP hidden dimension must be positive")
        if not (
            math.isfinite(self.gaussian_replay_learning_rate)
            and self.gaussian_replay_learning_rate > 0.0
        ):
            raise ValueError("ASCAL Gaussian replay MLP learning rate must be positive")
        if not (
            math.isfinite(self.gaussian_replay_variance_floor)
            and self.gaussian_replay_variance_floor > 0.0
        ):
            raise ValueError("ASCAL Gaussian replay variance floor must be positive")

        self._gaussian_replay_source_direction = (
            weight[1].detach().float().cpu().numpy()
            - weight[0].detach().float().cpu().numpy()
        ).astype(np.float64)
        if bias is None:
            self._gaussian_replay_source_bias = 0.0
        else:
            self._gaussian_replay_source_bias = float(
                bias[1].detach().float().cpu().item()
                - bias[0].detach().float().cpu().item()
            )
        self._gaussian_replay_rng = np.random.default_rng(
            self.gaussian_replay_seed
        )
        self._feature_route_query = None
        self._pending_gaussian_replay_features: np.ndarray | None = None
        self._pending_gaussian_replay_state: dict[str, Any] | None = None
        self._pending_gaussian_replay_assignment: (
            tuple[str, int | None] | None
        ) = None
        self._pending_gaussian_replay_mixture: dict[str, Any] | None = None
        self.gaussian_replay_updates = 0
        self.gaussian_replay_optimizer_steps = 0
        self.gaussian_replay_candidate_samples = 0
        self.gaussian_replay_generated_samples = 0
        self.gaussian_replay_applied_batches = 0
        self.gaussian_replay_applied_samples = 0
        self.gaussian_replay_cold_start_batches = 0
        self.gaussian_replay_label_changes = 0
        self.gaussian_replay_last_loss = 0.0
        self.gaussian_replay_last_effective_support = 0.0
        self.gaussian_replay_last_reliability = 0.0

    @property
    def trainable_parameters(self) -> int:
        if self.adaptation_mode == "static":
            return 0
        total = 0
        seen_heads: set[int] = set()
        for state in self._all_ordinal_ridge_states():
            head_state = self._gaussian_replay_head_state(state)
            head = head_state.get("mlp_head")
            if head is not None and id(head) not in seen_heads:
                seen_heads.add(id(head))
                total += sum(parameter.numel() for parameter in head.parameters())
        return total

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        getter = ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.reproduction_metadata.fget
        if getter is None:
            raise RuntimeError(
                "ASCAL Gaussian replay MLP lost the R12 metadata getter"
            )
        metadata = getter(self)
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_feature_routed_per_expert_class_conditional_"
                    "gaussian_replay_mlp"
                ),
                "research_name": "ASCAL-JMP-GaussianReplayMLP",
                "research_version": "R25",
                "routing_coordinate": (
                    "l2_normalized_frozen_clip_feature_orthogonal_to_"
                    "source_binary_head"
                ),
                "routing_rule": (
                    "maximum_mean_cosine_similarity_to_historical_expert_"
                    "prototype_with_active_tie_priority"
                ),
                "routing_score_used": False,
                "gmm_role": (
                    "prediction_time_selected_old_expert_equal_prior_pseudo_"
                    "label_and_continuous_reliability_only"
                ),
                "gmm_in_final_prediction": False,
                "feature_memory": (
                    "per_expert_reliability_weighted_real_and_fake_diagonal_"
                    "gaussian_sufficient_statistics"
                ),
                "raw_features_stored": False,
                "feature_replay": (
                    "balanced_real_fake_sampling_from_each_selected_experts_"
                    "cumulative_feature_distributions"
                ),
                "feature_replay_samples_per_update": (
                    "the_current_stream_batch_size_rounded_up_to_an_even_number"
                ),
                "expert_head": (
                    "one_hidden_layer_gelu_mlp_with_zero_initialized_output_"
                    "per_expert"
                ),
                "expert_hidden_dim": self.gaussian_replay_hidden_dim,
                "prediction_rule": (
                    "sigmoid_of_frozen_source_logit_plus_selected_expert_mlp_"
                    "residual_logit"
                ),
                "cold_start_rule": "exact_frozen_source_probability",
                "training_objective": (
                    "balanced_binary_cross_entropy_on_source_logit_plus_mlp_"
                    "using_generated_pseudo_features"
                ),
                "optimizer": "adam_one_step_after_each_predicted_batch",
                "learning_rate": self.gaussian_replay_learning_rate,
                "epochs": "none",
                "confidence_threshold": "none_continuous_gmm_reliability",
                "memory_capacity": "none_fixed_size_sufficient_statistics",
                "fusion_weight": "none_additive_logit_residual",
                "prediction_mutates_experts": False,
                "target_labels_used": False,
                "generator_boundaries_used": False,
                "semantic_features_used": True,
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "the old neural LoRA source detector and source score stay frozen",
                    "the R22 CLIP feature router selects one historical expert",
                    "the selected old GMM supplies pseudo-labels and reliability only",
                    "arrived features are discarded after weighted Gaussian moments update",
                    "balanced pseudo-features remove the observed stream class prior",
                    "the zero-output expert head is exactly neutral at expert birth",
                    "one post-prediction optimizer step updates only the selected expert",
                    "the final score contains Base plus one expert MLP and no GMM fusion",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_feature_routed_gaussian_replay_mlp"

    def _new_ordinal_ridge_state(self) -> dict[str, Any]:
        classifier = getattr(self.model, "classifier", None)
        feature_dim = int(getattr(classifier, "in_features", 0))
        if feature_dim < 1:
            raise TypeError(
                "ASCAL Gaussian replay MLP requires a finite feature dimension"
            )
        state = {
            "candidate_samples": 0,
            "route_feature_sum": np.zeros(feature_dim, dtype=np.float64),
            "route_feature_mass": 0.0,
            "class_samples": np.zeros(2, dtype=np.int64),
            "class_mass": np.zeros(2, dtype=np.float64),
            "feature_mean": np.zeros((2, feature_dim), dtype=np.float64),
            "feature_m2": np.zeros((2, feature_dim), dtype=np.float64),
        }
        state.update(self._new_gaussian_replay_head_state())
        return state

    @staticmethod
    def _new_gaussian_replay_head_state() -> dict[str, Any]:
        return {
            "mlp_head": None,
            "mlp_optimizer": None,
            "head_updates": 0,
            "optimizer_steps": 0,
            "generated_samples": 0,
            "last_loss": 0.0,
        }

    def _gaussian_replay_head_state(
        self,
        distribution_state: dict[str, Any],
    ) -> dict[str, Any]:
        return distribution_state

    def _batch_scores_and_gaussian_replay_features(
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
                "ASCAL Gaussian replay MLP expects (B, C, H, W) or "
                "(B, V, C, H, W) images"
            )
        forward_features = getattr(self.model, "forward_features", None)
        forward_classifier_features = getattr(
            self.model, "forward_classifier_features", None
        )
        classifier = getattr(self.model, "classifier", None)
        if not callable(forward_features) or not callable(classifier):
            raise TypeError(
                "ASCAL Gaussian replay MLP requires forward_features and classifier"
            )
        with torch.no_grad():
            features = forward_features(flat.to(self.device, non_blocking=True))
            classifier_features = (
                forward_classifier_features(features)
                if callable(forward_classifier_features)
                else features
            )
            logits = classifier(classifier_features)
        scores = (
            binary_score(logits)
            .view(batch, views)
            .mean(dim=1)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        feature_values = (
            classifier_features.detach()
            .float()
            .view(batch, views, -1)
            .mean(dim=1)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        if feature_values.shape != (batch, self.gaussian_replay_feature_dim):
            raise ValueError(
                "ASCAL Gaussian replay feature dimension does not match the source head"
            )
        if not np.all(np.isfinite(feature_values)):
            raise FloatingPointError(
                "ASCAL Gaussian replay received non-finite CLIP features"
            )
        norms = np.linalg.norm(feature_values, axis=1, keepdims=True)
        normalized = np.divide(
            feature_values,
            norms,
            out=np.zeros_like(feature_values),
            where=norms > np.finfo(np.float64).eps,
        )
        self._feature_route_query = self._feature_route_coordinates(normalized)
        return scores, feature_values

    def _gaussian_replay_ready(self, state: dict[str, Any] | None) -> bool:
        if state is None:
            return False
        head_state = self._gaussian_replay_head_state(state)
        if int(head_state.get("head_updates", 0)) <= 0:
            return False
        class_samples = np.asarray(
            state.get("class_samples", []), dtype=np.int64
        ).reshape(-1)
        class_mass = np.asarray(
            state.get("class_mass", []), dtype=np.float64
        ).reshape(-1)
        return bool(
            class_samples.shape == (2,)
            and class_mass.shape == (2,)
            and np.all(class_samples >= 2)
            and np.all(np.isfinite(class_mass))
            and np.all(class_mass > np.finfo(np.float64).eps)
            and head_state.get("mlp_head") is not None
        )

    @staticmethod
    def _normalized_feature_values(features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return np.divide(
            values,
            norms,
            out=np.zeros_like(values),
            where=norms > np.finfo(np.float64).eps,
        )

    def _gaussian_replay_head_features(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        return self._normalized_feature_values(features)

    def _ensure_gaussian_replay_head(self, state: dict[str, Any]) -> Any:
        head_state = self._gaussian_replay_head_state(state)
        head = head_state.get("mlp_head")
        if head is not None:
            return head

        import torch

        head = self._new_gaussian_replay_head()
        optimizer = torch.optim.Adam(
            head.parameters(),
            lr=self.gaussian_replay_learning_rate,
        )
        head.eval()
        head_state["mlp_head"] = head
        head_state["mlp_optimizer"] = optimizer
        return head

    def _new_gaussian_replay_head(self) -> Any:
        import torch
        import torch.nn as nn

        head = nn.Sequential(
            nn.Linear(
                self.gaussian_replay_feature_dim,
                self.gaussian_replay_hidden_dim,
            ),
            nn.GELU(),
            nn.Linear(self.gaussian_replay_hidden_dim, 1),
        ).to(self.device)
        first = head[0]
        output = head[2]
        initialization = self._gaussian_replay_rng.normal(
            0.0,
            math.sqrt(2.0 / float(self.gaussian_replay_feature_dim)),
            size=(
                self.gaussian_replay_hidden_dim,
                self.gaussian_replay_feature_dim,
            ),
        )
        with torch.no_grad():
            first.weight.copy_(
                torch.from_numpy(initialization).to(
                    device=first.weight.device,
                    dtype=first.weight.dtype,
                )
            )
            first.bias.zero_()
            output.weight.zero_()
            output.bias.zero_()
        return head

    def _gaussian_replay_residual(
        self,
        state: dict[str, Any],
        features: np.ndarray,
    ) -> np.ndarray:
        import torch

        head_state = self._gaussian_replay_head_state(state)
        head = head_state.get("mlp_head")
        if head is None:
            return np.zeros(int(features.shape[0]), dtype=np.float64)
        normalized = self._gaussian_replay_head_features(features)
        with torch.no_grad():
            residual = head(
                torch.from_numpy(normalized).to(
                    device=self.device,
                    dtype=torch.float32,
                )
            ).reshape(-1)
        values = residual.detach().cpu().numpy().astype(np.float64)
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(
                "ASCAL Gaussian replay MLP produced a non-finite residual"
            )
        return values

    def _gaussian_replay_prediction_residual(
        self,
        state: dict[str, Any],
        features: np.ndarray,
        scores: np.ndarray,
        mixture: dict[str, Any],
    ) -> np.ndarray:
        del scores, mixture
        return self._gaussian_replay_residual(state, features)

    @staticmethod
    def _update_weighted_diagonal_gaussian(
        state: dict[str, Any],
        class_index: int,
        features: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        positive = np.asarray(weights, dtype=np.float64).reshape(-1) > (
            np.finfo(np.float64).eps
        )
        if not np.any(positive):
            return
        values = np.asarray(features, dtype=np.float64)[positive]
        selected_weights = np.asarray(weights, dtype=np.float64).reshape(-1)[
            positive
        ]
        batch_mass = float(selected_weights.sum())
        batch_mean = np.average(values, axis=0, weights=selected_weights)
        centered = values - batch_mean[None, :]
        batch_m2 = np.sum(
            selected_weights[:, None] * centered * centered,
            axis=0,
        )

        old_mass = float(state["class_mass"][class_index])
        old_mean = np.asarray(
            state["feature_mean"][class_index], dtype=np.float64
        )
        old_m2 = np.asarray(state["feature_m2"][class_index], dtype=np.float64)
        total_mass = old_mass + batch_mass
        delta = batch_mean - old_mean
        combined_mean = old_mean + delta * (batch_mass / total_mass)
        combined_m2 = (
            old_m2
            + batch_m2
            + delta * delta * (old_mass * batch_mass / total_mass)
        )
        state["class_samples"][class_index] = int(
            state["class_samples"][class_index]
        ) + int(values.shape[0])
        state["class_mass"][class_index] = total_mass
        state["feature_mean"][class_index] = combined_mean
        state["feature_m2"][class_index] = combined_m2

    def _sample_gaussian_replay(
        self,
        state: dict[str, Any],
        samples: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        class_counts = self._gaussian_replay_class_counts(state, samples)
        generated: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for class_index in (0, 1):
            class_samples = int(class_counts[class_index])
            if class_samples < 1:
                raise ValueError(
                    "ASCAL Gaussian replay requires both pseudo-classes"
                )
            mass = float(state["class_mass"][class_index])
            mean = np.asarray(
                state["feature_mean"][class_index], dtype=np.float64
            )
            variance = np.asarray(
                state["feature_m2"][class_index], dtype=np.float64
            ) / mass
            variance = np.maximum(
                variance,
                self.gaussian_replay_variance_floor,
            )
            noise = self._gaussian_replay_rng.standard_normal(
                (class_samples, self.gaussian_replay_feature_dim)
            )
            generated.append(mean[None, :] + noise * np.sqrt(variance)[None, :])
            labels.append(
                np.full(class_samples, class_index, dtype=np.float32)
            )
        features = np.concatenate(generated, axis=0)
        targets = np.concatenate(labels, axis=0)
        order = self._gaussian_replay_rng.permutation(features.shape[0])
        return features[order], targets[order]

    def _gaussian_replay_class_counts(
        self,
        state: dict[str, Any],
        samples: int,
    ) -> tuple[int, int]:
        del state
        per_class = max(1, int(math.ceil(samples / 2.0)))
        return per_class, per_class

    def _gaussian_replay_samples_per_update(
        self,
        stream_batch_size: int,
    ) -> int:
        return int(stream_batch_size)

    def _gaussian_replay_minibatch_size(
        self,
        stream_batch_size: int,
        generated_samples: int,
    ) -> int:
        del stream_batch_size
        return int(generated_samples)

    def _gaussian_replay_training_samples(
        self,
        state: dict[str, Any],
        requested_samples: int,
        observed_features: np.ndarray,
        observed_labels: np.ndarray,
        observed_reliability: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        del observed_features, observed_labels, observed_reliability
        return self._sample_gaussian_replay(state, requested_samples)

    def _gaussian_replay_minibatch_loss(
        self,
        head: Any,
        features: Any,
        source_margin: Any,
        labels: Any,
    ) -> Any:
        import torch.nn.functional as functional

        residual = head(features).reshape(-1)
        return functional.binary_cross_entropy_with_logits(
            source_margin + residual,
            labels,
        )

    def _train_gaussian_replay_head(
        self,
        state: dict[str, Any],
        samples: int,
        observed_features: np.ndarray,
        observed_labels: np.ndarray,
        observed_reliability: np.ndarray,
    ) -> float | None:
        import torch

        requested_samples = self._gaussian_replay_samples_per_update(samples)
        if requested_samples < 1:
            raise ValueError("ASCAL Gaussian replay requires positive replay samples")
        training_samples = self._gaussian_replay_training_samples(
            state,
            requested_samples,
            observed_features,
            observed_labels,
            observed_reliability,
        )
        if training_samples is None:
            return None
        synthetic, labels = training_samples
        normalized = self._gaussian_replay_head_features(synthetic)
        source_margin = (
            synthetic @ self._gaussian_replay_source_direction
            + self._gaussian_replay_source_bias
        ) / self.temperature
        head = self._ensure_gaussian_replay_head(state)
        head_state = self._gaussian_replay_head_state(state)
        optimizer = head_state.get("mlp_optimizer")
        if optimizer is None:
            raise RuntimeError("ASCAL Gaussian replay MLP lost its optimizer")

        feature_tensor = torch.from_numpy(normalized).to(
            device=self.device,
            dtype=torch.float32,
        )
        source_tensor = torch.from_numpy(source_margin).to(
            device=self.device,
            dtype=torch.float32,
        )
        label_tensor = torch.from_numpy(labels).to(
            device=self.device,
            dtype=torch.float32,
        )
        generated_samples = int(labels.size)
        minibatch_size = self._gaussian_replay_minibatch_size(
            int(samples),
            generated_samples,
        )
        if minibatch_size < 1:
            raise ValueError(
                "ASCAL Gaussian replay requires a positive replay minibatch"
            )
        minibatch_size = min(minibatch_size, generated_samples)
        weighted_loss = 0.0
        optimizer_steps = 0
        head.train()
        for start in range(0, generated_samples, minibatch_size):
            stop = min(start + minibatch_size, generated_samples)
            optimizer.zero_grad(set_to_none=True)
            loss = self._gaussian_replay_minibatch_loss(
                head,
                feature_tensor[start:stop],
                source_tensor[start:stop],
                label_tensor[start:stop],
            )
            if not bool(torch.isfinite(loss)):
                head.eval()
                raise FloatingPointError(
                    "ASCAL Gaussian replay MLP produced a non-finite loss"
                )
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().cpu().item()) * float(
                stop - start
            )
            optimizer_steps += 1
        head.eval()
        head_state["head_updates"] = int(head_state["head_updates"]) + 1
        head_state["optimizer_steps"] = int(
            head_state["optimizer_steps"]
        ) + optimizer_steps
        head_state["generated_samples"] = int(
            head_state["generated_samples"]
        ) + generated_samples
        head_state["last_loss"] = weighted_loss / float(generated_samples)
        self.gaussian_replay_updates += 1
        self.gaussian_replay_optimizer_steps += optimizer_steps
        self.gaussian_replay_generated_samples += generated_samples
        self.gaussian_replay_last_loss = float(head_state["last_loss"])
        return float(head_state["last_loss"])

    def _update_gaussian_replay_state(
        self,
        state: dict[str, Any],
        mixture: dict[str, Any],
        scores: np.ndarray,
        features: np.ndarray,
    ) -> bool:
        labels, reliability, posterior = self._gaussian_replay_supervision(
            mixture,
            scores,
        )
        if not (
            posterior.shape == (int(scores.size),)
            and features.shape
            == (int(scores.size), self.gaussian_replay_feature_dim)
            and np.all(np.isfinite(posterior))
            and np.all(np.isfinite(reliability))
        ):
            raise RuntimeError(
                "ASCAL Gaussian replay received invalid pseudo-supervision"
            )

        route_features = self._feature_route_coordinates(
            self._normalized_feature_values(features)
        )
        state["route_feature_sum"] = np.asarray(
            state["route_feature_sum"], dtype=np.float64
        ) + np.sum(route_features, axis=0)
        state["route_feature_mass"] = float(
            state["route_feature_mass"]
        ) + float(route_features.shape[0])
        state["candidate_samples"] = int(state["candidate_samples"]) + int(
            scores.size
        )
        self.gaussian_replay_candidate_samples += int(scores.size)
        self.gaussian_replay_last_effective_support = float(reliability.sum())
        self.gaussian_replay_last_reliability = float(np.mean(reliability))
        for class_index in (0, 1):
            class_weights = np.where(labels == class_index, reliability, 0.0)
            self._update_weighted_diagonal_gaussian(
                state,
                class_index,
                features,
                class_weights,
            )

        class_samples = np.asarray(state["class_samples"], dtype=np.int64)
        class_mass = np.asarray(state["class_mass"], dtype=np.float64)
        distribution_ready = bool(
            np.all(class_samples >= 2)
            and np.all(np.isfinite(class_mass))
            and np.all(class_mass > np.finfo(np.float64).eps)
        )
        if not distribution_ready:
            return False
        self._train_gaussian_replay_head(
            state,
            int(scores.size),
            features,
            labels,
            reliability,
        )
        return self._gaussian_replay_ready(state)

    def _gaussian_replay_supervision(
        self,
        mixture: dict[str, Any],
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        posterior = np.asarray(
            joint_density_fake_posterior(scores, mixture),
            dtype=np.float64,
        ).reshape(-1)
        labels = (posterior >= 0.5).astype(np.int64)
        reliability = np.abs(2.0 * posterior - 1.0)
        return labels, reliability, posterior

    def _gaussian_replay_state_stats(self) -> dict[str, Any]:
        states = self._all_ordinal_ridge_states()
        class_masses = [
            np.asarray(state["class_mass"], dtype=np.float64) for state in states
        ]
        class_samples = [
            np.asarray(state["class_samples"], dtype=np.int64) for state in states
        ]
        return {
            "gaussian_replay_expert_count": len(states),
            "gaussian_replay_ready_experts": sum(
                self._gaussian_replay_ready(state) for state in states
            ),
            "gaussian_replay_updates": self.gaussian_replay_updates,
            "gaussian_replay_optimizer_steps": (
                self.gaussian_replay_optimizer_steps
            ),
            "gaussian_replay_candidate_samples": (
                self.gaussian_replay_candidate_samples
            ),
            "gaussian_replay_generated_samples": (
                self.gaussian_replay_generated_samples
            ),
            "gaussian_replay_applied_batches": (
                self.gaussian_replay_applied_batches
            ),
            "gaussian_replay_applied_samples": (
                self.gaussian_replay_applied_samples
            ),
            "gaussian_replay_cold_start_batches": (
                self.gaussian_replay_cold_start_batches
            ),
            "gaussian_replay_label_changes": self.gaussian_replay_label_changes,
            "gaussian_replay_real_samples": int(
                sum(int(samples[0]) for samples in class_samples)
            ),
            "gaussian_replay_fake_samples": int(
                sum(int(samples[1]) for samples in class_samples)
            ),
            "gaussian_replay_real_mass": float(
                sum(float(mass[0]) for mass in class_masses)
            ),
            "gaussian_replay_fake_mass": float(
                sum(float(mass[1]) for mass in class_masses)
            ),
            "gaussian_replay_last_loss": self.gaussian_replay_last_loss,
            "gaussian_replay_last_effective_support": (
                self.gaussian_replay_last_effective_support
            ),
            "gaussian_replay_last_reliability": (
                self.gaussian_replay_last_reliability
            ),
            "gaussian_replay_trainable_parameters": self.trainable_parameters,
            "gaussian_replay_distribution_values": (
                len(states) * self.gaussian_replay_feature_dim * 4
            ),
        }

    def _state_stats(self) -> dict[str, Any]:
        stats = ASCALGMMSegmentedMemoryPosteriorOrdinalRoute._state_stats(self)
        stats.update(self._gaussian_replay_state_stats())
        return stats

    def predict(self, images: Any) -> PredictionBatch:
        scores, features = self._batch_scores_and_gaussian_replay_features(images)
        self._ordinal_ridge_precomputed_scores = scores
        try:
            ordinal = ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.predict(
                self,
                images,
            )
        finally:
            self._ordinal_ridge_precomputed_scores = None
            self._feature_route_query = None
        if self._pending is None:
            raise RuntimeError(
                "ASCAL Gaussian replay MLP lost the routed prediction state"
            )

        context = self._ordinal_ridge_context()
        assignment = None if context is None else context[0]
        mixture = None if context is None else context[1]
        state = (
            None
            if assignment is None
            else self._peek_ordinal_ridge_state(assignment)
        )
        ready = self._gaussian_replay_ready(state)
        source_probability = np.asarray(
            self._source_probability(scores), dtype=np.float64
        ).reshape(-1)
        source_margin = np.asarray(scores, dtype=np.float64).reshape(-1) / (
            self.temperature
        )
        residual = np.zeros_like(source_margin)
        if ready:
            if state is None:
                raise RuntimeError(
                    "ASCAL Gaussian replay MLP lost its selected expert state"
                )
            if mixture is None:
                raise RuntimeError(
                    "ASCAL Gaussian replay MLP lost its selected GMM"
                )
            residual = self._gaussian_replay_prediction_residual(
                state,
                features,
                scores,
                mixture,
            )
            probability = self._stable_sigmoid(source_margin + residual)
        else:
            probability = source_probability.copy()

        source_labels = (source_probability >= 0.5).astype(np.int64)
        final_labels = (probability >= 0.5).astype(np.int64)
        label_changes = int(np.count_nonzero(source_labels != final_labels))
        self._pending_gaussian_replay_features = features.copy()
        self._pending_gaussian_replay_state = state
        self._pending_gaussian_replay_assignment = assignment
        self._pending_gaussian_replay_mixture = (
            None if mixture is None else _copy_gmm(mixture)
        )
        pending_state = dict(self._pending)
        pending_state.pop("scores")
        pending_state.update(
            {
                "prediction_gaussian_replay_routed": context is not None,
                "prediction_gaussian_replay_ready": ready,
                "prediction_gaussian_replay_applied": ready,
                "prediction_gaussian_replay_residual_mean": float(
                    np.mean(residual)
                ),
                "prediction_gaussian_replay_residual_abs_mean": float(
                    np.mean(np.abs(residual))
                ),
                "prediction_gaussian_replay_residual_max_abs": float(
                    np.max(np.abs(residual))
                ),
                "prediction_gaussian_replay_label_changes": label_changes,
                "prediction_gaussian_replay_internal_r12_fake_count": int(
                    ordinal.pred_label.sum().item()
                ),
                "prediction_gaussian_replay_source_fake_count": int(
                    source_labels.sum()
                ),
                "prediction_gaussian_replay_final_fake_count": int(
                    final_labels.sum()
                ),
            }
        )
        return self._prediction_batch(scores, probability, **pending_state)

    def adapt(self, images: Any) -> AdaptationStats:
        if self._pending is None:
            return ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.adapt(self, images)
        scores = np.asarray(self._pending["scores"], dtype=np.float64).reshape(-1)
        prediction_state = dict(self._pending)
        features = self._pending_gaussian_replay_features
        state = self._pending_gaussian_replay_state
        assignment = self._pending_gaussian_replay_assignment
        mixture = self._pending_gaussian_replay_mixture
        self._pending_gaussian_replay_features = None
        self._pending_gaussian_replay_state = None
        self._pending_gaussian_replay_assignment = None
        self._pending_gaussian_replay_mixture = None

        updated = False
        if self.adaptation_mode == "full" and assignment is not None:
            if features is None or features.shape != (
                int(scores.size),
                self.gaussian_replay_feature_dim,
            ):
                raise RuntimeError(
                    "ASCAL Gaussian replay MLP lost its prediction features"
                )
            if mixture is None:
                raise RuntimeError(
                    "ASCAL Gaussian replay MLP lost its selected GMM"
                )
            if state is None:
                state = self._ensure_ordinal_ridge_state(assignment)
            updated = self._update_gaussian_replay_state(
                state,
                mixture,
                scores,
                features,
            )

        stats = ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.adapt(self, images)
        routed = bool(prediction_state.get("prediction_gaussian_replay_routed"))
        applied = bool(prediction_state.get("prediction_gaussian_replay_applied"))
        if applied:
            self.gaussian_replay_applied_batches += 1
            self.gaussian_replay_applied_samples += int(scores.size)
        elif routed:
            self.gaussian_replay_cold_start_batches += 1
        self.gaussian_replay_label_changes += int(
            prediction_state.get("prediction_gaussian_replay_label_changes", 0)
            or 0
        )
        stats.extra.update(
            {
                **self._gaussian_replay_state_stats(),
                "gaussian_replay_updated": updated,
            }
        )
        return stats

    def discard_pending_prediction(self) -> None:
        self._ordinal_ridge_precomputed_scores = None
        self._feature_route_query = None
        self._pending_gaussian_replay_features = None
        self._pending_gaussian_replay_state = None
        self._pending_gaussian_replay_assignment = None
        self._pending_gaussian_replay_mixture = None
        ASCALGMMSegmentedMemoryPosteriorOrdinalRoute.discard_pending_prediction(
            self
        )


class ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorFeatureRoutedGaussianReplayMLP
):
    """Train each routed residual on a larger set of fresh Gaussian draws."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.expanded_replay_samples = int(
            self.config.get("feature_replay_samples_per_update", 256)
        )
        if (
            self.expanded_replay_samples < 2
            or self.expanded_replay_samples % 2 != 0
        ):
            raise ValueError(
                "ASCAL expanded Gaussian replay samples must be a positive even "
                "integer"
            )

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_feature_routed_per_expert_expanded_distinct_"
                    "gaussian_replay_mlp"
                ),
                "research_name": "ASCAL-JMP-ExpandedGaussianReplay",
                "research_version": "R26",
                "feature_replay_samples_per_update": (
                    self.expanded_replay_samples
                ),
                "feature_replay_balance": "equal_real_and_fake_draws",
                "feature_replay_draw_rule": (
                    "one_fresh_independent_draw_per_generated_pseudo_feature"
                ),
                "feature_replay_minibatch_rule": (
                    "current_unlabeled_stream_batch_size"
                ),
                "feature_replay_passes": 1,
                "generated_feature_reuse": False,
                "optimizer_steps_per_predicted_batch": (
                    "ceil_generated_replay_samples_over_stream_batch_size"
                ),
                "expert_initialization": (
                    "independent_random_hidden_layer_zero_output_residual_"
                    "without_base_parameter_copy"
                ),
                "frozen_base": True,
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "the R25 frozen Base feature router GMM and Gaussian moments stay unchanged",
                    "each ready expert draws one balanced set of fresh pseudo-features",
                    "each generated pseudo-feature is consumed exactly once",
                    "the replay set is split by the current stream batch size",
                    "only the selected zero-born residual MLP receives gradients",
                    "the final prediction remains frozen Base plus one expert residual",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_feature_routed_expanded_gaussian_replay_mlp"

    def _gaussian_replay_samples_per_update(
        self,
        stream_batch_size: int,
    ) -> int:
        del stream_batch_size
        return self.expanded_replay_samples

    def _gaussian_replay_minibatch_size(
        self,
        stream_batch_size: int,
        generated_samples: int,
    ) -> int:
        return min(int(stream_batch_size), int(generated_samples))


class ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSharedGaussianReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP
):
    """Share one residual MLP while preserving every routed feature distribution."""

    def _reset_state(self) -> None:
        self._shared_gaussian_replay_state = (
            self._new_gaussian_replay_head_state()
        )
        super()._reset_state()

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_feature_routed_per_expert_gaussian_memory_"
                    "with_one_shared_expanded_replay_residual_mlp"
                ),
                "research_name": "ASCAL-JMP-SharedResidualHead",
                "research_version": "R27",
                "ablation_parent": "ASCAL-JMP-ExpandedGaussianReplay-R26",
                "ablation_question": (
                    "whether_expert_specific_residual_parameters_are_needed_"
                    "when_routing_gmms_and_feature_distributions_remain_per_expert"
                ),
                "expert_head": (
                    "one_global_hidden_layer_gelu_mlp_with_zero_initialized_"
                    "output_shared_by_all_experts"
                ),
                "head_count": 1,
                "head_state_scope": "shared_across_all_routed_experts",
                "prediction_rule": (
                    "sigmoid_of_frozen_source_logit_plus_the_shared_mlp_"
                    "residual_logit"
                ),
                "optimizer": (
                    "one_shared_adam_state_updated_after_prediction_from_the_"
                    "selected_experts_generated_distribution"
                ),
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R26 routing segmentation GMMs feature moments and replay budget stay unchanged",
                    "all experts address one shared residual MLP and one optimizer state",
                    "the selected expert still supplies the only replay distribution for each update",
                    "the shared residual is zero at birth and Base remains frozen",
                    "no other R26 component or hyperparameter is changed",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_feature_routed_shared_gaussian_replay_mlp"

    def _gaussian_replay_head_state(
        self,
        distribution_state: dict[str, Any],
    ) -> dict[str, Any]:
        del distribution_state
        return self._shared_gaussian_replay_state

    def _gaussian_replay_state_stats(self) -> dict[str, Any]:
        stats = super()._gaussian_replay_state_stats()
        shared = self._shared_gaussian_replay_state
        stats.update(
            {
                "gaussian_replay_head_scope": "shared",
                "gaussian_replay_head_count": int(
                    shared.get("mlp_head") is not None
                ),
                "gaussian_replay_shared_head_updates": int(
                    shared["head_updates"]
                ),
                "gaussian_replay_shared_optimizer_steps": int(
                    shared["optimizer_steps"]
                ),
                "gaussian_replay_shared_generated_samples": int(
                    shared["generated_samples"]
                ),
            }
        )
        return stats


class ASCALGMMSegmentedMemoryPosteriorFeatureRoutedLinearGaussianReplay(
    ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP
):
    """Replace each R26 nonlinear residual MLP with one zero-born linear head."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_feature_routed_per_expert_expanded_gaussian_"
                    "replay_linear_residual"
                ),
                "research_name": "ASCAL-JMP-LinearResidualHead",
                "research_version": "R28",
                "ablation_parent": "ASCAL-JMP-ExpandedGaussianReplay-R26",
                "ablation_question": (
                    "whether_the_hidden_gelu_layer_is_needed_for_sample_level_"
                    "ranking_adaptation"
                ),
                "expert_head": (
                    "one_zero_initialized_linear_residual_logit_per_expert"
                ),
                "expert_head_parameters": self.gaussian_replay_feature_dim + 1,
                "prediction_rule": (
                    "sigmoid_of_frozen_source_logit_plus_selected_expert_"
                    "linear_residual_logit"
                ),
                "replay_rng_alignment": (
                    "consume_the_exact_r26_hidden_initialization_draw_count_"
                    "before_each_linear_head_birth"
                ),
                "fixed_method_hyperparameters": [
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R26 routing segmentation GMMs feature moments replay and expert scope stay unchanged",
                    "each 768 to 64 to 1 GELU residual is replaced by one 768 to 1 linear residual",
                    "the output remains exactly zero at expert birth and Base stays frozen",
                    "discarded random draws keep all subsequent Gaussian replay samples aligned with R26",
                    "no other R26 component or hyperparameter is changed",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_feature_routed_linear_gaussian_replay"

    def _new_gaussian_replay_head(self) -> Any:
        import torch
        import torch.nn as nn

        self._gaussian_replay_rng.normal(
            0.0,
            math.sqrt(2.0 / float(self.gaussian_replay_feature_dim)),
            size=(
                self.gaussian_replay_hidden_dim,
                self.gaussian_replay_feature_dim,
            ),
        )
        head = nn.Linear(self.gaussian_replay_feature_dim, 1).to(self.device)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        return head


class ASCALGMMSegmentedMemoryPosteriorFeatureRoutedUniformGaussianReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP
):
    """Ablate continuous GMM confidence while retaining its hard pseudo-class."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_feature_routed_per_expert_hard_gmm_label_"
                    "uniform_weight_expanded_gaussian_replay_mlp"
                ),
                "research_name": "ASCAL-JMP-UniformConfidence",
                "research_version": "R29",
                "ablation_parent": "ASCAL-JMP-ExpandedGaussianReplay-R26",
                "ablation_question": (
                    "whether_continuous_gmm_posterior_confidence_is_needed_"
                    "beyond_the_same_hard_pseudo_class"
                ),
                "gmm_role": (
                    "prediction_time_selected_old_expert_equal_prior_hard_"
                    "pseudo_label_only"
                ),
                "reliability_rule": "uniform_one_for_every_arrived_sample",
                "confidence_threshold": "none",
                "feature_memory": (
                    "per_expert_uniform_weight_real_and_fake_diagonal_"
                    "gaussian_sufficient_statistics"
                ),
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R26 routing segmentation GMM hard labels replay heads and prediction stay unchanged",
                    "every hard pseudo-labeled feature contributes unit mass instead of posterior confidence",
                    "no confidence threshold or replacement score is introduced",
                    "no other R26 component or hyperparameter is changed",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_feature_routed_uniform_gaussian_replay_mlp"

    def _gaussian_replay_supervision(
        self,
        mixture: dict[str, Any],
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels, _, posterior = super()._gaussian_replay_supervision(
            mixture,
            scores,
        )
        reliability = np.ones(labels.size, dtype=np.float64)
        return labels, reliability, posterior


class ASCALGMMSegmentedMemoryPosteriorActiveGaussianReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP
):
    """Use only the currently active learner and never recall a historical expert."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_active_state_only_expanded_gaussian_replay_"
                    "per_expert_mlp"
                ),
                "research_name": "ASCAL-JMP-ActiveOnly",
                "research_version": "R30",
                "ablation_parent": "ASCAL-JMP-ExpandedGaussianReplay-R26",
                "ablation_question": (
                    "whether_prediction_time_historical_clip_feature_routing_"
                    "is_needed_beyond_the_current_active_learning_state"
                ),
                "routing_coordinate": "none_active_learning_state_only",
                "routing_rule": (
                    "use_only_the_active_candidate_after_the_inherited_score_"
                    "segment_change_memory_callback"
                ),
                "per_batch_historical_feature_candidates": False,
                "segment_change_score_memory_recall": True,
                "historical_expert_recall": "segment_change_callback_only",
                "implementation_audit_status": (
                    "invalidated_as_a_complete_no_historical_recall_ablation"
                ),
                "episodic_memory_updated": True,
                "feature_route_statistics_updated": True,
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R26 segmentation GMMs feature moments replay heads and prediction stay unchanged",
                    "the per-batch candidate list excludes archived experts",
                    "the inherited segment-change callback can still recall archived score memory",
                    "a recalled state is exposed as the active candidate on following batches",
                    "no routing threshold replacement similarity or new hyperparameter is introduced",
                    "no other R26 component or hyperparameter is changed",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_active_only_expanded_gaussian_replay_mlp"

    def _routing_candidates(self, scores: np.ndarray) -> list[dict[str, Any]]:
        candidates = ASCALGMMSegmentedMemoryPosteriorPreRoute._routing_candidates(
            self,
            scores,
        )
        return [
            candidate
            for candidate in candidates
            if candidate["expert"] == "active_learning_state"
        ]


class ASCALGMMSegmentedMemoryPosteriorNoHistoricalRecallGaussianReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorActiveGaussianReplayMLP
):
    """Disable both per-batch and segment-change historical expert recall."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_segmented_current_state_only_expanded_"
                    "gaussian_replay_residual_mlp"
                ),
                "research_name": "ASCAL-JMP-NoHistoricalRecall",
                "research_version": "R34",
                "ablation_parent": "ASCAL-JMP-ExpandedGaussianReplay-R26",
                "ablation_question": (
                    "whether_any_historical_expert_recall_is_needed_beyond_"
                    "the_current_segment_learning_state"
                ),
                "routing_coordinate": "none_current_segment_state_only",
                "routing_rule": (
                    "current_active_gmm_when_ready_otherwise_exact_source_"
                    "fallback"
                ),
                "per_batch_historical_feature_candidates": False,
                "segment_change_score_memory_recall": False,
                "historical_expert_recall": False,
                "episodic_memory_updated": True,
                "episodic_memory_role": "shadow_archive_never_read",
                "completed_expert_state_rule": (
                    "archive_for_audit_then_start_a_fresh_current_segment_state"
                ),
                "implementation_audit_status": "valid_no_historical_recall_ablation",
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R26 segmentation GMM confidence feature replay heads and training budget stay unchanged",
                    "archived experts are excluded from every per-batch candidate list",
                    "the segment-change callback cannot select an archived score memory",
                    "each discovered segment therefore learns only its fresh active expert state",
                    "the archive remains shadow audit state and is never read for prediction or adaptation",
                    "no threshold similarity replacement or new hyperparameter is introduced",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_no_historical_recall_expanded_gaussian_replay_mlp"

    def _select_recalled_memory(
        self,
        scores: np.ndarray,
        new_mixture: dict[str, Any],
        *,
        excluded_index: int | None,
    ) -> int | None:
        del scores, excluded_index
        self.last_memory_fixed_score = None
        self.last_memory_new_bic = float(new_mixture["bic"])
        self.last_memory_identity_penalty = None
        self.last_memory_recall_gain = None
        return None


class ASCALGMMCurrentSegmentGaussianReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorNoHistoricalRecallGaussianReplayMLP
):
    """Keep only the current segment state and discard every completed expert."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.discarded_completed_segment_states = 0

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_current_segment_gmm_supervised_balanced_"
                    "gaussian_replay_residual_mlp"
                ),
                "research_name": "ASCAL-JMP-CurrentSegmentCore",
                "research_version": "R35",
                "ablation_parent": "ASCAL-JMP-NoHistoricalRecall-R34",
                "ablation_question": (
                    "whether_the_unused_shadow_archive_can_be_deleted_without_"
                    "changing_forward_online_predictions"
                ),
                "routing_coordinate": "none_current_segment_state_only",
                "routing_candidates": "current_active_gmm_only",
                "historical_expert_recall": False,
                "episodic_memory_updated": False,
                "episodic_memory_role": "none",
                "completed_expert_state_rule": "discard_immediately_at_segment_change",
                "state_scope": "one_current_segment_gmm_feature_distribution_and_mlp",
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R34 forward-online prediction supervision replay head and segment logic stay unchanged",
                    "a completed segment is discarded instead of copied into a shadow archive",
                    "the current-batch candidate builder never evaluates historical mixtures",
                    "only one current segment learning state can occupy memory",
                    "no threshold similarity replacement or new hyperparameter is introduced",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "current_segment_gmm_gaussian_replay_mlp"

    def _store_completed_segment(
        self,
        mixture: dict[str, Any],
        samples: int,
    ) -> int | None:
        del samples
        if int(mixture["components"]) < 2:
            return None
        self.discarded_completed_segment_states += 1
        self._novel_ordinal_ridge_state = self._new_ordinal_ridge_state()
        return None

    def _routing_candidates(self, scores: np.ndarray) -> list[dict[str, Any]]:
        if not self._mixture_active() or self._mixture is None:
            return []
        active_partition = self._candidate_partition()
        active_boundary = float(active_partition["decision_boundary"])
        return [
            {
                "expert": "active_learning_state",
                "memory_index": None,
                "mixture": self._mixture,
                "boundary": self._stabilized_boundary(active_boundary),
                "candidate_boundary": active_boundary,
                "real_components": int(active_partition["real_components"]),
                "fake_components": int(active_partition["fake_components"]),
                "deviance": _fixed_gmm_deviance(scores, self._mixture),
            }
        ]

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats["discarded_completed_segment_states"] = (
            self.discarded_completed_segment_states
        )
        return stats


class ASCALGMMGlobalStreamGaussianReplayMLP(
    ASCALGMMCurrentSegmentGaussianReplayMLP
):
    """Use one global online GMM, feature distribution, and residual head."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_global_stream_gmm_supervised_balanced_"
                    "gaussian_replay_residual_mlp"
                ),
                "research_name": "ASCAL-JMP-GlobalStreamCore",
                "research_version": "R36",
                "ablation_parent": "ASCAL-JMP-CurrentSegmentCore-R35",
                "ablation_question": (
                    "whether_parameter_free_causal_segment_resets_are_needed_"
                    "beyond_one_global_accumulating_online_state"
                ),
                "segmentation_rule": "none",
                "segment_reset_scope": "none",
                "state_scope": "one_global_stream_gmm_feature_distribution_and_mlp",
                "score_history": "all_causally_arrived_source_scores",
                "completed_expert_state_rule": "not_applicable_no_segment_change",
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R35 GMM supervision confidence replay head and prediction stay unchanged",
                    "the BIC change-point scan and every current-state reset are disabled",
                    "one GMM and one feature distribution and MLP accumulate the full stream",
                    "no historical archive route threshold or new hyperparameter is introduced",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "global_stream_gmm_gaussian_replay_mlp"

    def _detect_segment_change(self) -> None:
        self.last_segment_gain = None


class ASCALGMMSegmentedMemoryPosteriorCLIPRoutedGaussianReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP
):
    """Recall historical experts only through frozen CLIP feature routing."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_only_routed_historical_expert_memory_with_"
                    "gmm_supervised_balanced_gaussian_replay_residual_mlp"
                ),
                "research_name": "ASCAL-JMP-CLIPExpertMemory",
                "research_version": "R37",
                "ablation_parent": "ASCAL-JMP-ExpandedGaussianReplay-R26",
                "ablation_question": (
                    "whether_segment_change_score_recall_can_be_removed_while_"
                    "retaining_clip_routed_expert_memory_and_continual_retention"
                ),
                "routing_coordinate": (
                    "l2_normalized_frozen_clip_feature_orthogonal_to_source_"
                    "binary_head"
                ),
                "routing_rule": (
                    "maximum_mean_cosine_similarity_to_active_or_historical_"
                    "expert_prototype"
                ),
                "routing_score_used": False,
                "segment_change_score_memory_recall": False,
                "historical_expert_recall": "clip_feature_route_only",
                "expert_memory": (
                    "gmm_route_prototype_real_fake_diagonal_gaussians_"
                    "residual_mlp_and_adam_state"
                ),
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R26 frozen Base GMM supervision feature route replay and expert heads stay unchanged",
                    "historical prediction and learning assignments use only frozen CLIP feature similarity",
                    "the BIC segment-change callback always starts a novel state and never score-recalls memory",
                    "completed expert states remain archived for later CLIP-routed reuse",
                    "no route threshold memory capacity fusion weight or target-selected parameter is added",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_clip_routed_gaussian_replay_mlp"

    def _select_recalled_memory(
        self,
        scores: np.ndarray,
        new_mixture: dict[str, Any],
        *,
        excluded_index: int | None,
    ) -> int | None:
        del scores, excluded_index
        self.last_memory_fixed_score = None
        self.last_memory_new_bic = float(new_mixture["bic"])
        self.last_memory_identity_penalty = None
        self.last_memory_recall_gain = None
        return None


class ASCALGMMSegmentedMemoryPosteriorCLIPRoutedDecoupledRankReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorCLIPRoutedGaussianReplayMLP
):
    """Separate threshold calibration from sample-level ranking gradients."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.decoupled_rank_minibatches = 0
        self.decoupled_rank_pairs = 0
        self.decoupled_rank_last_calibration_loss = 0.0
        self.decoupled_rank_last_ranking_loss = 0.0

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "clip_routed_expert_memory_with_decoupled_calibration_"
                    "and_pairwise_ranking_replay"
                ),
                "research_name": "ASCAL-JMP-DecoupledRank",
                "research_version": "R39",
                "ablation_parent": "ASCAL-JMP-CLIPExpertMemory-R37",
                "ablation_question": (
                    "whether_separating_global_threshold_calibration_from_"
                    "feature_dependent_ranking_improves_auc_without_losing_"
                    "r37_accuracy"
                ),
                "training_objective": (
                    "equal_mean_of_bias_only_balanced_bce_and_feature_only_"
                    "pairwise_logistic_auc_surrogate"
                ),
                "calibration_gradient_scope": (
                    "expert_output_bias_only_with_feature_residual_detached"
                ),
                "ranking_gradient_scope": (
                    "expert_hidden_and_output_weights_only_with_bias_cancelled"
                ),
                "ranking_pairs": (
                    "all_pseudo_fake_real_pairs_within_each_replay_minibatch"
                ),
                "ranking_margin_hyperparameter": "none_logistic_softplus",
                "objective_mix": "fixed_equal_mean_without_tuned_coefficient",
                "prediction_rule": (
                    "sigmoid_of_frozen_source_logit_plus_selected_expert_"
                    "calibration_bias_and_feature_ranking_residual"
                ),
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R37 segmentation routing GMM supervision Gaussian replay and prediction stay unchanged",
                    "balanced BCE updates only the scalar expert calibration bias",
                    "a parameter-free pairwise logistic loss updates only the feature-dependent residual",
                    "the two mean losses are averaged equally without a tunable mixing coefficient",
                    "no target label confidence threshold fusion weight or ranking margin is introduced",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_clip_routed_decoupled_rank_replay_mlp"

    def _gaussian_replay_minibatch_loss(
        self,
        head: Any,
        features: Any,
        source_margin: Any,
        labels: Any,
    ) -> Any:
        import torch
        import torch.nn.functional as functional

        if len(head) != 3 or not hasattr(head[2], "weight"):
            raise TypeError(
                "ASCAL decoupled ranking requires the 768-to-hidden-to-1 MLP"
            )
        hidden = head[1](head[0](features))
        feature_residual = functional.linear(
            hidden,
            head[2].weight,
            bias=None,
        ).reshape(-1)
        calibration_bias = head[2].bias.reshape(())
        calibration_loss = functional.binary_cross_entropy_with_logits(
            source_margin + feature_residual.detach() + calibration_bias,
            labels,
        )

        real = labels < 0.5
        fake = labels >= 0.5
        real_count = int(torch.count_nonzero(real).item())
        fake_count = int(torch.count_nonzero(fake).item())
        if real_count < 1 or fake_count < 1:
            ranking_loss = calibration_loss.detach().new_zeros(())
            loss = calibration_loss
        else:
            ranking_logit = source_margin + feature_residual
            pair_margin = (
                ranking_logit[fake][:, None] - ranking_logit[real][None, :]
            )
            ranking_loss = functional.softplus(-pair_margin).mean()
            loss = 0.5 * (calibration_loss + ranking_loss)
            self.decoupled_rank_minibatches += 1
            self.decoupled_rank_pairs += real_count * fake_count

        self.decoupled_rank_last_calibration_loss = float(
            calibration_loss.detach().cpu().item()
        )
        self.decoupled_rank_last_ranking_loss = float(
            ranking_loss.detach().cpu().item()
        )
        return loss

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(
            {
                "decoupled_rank_minibatches": self.decoupled_rank_minibatches,
                "decoupled_rank_pairs": self.decoupled_rank_pairs,
                "decoupled_rank_last_calibration_loss": (
                    self.decoupled_rank_last_calibration_loss
                ),
                "decoupled_rank_last_ranking_loss": (
                    self.decoupled_rank_last_ranking_loss
                ),
            }
        )
        return stats


class ASCALGMMSegmentedMemoryPosteriorCLIPRoutedConservativeRankReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorCLIPRoutedDecoupledRankReplayMLP
):
    """Use the R39 objective with a pre-registered conservative update rate."""

    def _reset_state(self) -> None:
        self.config.setdefault("feature_replay_learning_rate", 3e-4)
        super()._reset_state()

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "research_name": "ASCAL-JMP-ConservativeRank",
                "research_version": "R40",
                "ablation_parent": "ASCAL-JMP-DecoupledRank-R39",
                "ablation_question": (
                    "whether_a_three_times_smaller_pre_registered_learning_"
                    "rate_preserves_high_source_auc_more_reliably"
                ),
                "learning_rate": self.gaussian_replay_learning_rate,
                "intentional_changes": [
                    "R39 architecture objective routing replay and expert memory stay unchanged",
                    "the Adam learning rate is reduced from 0.001 to 0.0003",
                    "no other method setting or target-selected parameter changes",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_clip_routed_conservative_rank_replay_mlp"


class ASCALGMMSegmentedMemoryPosteriorCLIPRoutedCompactRankReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorCLIPRoutedDecoupledRankReplayMLP
):
    """Use the R39 objective with a smaller per-expert ranking head."""

    def _reset_state(self) -> None:
        self.config.setdefault("feature_replay_hidden_dim", 32)
        super()._reset_state()

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "research_name": "ASCAL-JMP-CompactRank",
                "research_version": "R41",
                "ablation_parent": "ASCAL-JMP-DecoupledRank-R39",
                "ablation_question": (
                    "whether_halving_the_hidden_width_reduces_pseudo_label_"
                    "overfit_while_retaining_ranking_capacity"
                ),
                "expert_hidden_dim": self.gaussian_replay_hidden_dim,
                "expert_head": (
                    "one_768_to_32_to_1_gelu_residual_mlp_per_expert"
                ),
                "intentional_changes": [
                    "R39 objective learning rate routing replay and expert memory stay unchanged",
                    "the per-expert hidden width is reduced from 64 to 32",
                    "no other method setting or target-selected parameter changes",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_clip_routed_compact_rank_replay_mlp"


class ASCALGMMSegmentedMemoryPosteriorCLIPRoutedConfidenceGatedReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorCLIPRoutedGaussianReplayMLP
):
    """Gate only the feature-dependent residual by GMM confidence."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.confidence_gate_batches = 0
        self.confidence_gate_samples = 0
        self.confidence_gate_last_mean = 0.0
        self.confidence_gate_last_min = 0.0
        self.confidence_gate_last_max = 0.0

    @property
    def _confidence_gate_exponent(self) -> float:
        return 1.0

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "clip_routed_expert_memory_with_gmm_confidence_gated_"
                    "feature_residual"
                ),
                "research_name": "ASCAL-JMP-GMMConfidenceGate",
                "research_version": "R42",
                "ablation_parent": "ASCAL-JMP-CLIPExpertMemory-R37",
                "ablation_question": (
                    "whether_continuous_gmm_confidence_can_protect_source_"
                    "ordering_from_uncertain_feature_residuals"
                ),
                "training_objective": "unchanged_r37_balanced_replay_bce",
                "gmm_in_final_prediction": True,
                "gmm_final_prediction_role": (
                    "continuous_feature_residual_gate_only_not_score_fusion"
                ),
                "confidence_gate": "absolute_two_posterior_minus_one",
                "confidence_gate_exponent": self._confidence_gate_exponent,
                "prediction_rule": (
                    "frozen_source_logit_plus_expert_bias_plus_gmm_confidence_"
                    "gated_feature_residual"
                ),
                "confidence_threshold": "none",
                "fusion_weight": "none",
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R37 training routing segmentation replay and expert memory stay unchanged",
                    "the scalar expert bias is always retained because it cannot reorder samples within an expert",
                    "only the feature-dependent residual is multiplied by absolute two-posterior-minus-one",
                    "uncertain samples therefore fall back to the frozen Source ordering plus expert bias",
                    "no confidence threshold learned fusion coefficient or target label is introduced",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_clip_routed_confidence_gated_replay_mlp"

    def _gaussian_replay_prediction_residual(
        self,
        state: dict[str, Any],
        features: np.ndarray,
        scores: np.ndarray,
        mixture: dict[str, Any],
    ) -> np.ndarray:
        head_state = self._gaussian_replay_head_state(state)
        head = head_state.get("mlp_head")
        if head is None or len(head) != 3 or not hasattr(head[2], "bias"):
            raise TypeError(
                "ASCAL confidence gating requires the 768-to-hidden-to-1 MLP"
            )
        full_residual = self._gaussian_replay_residual(state, features)
        bias_parameter = head[2].bias
        if bias_parameter is None or int(bias_parameter.numel()) != 1:
            raise TypeError(
                "ASCAL confidence gating requires one scalar output bias"
            )
        bias = float(bias_parameter.detach().cpu().reshape(-1)[0].item())
        posterior = np.asarray(
            joint_density_fake_posterior(scores, mixture),
            dtype=np.float64,
        ).reshape(-1)
        if posterior.shape != full_residual.shape or not np.all(
            np.isfinite(posterior)
        ):
            raise RuntimeError(
                "ASCAL confidence gating received an invalid GMM posterior"
            )
        confidence = np.clip(np.abs(2.0 * posterior - 1.0), 0.0, 1.0)
        gate = np.power(confidence, self._confidence_gate_exponent)
        residual = bias + gate * (full_residual - bias)
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError(
                "ASCAL confidence-gated MLP produced a non-finite residual"
            )
        self.confidence_gate_batches += 1
        self.confidence_gate_samples += int(gate.size)
        self.confidence_gate_last_mean = float(np.mean(gate))
        self.confidence_gate_last_min = float(np.min(gate))
        self.confidence_gate_last_max = float(np.max(gate))
        return residual

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(
            {
                "confidence_gate_batches": self.confidence_gate_batches,
                "confidence_gate_samples": self.confidence_gate_samples,
                "confidence_gate_last_mean": self.confidence_gate_last_mean,
                "confidence_gate_last_min": self.confidence_gate_last_min,
                "confidence_gate_last_max": self.confidence_gate_last_max,
                "confidence_gate_exponent": self._confidence_gate_exponent,
            }
        )
        return stats


class ASCALGMMSegmentedMemoryPosteriorCLIPRoutedQuadraticConfidenceGatedReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorCLIPRoutedConfidenceGatedReplayMLP
):
    """Use a squared confidence gate as a conservative sensitivity check."""

    @property
    def _confidence_gate_exponent(self) -> float:
        return 2.0

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "research_name": "ASCAL-JMP-GMMConfidenceGateSquared",
                "research_version": "R43",
                "ablation_parent": "ASCAL-JMP-GMMConfidenceGate-R42",
                "ablation_question": (
                    "whether_a_more_conservative_squared_confidence_gate_"
                    "better_preserves_source_auc"
                ),
                "confidence_gate_exponent": self._confidence_gate_exponent,
                "intentional_changes": [
                    "R42 training prediction decomposition routing replay and memory stay unchanged",
                    "the parameter-free confidence is squared before gating the feature residual",
                    "no other method setting or target-selected parameter changes",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return (
            "segmented_memory_clip_routed_quadratic_confidence_gated_"
            "replay_mlp"
        )


class ASCALGMMSegmentedMemoryPosteriorCLIPRoutedOrthogonalResidualReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorCLIPRoutedGaussianReplayMLP
):
    """Train the residual only in the subspace unseen by the Source head."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "clip_routed_expert_memory_with_source_orthogonal_"
                    "gaussian_replay_residual"
                ),
                "research_name": "ASCAL-JMP-OrthogonalResidual",
                "research_version": "R44",
                "ablation_parent": "ASCAL-JMP-CLIPExpertMemory-R37",
                "ablation_question": (
                    "whether_removing_the_frozen_source_decision_direction_"
                    "from_the_expert_input_prevents_duplicate_ranking"
                ),
                "residual_coordinate": (
                    "l2_normalized_frozen_clip_feature_orthogonal_to_source_"
                    "binary_head"
                ),
                "source_coordinate_in_residual_input": False,
                "source_coordinate_in_base_logit": True,
                "training_objective": "unchanged_r37_balanced_replay_bce",
                "prediction_rule": (
                    "frozen_source_logit_plus_selected_expert_residual_from_"
                    "the_source_orthogonal_clip_subspace"
                ),
                "gmm_in_final_prediction": False,
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R37 segmentation routing GMM supervision replay optimizer and expert memory stay unchanged",
                    "the Base logit retains the complete frozen source decision direction",
                    "the expert MLP sees only normalized CLIP features orthogonal to that direction",
                    "the same parameter-free coordinate already used by R37 routing is reused for the residual",
                    "no new threshold loss coefficient fusion weight or target label is introduced",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_clip_routed_orthogonal_residual_replay_mlp"

    def _gaussian_replay_head_features(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        normalized = self._normalized_feature_values(features)
        return self._feature_route_coordinates(normalized)


class ASCALGMMSegmentedMemoryPosteriorSegmentCLIPRoutedGaussianReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorCLIPRoutedGaussianReplayMLP
):
    """Keep BIC segment identity independent from per-batch expert routing."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.routing_learning_state_handoffs_suppressed = 0

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "bic_segment_created_clip_routed_historical_expert_memory_"
                    "with_selected_expert_gaussian_replay_residual_learning"
                ),
                "research_name": "ASCAL-JMP-SegmentExpertMemory",
                "research_version": "R38",
                "ablation_parent": "ASCAL-JMP-CLIPExpertMemory-R37",
                "ablation_question": (
                    "whether_per_batch_route_induced_learning_state_handoffs_"
                    "are_needed_beyond_bic_created_experts"
                ),
                "expert_creation_rule": (
                    "one_novel_state_only_after_parameter_free_causal_bic_"
                    "segment_change"
                ),
                "prediction_route_state_mutation": False,
                "adaptation_rule": (
                    "update_the_prediction_selected_experts_feature_"
                    "distribution_and_residual_without_restarting_score_history"
                ),
                "score_segmentation_state": (
                    "one_independent_current_stream_state_changed_only_by_bic"
                ),
                "segment_change_score_memory_recall": False,
                "historical_expert_recall": "clip_feature_route_only",
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R37 Base feature route GMM teacher Gaussian replay and expert archive stay unchanged",
                    "a batch may select and update a historical expert without changing the active BIC segment identity",
                    "only the parameter-free BIC change detector can finalize a segment and create a novel expert",
                    "per-batch feature routing no longer clears score history or increments segment changes",
                    "no hysteresis route threshold cooldown memory cap or new target parameter is introduced",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_segment_clip_routed_gaussian_replay_mlp"

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

        self.last_routing_proposed_expert = proposed_expert
        self.last_routing_proposed_memory_index = proposed_memory_index
        self.last_routing_admission_reason = routing_state[
            "prediction_routing_admission_reason"
        ]
        if proposed_expert == "episodic_memory":
            self.routing_memory_proposals += 1
            if selected_expert != "episodic_memory":
                self.routing_memory_admission_fallbacks += 1

        if selected_expert != "episodic_memory":
            return
        if selected_memory_index is None:
            raise RuntimeError(
                "ASCAL segment expert route selected memory without an index"
            )
        if int(selected_memory_index) == self.active_memory_index:
            self.routing_active_memory_identity_reuses += 1
            return
        self.routing_learning_state_handoffs_suppressed += 1

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats["routing_learning_state_handoffs_suppressed"] = (
            self.routing_learning_state_handoffs_suppressed
        )
        return stats


class ASCALGMMSegmentedMemoryPosteriorFeatureRoutedCurrentBatchReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP
):
    """Replace cumulative Gaussian draws with balanced current-batch resampling."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.current_batch_replay_skipped_updates = 0

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_feature_routed_balanced_current_batch_"
                    "resampled_residual_mlp"
                ),
                "research_name": "ASCAL-JMP-CurrentBatchReplay",
                "research_version": "R31",
                "ablation_parent": "ASCAL-JMP-ExpandedGaussianReplay-R26",
                "ablation_question": (
                    "whether_cumulative_class_conditional_gaussian_feature_"
                    "replay_is_needed_beyond_current_batch_pseudo_features"
                ),
                "feature_memory_role": "shadow_statistics_not_used_for_training",
                "feature_replay": (
                    "balanced_reliability_weighted_resampling_with_replacement_"
                    "from_the_current_arrived_batch_only"
                ),
                "feature_replay_samples_per_update": self.expanded_replay_samples,
                "missing_current_class_rule": "skip_the_post_prediction_head_update",
                "raw_features_stored": False,
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R26 routing segmentation GMM supervision heads prediction and replay budget stay unchanged",
                    "the head trains on balanced resamples of only the just-arrived pseudo-labeled features",
                    "cumulative Gaussian moments remain shadow audit state and never generate training features",
                    "a batch without reliable samples from both pseudo-classes performs no head update",
                    "no image or current feature is retained after adaptation",
                    "no other R26 component or hyperparameter is changed",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_feature_routed_current_batch_replay_mlp"

    def _gaussian_replay_training_samples(
        self,
        state: dict[str, Any],
        requested_samples: int,
        observed_features: np.ndarray,
        observed_labels: np.ndarray,
        observed_reliability: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        del state
        features = np.asarray(observed_features, dtype=np.float64)
        labels = np.asarray(observed_labels, dtype=np.int64).reshape(-1)
        reliability = np.asarray(
            observed_reliability,
            dtype=np.float64,
        ).reshape(-1)
        per_class = max(1, int(math.ceil(requested_samples / 2.0)))
        replay_features: list[np.ndarray] = []
        replay_labels: list[np.ndarray] = []
        for class_index in (0, 1):
            indices = np.flatnonzero(
                (labels == class_index)
                & (reliability > np.finfo(np.float64).eps)
            )
            if indices.size == 0:
                self.current_batch_replay_skipped_updates += 1
                return None
            probabilities = reliability[indices]
            probabilities = probabilities / float(probabilities.sum())
            selected = self._gaussian_replay_rng.choice(
                indices,
                size=per_class,
                replace=True,
                p=probabilities,
            )
            replay_features.append(features[selected])
            replay_labels.append(
                np.full(per_class, class_index, dtype=np.float32)
            )
        synthetic = np.concatenate(replay_features, axis=0)
        targets = np.concatenate(replay_labels, axis=0)
        order = self._gaussian_replay_rng.permutation(targets.size)
        return synthetic[order], targets[order]

    def _gaussian_replay_state_stats(self) -> dict[str, Any]:
        stats = super()._gaussian_replay_state_stats()
        stats.update(
            {
                "current_batch_replay_skipped_updates": (
                    self.current_batch_replay_skipped_updates
                ),
                "gaussian_replay_feature_source": "current_batch_resampling",
            }
        )
        return stats


class ASCALGMMSegmentedMemoryPosteriorFeatureRoutedPriorGaussianReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP
):
    """Replay according to accumulated pseudo-class mass instead of balancing it."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_feature_routed_empirical_prior_gaussian_"
                    "replay_residual_mlp"
                ),
                "research_name": "ASCAL-JMP-PriorReplay",
                "research_version": "R32",
                "ablation_parent": "ASCAL-JMP-ExpandedGaussianReplay-R26",
                "ablation_question": (
                    "whether_equal_real_fake_feature_replay_is_needed_instead_"
                    "of_the_accumulated_pseudo_class_mass"
                ),
                "feature_replay_balance": (
                    "accumulated_reliability_weighted_pseudo_class_mass"
                ),
                "class_count_rule": (
                    "nearest_integer_empirical_mass_allocation_with_one_sample_"
                    "minimum_per_class"
                ),
                "feature_replay_samples_per_update": self.expanded_replay_samples,
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R26 routing segmentation GMM confidence Gaussian moments heads and prediction stay unchanged",
                    "the same total replay budget follows each selected experts accumulated pseudo-class mass",
                    "at least one generated feature per class keeps the binary objective defined",
                    "no target class prior or target label is read",
                    "no other R26 component or hyperparameter is changed",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_feature_routed_prior_gaussian_replay_mlp"

    def _gaussian_replay_class_counts(
        self,
        state: dict[str, Any],
        samples: int,
    ) -> tuple[int, int]:
        total_samples = int(samples)
        if total_samples < 2:
            raise ValueError("ASCAL prior replay requires at least two samples")
        mass = np.asarray(state["class_mass"], dtype=np.float64).reshape(-1)
        total_mass = float(mass.sum())
        if (
            mass.shape != (2,)
            or not np.all(np.isfinite(mass))
            or np.any(mass <= np.finfo(np.float64).eps)
            or not math.isfinite(total_mass)
            or total_mass <= np.finfo(np.float64).eps
        ):
            raise ValueError("ASCAL prior replay received invalid class mass")
        fake_samples = int(math.floor(total_samples * mass[1] / total_mass + 0.5))
        fake_samples = min(max(fake_samples, 1), total_samples - 1)
        return total_samples - fake_samples, fake_samples


class ASCALGMMSegmentedMemoryPosteriorFeatureRoutedSourceReplayMLP(
    ASCALGMMSegmentedMemoryPosteriorFeatureRoutedExpandedGaussianReplayMLP
):
    """Use frozen Source pseudo-supervision instead of the selected expert GMM."""

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        metadata = super().reproduction_metadata
        metadata.update(
            {
                "adaptive_role": (
                    "frozen_clip_feature_routed_source_confidence_gaussian_"
                    "replay_residual_mlp"
                ),
                "research_name": "ASCAL-JMP-SourceSupervision",
                "research_version": "R33",
                "ablation_parent": "ASCAL-JMP-ExpandedGaussianReplay-R26",
                "ablation_question": (
                    "whether_expert_gmm_pseudo_supervision_is_needed_beyond_"
                    "the_frozen_source_probability"
                ),
                "gmm_role": "segmentation_and_expert_identity_only",
                "pseudo_label": "frozen_source_probability_at_one_half",
                "reliability_rule": (
                    "absolute_frozen_source_signed_probability_without_threshold"
                ),
                "gmm_in_feature_supervision": False,
                "gmm_in_final_prediction": False,
                "fixed_method_hyperparameters": [
                    "feature_replay_hidden_dim",
                    "feature_replay_learning_rate",
                    "feature_replay_samples_per_update",
                ],
                "target_selected_hyperparameters": 0,
                "intentional_changes": [
                    "R26 routing segmentation feature moments replay heads and prediction stay unchanged",
                    "the frozen Source probability replaces the selected GMM posterior for labels and reliability",
                    "GMMs remain only to define causal expert state and historical routing",
                    "no source threshold confidence threshold or new hyperparameter is introduced",
                    "no other R26 component or hyperparameter is changed",
                ],
            }
        )
        return metadata

    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_feature_routed_source_gaussian_replay_mlp"

    def _gaussian_replay_supervision(
        self,
        mixture: dict[str, Any],
        scores: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del mixture
        probability = np.asarray(
            self._source_probability(scores),
            dtype=np.float64,
        ).reshape(-1)
        labels = (probability >= 0.5).astype(np.int64)
        reliability = np.abs(2.0 * probability - 1.0)
        return labels, reliability, probability


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
    "ASCALGMMSegmentedMemoryPosteriorAnalyticExpert",
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
