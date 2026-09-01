"""The single retained Ours implementation and its readout ablation.

The method keeps a frozen CLIP LoRA source detector, detects persistent score
changes with BIC-selected one-dimensional mixtures, routes compact expert state
with frozen CLIP features, and learns one residual MLP per expert from balanced
class-conditional Gaussian replay. ``readout_mode=base`` is the retained R37
ablation. ``readout_mode=calibrated`` is the final R47 setting: the learned
feature residual is scaled by 0.75 and one scalar expert intercept is refit on
the same replay batch without another optimizer or additional samples.

Both settings use this class directly; neither method variant inherits from the
other. All earlier research variants have been removed from the public code.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from src.types import AdaptationStats, PredictionBatch

from .base import TTAMethod


def binary_score(logits: Any) -> Any:
    """Signed real/fake score from two logits; positive means fake."""

    return logits[:, 1].float() - logits[:, 0].float()


def _sigmoid(values: Any) -> Any:
    values = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def fit_temperature(scores: Any, labels: Any, *, grid_size: int = 60) -> float:
    """Fit a scalar temperature on source scores by binary NLL."""

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if scores.size == 0 or np.unique(labels).size < 2:
        raise ValueError("Temperature fitting requires scores from both classes")

    def nll(tau: float) -> float:
        probabilities = np.clip(_sigmoid(scores / tau), 1e-7, 1.0 - 1e-7)
        return float(
            -np.mean(labels * np.log(probabilities) + (1.0 - labels) * np.log(1.0 - probabilities))
        )

    coarse = np.geomspace(0.25, 8.0, grid_size)
    best = float(min(coarse, key=nll))
    refined = np.linspace(max(0.25, best / 1.5), min(8.0, best * 1.5), grid_size)
    return float(min(refined, key=nll))


def fit_gaussian_ml(values: Any) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise ValueError("Gaussian anchor fitting requires at least two samples")
    mu = float(values.mean())
    sigma = float(values.std())
    return mu, max(sigma, 1e-6)


def fit_gmm_bic(values: Any, *, max_components: int, seed: int = 0) -> dict[str, Any]:
    """Fit a 1-D GMM with BIC model selection over the component count."""

    from sklearn.mixture import GaussianMixture

    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if values.shape[0] < 2:
        raise ValueError("GMM anchor fitting requires at least two samples")
    if max_components < 1:
        raise ValueError("max_components must be positive")
    best_model = None
    best_bic = math.inf
    for components in range(1, int(max_components) + 1):
        model = GaussianMixture(
            n_components=components,
            covariance_type="full",
            reg_covar=1e-6,
            n_init=4,
            max_iter=200,
            random_state=seed,
        )
        model.fit(values)
        bic = float(model.bic(values))
        if bic < best_bic:
            best_bic = bic
            best_model = model
    if best_model is None:
        raise RuntimeError("GMM anchor fitting failed")
    weights = np.asarray(best_model.weights_, dtype=np.float64)
    mus = best_model.means_.reshape(-1).astype(np.float64)
    sigmas = np.sqrt(best_model.covariances_.reshape(-1)).astype(np.float64)
    order = np.argsort(mus)
    return {
        "weights": [float(value) for value in weights[order]],
        "mus": [float(value) for value in mus[order]],
        "sigmas": [float(max(value, 1e-6)) for value in sigmas[order]],
        "components": int(len(mus)),
        "bic": float(best_bic),
    }


def validate_score_anchors(anchors: Any) -> dict[str, Any]:
    """Type-check and normalize the anchor block stored in a checkpoint."""

    if not isinstance(anchors, dict):
        raise ValueError("score_anchors must be a mapping")
    temperature = float(anchors.get("temperature", 0.0))
    real = anchors.get("real")
    fake = anchors.get("fake")
    if not isinstance(real, dict) or not isinstance(fake, dict):
        raise ValueError("score_anchors require real and fake blocks")
    real_mu = float(real.get("mu", 0.0))
    real_sigma = float(real.get("sigma", 0.0))
    weights = [float(value) for value in fake.get("weights", [])]
    mus = [float(value) for value in fake.get("mus", [])]
    sigmas = [float(value) for value in fake.get("sigmas", [])]
    if temperature <= 0.0:
        raise ValueError("score_anchors.temperature must be positive")
    if real_sigma <= 0.0:
        raise ValueError("score_anchors.real.sigma must be positive")
    if not weights or not (len(weights) == len(mus) == len(sigmas)):
        raise ValueError("score_anchors.fake blocks must have matching non-empty lengths")
    if any(sigma <= 0.0 for sigma in sigmas):
        raise ValueError("score_anchors.fake.sigmas must be positive")
    if abs(sum(weights) - 1.0) > 1e-3:
        raise ValueError("score_anchors.fake.weights must sum to one")
    return {
        **anchors,
        "temperature": temperature,
        "real": {"mu": real_mu, "sigma": real_sigma},
        "fake": {"weights": weights, "mus": mus, "sigmas": sigmas},
    }


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


class _ScoreMixture(TTAMethod):
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
            raise ValueError("Ours adaptation_mode must be static or full")

        self.anchors = validate_score_anchors(self.config.get("score_anchors"))
        self.temperature = float(self.config.get("temperature", self.anchors["temperature"]))
        if self.temperature <= 0.0:
            raise ValueError("Ours temperature must be positive")

        source_fake_mean = float(
            np.dot(self.anchors["fake"]["weights"], self.anchors["fake"]["mus"])
        )
        if source_fake_mean <= float(self.anchors["real"]["mu"]):
            raise ValueError("Ours requires source scores to increase toward fake")

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
                "Ours expects (B, C, H, W) or (B, V, C, H, W) images"
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
            raise RuntimeError("Ours adapt requires a matching predict call")

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


class _MonotoneBoundary(_ScoreMixture):
    """Use the target mixture only to shift the frozen detector's score boundary."""


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
            raise RuntimeError("Ours boundary requires an active mixture")
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


class _StableBoundary(_MonotoneBoundary):
    """Stabilize causal GMM boundaries with their cumulative median."""

    def _reset_state(self) -> None:
        super()._reset_state()
        self.boundary_history: list[float] = []


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


class _CausalSegments(_StableBoundary):
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


class _EpisodicScoreMemory(_CausalSegments):
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
            raise RuntimeError("Active Ours memory index is out of range")
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


class _ProjectedPosterior(_EpisodicScoreMemory):
    """Project a joint-density Bayes boundary onto the monotone source score."""


    @property
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_projection"

    def _candidate_partition(self) -> dict[str, Any]:
        if self._mixture is None:
            raise RuntimeError(
                "Ours posterior projection requires an active mixture"
            )
        return equal_density_boundary(self._mixture)

    def _memory_boundary(self, mixture: dict[str, Any]) -> float:
        return float(equal_density_boundary(mixture)["decision_boundary"])


class _PreRoutedExperts(
    _ProjectedPosterior
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
            raise RuntimeError("Ours pre-route handoff has no eligible memory")

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
            raise RuntimeError("Ours pre-route selected memory without an index")
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
                    raise RuntimeError("Ours pre-route selected memory without an index")
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


class _MDLRoutedExperts(
    _PreRoutedExperts
):
    """Use one MDL-admitted expert assignment for prediction and adaptation."""

    _ROUTING_PENDING_FIELDS = (
        *_PreRoutedExperts._ROUTING_PENDING_FIELDS,
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
            raise RuntimeError("Ours MDL route has no eligible memory")
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
                raise RuntimeError("Ours MDL route checked memory without an index")
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
            raise RuntimeError("Ours MDL route selected memory without an index")
        selected_memory_index = int(selected_memory_index)
        if selected_memory_index == self.active_memory_index:
            if not admission_checked:
                self.routing_active_memory_identity_reuses += 1
            return
        if not admission_checked or not admission_accepted:
            raise RuntimeError(
                "Ours MDL route cannot adapt with an unconfirmed memory assignment"
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


class _LiveExpertRouting(
    _MDLRoutedExperts
):
    """Expose only one routable state for the currently active expert."""


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


class _OrdinalExpertReadout(
    _LiveExpertRouting
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
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_posterior_ordinal_route"

    def predict(self, images: Any) -> PredictionBatch:
        routed = super().predict(images)
        if self._pending is None:
            raise RuntimeError("Ours ordinal route lost the routed prediction state")

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
                "Ours ordinal readout changed a routed expert hard decision"
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


class _OrdinalRidgeState(
    _OrdinalExpertReadout
):
    """Keep ordinal-route decisions and learn one causal within-class ranker per expert."""

    _ORDINAL_RIDGE_MEMORY_KEY = "ordinal_ridge_state"

    def _reset_state(self) -> None:
        super()._reset_state()
        classifier = getattr(self.model, "classifier", None)
        weight = getattr(classifier, "weight", None)
        if weight is None or weight.ndim != 2 or int(weight.shape[0]) != 2:
            raise TypeError(
                "Ours ordinal ridge requires a two-class linear classifier"
            )
        direction = (
            weight[1].detach().float().cpu().numpy()
            - weight[0].detach().float().cpu().numpy()
        ).astype(np.float64)
        direction_norm = float(np.linalg.norm(direction))
        if not math.isfinite(direction_norm) or direction_norm <= 0.0:
            raise ValueError(
                "Ours ordinal ridge requires a nonzero source score direction"
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
                "Ours ordinal ridge expects (B, C, H, W) or "
                "(B, V, C, H, W) images"
            )
        forward_features = getattr(self.model, "forward_features", None)
        forward_classifier_features = getattr(
            self.model, "forward_classifier_features", None
        )
        classifier = getattr(self.model, "classifier", None)
        if not callable(forward_features) or not callable(classifier):
            raise TypeError(
                "Ours ordinal ridge requires forward_features and classifier"
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
                "Ours ordinal ridge feature dimension does not match the source head"
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
            raise RuntimeError("Ours ordinal ridge memory index is out of range")
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
            raise RuntimeError("Ours ordinal ridge lost its novel state")
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
            raise RuntimeError("Ours ordinal ridge lost its ordinal-route prediction state")
        selected_expert = self._pending.get("prediction_routing_expert")
        if selected_expert is None:
            return None
        if selected_expert == "active_learning_state":
            if self._mixture is None or not self._mixture_active():
                raise RuntimeError("Ours ordinal ridge selected no eligible live GMM")
            memory_index = self.active_memory_index
            assignment = (
                "novel" if memory_index is None else "episodic_memory",
                memory_index,
            )
            return assignment, self._mixture
        if selected_expert != "episodic_memory":
            raise RuntimeError("Ours ordinal ridge received an unknown ordinal-route expert")
        memory_index = self._pending.get("prediction_routing_memory_index")
        if memory_index is None:
            raise RuntimeError("Ours ordinal ridge selected memory without an index")
        memory_index = int(memory_index)
        if not 0 <= memory_index < len(self.segment_memories):
            raise RuntimeError("Ours ordinal ridge selected memory out of range")
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
                "Ours ordinal ridge produced non-finite supervision"
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
            raise RuntimeError("Ours ordinal ridge lost the ordinal-route pending state")

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
                raise RuntimeError("Ours ordinal ridge lost its selected state")
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
            raise RuntimeError("Ours ordinal ridge changed an ordinal-route hard decision")
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
                    "Ours ordinal ridge lost its matching prediction features"
                )
            if mixture is None:
                raise RuntimeError(
                    "Ours ordinal ridge lost its prediction-time selected GMM"
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


class _RidgeExpertState(
    _OrdinalRidgeState
):
    """Fuse ordinal-route with one confidence-weighted binary Ridge per routed expert."""

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
                "Ours RMS Ridge expert expects (B, C, H, W) or "
                "(B, V, C, H, W) images"
            )
        forward_features = getattr(self.model, "forward_features", None)
        forward_classifier_features = getattr(
            self.model, "forward_classifier_features", None
        )
        classifier = getattr(self.model, "classifier", None)
        if not callable(forward_features) or not callable(classifier):
            raise TypeError(
                "Ours RMS Ridge expert requires forward_features and classifier"
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
                "Ours RMS Ridge expert feature dimension does not match the source head"
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
            raise RuntimeError("Ours RMS Ridge expert built the wrong dimension")
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
                "Ours RMS Ridge expert posterior and ordinal-route labels do not align"
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
                "Ours RMS Ridge expert produced non-finite supervision"
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
                "Ours RMS Ridge expert received misaligned prediction statistics"
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
            raise RuntimeError("Ours RMS Ridge expert has no valid historical scale")
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
                "Ours RMS Ridge expert produced a non-finite prediction"
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
        stats = _OrdinalExpertReadout._state_stats(self)
        stats.update(self._rms_ridge_expert_state_stats())
        return stats

    def predict(self, images: Any) -> PredictionBatch:
        scores, features = self._batch_scores_and_rms_ridge_expert_features(images)
        self._ordinal_ridge_precomputed_scores = scores
        try:
            ordinal = _OrdinalExpertReadout.predict(
                self,
                images,
            )
        finally:
            self._ordinal_ridge_precomputed_scores = None
        if self._pending is None:
            raise RuntimeError("Ours RMS Ridge expert lost the ordinal-route pending state")

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
                raise RuntimeError("Ours RMS Ridge expert lost its selected state")
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
            return _OrdinalExpertReadout.adapt(self, images)
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
                    "Ours RMS Ridge expert lost its prediction features"
                )
            if mixture is None:
                raise RuntimeError(
                    "Ours RMS Ridge expert lost its prediction-time selected GMM"
                )
            if routed_labels is None or int(routed_labels.size) != int(scores.size):
                raise RuntimeError("Ours RMS Ridge expert lost its ordinal-route decisions")
            if base_margins is None or int(base_margins.size) != int(scores.size):
                raise RuntimeError("Ours RMS Ridge expert lost its ordinal-route margins")
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

        stats = _OrdinalExpertReadout.adapt(self, images)
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
        _OrdinalExpertReadout.discard_pending_prediction(self)


class _FeatureRoutedExperts(
    _RidgeExpertState
):
    """Route by frozen CLIP features and train direct Ridge with GMM trust."""

    def _reset_state(self) -> None:
        self._feature_route_query: np.ndarray | None = None
        super()._reset_state()


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
            raise ValueError("Ours feature route received invalid CLIP features")
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
            raise RuntimeError("Ours feature route lost the current CLIP features")

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
                "Ours feature-routed trusted Ridge produced non-finite supervision"
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
            raise RuntimeError("Ours feature-routed trusted Ridge lost Base scores")
        scores = np.asarray(self._pending["scores"], dtype=np.float64).reshape(-1)
        base_probability = np.asarray(base_probability, dtype=np.float64).reshape(-1)
        if base_probability.shape != scores.shape:
            raise RuntimeError(
                "Ours feature-routed trusted Ridge received misaligned Base scores"
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
                "Ours feature-routed trusted Ridge produced a non-finite prediction"
            )
        return probability, base_margin, ridge_margin, 1.0, 1.0

    def predict(self, images: Any) -> PredictionBatch:
        self._feature_route_query = None
        try:
            return super().predict(images)
        finally:
            self._feature_route_query = None


class _GaussianReplayExperts(
    _FeatureRoutedExperts
):
    """Learn one Base-anchored MLP from each expert's Gaussian feature replay."""

    _ORDINAL_RIDGE_MEMORY_KEY = "gaussian_replay_mlp_state"

    def _reset_state(self) -> None:
        super()._reset_state()
        classifier = getattr(self.model, "classifier", None)
        weight = getattr(classifier, "weight", None)
        if weight is None or weight.ndim != 2 or int(weight.shape[0]) != 2:
            raise TypeError(
                "Ours Gaussian replay MLP requires a two-class linear classifier"
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
            raise ValueError("Ours Gaussian replay MLP hidden dimension must be positive")
        if not (
            math.isfinite(self.gaussian_replay_learning_rate)
            and self.gaussian_replay_learning_rate > 0.0
        ):
            raise ValueError("Ours Gaussian replay MLP learning rate must be positive")
        if not (
            math.isfinite(self.gaussian_replay_variance_floor)
            and self.gaussian_replay_variance_floor > 0.0
        ):
            raise ValueError("Ours Gaussian replay variance floor must be positive")

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
    def _prediction_mode_name(self) -> str:
        return "segmented_memory_feature_routed_gaussian_replay_mlp"

    def _new_ordinal_ridge_state(self) -> dict[str, Any]:
        classifier = getattr(self.model, "classifier", None)
        feature_dim = int(getattr(classifier, "in_features", 0))
        if feature_dim < 1:
            raise TypeError(
                "Ours Gaussian replay MLP requires a finite feature dimension"
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
                "Ours Gaussian replay MLP expects (B, C, H, W) or "
                "(B, V, C, H, W) images"
            )
        forward_features = getattr(self.model, "forward_features", None)
        forward_classifier_features = getattr(
            self.model, "forward_classifier_features", None
        )
        classifier = getattr(self.model, "classifier", None)
        if not callable(forward_features) or not callable(classifier):
            raise TypeError(
                "Ours Gaussian replay MLP requires forward_features and classifier"
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
                "Ours Gaussian replay feature dimension does not match the source head"
            )
        if not np.all(np.isfinite(feature_values)):
            raise FloatingPointError(
                "Ours Gaussian replay received non-finite CLIP features"
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
                "Ours Gaussian replay MLP produced a non-finite residual"
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
                    "Ours Gaussian replay requires both pseudo-classes"
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

    def _after_gaussian_replay_head_update(
        self,
        state: dict[str, Any],
        head: Any,
        features: Any,
        source_margin: Any,
        labels: Any,
    ) -> None:
        del state, head, features, source_margin, labels

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
            raise ValueError("Ours Gaussian replay requires positive replay samples")
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
            raise RuntimeError("Ours Gaussian replay MLP lost its optimizer")

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
                "Ours Gaussian replay requires a positive replay minibatch"
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
                    "Ours Gaussian replay MLP produced a non-finite loss"
                )
            loss.backward()
            optimizer.step()
            weighted_loss += float(loss.detach().cpu().item()) * float(
                stop - start
            )
            optimizer_steps += 1
        head.eval()
        self._after_gaussian_replay_head_update(
            state,
            head,
            feature_tensor,
            source_tensor,
            label_tensor,
        )
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
                "Ours Gaussian replay received invalid pseudo-supervision"
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
        if not self._gaussian_replay_parameter_update_enabled():
            return False
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

    def _gaussian_replay_parameter_update_enabled(self) -> bool:
        return True

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
        stats = _OrdinalExpertReadout._state_stats(self)
        stats.update(self._gaussian_replay_state_stats())
        return stats

    def predict(self, images: Any) -> PredictionBatch:
        scores, features = self._batch_scores_and_gaussian_replay_features(images)
        self._ordinal_ridge_precomputed_scores = scores
        try:
            ordinal = _OrdinalExpertReadout.predict(
                self,
                images,
            )
        finally:
            self._ordinal_ridge_precomputed_scores = None
            self._feature_route_query = None
        if self._pending is None:
            raise RuntimeError(
                "Ours Gaussian replay MLP lost the routed prediction state"
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
                    "Ours Gaussian replay MLP lost its selected expert state"
                )
            if mixture is None:
                raise RuntimeError(
                    "Ours Gaussian replay MLP lost its selected GMM"
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
                "prediction_gaussian_replay_fake_count": int(
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
            return _OrdinalExpertReadout.adapt(self, images)
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
                    "Ours Gaussian replay MLP lost its prediction features"
                )
            if mixture is None:
                raise RuntimeError(
                    "Ours Gaussian replay MLP lost its selected GMM"
                )
            if state is None:
                state = self._ensure_ordinal_ridge_state(assignment)
            updated = self._update_gaussian_replay_state(
                state,
                mixture,
                scores,
                features,
            )

        stats = _OrdinalExpertReadout.adapt(self, images)
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
        _OrdinalExpertReadout.discard_pending_prediction(
            self
        )


class _BalancedGaussianReplayExperts(
    _GaussianReplayExperts
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
                "Ours expanded Gaussian replay samples must be a positive even "
                "integer"
            )


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


class _CLIPExpertMemory(
    _BalancedGaussianReplayExperts
):
    """Recall historical experts only through frozen CLIP feature routing."""


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


class Ours(
    _CLIPExpertMemory
):
    """Shrink feature ranking while re-solving one expert intercept."""

    def _reset_state(self) -> None:
        self.ablation_mode = str(
            self.config.get("ablation_mode", "none")
        ).lower().replace("-", "_")
        allowed_ablations = {"none", "no_detect", "no_route", "no_update"}
        if self.ablation_mode not in allowed_ablations:
            raise ValueError(
                "Ours ablation_mode must be one of none, no_detect, no_route, "
                "or no_update"
            )
        self.readout_mode = str(
            self.config.get("readout_mode", "calibrated")
        ).lower()
        if self.readout_mode not in {"base", "calibrated"}:
            raise ValueError("Ours readout_mode must be base or calibrated")
        if self.readout_mode == "base":
            self.feature_residual_scale = 1.0
        else:
            self.feature_residual_scale = float(
                self.config.get("feature_residual_scale", 0.75)
            )
            if not math.isclose(
                self.feature_residual_scale, 0.75, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    "The final Ours calibrated readout fixes feature_residual_scale at 0.75"
                )
        super()._reset_state()
        if self.ablation_mode != "none" and (
            self.readout_mode != "calibrated" or self.adaptation_mode != "full"
        ):
            raise ValueError(
                "Ours component ablations require the full calibrated method"
            )
        self.intercept_refit_updates = 0
        self.intercept_refit_last_bias = 0.0
        self.intercept_refit_last_learned_bias = 0.0
        self.intercept_refit_last_bias_delta = 0.0
        self.intercept_refit_last_balance_error = 0.0

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        calibrated = self.readout_mode == "calibrated"
        component_ablation = (
            None if self.ablation_mode == "none" else self.ablation_mode
        )
        return {
            "protocol": self.protocol_name,
            "research_name": "Ours",
            "internal_version": "R47" if calibrated else "R37",
            "readout_mode": self.readout_mode,
            "adaptation_mode": self.adaptation_mode,
            "target_labels_used": False,
            "generator_boundaries_used": False,
            "source_model_frozen": True,
            "component_ablation": component_ablation,
            "segmentation": (
                "disabled_single_continuous_score_state"
                if self.ablation_mode == "no_detect"
                else "causal_bic_score_segments"
            ),
            "pseudo_supervision": "selected_expert_equal_prior_gmm",
            "expert_routing": (
                "disabled_active_expert_only"
                if self.ablation_mode == "no_route"
                else "frozen_clip_feature_similarity"
            ),
            "expert_memory": (
                "routing_prototype_only_no_class_statistics_or_residual_update"
                if self.ablation_mode == "no_update"
                else "class_conditional_diagonal_gaussians_and_residual_mlp"
            ),
            "replay_rule": "128_real_and_128_fake_fresh_samples_per_update",
            "training_objective": "balanced_bce_on_source_plus_residual",
            "feature_residual_scale": self.feature_residual_scale,
            "intercept_refit": calibrated,
            "intercept_refit_solver": (
                "deterministic_monotone_bisection_on_current_balanced_replay"
                if calibrated
                else "none"
            ),
            "target_selected_hyperparameters": 1 if calibrated else 0,
            "formal_status": (
                "component_ablation"
                if component_ablation is not None
                else "final_method_fixed_pending_formal_seed_validation"
                if calibrated
                else "retained_readout_ablation"
            ),
        }


    @property
    def _prediction_mode_name(self) -> str:
        suffix = "" if self.ablation_mode == "none" else f"_{self.ablation_mode}"
        return f"ours_{self.readout_mode}_readout{suffix}"

    def _detect_segment_change(self) -> None:
        if self.ablation_mode == "no_detect":
            return
        super()._detect_segment_change()

    def _routing_candidates(self, scores: np.ndarray) -> list[dict[str, Any]]:
        candidates = super()._routing_candidates(scores)
        if self.ablation_mode == "no_route":
            return [
                candidate
                for candidate in candidates
                if candidate["expert"] == "active_learning_state"
            ]
        return candidates

    def _gaussian_replay_parameter_update_enabled(self) -> bool:
        return self.ablation_mode != "no_update"

    def _after_gaussian_replay_head_update(
        self,
        state: dict[str, Any],
        head: Any,
        features: Any,
        source_margin: Any,
        labels: Any,
    ) -> None:
        if self.readout_mode == "base":
            return super()._after_gaussian_replay_head_update(
                state, head, features, source_margin, labels
            )

        import torch
        import torch.nn.functional as functional

        if len(head) != 3 or not hasattr(head[2], "weight"):
            raise TypeError("Ours calibrated readout requires a hidden residual MLP")
        with torch.no_grad():
            hidden = head[1](head[0](features))
            feature_residual = functional.linear(
                hidden,
                head[2].weight,
                bias=None,
            ).reshape(-1)
            fixed_margin = source_margin + self.feature_residual_scale * feature_residual
            lower = -float(torch.max(fixed_margin).cpu().item()) - 32.0
            upper = -float(torch.min(fixed_margin).cpu().item()) + 32.0
            for _ in range(64):
                midpoint = 0.5 * (lower + upper)
                balance = torch.mean(torch.sigmoid(fixed_margin + midpoint) - labels)
                if float(balance.cpu().item()) > 0.0:
                    upper = midpoint
                else:
                    lower = midpoint
            calibrated_bias = 0.5 * (lower + upper)
            balance_error = float(
                torch.mean(torch.sigmoid(fixed_margin + calibrated_bias) - labels)
                .cpu()
                .item()
            )
        learned_bias = float(head[2].bias.detach().cpu().reshape(-1)[0].item())
        head_state = self._gaussian_replay_head_state(state)
        head_state["calibrated_prediction_bias"] = calibrated_bias
        head_state["calibrated_prediction_scale"] = self.feature_residual_scale
        self.intercept_refit_updates += 1
        self.intercept_refit_last_bias = calibrated_bias
        self.intercept_refit_last_learned_bias = learned_bias
        self.intercept_refit_last_bias_delta = calibrated_bias - learned_bias
        self.intercept_refit_last_balance_error = balance_error

    def _gaussian_replay_prediction_residual(
        self,
        state: dict[str, Any],
        features: np.ndarray,
        scores: np.ndarray,
        mixture: dict[str, Any],
    ) -> np.ndarray:
        if self.readout_mode == "base":
            return super()._gaussian_replay_prediction_residual(
                state, features, scores, mixture
            )
        full_residual = self._gaussian_replay_residual(state, features)
        head_state = self._gaussian_replay_head_state(state)
        head = head_state.get("mlp_head")
        if head is None or len(head) != 3 or head[2].bias is None:
            raise TypeError("Ours calibrated readout requires the standard MLP bias")
        learned_bias = float(head[2].bias.detach().cpu().reshape(-1)[0].item())
        calibrated_bias = float(
            head_state.get("calibrated_prediction_bias", learned_bias)
        )
        residual = calibrated_bias + self.feature_residual_scale * (
            full_residual - learned_bias
        )
        if not np.all(np.isfinite(residual)):
            raise FloatingPointError("Ours produced a non-finite residual")
        return residual

    def _state_stats(self) -> dict[str, Any]:
        stats = super()._state_stats()
        stats.update(
            {
                "ablation_mode": self.ablation_mode,
                "detect_enabled": self.ablation_mode != "no_detect",
                "route_enabled": self.ablation_mode != "no_route",
                "update_enabled": self.ablation_mode != "no_update",
                "readout_mode": self.readout_mode,
                "feature_residual_scale": self.feature_residual_scale,
                "intercept_refit_updates": self.intercept_refit_updates,
                "intercept_refit_last_bias": self.intercept_refit_last_bias,
                "intercept_refit_last_learned_bias": (
                    self.intercept_refit_last_learned_bias
                ),
                "intercept_refit_last_bias_delta": (
                    self.intercept_refit_last_bias_delta
                ),
                "intercept_refit_last_balance_error": (
                    self.intercept_refit_last_balance_error
                ),
            }
        )
        return stats

__all__ = [
    "Ours",
    "binary_score",
    "fit_gaussian_ml",
    "fit_gmm_bic",
    "fit_temperature",
    "validate_score_anchors",
]
