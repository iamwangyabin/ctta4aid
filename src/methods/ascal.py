"""ASCAL: Anchored Score Calibration for continual decision-layer adaptation.

The detector's parameters stay frozen during deployment. ASCAL treats the
binary score margin ``s = logit_fake - logit_real`` as the observable and
maintains an online estimate of the real/fake score distributions: a single
Gaussian for real samples and a GMM for fake samples (multiple generators).
Every estimate is a windowed MAP update whose prior always points at the
source-calibrated anchors, never at the method's own history. Gate checks
decide when a window is not learned at all.

The anchors (temperature, Gaussian/GMM parameters, admission thresholds) are
computed offline on the source validation split by ``train_source.py`` and
stored in the source checkpoint metadata. Target hidden labels never enter
this class.
"""

from __future__ import annotations

import math
from copy import deepcopy
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


def normal_density(values: Any, mu: float, sigma: float) -> Any:
    sigma = max(float(sigma), 1e-8)
    z = (np.asarray(values, dtype=np.float64) - float(mu)) / sigma
    return np.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def gmm_density(values: Any, weights: Any, mus: Any, sigmas: Any) -> Any:
    values = np.asarray(values, dtype=np.float64)
    density = np.zeros_like(values)
    for weight, mu, sigma in zip(weights, mus, sigmas):
        density += float(weight) * normal_density(values, float(mu), float(sigma))
    return density


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
    theta_a = float(anchors.get("theta_a", -1.0))
    theta_q = float(anchors.get("theta_q", 2.0))
    if theta_a < 0.0:
        raise ValueError("score_anchors.theta_a must be non-negative")
    if not -1.0 <= theta_q <= 1.0:
        raise ValueError("score_anchors.theta_q must be in [-1, 1]")
    return {
        **anchors,
        "temperature": temperature,
        "real": {"mu": real_mu, "sigma": real_sigma},
        "fake": {"weights": weights, "mus": mus, "sigmas": sigmas},
        "theta_a": theta_a,
        "theta_q": theta_q,
    }


class ASCAL(TTAMethod):
    """Anchored score-distribution memory with gated windowed MAP updates."""

    protocol_name = "predict_then_adapt"

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        config.setdefault("capture_initial_state", False)
        super().__init__(model, device, config)
        self.model.eval()
        self.model.requires_grad_(False)

        self.adaptation_mode = str(self.config.get("adaptation_mode", "full")).lower()
        if self.adaptation_mode not in {"static", "full"}:
            raise ValueError("ASCAL adaptation_mode must be static or full")

        self.anchors = validate_score_anchors(self.config.get("score_anchors"))
        self.temperature = float(self.config.get("temperature", self.anchors["temperature"]))
        if self.temperature <= 0.0:
            raise ValueError("ASCAL temperature must be positive")
        self.theta_a = float(self.config.get("theta_a", self.anchors["theta_a"]))
        self.theta_q = float(self.config.get("theta_q", self.anchors["theta_q"]))
        self.kappa = float(self.config.get("anchor_kappa", self.anchors.get("anchor_kappa", 100.0)))
        self.fuse_multiplier = float(
            self.config.get(
                "fuse_sigma_multiplier", self.anchors.get("fuse_sigma_multiplier", 3.0)
            )
        )
        if self.kappa < 0.0:
            raise ValueError("ASCAL anchor_kappa must be non-negative")
        if self.fuse_multiplier <= 0.0:
            raise ValueError("ASCAL fuse_sigma_multiplier must be positive")
        if self.theta_a < 0.0 or not -1.0 <= self.theta_q <= 1.0:
            raise ValueError("ASCAL admission thresholds out of range")

        self.memory_capacity = int(self.config.get("memory_capacity", 2000))
        self.window_size = int(self.config.get("window_size", 200))
        if self.window_size < 2 or self.memory_capacity < self.window_size:
            raise ValueError("ASCAL requires memory_capacity >= window_size >= 2")
        self.admission_rate_floor = float(self.config.get("admission_rate_floor", 0.5))
        if not 0.0 <= self.admission_rate_floor <= 1.0:
            raise ValueError("ASCAL admission_rate_floor must be in [0, 1]")
        gates = dict(self.config.get("gates", {}) or {})
        self.gate_bimodality = bool(gates.get("bimodality", True))
        self.gate_admission = bool(gates.get("admission", True))
        self.gate_drift_fuse = bool(gates.get("drift_fuse", True))
        self.min_sigma = float(self.config.get("min_sigma", 1e-3))
        if self.min_sigma <= 0.0:
            raise ValueError("ASCAL min_sigma must be positive")

        self.lambda_initial = float(self.config.get("lambda_initial", 1.0))
        self.lambda_min = float(self.config.get("lambda_min", 0.5))
        self.lambda_min_entries = int(self.config.get("lambda_min_entries_per_class", 64))
        self.lambda_stable_windows = int(self.config.get("lambda_stable_windows", 2))
        self.lambda_anneal_windows = int(self.config.get("lambda_anneal_windows", 4))
        if not 0.0 <= self.lambda_min <= self.lambda_initial <= 1.0:
            raise ValueError("ASCAL requires 0 <= lambda_min <= lambda_initial <= 1")
        if min(
            self.lambda_min_entries,
            self.lambda_stable_windows,
            self.lambda_anneal_windows,
        ) < 1:
            raise ValueError("ASCAL lambda schedule parameters must be positive")
        self.pi_prior_strength = float(self.config.get("pi_prior_strength", 50.0))
        if self.pi_prior_strength < 0.0:
            raise ValueError("ASCAL pi_prior_strength must be non-negative")

        self.views = int(self.config.get("views", 5))
        if self.views < 1:
            raise ValueError("ASCAL views must be positive")
        if self.views > 1:
            from src.data.transforms import CLIP_MEAN, CLIP_STD
            from src.data.views import ASCALViewTransform

            self.input_transform = ASCALViewTransform(
                views=self.views,
                image_size=int(self.config.get("image_size", 224)),
                resize_size=int(self.config.get("resize_size", 256)),
                mean=CLIP_MEAN,
                std=CLIP_STD,
                jpeg_qualities=tuple(self.config.get("jpeg_qualities", (75, 85, 95))),
            )

        self._initial_params = {
            "real_mu": self.anchors["real"]["mu"],
            "real_sigma": self.anchors["real"]["sigma"],
            "fake_weights": np.asarray(self.anchors["fake"]["weights"], dtype=np.float64),
            "fake_mus": np.asarray(self.anchors["fake"]["mus"], dtype=np.float64),
            "fake_sigmas": np.asarray(self.anchors["fake"]["sigmas"], dtype=np.float64),
        }
        self._pending: dict[str, Any] | None = None
        self._reset_state()

    def _reset_state(self) -> None:
        self._params = deepcopy(self._initial_params)
        self.queue: list[dict[str, Any]] = []
        self._window_entries: list[dict[str, Any]] = []
        self._window_arrived = 0
        self._clock = 0
        self.lambda_value = self.lambda_initial
        self._consecutive_accepted = 0
        self._updates = 0
        self._frozen_windows = 0
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
            "adaptive_role": "score_distribution_memory_only_no_parameter_gradients",
            "anchor_rule": "map_prior_always_points_at_source_anchors",
            "intentional_changes": [
                "decision-layer adaptation: the detector stays frozen during deployment",
                "window updates are gated by bimodality, admission rate, and a 3-sigma drift fuse",
                "predictions fuse source and memory scores with a lambda that never drops its source fallback",
            ],
        }

    # ------------------------------------------------------------- prediction

    def _batch_scores(self, images: Any) -> tuple[Any, Any]:
        import torch

        if images.dim() == 5:
            batch, views = int(images.shape[0]), int(images.shape[1])
            flat = images.reshape(batch * views, *images.shape[2:])
        elif images.dim() == 4:
            batch, views = int(images.shape[0]), 1
            flat = images
        else:
            raise ValueError("ASCAL expects (B, C, H, W) or (B, V, C, H, W) images")
        with torch.no_grad():
            logits = self.model(flat.to(self.device, non_blocking=True))
        margins = (
            binary_score(logits)
            .view(batch, views)
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        scores = margins.mean(axis=1)
        consistency = margins.var(axis=1) if views > 1 else np.zeros(batch)
        return scores, consistency

    def _class_priors(self) -> tuple[float, float]:
        real_count = sum(item["pseudo_label"] == 0 for item in self.queue)
        fake_count = len(self.queue) - real_count
        total = real_count + fake_count + 2.0 * self.pi_prior_strength
        return (
            (real_count + self.pi_prior_strength) / total,
            (fake_count + self.pi_prior_strength) / total,
        )

    def _memory_posterior(self, scores: Any) -> Any:
        params = self._params
        density_real = normal_density(scores, params["real_mu"], params["real_sigma"])
        density_fake = gmm_density(
            scores, params["fake_weights"], params["fake_mus"], params["fake_sigmas"]
        )
        prior_real, prior_fake = self._class_priors()
        numerator = prior_fake * density_fake
        denominator = prior_real * density_real + numerator
        return np.where(denominator > 0.0, numerator / np.maximum(denominator, 1e-300), 0.5)

    def predict(self, images: Any) -> PredictionBatch:
        import torch

        scores, consistency = self._batch_scores(images)
        p_source = _sigmoid(scores / self.temperature)
        if self.adaptation_mode == "static":
            probability = p_source
        else:
            probability = (
                self.lambda_value * p_source
                + (1.0 - self.lambda_value) * self._memory_posterior(scores)
            )
        probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
        logit_margin = np.log(probability / (1.0 - probability))
        logits = torch.from_numpy(
            np.stack([-0.5 * logit_margin, 0.5 * logit_margin], axis=1)
        ).float()
        prob_fake = torch.from_numpy(probability).float()
        self._pending = {
            "scores": scores,
            "quality": np.abs(probability - 0.5),
            "consistency": consistency,
            "pseudo_label": (probability >= 0.5).astype(np.int64),
        }
        return PredictionBatch(
            logits=logits,
            prob_fake=prob_fake,
            pred_label=(prob_fake >= 0.5).long(),
        )

    def discard_pending_prediction(self) -> None:
        self._pending = None

    # ------------------------------------------------------------- adaptation

    def _memory_ready(self) -> bool:
        real_count = sum(item["pseudo_label"] == 0 for item in self.queue)
        fake_count = len(self.queue) - real_count
        return (
            real_count >= self.lambda_min_entries
            and fake_count >= self.lambda_min_entries
        )

    def _assign_window(self, scores: Any) -> tuple[Any, Any]:
        """Return real posterior and fake component assignment under current params."""

        params = self._params
        density_real = normal_density(scores, params["real_mu"], params["real_sigma"])
        component_densities = [
            normal_density(scores, mu, sigma)
            for mu, sigma in zip(params["fake_mus"], params["fake_sigmas"])
        ]
        density_fake = sum(
            weight * density
            for weight, density in zip(params["fake_weights"], component_densities)
        )
        prior_real, prior_fake = self._class_priors()
        denominator = prior_real * density_real + prior_fake * density_fake
        posterior_real = np.where(
            denominator > 0.0,
            prior_real * density_real / np.maximum(denominator, 1e-300),
            0.5,
        )
        stacked = np.stack(component_densities, axis=0)
        assignment = np.argmax(stacked, axis=0)
        return posterior_real, assignment

    def _anchored_gaussian(
        self, values: Any, weights: Any, anchor_mu: float, anchor_sigma: float
    ) -> tuple[float, float] | None:
        weight_sum = float(np.sum(weights))
        if values.size == 0 or weight_sum <= 0.0:
            return None
        mu = (self.kappa * anchor_mu + float(np.sum(weights * values))) / (
            self.kappa + weight_sum
        )
        variance = (
            self.kappa * anchor_sigma**2 + float(np.sum(weights * (values - mu) ** 2))
        ) / (self.kappa + weight_sum)
        return mu, max(math.sqrt(max(variance, 0.0)), self.min_sigma)

    def _window_bics(self, scores: Any) -> tuple[float, float]:
        from sklearn.mixture import GaussianMixture

        column = np.asarray(scores, dtype=np.float64).reshape(-1, 1)
        bics = []
        for components in (1, 2):
            model = GaussianMixture(
                n_components=components,
                covariance_type="full",
                reg_covar=1e-6,
                n_init=2,
                max_iter=200,
                random_state=0,
            )
            model.fit(column)
            bics.append(float(model.bic(column)))
        return bics[0], bics[1]

    def _candidate_params(self, scores: Any, qualities: Any) -> dict[str, Any]:
        anchors = self.anchors
        posterior_real, assignment = self._assign_window(scores)
        real_mask = posterior_real >= 0.5
        real = self._anchored_gaussian(
            scores[real_mask],
            qualities[real_mask],
            anchors["real"]["mu"],
            anchors["real"]["sigma"],
        )
        fake_mus = list(self._params["fake_mus"])
        fake_sigmas = list(self._params["fake_sigmas"])
        fake_weights = list(self._params["fake_weights"])
        fake_mask = ~real_mask
        fake_count = int(fake_mask.sum())
        component_counts = []
        for index, (anchor_mu, anchor_sigma) in enumerate(
            zip(anchors["fake"]["mus"], anchors["fake"]["sigmas"])
        ):
            component_mask = fake_mask & (assignment == index)
            count = int(component_mask.sum())
            component_counts.append(count)
            updated = self._anchored_gaussian(
                scores[component_mask],
                qualities[component_mask],
                anchor_mu,
                anchor_sigma,
            )
            if updated is not None:
                fake_mus[index], fake_sigmas[index] = updated
            fake_weights[index] = (
                self.kappa * anchors["fake"]["weights"][index] + count
            ) / (self.kappa + max(fake_count, 1))
        return {
            "real": real,
            "fake_mus": fake_mus,
            "fake_sigmas": fake_sigmas,
            "fake_weights": fake_weights,
            "window_real_samples": int(real_mask.sum()),
            "window_fake_samples": fake_count,
            "component_counts": component_counts,
        }

    def _drift_within_fuse(self, candidates: dict[str, Any]) -> bool:
        anchors = self.anchors
        if candidates["real"] is not None:
            anchor_mu = anchors["real"]["mu"]
            fuse = self.fuse_multiplier * anchors["real"]["sigma"]
            if abs(candidates["real"][0] - anchor_mu) > fuse:
                return False
        for index, count in enumerate(candidates["component_counts"]):
            if count == 0:
                continue
            anchor_mu = anchors["fake"]["mus"][index]
            fuse = self.fuse_multiplier * anchors["fake"]["sigmas"][index]
            if abs(candidates["fake_mus"][index] - anchor_mu) > fuse:
                return False
        return True

    def _update_from_window(self) -> dict[str, Any]:
        entries = self._window_entries
        arrived = self._window_arrived
        self._window_entries = []
        self._window_arrived = 0
        scores = np.asarray([entry["score"] for entry in entries], dtype=np.float64)
        qualities = np.asarray([entry["quality"] for entry in entries], dtype=np.float64)
        report: dict[str, Any] = {
            "window_entries": len(entries),
            "window_arrived": arrived,
            "admission_rate": len(entries) / max(arrived, 1),
        }
        frozen = False
        if self.gate_admission and report["admission_rate"] < self.admission_rate_floor:
            report["freeze_reason"] = "admission_rate"
            frozen = True
        if not frozen and self.gate_bimodality:
            bic_single, bic_pair = self._window_bics(scores)
            report["bic_single"] = bic_single
            report["bic_pair"] = bic_pair
            if not bic_pair < bic_single:
                report["freeze_reason"] = "no_bimodal_structure"
                frozen = True
        candidates = None
        if not frozen:
            candidates = self._candidate_params(scores, qualities)
            if candidates["real"] is None or candidates["window_fake_samples"] == 0:
                report["freeze_reason"] = "single_class_window"
                frozen = True
        if not frozen and self.gate_drift_fuse and not self._drift_within_fuse(candidates):
            report["freeze_reason"] = "drift_fuse"
            frozen = True

        if frozen:
            self._frozen_windows += 1
            self._consecutive_accepted = 0
            report["accepted"] = False
            return report

        self._params["real_mu"], self._params["real_sigma"] = candidates["real"]
        self._params["fake_mus"] = np.asarray(candidates["fake_mus"], dtype=np.float64)
        self._params["fake_sigmas"] = np.asarray(candidates["fake_sigmas"], dtype=np.float64)
        self._params["fake_weights"] = np.asarray(
            candidates["fake_weights"], dtype=np.float64
        )
        self._updates += 1
        self._consecutive_accepted += 1
        if (
            self._memory_ready()
            and self._consecutive_accepted >= self.lambda_stable_windows
            and self.lambda_value > self.lambda_min
        ):
            step = (self.lambda_initial - self.lambda_min) / self.lambda_anneal_windows
            self.lambda_value = max(self.lambda_min, self.lambda_value - step)
        report["accepted"] = True
        report["real_mu"] = float(self._params["real_mu"])
        report["real_sigma"] = float(self._params["real_sigma"])
        report["fake_mus"] = [float(value) for value in self._params["fake_mus"]]
        report["window_real_samples"] = candidates["window_real_samples"]
        report["window_fake_samples"] = candidates["window_fake_samples"]
        return report

    def adapt(self, images: Any) -> AdaptationStats:
        if self.adaptation_mode == "static":
            self._pending = None
            return AdaptationStats(
                selected=0,
                extra={
                    "adaptation_mode": "static",
                    "memory_entries": 0,
                    "lambda": 1.0,
                },
            )
        if self._pending is None:
            raise RuntimeError("ASCAL adapt requires a matching predict call")
        pending = self._pending
        self._pending = None

        admitted = 0
        arrived = int(pending["scores"].shape[0])
        self._window_arrived += arrived
        for index in range(arrived):
            quality = float(pending["quality"][index])
            consistency = float(pending["consistency"][index])
            if quality > self.theta_q and consistency <= self.theta_a:
                self.queue.append(
                    {
                        "score": float(pending["scores"][index]),
                        "quality": quality,
                        "consistency": consistency,
                        "pseudo_label": int(pending["pseudo_label"][index]),
                        "timestamp": self._clock + index,
                    }
                )
                self._window_entries.append(self.queue[-1])
                admitted += 1
        self._clock += arrived
        if len(self.queue) > self.memory_capacity:
            del self.queue[: len(self.queue) - self.memory_capacity]

        update: dict[str, Any] | None = None
        if len(self._window_entries) >= self.window_size:
            update = self._update_from_window()

        extra: dict[str, Any] = {
            "adaptation_mode": "full",
            "candidates_admitted": admitted,
            "memory_entries": len(self.queue),
            "lambda": self.lambda_value,
            "updates": self._updates,
            "frozen_windows": self._frozen_windows,
            "memory_ready": self._memory_ready(),
        }
        if update is not None:
            extra["window_accepted"] = update["accepted"]
            extra["window_freeze_reason"] = update.get("freeze_reason")
            extra["window_admission_rate"] = update["admission_rate"]
        return AdaptationStats(loss=None, selected=admitted, extra=extra)

    def reset(self) -> None:
        self._reset_state()


__all__ = [
    "ASCAL",
    "binary_score",
    "fit_gaussian_ml",
    "fit_gmm_bic",
    "fit_temperature",
    "gmm_density",
    "normal_density",
    "validate_score_anchors",
]
