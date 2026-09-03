"""Framework adapter for TDA's online feature-cache adaptation."""

from __future__ import annotations

import math
from typing import Any

from src.types import AdaptationStats, PredictionBatch

from .base import TTAMethod


class TDA(TTAMethod):
    """Run the authors' batch-one cache update before each online prediction."""

    protocol_name = "online_adapt_then_predict"

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        config.setdefault("capture_initial_state", False)
        super().__init__(model, device, config)
        if not hasattr(self.model, "forward_features") or not hasattr(
            self.model, "classifier"
        ):
            raise TypeError("TDA requires a VLM with forward_features and classifier")

        self.model.eval()
        self.model.requires_grad_(False)
        self.positive_cache: dict[int, list[tuple[Any, float, Any | None]]] = {}
        self.negative_cache: dict[int, list[tuple[Any, float, Any | None]]] = {}
        self.positive_shot_capacity = int(self.config.get("positive_shot_capacity", 3))
        self.negative_shot_capacity = int(self.config.get("negative_shot_capacity", 2))
        self.positive_alpha = float(self.config.get("positive_alpha", 2.0))
        self.positive_beta = float(self.config.get("positive_beta", 5.0))
        self.negative_alpha = float(self.config.get("negative_alpha", 0.117))
        self.negative_beta = float(self.config.get("negative_beta", 1.0))
        self.entropy_threshold = tuple(
            float(value)
            for value in self.config.get("negative_entropy_threshold", (0.2, 0.5))
        )
        self.mask_threshold = tuple(
            float(value)
            for value in self.config.get("negative_mask_threshold", (0.03, 1.0))
        )
        if self.positive_shot_capacity < 1 or self.negative_shot_capacity < 1:
            raise ValueError("TDA cache capacities must be positive")
        if len(self.entropy_threshold) != 2 or len(self.mask_threshold) != 2:
            raise ValueError("TDA threshold ranges must contain lower and upper values")
        self._last_adaptation = AdaptationStats(selected=0)

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "vendored_official_core_with_protocol_wrapper",
            "official_commit": "e697fb0c8078cdeff93daa56bcf8860702542069",
            "official_core": "src.official.tda",
            "numerical_validation": "not_run_requires_framework_native_benchmark",
            "protocol_wrapper": self.protocol_name,
            "intentional_changes": [
                "official CLIP loader is replaced by the shared local OpenAI CLIP ViT-L/14 loader",
                "the official clean/custom-dataset contract is enforced as exactly "
                "one deterministic global view of one target sample per online step",
                "hard-coded CUDA and half casts are replaced by input-device and "
                "input-dtype handling",
                "the framework records Arrow sample identity and evaluator-only metrics",
            ],
        }

    def predict(self, images: Any) -> PredictionBatch:
        import torch

        from src.official import tda as official_tda

        if getattr(images, "ndim", None) != 4 or int(images.shape[0]) != 1:
            raise ValueError(
                "TDA requires exactly one target image per online step; distinct "
                "target images must never be interpreted as augmentation views"
            )

        device_images = images.to(self.device, non_blocking=True)
        with torch.no_grad():
            features = self.model.forward_features(device_images)
            logits = self.model.classifier(features)
            (
                cache_feature,
                _base_logits,
                loss,
                probability_map,
                prediction,
                selected_count,
            ) = official_tda.process_features(logits, features)
            official_tda.update_cache(
                self.positive_cache,
                prediction,
                cache_feature,
                loss,
                self.positive_shot_capacity,
            )
            normalized_entropy = float(loss.detach().mean().item()) / math.log2(
                int(logits.shape[1])
            )
            negative_selected = (
                self.entropy_threshold[0] < normalized_entropy < self.entropy_threshold[1]
            )
            if negative_selected:
                official_tda.update_cache(
                    self.negative_cache,
                    prediction,
                    cache_feature,
                    loss,
                    self.negative_shot_capacity,
                    probability_map,
                )

            final_logits = logits.clone()
            final_logits += official_tda.compute_cache_logits(
                cache_feature,
                self.positive_cache,
                alpha=self.positive_alpha,
                beta=self.positive_beta,
                classes=int(logits.shape[1]),
            )
            final_logits -= official_tda.compute_cache_logits(
                cache_feature,
                self.negative_cache,
                alpha=self.negative_alpha,
                beta=self.negative_beta,
                classes=int(logits.shape[1]),
                negative_mask_thresholds=(self.mask_threshold[0], self.mask_threshold[1]),
            )
            probabilities = final_logits.softmax(dim=1)

        self._last_adaptation = AdaptationStats(
            loss=float(loss.detach().mean().item()),
            selected=selected_count,
            extra={
                "adaptation_inside_predict": True,
                "positive_cache_entries": sum(len(items) for items in self.positive_cache.values()),
                "negative_cache_entries": sum(len(items) for items in self.negative_cache.values()),
                "negative_cache_updated": negative_selected,
                "normalized_entropy": normalized_entropy,
            },
        )
        return PredictionBatch(
            logits=final_logits.detach(),
            prob_fake=probabilities[:, 1].detach(),
            pred_label=probabilities.argmax(dim=1).detach(),
        )

    def adapt(self, images: Any) -> AdaptationStats:
        del images
        stats = self._last_adaptation
        self._last_adaptation = AdaptationStats(selected=0)
        return stats

    def discard_pending_prediction(self) -> None:
        self._last_adaptation = AdaptationStats(selected=0)

    def reset(self) -> None:
        self.positive_cache.clear()
        self.negative_cache.clear()
        self._last_adaptation = AdaptationStats(selected=0)
