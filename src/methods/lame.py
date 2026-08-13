"""Framework wrapper around the pinned authors' LAME core."""

from __future__ import annotations

from typing import Any

from src.types import AdaptationStats, PredictionBatch

from .base import TTAMethod


class LAME(TTAMethod):
    """Parameter-free batch output adaptation through Laplacian optimization."""

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(model, device, config)
        from src.official import lame as official_lame

        if not hasattr(self.model, "forward_features") or not hasattr(
            self.model, "classifier"
        ):
            raise RuntimeError(
                "LAME requires the framework detector's forward_features and classifier"
            )
        self.official_module = official_lame
        self.affinity_name = str(self.config.get("affinity", "rbf"))
        self.knn = int(self.config.get("knn", 5))
        self.sigma = float(self.config.get("sigma", 1.0))
        self.force_symmetry = bool(self.config.get("force_symmetry", False))
        self.bound_lambda = float(self.config.get("bound_lambda", 1.0))
        self.max_steps = int(self.config.get("max_steps", 100))
        self.affinity = official_lame.build_affinity(
            self.affinity_name, sigma=self.sigma, knn=self.knn
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.last_batch_guarded = False

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "vendored_official_core_with_protocol_wrapper",
            "official_commit": "d2e5f63090bc1c8129bf7cbd781029a5955e1a67",
            "official_core": "src.official.lame",
            "numerical_validation": "not_run_requires_official_data_and_weights",
            "protocol_wrapper": "parameter_free_output_adaptation_in_predict",
            "intentional_changes": [
                "Detectron2 model I/O replaced by framework feature/classifier I/O",
                "source prediction returned unchanged for a singleton batch to avoid undefined RBF bandwidth",
                "official hard-coded lambda and iteration cap exposed with unchanged defaults",
            ],
            "license": "CC-BY-NC-SA-4.0",
        }

    def predict(self, images: Any) -> PredictionBatch:
        import torch
        import torch.nn.functional as F

        device_images = images.to(self.device, non_blocking=True)
        with torch.no_grad():
            features = self.model.forward_features(device_images)
            logits = self.model.classifier(features)
            probabilities = logits.softmax(dim=1)
            self.last_batch_guarded = features.shape[0] < 2
            if not self.last_batch_guarded:
                normalized_features = F.normalize(features, p=2, dim=-1)
                unary = -torch.log(probabilities + 1e-10)
                kernel = self.affinity(normalized_features)
                if self.force_symmetry:
                    kernel = 0.5 * (kernel + kernel.t())
                probabilities = self.official_module.laplacian_optimization(
                    unary,
                    kernel,
                    bound_lambda=self.bound_lambda,
                    max_steps=self.max_steps,
                )
            adapted_logits = torch.log(probabilities.clamp_min(1e-20))
        return PredictionBatch(
            logits=adapted_logits.detach(),
            prob_fake=probabilities[:, 1].detach(),
            pred_label=probabilities.argmax(dim=1).detach(),
        )

    def adapt(self, images: Any) -> AdaptationStats:
        return AdaptationStats(
            selected=0,
            extra={
                "output_adaptation": True,
                "state_update": False,
                "singleton_identity_guard": self.last_batch_guarded,
            },
        )
