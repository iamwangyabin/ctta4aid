"""Framework wrapper around the pinned official CoTTA ImageNet core."""

from __future__ import annotations

from typing import Any

from src.types import AdaptationStats, PredictionBatch

from .base import TTAMethod
from .utils import NormalizedInputTransform, build_optimizer


class CoTTA(TTAMethod):
    """Expose authors' CoTTA core through Predict-Then-Adapt."""

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(model, device, config)
        from src.official import cotta as official_cotta

        variant = str(self.config.get("official_variant", "imagenet")).lower()
        if variant != "imagenet":
            raise ValueError("This wrapper pins the official CoTTA ImageNet branch")
        if not bool(self.config.get("symmetric_loss", True)):
            raise ValueError("Official CoTTA ImageNet requires symmetric_loss=true")
        if not bool(self.config.get("augment_only_if_low_confidence", True)):
            raise ValueError(
                "Official CoTTA ImageNet augments only below the anchor threshold"
            )
        self.official_module = official_cotta
        official_cotta.configure_model(self.model)
        parameters, self.official_parameter_names = official_cotta.collect_params(
            self.model
        )
        if not parameters:
            raise RuntimeError("Official CoTTA did not collect trainable parameters")
        self.optimizer = build_optimizer(parameters, self.config)
        self.core = official_cotta.CoTTA(
            self.model,
            self.optimizer,
            steps=int(self.config.get("steps", 1)),
            episodic=bool(self.config.get("episodic", False)),
            image_size=int(self.config.get("image_size", 224)),
            gaussian_std=float(self.config.get("gaussian_std", 0.005)),
            soft=bool(self.config.get("soft_augmentation", False)),
            augmentations=int(self.config.get("augmentations", 32)),
            anchor_confidence=float(self.config.get("anchor_confidence", 0.1)),
            ema_decay=float(self.config.get("ema_decay", 0.999)),
            restore_probability=float(
                self.config.get("restore_probability", 0.001)
            ),
        )
        self.core.transform = NormalizedInputTransform(self.core.transform)
        self._pending_teacher_target: Any = None
        self._pending_batch_id: int | None = None

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "vendored_official_core_with_protocol_wrapper",
            "official_commit": "c212a204b32be4005092e4323105a24a29ad2952",
            "official_core": "src.official.cotta",
            "official_variant": "imagenet",
            "numerical_validation": "not_run_requires_official_data_and_weights",
            "protocol_wrapper": "predict_then_adapt",
            "intentional_changes": [
                "modern torchvision interpolation argument",
                "device-safe stochastic restoration instead of hard-coded CUDA",
                "ImageNet normalization bridge around official pixel augmentation",
                "teacher prediction split and cached for Predict-Then-Adapt",
                "hard-coded official constants exposed without changing defaults",
            ],
        }

    def _augment(self, images: Any) -> Any:
        return self.core.transform(images)

    def predict(self, images: Any) -> PredictionBatch:
        import torch

        device_images = images.to(self.device, non_blocking=True)
        with torch.no_grad():
            logits = self.core.teacher_prediction(device_images).detach()
            probabilities = logits.softmax(dim=1)
        self._pending_teacher_target = logits
        self._pending_batch_id = id(images)
        return PredictionBatch(
            logits=logits,
            prob_fake=probabilities[:, 1].detach(),
            pred_label=probabilities.argmax(dim=1).detach(),
        )

    def adapt(self, images: Any) -> AdaptationStats:
        original_batch_id = id(images)
        device_images = images.to(self.device, non_blocking=True)
        teacher_target = None
        if self._pending_batch_id == original_batch_id:
            teacher_target = self._pending_teacher_target
        self._pending_teacher_target = None
        self._pending_batch_id = None

        for step in range(self.core.steps):
            self.core.forward_and_adapt(
                device_images,
                self.model,
                self.optimizer,
                outputs_ema=teacher_target if step == 0 else None,
            )
        return AdaptationStats(
            loss=self.core.last_loss,
            selected=int(device_images.shape[0]),
        )

    def reset(self) -> None:
        self.core.reset()
        self.core.transform = NormalizedInputTransform(
            self.official_module.get_tta_transforms(
                gaussian_std=float(self.config.get("gaussian_std", 0.005)),
                soft=bool(self.config.get("soft_augmentation", False)),
                image_size=int(self.config.get("image_size", 224)),
            )
        )
        self._pending_teacher_target = None
        self._pending_batch_id = None

    def discard_pending_prediction(self) -> None:
        self._pending_teacher_target = None
        self._pending_batch_id = None
