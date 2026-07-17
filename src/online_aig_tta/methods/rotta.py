"""Framework wrapper around the pinned official RoTTA core."""

from __future__ import annotations

from typing import Any

from online_aig_tta.types import AdaptationStats, PredictionBatch

from .base import TTAMethod
from .utils import NormalizedInputTransform, build_optimizer


class RoTTA(TTAMethod):
    """Expose RobustBN/CSTU/teacher-student RoTTA through Predict-Then-Adapt."""

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(model, device, config)
        from online_aig_tta.official import rotta as official_rotta

        self.official_module = official_rotta
        official_rotta.configure_model(
            self.model, alpha=float(self.config.get("alpha", 0.05))
        )
        parameters, self.official_parameter_names = official_rotta.collect_params(
            self.model
        )
        if not parameters:
            raise RuntimeError("Official RoTTA did not collect RobustBN parameters")
        self.optimizer = build_optimizer(parameters, self.config)
        self.steps = int(self.config.get("steps", 1))
        if self.steps < 1:
            raise ValueError("Official RoTTA requires steps >= 1")
        self.core = official_rotta.RoTTA(
            self.model,
            self.optimizer,
            num_classes=int(self.config.get("num_classes", 2)),
            memory_size=int(self.config.get("memory_size", 64)),
            lambda_t=float(self.config.get("lambda_t", 1.0)),
            lambda_u=float(self.config.get("lambda_u", 1.0)),
            nu=float(self.config.get("nu", 0.001)),
            update_frequency=int(self.config.get("update_frequency", 64)),
            image_size=int(self.config.get("image_size", 224)),
            gaussian_std=float(self.config.get("gaussian_std", 0.005)),
            soft=bool(self.config.get("soft_augmentation", False)),
        )
        self.core.transform = NormalizedInputTransform(self.core.transform)
        self._pending_teacher_output: Any = None
        self._pending_batch_id: int | None = None

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "vendored_official_core_with_binary_protocol_wrapper",
            "official_commit": "67e34c900cdd355fc07e55edd4c577ea7b8ebcc9",
            "official_core": "online_aig_tta.official.rotta",
            "numerical_validation": "not_run_requires_official_data_and_weights",
            "protocol_wrapper": "predict_then_adapt",
            "intentional_changes": [
                "class count changed from CIFAR-10/100 to binary detection",
                "input resolution changed from CIFAR 32 to common AIGC 224",
                "torchvision interpolation and hard-coded CUDA compatibility patches",
                "ImageNet normalization bridge around official pixel augmentation",
                "EMA prediction cached before the official memory/update path",
            ],
        }

    def predict(self, images: Any) -> PredictionBatch:
        import torch

        device_images = images.to(self.device, non_blocking=True)
        with torch.no_grad():
            logits = self.core.teacher_prediction(device_images).detach()
            probabilities = logits.softmax(dim=1)
        self._pending_teacher_output = logits
        self._pending_batch_id = id(images)
        return PredictionBatch(
            logits=logits,
            prob_fake=probabilities[:, 1].detach(),
            pred_label=probabilities.argmax(dim=1).detach(),
        )

    def adapt(self, images: Any) -> AdaptationStats:
        original_batch_id = id(images)
        device_images = images.to(self.device, non_blocking=True)
        cached = (
            self._pending_teacher_output
            if self._pending_batch_id == original_batch_id
            else None
        )
        updates_before = self.core.update_count
        for step in range(self.steps):
            self.core.forward_and_adapt(
                device_images, ema_out=cached if step == 0 else None
            )
        self._pending_teacher_output = None
        self._pending_batch_id = None
        return AdaptationStats(
            loss=self.core.last_loss,
            selected=int(device_images.shape[0]),
            extra={
                "memory_occupancy": self.core.mem.get_occupancy(),
                "memory_per_class": self.core.mem.per_class_dist(),
                "optimizer_updates": self.core.update_count - updates_before,
                "instances_seen": self.core.current_instance,
            },
        )

    def reset(self) -> None:
        self.core.reset()
        self.core.transform = NormalizedInputTransform(self.core.transform)
        self._pending_teacher_output = None
        self._pending_batch_id = None

    def discard_pending_prediction(self) -> None:
        self._pending_teacher_output = None
        self._pending_batch_id = None
