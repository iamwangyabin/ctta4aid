"""Framework wrapper around the pinned official RoTTA core."""

from __future__ import annotations

from typing import Any

from src.types import AdaptationStats, PredictionBatch

from .base import TTAMethod
from .utils import (
    NormalizedInputTransform,
    build_optimizer,
    model_input_normalization,
    select_clip_visual_norm_parameters,
)


def _layernorm_core_class(official_rotta: Any) -> type:
    class RoTTALayerNormCore(official_rotta.RoTTA):
        """Preserve the RoTTA update while accumulating its memory loss."""

        def __init__(self, *args: Any, update_micro_batch_size: int, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.update_micro_batch_size = update_micro_batch_size
            self.last_update_microbatches = 0

        def reset(self) -> None:
            super().reset()
            self.last_update_microbatches = 0

        def update_model(self) -> None:
            import torch

            self.model.train()
            self.model_ema.train()
            memory_data, ages = self.mem.get_memory()
            if not memory_data:
                self.last_update_microbatches = 0
                self.update_ema_variables(self.model_ema, self.model, self.nu)
                return

            memory_data = torch.stack(memory_data)
            strong_augmented = self.transform(memory_data)
            weights = official_rotta.timeliness_reweighting(
                ages, memory_data.device
            )
            sample_count = int(memory_data.shape[0])
            self.optimizer.zero_grad()
            total_loss = torch.zeros((), device=memory_data.device)
            self.last_update_microbatches = 0
            for start in range(0, sample_count, self.update_micro_batch_size):
                stop = min(start + self.update_micro_batch_size, sample_count)
                with torch.no_grad():
                    ema_output = self.model_ema(memory_data[start:stop])
                student_output = self.model(strong_augmented[start:stop])
                loss_sum = (
                    official_rotta.softmax_entropy(student_output, ema_output)
                    * weights[start:stop]
                ).sum()
                (loss_sum / sample_count).backward()
                total_loss += loss_sum.detach()
                self.last_update_microbatches += 1
            self.optimizer.step()
            self.last_loss = float((total_loss / sample_count).cpu())
            self.update_count += 1
            self.update_ema_variables(self.model_ema, self.model, self.nu)

    return RoTTALayerNormCore


class RoTTA(TTAMethod):
    """Expose RoTTA through Predict-Then-Adapt, with an explicit CLIP-LN port."""

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(model, device, config)
        from src.official import rotta as official_rotta

        self.official_module = official_rotta
        self.clip_visual_layernorm = bool(
            self.config.get("clip_visual_layernorm", False)
        )
        self.normalization_mapping = None
        if self.clip_visual_layernorm:
            self.model.train()
            self.model.requires_grad_(False)
            parameters, self.official_parameter_names = (
                select_clip_visual_norm_parameters(self.model)
            )
            self.normalization_mapping = (
                "RobustBatchNorm_to_CLIP_visual_LayerNorm_affine"
            )
        else:
            official_rotta.configure_model(
                self.model, alpha=float(self.config.get("alpha", 0.05))
            )
            parameters, self.official_parameter_names = official_rotta.collect_params(
                self.model
            )
        if not parameters:
            raise RuntimeError("RoTTA did not collect adaptation parameters")
        self.optimizer = build_optimizer(parameters, self.config)
        self.steps = int(self.config.get("steps", 1))
        if self.steps < 1:
            raise ValueError("Official RoTTA requires steps >= 1")
        configured_micro_batch_size = self.config.get(
            "update_micro_batch_size", 2 if self.clip_visual_layernorm else None
        )
        self.update_micro_batch_size = (
            None
            if configured_micro_batch_size is None
            else int(configured_micro_batch_size)
        )
        if self.update_micro_batch_size is not None and self.update_micro_batch_size < 1:
            raise ValueError("RoTTA update_micro_batch_size must be positive")
        core_class = (
            _layernorm_core_class(official_rotta)
            if self.update_micro_batch_size is not None
            else official_rotta.RoTTA
        )
        core_kwargs = {
            "update_micro_batch_size": self.update_micro_batch_size
        } if self.update_micro_batch_size is not None else {}
        self.core = core_class(
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
            **core_kwargs,
        )
        self._input_mean, self._input_std = model_input_normalization(self.model)
        self._bridge_pixel_transform()
        self._pending_teacher_output: Any = None
        self._pending_batch_id: int | None = None

    def _bridge_pixel_transform(self) -> None:
        self.core.transform = NormalizedInputTransform(
            self.core.transform,
            mean=self._input_mean,
            std=self._input_std,
        )

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        layernorm_changes = (
            [
                "RobustBN is unavailable in ViT-L/14 and is explicitly replaced "
                "by OpenAI CLIP visual LayerNorm affine adaptation",
                "RobustBN source/target statistic interpolation has no LayerNorm "
                "equivalent and is therefore absent",
                "the unchanged full CSTU weighted-mean loss is accumulated in "
                "execution microbatches before one optimizer step and EMA update",
            ]
            if self.clip_visual_layernorm
            else []
        )
        return {
            "level": (
                "vendored_official_core_with_explicit_layernorm_transfer"
                if self.clip_visual_layernorm
                else "vendored_official_core_with_binary_protocol_wrapper"
            ),
            "official_commit": "67e34c900cdd355fc07e55edd4c577ea7b8ebcc9",
            "official_core": "src.official.rotta",
            "numerical_validation": "not_run_requires_official_data_and_weights",
            "protocol_wrapper": "predict_then_adapt",
            "reported_method_name": (
                "RoTTA-LN" if self.clip_visual_layernorm else "RoTTA"
            ),
            "stream_batch_size": int(
                self.config.get("data", {}).get("batch_size", 64)
            ),
            "update_micro_batch_size": self.update_micro_batch_size,
            "microbatch_validation": (
                "unit_tested_against_single_batch_update"
                if self.update_micro_batch_size is not None
                else None
            ),
            "intentional_changes": [
                "class count changed from CIFAR-10/100 to binary detection",
                "input resolution changed from CIFAR 32 to common AIGC 224",
                "torchvision interpolation and hard-coded CUDA compatibility patches",
                "model-specific normalization bridge around official pixel augmentation",
                "EMA prediction cached before the official memory/update path",
                *layernorm_changes,
            ],
            "normalization_mapping": self.normalization_mapping,
            "retained_rotta_components": [
                "CSTU memory",
                "teacher/student update",
                "EMA update",
                "timeliness reweighting",
                "entropy objective",
                "official optimizer and update frequency",
                "single optimizer step and EMA update per full CSTU memory loss",
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
                "last_update_microbatches": getattr(
                    self.core, "last_update_microbatches", 1
                ),
                "instances_seen": self.core.current_instance,
            },
        )

    def reset(self) -> None:
        self.core.reset()
        self._bridge_pixel_transform()
        self._pending_teacher_output = None
        self._pending_batch_id = None

    def discard_pending_prediction(self) -> None:
        self._pending_teacher_output = None
        self._pending_batch_id = None
