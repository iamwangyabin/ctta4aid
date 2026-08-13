"""Framework wrapper around the patched authors' public T²A adapter."""

from __future__ import annotations

import math
from typing import Any

from src.types import AdaptationStats

from .base import TTAMethod
from .utils import preserve_batch_norm_buffers


class T2A(TTAMethod):
    """Expose the authors' repaired adapter through Predict-Then-Adapt."""

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        super().__init__(model, device, config)
        import torch.nn as nn

        from src.official import t2a as official_t2a
        from src.official.t2a_losses import Entropy

        class DictOutputModel(nn.Module):
            def __init__(self, detector: Any) -> None:
                super().__init__()
                self.detector = detector

            def forward(self, data_dict: dict[str, Any]) -> dict[str, Any]:
                return {"cls": self.detector(data_dict["images"])}

        self.model.train()
        self.model.requires_grad_(True)
        self.official_module = official_t2a
        self.official_model = DictOutputModel(self.model)
        optimizer_name = str(self.config.get("optimizer", "adam")).lower()
        official_optimizer_config = self.config.get("optimizer_config", {})
        release_repairs = self.config.get("release_repairs", {})
        official_optimizer_name = {
            "adam": "Adam",
            "adamw": "AdamW",
            "sgd": "SGD",
        }.get(optimizer_name)
        if official_optimizer_name is None:
            raise ValueError(f"Unsupported T²A optimizer: {optimizer_name}")
        optimizer_config = {
            "lr": float(
                official_optimizer_config.get(
                    "lr", self.config.get("lr", self.config.get("learning_rate", 1e-4))
                )
            ),
            "beta1": float(
                official_optimizer_config.get("beta1", self.config.get("beta1", 0.9))
            ),
            "beta2": float(
                official_optimizer_config.get("beta2", self.config.get("beta2", 0.999))
            ),
            "weight_decay": float(
                official_optimizer_config.get(
                    "weight_decay", self.config.get("weight_decay", 0.0)
                )
            ),
            "momentum": float(
                official_optimizer_config.get(
                    "momentum", self.config.get("momentum", 0.9)
                )
            ),
            "dampening": float(
                official_optimizer_config.get(
                    "dampening", self.config.get("dampening", 0.0)
                )
            ),
            "nesterov": bool(
                official_optimizer_config.get(
                    "nesterov", self.config.get("nesterov", False)
                )
            ),
        }
        self.core = official_t2a.T2AAdapter(
            self.official_model,
            self.device,
            steps=int(self.config.get("steps", 1)),
            episodic=bool(self.config.get("episodic", False)),
            optimizer=official_optimizer_name,
            optimizer_config=optimizer_config,
            l1_lambda=float(self.config.get("l1_lambda", 0.0)),
            entropy_fn=Entropy(),
            e_margin=float(
                release_repairs.get(
                    "e_margin",
                    self.config.get("entropy_margin", math.log(2.0) * 0.4),
                )
            ),
            noise_type=str(self.config.get("noise_type", "bernoulli")),
            psi=float(
                self.config.get(
                    "psi", self.config.get("gradient_similarity_threshold", 0.01)
                )
            ),
            gamma=float(self.config.get("gamma", 2.0)),
            alpha=float(self.config.get("alpha", 1.0)),
            beta=float(self.config.get("beta", 1.0)),
            filter_grad=bool(
                release_repairs.get(
                    "filter_grad", self.config.get("gradient_masking", True)
                )
            ),
            cosine_strategy=str(release_repairs.get("cosine_strategy", "zero_pad")),
        )
        self.optimizer = self.core.optimizer

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "patched_vendored_official_core_with_protocol_wrapper",
            "official_commit": "33c8ccc64afdda260564123d6c790d030a89ff81",
            "official_core": "src.official.t2a",
            "numerical_validation": "not_run_requires_official_data_and_weights",
            "protocol_wrapper": "predict_then_adapt",
            "intentional_changes": [
                "framework tensor model is wrapped as authors' dict-output model",
                "missing public-adapter attributes are initialized",
                "unreported e_margin is isolated under release_repairs",
                "one non-pseudo complementary class is sampled from released 1-p weights",
                "log-softmax losses receive logits instead of probabilities",
                "model and optimizer reset states are deep-copied",
                "BN/non-BN gradient identification uses parameter identity",
                "normalized-loss denominators are guarded against numerical zero",
                "authors' adapt-then-predict forward is split by the framework",
                "predict preserves BatchNorm running buffers until adapt",
            ],
        }

    def predict(self, images: Any):
        with preserve_batch_norm_buffers(self.model):
            return super().predict(images)

    def adapt(self, images: Any) -> AdaptationStats:
        device_images = images.to(self.device, non_blocking=True)
        self.core.adapt({"images": device_images})
        return AdaptationStats(
            loss=self.core.last_loss,
            selected=int(device_images.shape[0]),
            extra={"masked_parameter_tensors": int(self.core.last_masked)},
        )

    def reset(self) -> None:
        self.core.reset()
