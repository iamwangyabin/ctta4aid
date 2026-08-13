"""Framework wrapper for OST's episodic one-shot test-time training."""

from __future__ import annotations

import random
import time
from typing import Any

from src.data.ost import build_ost_transform
from src.models.ost import OST_COMMIT
from src.official.ost import OSTInferenceCore
from src.types import AdaptationStats, PredictionBatch

from .base import TTAMethod


class OST(TTAMethod):
    """Adapt fast weights on a synthesized sample before each prediction."""

    protocol_name = "episodic_adapt_then_predict"

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config or {}
        self._initial_model_state = {
            name: value.detach().cpu().clone()
            for name, value in self.model.state_dict().items()
        }
        self.steps = int(self.config.get("steps", 1))
        self.alpha_min = float(self.config.get("alpha_min", 0.65))
        self.alpha_max = float(self.config.get("alpha_max", 0.90))
        if not 0.0 < self.alpha_min <= self.alpha_max < 1.0:
            raise ValueError("OST alpha range must satisfy 0 < min <= max < 1")
        self._rng = random.Random(int(self.config.get("synthesis_seed", 0)))
        self.input_transform = build_ost_transform(
            int(self.config.get("image_size", 256))
        )
        self.core = OSTInferenceCore(
            self.model,
            self.device,
            learning_rate=float(self.config.get("task_learning_rate", 0.0005)),
            steps=self.steps,
            second_order=bool(self.config.get("second_order", True)),
            enable_inner_loop_optimizable_bn_params=bool(
                self.config.get("enable_inner_loop_optimizable_bn_params", True)
            ),
        )
        self.template_sampler: Any = None
        self._last_adaptation = AdaptationStats(selected=0)

    def set_template_sampler(self, sampler: Any) -> None:
        self.template_sampler = sampler

    def _pseudo_sample(self, target: Any, template: Any) -> tuple[Any, float]:
        alpha = self._rng.uniform(self.alpha_min, self.alpha_max)
        return alpha * target + (1.0 - alpha) * template, alpha

    def predict(self, images: Any) -> PredictionBatch:
        import torch

        if self.template_sampler is None:
            raise RuntimeError("OST requires a labeled source template sampler")
        prediction_logits = []
        losses = []
        alphas = []
        template_ids = []
        started = time.perf_counter()
        for image in images:
            target = image.unsqueeze(0).to(self.device, non_blocking=True)
            template, template_label, template_id = self.template_sampler.sample()
            template = template.unsqueeze(0).to(self.device, non_blocking=True)
            template_label = torch.tensor(
                [int(template_label)], dtype=torch.long, device=self.device
            )
            pseudo_sample, alpha = self._pseudo_sample(target, template)
            scores, support_loss = self.core.infer(
                target, pseudo_sample, template, template_label
            )
            prediction_logits.append(scores[0].detach())
            losses.append(float(support_loss))
            alphas.append(alpha)
            template_ids.append(str(template_id))

        logits = torch.stack(prediction_logits).float()
        probabilities = logits.softmax(dim=1)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._last_adaptation = AdaptationStats(
            loss=sum(losses) / len(losses),
            selected=2 * len(losses),
            elapsed_ms=elapsed_ms,
            extra={
                "adaptation_inside_predict": True,
                "adaptation_inside_predict_ms": elapsed_ms,
                "optimizer_updates": len(losses) * self.steps,
                "source_templates": len(template_ids),
                "template_sample_ids": template_ids,
                "mean_target_blend_alpha": sum(alphas) / len(alphas),
            },
        )
        return PredictionBatch(
            logits=logits,
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
        self.model.load_state_dict(self._initial_model_state)
        if self.template_sampler is not None and hasattr(self.template_sampler, "reset"):
            self.template_sampler.reset()
        self._rng.seed(int(self.config.get("synthesis_seed", 0)))
        self._last_adaptation = AdaptationStats(selected=0)

    @property
    def trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "patched_vendored_official_core_with_cross_task_data_adapter",
            "official_commit": OST_COMMIT,
            "official_core": "src.official.ost",
            "protocol_wrapper": "episodic_adapt_then_predict_inside_predict",
            "numerical_validation": "not_equivalent_to_official_face_benchmark",
            "source_free_during_test": False,
            "intentional_changes": [
                "framework reads labeled source templates from canonical Arrow data",
                "full-frame alpha blending replaces landmark and SimSwap face blending",
                "single-device fast weights replace the authors' DataParallel tensor layout",
                "target hidden labels remain isolated in the evaluator",
                "the official MetaXception, AM-Softmax, and one-step update are preserved",
            ],
        }
