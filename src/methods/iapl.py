"""Framework adapter for IAPL's episodic image-adaptive prompt tuning."""

from __future__ import annotations

import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Any

from src.models.iapl import IAPL_COMMIT
from src.types import AdaptationStats, PredictionBatch

from .base import TTAMethod


class IAPL(TTAMethod):
    """Run per-image prompt adaptation inside the prediction lifecycle."""

    protocol_name = "episodic_adapt_then_predict"

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        import torch

        from src.data.views import GlobalLocalViewTransform

        # A full CLIP snapshot is prohibitively large. IAPL only mutates the shallow
        # prompt, optimizer state, and module buffers during test-time adaptation.
        self.model = model.to(device)
        self.device = device
        self.config = config or {}
        if not hasattr(self.model, "prompt_learner") or not hasattr(
            self.model.prompt_learner, "ctx"
        ):
            raise TypeError("IAPL requires a model with prompt_learner.ctx")

        if hasattr(self.model, "freeze_tta"):
            self.model.freeze_tta()
        else:
            self.model.requires_grad_(False)
            self.model.prompt_learner.ctx.requires_grad_(True)

        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("IAPL did not expose any trainable prompt parameters")

        self.views = int(self.config.get("views", 32))
        self.steps = int(self.config.get("steps", 2))
        self.selection_fraction = float(self.config.get("selection_fraction", 0.2))
        self.selection_count = self.config.get("selection_count")
        self.optimal_input_selection = bool(
            self.config.get("optimal_input_selection", True)
        )
        if self.steps < 1:
            raise ValueError("IAPL steps must be positive")
        if not 0.0 < self.selection_fraction <= 1.0:
            raise ValueError("IAPL selection_fraction must be in (0, 1]")
        if self.selection_count is not None and not (
            1 <= int(self.selection_count) <= self.views
        ):
            raise ValueError("IAPL selection_count must be between 1 and views")

        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=float(self.config.get("lr", 0.005)),
            weight_decay=float(self.config.get("weight_decay", 0.01)),
        )
        self._initial_prompt = self.model.prompt_learner.ctx.detach().clone()
        self._initial_optimizer_state = deepcopy(self.optimizer.state_dict())
        self._initial_buffers = {
            name: value.detach().clone() for name, value in self.model.named_buffers()
        }
        self._last_adaptation = AdaptationStats(selected=0)

        mean = tuple(
            float(value)
            for value in self.config.get("normalization_mean", (0.485, 0.456, 0.406))
        )
        std = tuple(
            float(value)
            for value in self.config.get("normalization_std", (0.229, 0.224, 0.225))
        )
        if len(mean) != 3 or len(std) != 3:
            raise ValueError("IAPL normalization_mean/std must contain three values")
        self.input_transform = GlobalLocalViewTransform(
            views=self.views,
            image_size=int(self.config.get("image_size", 224)),
            resize_size=int(self.config.get("resize_size", 256)),
            mean=mean,
            std=std,
        )

    def _autocast(self):
        import torch

        if str(self.device).startswith("cuda") and bool(self.config.get("amp", True)):
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _restore_prompt(self) -> None:
        import torch

        with torch.no_grad():
            self.model.prompt_learner.ctx.copy_(self._initial_prompt)
        self.optimizer.load_state_dict(deepcopy(self._initial_optimizer_state))

    @staticmethod
    def _binary_logits(output: Any) -> Any:
        if isinstance(output, (list, tuple)):
            output = output[0]
        return output.reshape(-1)

    def _selected_indices(self, logits: Any) -> Any:
        import torch

        available = int(logits.numel())
        requested = (
            int(self.selection_count)
            if self.selection_count is not None
            else int(available * self.selection_fraction)
        )
        count = min(available, max(1, requested))
        confidence = (torch.sigmoid(logits) - 0.5).abs()
        if self.optimal_input_selection:
            return confidence.topk(count).indices
        local_count = max(0, count - 1)
        local = confidence.topk(local_count).indices if local_count else confidence[:0].long()
        return torch.cat([torch.zeros(1, device=logits.device, dtype=torch.long), local])

    def _adapt_one(self, views: Any) -> tuple[Any, float, int, float]:
        import torch

        self._restore_prompt()
        final_indices = None
        final_loss = None
        started = time.perf_counter()
        for _ in range(self.steps):
            self.model.train()
            with self._autocast():
                logits = self._binary_logits(self.model(views))
                final_indices = self._selected_indices(logits)
                mean_probability = torch.sigmoid(logits[final_indices]).mean()
                final_loss = -(
                    mean_probability * torch.log(mean_probability + 1e-8)
                    + (1.0 - mean_probability)
                    * torch.log(1.0 - mean_probability + 1e-8)
                )
            self.optimizer.zero_grad(set_to_none=True)
            final_loss.backward()
            self.optimizer.step()

        assert final_indices is not None and final_loss is not None
        self.model.eval()
        with torch.no_grad(), self._autocast():
            if self.optimal_input_selection:
                candidate_logits = self._binary_logits(self.model(views[final_indices]))
                confidence = (torch.sigmoid(candidate_logits) - 0.5).abs()
                prediction_logit = candidate_logits[confidence.argmax()]
            else:
                prediction_logit = self._binary_logits(self.model(views[:1]))[0]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return (
            prediction_logit.detach(),
            float(final_loss.detach()),
            int(final_indices.numel()),
            elapsed_ms,
        )

    def predict(self, images: Any) -> PredictionBatch:
        import torch

        if images.ndim != 5:
            raise ValueError("IAPL expects images shaped [batch, views, channels, height, width]")
        if int(images.shape[1]) != self.views:
            raise ValueError(
                f"IAPL expected {self.views} views per image, got {int(images.shape[1])}"
            )

        prediction_logits = []
        losses = []
        selected = 0
        adaptation_ms = 0.0
        for sample_views in images:
            logit, loss, sample_selected, sample_ms = self._adapt_one(
                sample_views.to(self.device, non_blocking=True)
            )
            prediction_logits.append(logit)
            losses.append(loss)
            selected += sample_selected
            adaptation_ms += sample_ms

        fake_logits = torch.stack(prediction_logits).float()
        logits = torch.stack([torch.zeros_like(fake_logits), fake_logits], dim=1)
        probabilities = logits.softmax(dim=1)
        self._last_adaptation = AdaptationStats(
            loss=sum(losses) / len(losses),
            selected=selected,
            elapsed_ms=adaptation_ms,
            extra={
                "adaptation_inside_predict": True,
                "adaptation_inside_predict_ms": adaptation_ms,
                "optimizer_updates": len(losses) * self.steps,
                "views_per_image": int(images.shape[1]),
            },
        )
        return PredictionBatch(
            logits=logits.detach(),
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
        import torch

        self._restore_prompt()
        current_buffers = dict(self.model.named_buffers())
        with torch.no_grad():
            for name, initial in self._initial_buffers.items():
                current_buffers[name].copy_(initial)
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
            "level": "framework_adapter_over_vendored_upstream_core",
            "official_commit": IAPL_COMMIT,
            "protocol_wrapper": "episodic_adapt_then_predict_inside_predict",
            "numerical_validation": "requires_framework_native_benchmark_run",
            "intentional_changes": [
                "framework owns dataset loading, metrics, and result serialization",
                "official per-image prompt reset, entropy tuning, and OIS are preserved",
                "the pinned upstream model core is vendored under src/official/iapl",
                "single-process framework execution replaces the authors' DDP runner",
                "each target rebuilds the model to preserve single-target independence",
            ],
        }
