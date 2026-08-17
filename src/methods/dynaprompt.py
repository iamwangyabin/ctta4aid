"""Framework adapter for DynaPrompt's online multi-prompt test-time tuning."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import Any

from src.types import AdaptationStats, PredictionBatch

from .base import TTAMethod


class DynaPrompt(TTAMethod):
    """Adapt dynamic text prompts on one global image plus its AugMix views."""

    protocol_name = "online_adapt_then_predict"

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        import torch

        from src.data.views import DynaPromptViewTransform
        from src.official import dynaprompt as official_dynaprompt

        config = dict(config or {})
        config.setdefault("capture_initial_state", False)
        if not hasattr(model, "clip"):
            raise TypeError("DynaPrompt requires the shared CLIP binary VLM adapter")
        self.views = int(config.get("views", 64))
        self.steps = int(config.get("steps", 1))
        self.selection_fraction = float(config.get("selection_fraction", 0.1))
        self.num_prompts = int(config.get("num_prompts", 4))
        self.online_prompt_tuning = bool(config.get("online_prompt_tuning", True))
        if self.steps < 1 or self.num_prompts < 2:
            raise ValueError("DynaPrompt requires positive steps and at least two prompts")
        if not 0.0 < self.selection_fraction <= 1.0:
            raise ValueError("DynaPrompt selection_fraction must be in (0, 1]")
        if int(self.views * self.selection_fraction) < 1:
            raise ValueError("DynaPrompt views and selection_fraction select no adaptation view")

        classnames = list(config.get("class_names", ("real photograph", "AI-generated image")))
        if len(classnames) != 2:
            raise ValueError("DynaPrompt main-track class_names must be [real, fake]")
        prompt_model = official_dynaprompt.build_prompt_model(
            model.clip,
            classnames,
            n_ctx=int(config.get("n_ctx", 4)),
            ctx_init=str(config.get("ctx_init", "a photo of a")),
            num_prompts=self.num_prompts,
        )
        super().__init__(prompt_model, device, config)
        # OpenAI CLIP loads CUDA weights in fp16. Keep DynaPrompt's trainable
        # context in fp32 so PyTorch AMP can safely unscale its gradients.
        self.model.float()
        self.model.eval()
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(name.startswith("prompt_learner."))
        trainable = list(self.model.prompt_learner.parameters())
        if not trainable:
            raise RuntimeError("DynaPrompt did not expose prompt learner parameters")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(self.config.get("lr", 0.005)),
            weight_decay=float(self.config.get("weight_decay", 0.01)),
        )
        self._initial_optimizer_state = deepcopy(self.optimizer.state_dict())
        self._amp_enabled = str(device).startswith("cuda") and bool(
            self.config.get("amp", True)
        )
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=float(self.config.get("amp_init_scale", 1000.0)),
            enabled=self._amp_enabled,
        )
        self.input_transform = DynaPromptViewTransform(
            views=self.views,
            image_size=int(self.config.get("image_size", 224)),
            augmix=bool(self.config.get("augmix", True)),
            severity=int(self.config.get("augmix_severity", 1)),
        )
        self._last_adaptation = AdaptationStats(selected=0)

    def _autocast(self) -> Any:
        import torch

        if self._amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "vendored_public_core_with_framework_wrapper",
            "official_commit": "acd33cf71f5be817512f99ba3b81ec019595ad59",
            "official_core": "src.official.dynaprompt",
            "upstream_license": "none_declared",
            "numerical_validation": "not_run_requires_framework_native_benchmark",
            "protocol_wrapper": self.protocol_name,
            "intentional_changes": [
                "the official ImageFolder loader is replaced by the canonical Arrow loader",
                "the official ViT-B command-line restriction is lifted for OpenAI CLIP ViT-L/14",
                "fixed binary class names replace the repository's ImageNet class list",
                "the original global-plus-AugMix multi-view transform is retained",
                "trainable prompt state is held in fp32 before CUDA AMP so "
                "its gradients are scaler-compatible",
            ],
        }

    def _prepare_online_prompt_state(self) -> None:
        import torch

        learner = self.model.prompt_learner
        if not self.online_prompt_tuning:
            self.model.reset()
        elif learner.ctx_use.count(0) == 0:
            prompt_index = learner.ctx_order[0]
            learner.ctx_use[prompt_index] = 0
            with torch.no_grad():
                learner.ctx[prompt_index].mul_(0)
        self.optimizer.load_state_dict(deepcopy(self._initial_optimizer_state))

    def predict(self, images: Any) -> PredictionBatch:
        import torch

        from src.official import dynaprompt as official_dynaprompt

        if images.ndim != 5 or images.shape[0] != 1:
            raise ValueError(
                "DynaPrompt requires data.batch_size=1 and [batch, views, C, H, W] input"
            )
        if int(images.shape[1]) != self.views:
            raise ValueError(f"DynaPrompt expected {self.views} views per image")
        views = images[0].to(self.device, non_blocking=True)
        self._prepare_online_prompt_state()
        selected_prompts = None
        raw_prediction = None
        raw_entropy = None
        selected_view_count = 0
        losses = []
        for _ in range(self.steps):
            with self._autocast():
                view_logits = self.model(views)
                (
                    selected_prompts,
                    raw_prediction,
                    raw_entropy,
                    prompt_logits,
                ) = official_dynaprompt.select_dynamic_prompts(
                    view_logits, self.model.prompt_learner, self.num_prompts
                )
                confident_logits, selected_indices = official_dynaprompt.select_confident_samples(
                    prompt_logits, self.selection_fraction
                )
                loss = official_dynaprompt.avg_entropy(confident_logits).mean()
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            selected_view_count = int(selected_indices.numel())
            losses.append(float(loss.detach().float().item()))

        assert selected_prompts is not None
        with torch.no_grad(), self._autocast():
            logits = self.model(views[:1], selected_prompts + 1)
            logits = logits.view(1, selected_prompts.numel(), -1).mean(1)
            probabilities = logits.softmax(dim=1)
        self._last_adaptation = AdaptationStats(
            loss=sum(losses) / len(losses),
            selected=selected_view_count,
            extra={
                "adaptation_inside_predict": True,
                "optimizer_updates": self.steps,
                "views_per_image": self.views,
                "selected_prompts": selected_prompts.detach().cpu().tolist(),
                "raw_entropy": (
                    None if raw_entropy is None else float(raw_entropy.detach().cpu().item())
                ),
                "raw_prediction": None
                if raw_prediction is None
                else int(raw_prediction.argmax().detach().cpu().item()),
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
        self.model.reset()
        self.optimizer.load_state_dict(deepcopy(self._initial_optimizer_state))
        self._last_adaptation = AdaptationStats(selected=0)
