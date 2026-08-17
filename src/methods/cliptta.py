"""Framework adapter for CLIPTTA's closed-set visual norm adaptation profile."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from src.types import AdaptationStats, PredictionBatch

from .base import TTAMethod
from .utils import build_optimizer


class CLIPTTA(TTAMethod):
    """Apply the authors' CLIPTTA objective, then predict with the updated visual encoder."""

    protocol_name = "online_adapt_then_predict"

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        config.setdefault("capture_initial_state", False)
        super().__init__(model, device, config)
        if not hasattr(self.model, "clip") or not hasattr(self.model, "text_features"):
            raise TypeError("CLIPTTA requires the shared CLIP binary VLM adapter")
        if bool(self.config.get("use_memory", False)):
            raise ValueError("The main-track CLIPTTA profile fixes use_memory=false")
        if bool(self.config.get("update_text", False)):
            raise ValueError("The main-track CLIPTTA profile fixes update_text=false")
        if bool(self.config.get("update_all_params", False)):
            raise ValueError("The main-track CLIPTTA profile updates visual norm parameters only")
        if bool(self.config.get("use_ood_loss", False)) or bool(
            self.config.get("detect_ood", False)
        ):
            raise ValueError("The binary main track uses CLIPTTA's closed-set profile")

        import torch.nn as nn

        self.model.eval()
        self.model.requires_grad_(False)
        parameters = []
        names = []
        seen = set()
        for module_name, module in self.model.clip.visual.named_modules():
            if not isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                continue
            module.requires_grad_(True)
            if isinstance(module, nn.BatchNorm2d):
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
            for parameter_name, parameter in module.named_parameters(recurse=False):
                if parameter_name not in {"weight", "bias"} or id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                parameters.append(parameter)
                names.append(f"clip.visual.{module_name}.{parameter_name}")
        if not parameters:
            raise RuntimeError("CLIPTTA could not find visual normalization parameters")

        self.official_parameter_names = names
        self._adaptation_parameters = parameters
        self.optimizer = build_optimizer(parameters, self.config)
        self._initial_adaptation_state = [parameter.detach().clone() for parameter in parameters]
        self._initial_optimizer_state = deepcopy(self.optimizer.state_dict())
        self.steps = int(self.config.get("steps", 1))
        if self.steps < 1:
            raise ValueError("CLIPTTA steps must be positive")
        self.logit_scale = float(self.config.get("logit_scale", 100.0))
        self.beta_tta = float(self.config.get("beta_tta", 1.0))
        self.beta_reg = float(self.config.get("beta_reg", 0.1))
        self.use_softmax_entropy = bool(self.config.get("use_softmax_entropy", False))
        self.use_tent = bool(self.config.get("use_tent", False))
        self._last_adaptation = AdaptationStats(selected=0)

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "vendored_public_core_with_framework_wrapper",
            "official_commit": "ef0e6797f7618959ca85be36816a5e01299a522f",
            "official_core": "src.official.cliptta",
            "upstream_license": "none_declared",
            "numerical_validation": "not_run_requires_framework_native_benchmark",
            "protocol_wrapper": self.protocol_name,
            "intentional_changes": [
                "the official parser restriction to ViT-B variants is lifted for "
                "the documented ViT-L/14 loader",
                "the closed-set, no-memory, vision-only profile is used for binary AIGC detection",
                "official OpenAI CLIP loading is replaced by the shared local checkpoint loader",
                "the framework owns Arrow I/O and evaluator serialization",
            ],
        }

    def _adapt_once(self, images: Any) -> tuple[float, dict[str, float]]:
        from src.official import cliptta as official_cliptta

        features = self.model.forward_features(images)
        loss, components = official_cliptta.cliptta_loss(
            features,
            self.model.text_features.to(dtype=features.dtype),
            logit_scale=self.logit_scale,
            beta_tta=self.beta_tta,
            beta_reg=self.beta_reg,
            use_softmax_entropy=self.use_softmax_entropy,
            use_tent=self.use_tent,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().float().item()), {
            name: float(value.detach().float().item()) for name, value in components.items()
        }

    def predict(self, images: Any) -> PredictionBatch:
        import torch

        device_images = images.to(self.device, non_blocking=True)
        losses = []
        components: dict[str, float] = {}
        for _ in range(self.steps):
            loss, components = self._adapt_once(device_images)
            losses.append(loss)
        with torch.no_grad():
            logits = self.model(device_images)
            probabilities = logits.softmax(dim=1)
        self._last_adaptation = AdaptationStats(
            loss=sum(losses) / len(losses),
            selected=int(device_images.shape[0]),
            extra={"adaptation_inside_predict": True, "steps": self.steps, **components},
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

        with torch.no_grad():
            for parameter, initial in zip(
                self._adaptation_parameters, self._initial_adaptation_state
            ):
                parameter.copy_(initial)
        self.optimizer.load_state_dict(deepcopy(self._initial_optimizer_state))
        self._last_adaptation = AdaptationStats(selected=0)
