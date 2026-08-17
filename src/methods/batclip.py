"""Framework adapter for BATCLIP's bimodal online norm adaptation."""

from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from typing import Any

from src.types import AdaptationStats

from .base import TTAMethod
from .utils import build_optimizer


class BATCLIP(TTAMethod):
    """Use BATCLIP's entropy, image-to-text, and inter-mean objective."""

    protocol_name = "predict_then_adapt"

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        config.setdefault("capture_initial_state", False)
        super().__init__(model, device, config)
        if not hasattr(self.model, "forward_with_features"):
            raise TypeError("BATCLIP requires a VLM with forward_with_features")

        import torch
        import torch.nn as nn

        self.model.eval()
        self.model.requires_grad_(False)
        parameters = []
        names = []
        seen = set()
        for module_name, module in self.model.named_modules():
            if not isinstance(
                module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)
            ):
                continue
            module.train()
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
                names.append(f"{module_name}.{parameter_name}")
        if not parameters:
            raise RuntimeError("BATCLIP requires LayerNorm, GroupNorm, or BatchNorm parameters")

        self.official_parameter_names = names
        self._adaptation_parameters = parameters
        self.optimizer = build_optimizer(parameters, self.config)
        self._initial_adaptation_state = [parameter.detach().clone() for parameter in parameters]
        self._initial_optimizer_state = deepcopy(self.optimizer.state_dict())
        self.steps = int(self.config.get("steps", 1))
        if self.steps < 1:
            raise ValueError("BATCLIP steps must be positive")
        self._amp_enabled = str(device).startswith("cuda") and bool(
            self.config.get("amp", True)
        )
        self.scaler = torch.cuda.amp.GradScaler(
            init_scale=float(self.config.get("amp_init_scale", 1000.0)),
            enabled=self._amp_enabled,
        )

    def _autocast(self) -> Any:
        import torch

        if self._amp_enabled:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return nullcontext()

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "vendored_official_core_with_protocol_wrapper",
            "official_commit": "ba2e3381873ef58e76a90148ee3835864349e985",
            "official_core": "src.official.batclip",
            "numerical_validation": "not_run_requires_framework_native_benchmark",
            "protocol_wrapper": self.protocol_name,
            "intentional_changes": [
                "official OpenCLIP loading is replaced by the shared local OpenAI "
                "CLIP ViT-L/14 loader",
                "the original mixed-precision loss scaling remains enabled on CUDA "
                "but is device-safe on CPU",
                "the framework separates pre-update prediction from the official "
                "update and owns Arrow I/O",
            ],
        }

    def adapt(self, images: Any) -> AdaptationStats:
        from src.official import batclip as official_batclip

        device_images = images.to(self.device, non_blocking=True)
        losses = []
        components: dict[str, float] = {}
        for _ in range(self.steps):
            with self._autocast():
                logits, _features, text_features, image_pre_features, _text_pre_features = (
                    self.model.forward_with_features(device_images)
                )
                loss, loss_components = official_batclip.batclip_loss(
                    logits, image_pre_features, text_features
                )
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses.append(float(loss.detach().float().item()))
            components = {
                name: float(value.detach().float().item())
                for name, value in loss_components.items()
            }
        return AdaptationStats(
            loss=sum(losses) / len(losses),
            selected=int(device_images.shape[0]),
            extra={"steps": self.steps, **components},
        )

    def reset(self) -> None:
        import torch

        with torch.no_grad():
            for parameter, initial in zip(
                self._adaptation_parameters, self._initial_adaptation_state
            ):
                parameter.copy_(initial)
        self.optimizer.load_state_dict(deepcopy(self._initial_optimizer_state))
