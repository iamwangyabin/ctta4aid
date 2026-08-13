from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch


IAPL_COMMIT = "a173e7783bbafaa00d60e6e31774a0bc14411a23"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _model_arguments(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        n_ctx=int(config.get("n_ctx", 2)),
        prompt_depth=int(config.get("prompt_depth", 9)),
        image_size=int(config.get("image_size", 224)),
        vision_width=int(config.get("vision_width", 1024)),
        vit_adapter_list=list(config.get("vit_adapter_list", [3, 7, 11, 15, 19, 23])),
        text_adapter_list=list(config.get("text_adapter_list", [])),
        gate=bool(config.get("gate", True)),
        condition=bool(config.get("condition", True)),
        tta=True,
        loss_adapter=float(config.get("loss_adapter", 1.0)),
        loss_contrast=float(config.get("loss_contrast", 1.0)),
        loss_condition=float(config.get("loss_condition", 1.0)),
        use_contrast=bool(config.get("use_contrast", False)),
        smooth=bool(config.get("smooth", False)),
    )


def build_iapl_detector(
    config: dict[str, Any], *, device: str | Any = "cpu"
) -> tuple[Any, dict[str, Any]]:
    import torch

    from src.official.iapl import CLIPModel
    from src.official.iapl import clip_models

    for key in ("clip_path", "checkpoint"):
        if not config.get(key):
            raise ValueError(f"IAPL method config requires {key}")
    if not bool(config.get("condition", True)):
        raise ValueError("The pinned IAPL release requires condition=true for TTA")

    clip_path = _resolve_path(config["clip_path"])
    checkpoint_path = _resolve_path(config["checkpoint"])
    if not clip_path.is_file():
        raise FileNotFoundError(f"CLIP checkpoint does not exist: {clip_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"IAPL checkpoint does not exist: {checkpoint_path}")

    upstream_clip_loader = clip_models.load_clip_to_cpu
    torch_load = torch.load

    def compatible_torch_load(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return torch_load(*args, **kwargs)

    def configured_clip_loader(_ignored_path: str, *args: Any, **kwargs: Any) -> Any:
        with patch.object(torch, "load", compatible_torch_load):
            return upstream_clip_loader(str(clip_path), *args, **kwargs)

    with patch.object(clip_models, "load_clip_to_cpu", configured_clip_loader):
        model = CLIPModel(_model_arguments(config))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if state_dict and all(name.startswith("module.") for name in state_dict):
        state_dict = {name.removeprefix("module."): value for name, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    metadata = {
        "official_commit": IAPL_COMMIT,
        "checkpoint_keys": sorted(checkpoint) if isinstance(checkpoint, dict) else [],
    }
    del checkpoint, state_dict
    return model.to(device), metadata
