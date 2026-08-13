from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any


OST_COMMIT = "1e4518b9e560baf9c5693f13a402fa5d7104190f"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _model_arguments(config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        learnable_bn_beta=bool(config.get("learnable_bn_beta", True)),
        learnable_bn_gamma=bool(config.get("learnable_bn_gamma", True)),
        enable_inner_loop_optimizable_bn_params=bool(
            config.get("enable_inner_loop_optimizable_bn_params", True)
        ),
        number_of_training_steps_per_iter=int(config.get("steps", 1)),
        per_step_bn_statistics=bool(config.get("per_step_bn_statistics", False)),
    )


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    if state_dict and all(name.startswith("module.") for name in state_dict):
        return {name.removeprefix("module."): value for name, value in state_dict.items()}
    return state_dict


def _checkpoint_state(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("OST checkpoint must contain a state dictionary")
    return _clean_state_dict(state_dict)


def _new_ost_detector(config: dict[str, Any], device: str | Any) -> Any:
    from src.official.ost import MetaXception

    return MetaXception(
        inc=3,
        num_output_classes=int(config.get("num_classes", 2)),
        args=_model_arguments(config),
        device=device,
        direct=False,
    )


def build_ost_training_detector(
    config: dict[str, Any], *, device: str | Any = "cpu"
) -> tuple[Any, dict[str, Any]]:
    """Build OST from the authors' Xception initialization for meta-training."""
    import torch

    initialization = config.get("initialization_checkpoint")
    if not initialization:
        raise ValueError("OST training requires model.initialization_checkpoint")
    checkpoint_path = _resolve_path(initialization)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"OST Xception initialization does not exist: {checkpoint_path}"
        )
    model = _new_ost_detector(config, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(
        _checkpoint_state(checkpoint),
        strict=bool(config.get("strict_initialization", False)),
    )
    metadata = {
        "official_commit": OST_COMMIT,
        "initialization_checkpoint": str(checkpoint_path),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    del checkpoint
    return model.to(device), metadata


def build_ost_detector(
    config: dict[str, Any], *, device: str | Any = "cpu"
) -> tuple[Any, dict[str, Any]]:
    import torch

    if not config.get("checkpoint"):
        raise ValueError("OST method config requires checkpoint")
    checkpoint_path = _resolve_path(config["checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"OST checkpoint does not exist: {checkpoint_path}")

    model = _new_ost_detector(config, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(
        _checkpoint_state(checkpoint),
        strict=bool(config.get("strict_checkpoint", False)),
    )
    metadata = {
        "official_commit": OST_COMMIT,
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    del checkpoint
    return model.to(device), metadata
