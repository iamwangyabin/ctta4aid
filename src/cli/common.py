from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.config import deep_merge, method_config, require
from src.data.ost import OSTTemplateSampler
from src.methods import build_method
from src.models import (
    build_clip_lora_detector,
    build_clip_source_detector,
    build_clip_vlm_detector,
    build_detector,
    build_iapl_detector,
    build_ost_detector,
    load_checkpoint,
)


@lru_cache(maxsize=8)
def checkpoint_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(requested: str) -> Any:
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_config_for_method(experiment_config: dict[str, Any], name: str) -> dict[str, Any]:
    """Apply a method's declared loader overrides without changing sample identity."""

    base = deepcopy(experiment_config["data"])
    overrides = method_config(experiment_config, name).get("data", {})
    if not isinstance(overrides, dict):
        raise ValueError(f"method_configs.{name}.data must be a mapping")
    return deep_merge(base, overrides)


def build_fresh_method(
    experiment_config: dict[str, Any], name: str, device: Any
) -> tuple[Any, dict[str, Any]]:
    effective_method_config = method_config(experiment_config, name)
    normalized_name = name.lower().replace("_", "").replace("-", "")
    if normalized_name == "iapl":
        model, metadata = build_iapl_detector(effective_method_config, device=device)
        method = build_method(name, model, device, effective_method_config)
        checkpoint_path = str(
            Path(effective_method_config["checkpoint"]).expanduser().resolve()
        )
        method.source_checkpoint_identity = {
            "path": checkpoint_path,
            "sha256": checkpoint_sha256(checkpoint_path),
        }
        clip_checkpoint_path = str(
            Path(effective_method_config["clip_path"]).expanduser().resolve()
        )
        method.source_model_metadata = {
            **metadata,
            "source_setup": "authors_iapl_task_checkpoint_with_openai_clip_vitl14",
            "task_checkpoint": method.source_checkpoint_identity,
            "clip_checkpoint": {
                "path": clip_checkpoint_path,
                "sha256": checkpoint_sha256(clip_checkpoint_path),
            },
        }
        return method, metadata
    if normalized_name == "ost":
        effective_method_config.setdefault(
            "synthesis_seed", int(experiment_config.get("seed", 0))
        )
        model, metadata = build_ost_detector(effective_method_config, device=device)
        method = build_method(name, model, device, effective_method_config)
        if (
            str(effective_method_config.get("adaptation_mode", "full")).lower()
            == "full"
        ):
            method.set_template_sampler(
                OSTTemplateSampler.from_data_config(
                    experiment_config["data"],
                    transform=method.input_transform,
                    seed=int(experiment_config.get("seed", 0)),
                )
            )
        checkpoint_path = str(
            Path(effective_method_config["checkpoint"]).expanduser().resolve()
        )
        method.source_checkpoint_identity = {
            "path": checkpoint_path,
            "sha256": checkpoint_sha256(checkpoint_path),
        }
        method.source_model_metadata = metadata
        return method, metadata
    if normalized_name in {"ours", "oursnocalibratedreadout", "oursstatic"}:
        expected_readout = (
            "base"
            if normalized_name == "oursnocalibratedreadout"
            else "calibrated"
        )
        configured_readout = str(
            effective_method_config.get("readout_mode", expected_readout)
        ).lower()
        if configured_readout != expected_readout:
            raise ValueError(
                f"{name} fixes readout_mode={expected_readout}, got "
                f"{configured_readout}"
            )
        effective_method_config["readout_mode"] = expected_readout
        if normalized_name == "oursstatic":
            configured_adaptation = str(
                effective_method_config.get("adaptation_mode", "static")
            ).lower()
            if configured_adaptation != "static":
                raise ValueError(
                    f"{name} fixes adaptation_mode=static, got "
                    f"{configured_adaptation}"
                )
            effective_method_config["adaptation_mode"] = "static"
        model, metadata = build_clip_lora_detector(
            effective_method_config, device=device
        )
        checkpoint_path = str(
            Path(effective_method_config["source_checkpoint"]).expanduser().resolve()
        )
        checkpoint_metadata = load_checkpoint(model, checkpoint_path, device=device)
        anchors = checkpoint_metadata.get("score_anchors")
        if not isinstance(anchors, dict):
            raise ValueError(
                "Ours requires a calibrated LoRA source checkpoint carrying "
                "score_anchors; train one with "
                "configs/train/genimage_sd14_clip_vitl14_lora_ours.yaml"
            )
        checkpoint_rank = checkpoint_metadata.get("lora_rank")
        if checkpoint_rank is not None and int(checkpoint_rank) != int(
            effective_method_config.get("lora_rank", 4)
        ):
            raise ValueError("Ours lora_rank does not match its source checkpoint")
        effective_method_config["score_anchors"] = anchors
        method = build_method(name, model, device, effective_method_config)
        clip_checkpoint_path = str(
            Path(effective_method_config["checkpoint"]).expanduser().resolve()
        )
        method.source_checkpoint_identity = {
            "path": checkpoint_path,
            "sha256": checkpoint_sha256(checkpoint_path),
        }
        method.source_model_metadata = {
            **metadata,
            **checkpoint_metadata,
            "task_checkpoint": method.source_checkpoint_identity,
            "clip_checkpoint": {
                "path": clip_checkpoint_path,
                "sha256": checkpoint_sha256(clip_checkpoint_path),
            },
        }
        if hasattr(model, "input_transform") and not hasattr(method, "input_transform"):
            method.input_transform = model.input_transform
        return method, metadata

    require(experiment_config, "model")
    model_config = experiment_config["model"]
    model_family = str(model_config.get("family", "detector")).lower().replace("-", "_")
    if model_family == "clip_vlm_main":
        source_trained_methods = {
            "sourceft",
            "tent",
            "eata",
            "sar",
            "cotta",
            "rotta",
            "lame",
            "t2a",
        }
        clip_native_methods = {
            "frozenclip",
            "tda",
            "dynaprompt",
            "cliptta",
            "batclip",
        }
        if normalized_name in source_trained_methods:
            source_config = deepcopy(model_config.get("source_detector", {}))
            require(source_config, "checkpoint", "source_checkpoint")
            model, initialization = build_clip_source_detector(source_config, device=device)
            checkpoint_path = str(
                Path(source_config["source_checkpoint"]).expanduser().resolve()
            )
            checkpoint_metadata = load_checkpoint(model, checkpoint_path, device=device)
            metadata = {**initialization, **checkpoint_metadata}
        elif normalized_name in clip_native_methods:
            native_config = deep_merge(
                model_config.get("clip_native", {}), effective_method_config.get("vlm", {})
            )
            require(native_config, "checkpoint")
            model, metadata = build_clip_vlm_detector(native_config, device=device)
            checkpoint_path = str(Path(native_config["checkpoint"]).expanduser().resolve())
        else:
            raise ValueError(
                f"CLIP VLM main track has no model profile for method: {name}"
            )
    elif model_family == "clip_vlm":
        require(model_config, "checkpoint")
        model, metadata = build_clip_vlm_detector(model_config, device=device)
        checkpoint_path = str(Path(model_config["checkpoint"]).expanduser().resolve())
    else:
        require(model_config, "architecture", "checkpoint")
        model = build_detector(
            model_config["architecture"],
            pretrained=bool(model_config.get("pretrained", False)),
            pretrained_weights=str(model_config.get("pretrained_weights", "default")),
            num_classes=int(model_config.get("num_classes", 2)),
        )
        checkpoint_path = str(Path(model_config["checkpoint"]).expanduser().resolve())
        metadata = load_checkpoint(model, checkpoint_path, device=device)
    method = build_method(
        name,
        model,
        device,
        effective_method_config,
        checkpoint_metadata=metadata,
    )
    method.source_checkpoint_identity = {
        "path": checkpoint_path,
        "sha256": checkpoint_sha256(checkpoint_path),
    }
    method.source_model_metadata = metadata
    if hasattr(model, "input_transform") and not hasattr(method, "input_transform"):
        method.input_transform = model.input_transform
    return method, metadata


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
