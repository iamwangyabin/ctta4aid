from __future__ import annotations

import hashlib
import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from online_aig_tta.config import method_config, require
from online_aig_tta.methods import build_method
from online_aig_tta.models import build_detector, load_checkpoint


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


def build_fresh_method(
    experiment_config: dict[str, Any], name: str, device: Any
) -> tuple[Any, dict[str, Any]]:
    require(experiment_config, "model")
    model_config = experiment_config["model"]
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
        method_config(experiment_config, name),
        checkpoint_metadata=metadata,
    )
    method.source_checkpoint_identity = {
        "path": checkpoint_path,
        "sha256": checkpoint_sha256(checkpoint_path),
    }
    return method, metadata


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
