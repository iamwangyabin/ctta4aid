from __future__ import annotations

from typing import Any

from .base import TTAMethod
from .cotta import CoTTA
from .eata import EATA
from .lame import LAME
from .rotta import RoTTA
from .source import SourceOnly
from .t2a import T2A
from .tent import Tent


def build_method(
    name: str,
    model: Any,
    device: Any,
    config: dict[str, Any] | None = None,
    *,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> TTAMethod:
    normalized = name.lower().replace("²", "2").replace("_", "").replace("-", "")
    config = config or {}
    if normalized in {"source", "sourceonly"}:
        return SourceOnly(model, device, config)
    if normalized == "tent":
        return Tent(model, device, config)
    if normalized == "eata":
        fishers = (checkpoint_metadata or {}).get("fishers")
        return EATA(model, device, config, fishers=fishers)
    if normalized == "cotta":
        return CoTTA(model, device, config)
    if normalized == "rotta":
        return RoTTA(model, device, config)
    if normalized == "lame":
        return LAME(model, device, config)
    if normalized in {"t2a", "thinktwicebeforeadaptation"}:
        return T2A(model, device, config)
    raise ValueError(f"Unknown TTA method: {name}")


__all__ = [
    "TTAMethod",
    "SourceOnly",
    "Tent",
    "EATA",
    "CoTTA",
    "RoTTA",
    "LAME",
    "T2A",
    "build_method",
]
