from __future__ import annotations

from typing import Any

from .base import TTAMethod
from .batclip import BATCLIP
from .cotta import CoTTA
from .cliptta import CLIPTTA
from .dynaprompt import DynaPrompt
from .eata import EATA
from .iapl import IAPL
from .lame import LAME
from .ost import OST
from .rotta import RoTTA
from .sar import SAR
from .source import SourceOnly
from .t2a import T2A
from .tda import TDA
from .tent import Tent
from .tent_ln import TentLayerNorm


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
    if normalized in {"tentln", "tentlayernorm"}:
        return TentLayerNorm(model, device, config)
    if normalized == "eata":
        fishers = (checkpoint_metadata or {}).get("fishers")
        return EATA(model, device, config, fishers=fishers)
    if normalized == "iapl":
        return IAPL(model, device, config)
    if normalized == "ost":
        return OST(model, device, config)
    if normalized == "cotta":
        return CoTTA(model, device, config)
    if normalized == "rotta":
        return RoTTA(model, device, config)
    if normalized == "lame":
        return LAME(model, device, config)
    if normalized in {"t2a", "thinktwicebeforeadaptation"}:
        return T2A(model, device, config)
    if normalized == "tda":
        return TDA(model, device, config)
    if normalized == "batclip":
        return BATCLIP(model, device, config)
    if normalized == "cliptta":
        return CLIPTTA(model, device, config)
    if normalized == "dynaprompt":
        return DynaPrompt(model, device, config)
    if normalized == "sar":
        return SAR(model, device, config)
    raise ValueError(f"Unknown TTA method: {name}")


__all__ = [
    "TTAMethod",
    "BATCLIP",
    "CLIPTTA",
    "DynaPrompt",
    "SourceOnly",
    "Tent",
    "TentLayerNorm",
    "EATA",
    "IAPL",
    "OST",
    "CoTTA",
    "RoTTA",
    "LAME",
    "T2A",
    "TDA",
    "SAR",
    "build_method",
]
