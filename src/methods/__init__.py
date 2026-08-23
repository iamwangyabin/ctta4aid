from __future__ import annotations

from typing import Any

from .ascal import ASCAL
from .ascal_gmm import (
    ASCALGMM,
    ASCALGMMDensityShift,
    ASCALGMMMedianShift,
    ASCALGMMSegmentedHandoffShift,
    ASCALGMMSegmentedMemoryShift,
    ASCALGMMSegmentedShift,
    ASCALGMMShift,
)
from .base import TTAMethod
from .batclip import BATCLIP
from .cotta import CoTTA
from .cliptta import CLIPTTA
from .dynaprompt import DynaPrompt
from .eata import EATA
from .iapl import IAPL
from .lame import LAME
from .ost import OST
from .pound_tta import PoundTTA
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
    if normalized in {"source", "sourceonly", "sourceft", "frozenclip"}:
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
    if normalized in {"ours", "oursstatic", "poundtta", "poundttastatic"}:
        effective_config = dict(config)
        if normalized in {"oursstatic", "poundttastatic"}:
            effective_config.setdefault("adaptation_mode", "static")
        return PoundTTA(model, device, effective_config)
    if normalized in {"ascal", "ascalstatic"}:
        effective_config = dict(config)
        if normalized == "ascalstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCAL(model, device, effective_config)
    if normalized in {"ascalgmm", "ascalgmmstatic"}:
        effective_config = dict(config)
        if normalized == "ascalgmmstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMM(model, device, effective_config)
    if normalized in {"ascalgmmshift", "ascalgmmshiftstatic"}:
        effective_config = dict(config)
        if normalized == "ascalgmmshiftstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMShift(model, device, effective_config)
    if normalized in {"ascalgmmmedianshift", "ascalgmmmedianshiftstatic"}:
        effective_config = dict(config)
        if normalized == "ascalgmmmedianshiftstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMMedianShift(model, device, effective_config)
    if normalized in {"ascalgmmdensityshift", "ascalgmmdensityshiftstatic"}:
        effective_config = dict(config)
        if normalized == "ascalgmmdensityshiftstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMDensityShift(model, device, effective_config)
    if normalized in {"ascalgmmsegmentedshift", "ascalgmmsegmentedshiftstatic"}:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedshiftstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedShift(model, device, effective_config)
    if normalized in {
        "ascalgmmsegmentedhandoffshift",
        "ascalgmmsegmentedhandoffshiftstatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedhandoffshiftstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedHandoffShift(model, device, effective_config)
    if normalized in {
        "ascalgmmsegmentedmemoryshift",
        "ascalgmmsegmentedmemoryshiftstatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryshiftstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryShift(model, device, effective_config)
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
    "ASCAL",
    "ASCALGMM",
    "ASCALGMMDensityShift",
    "ASCALGMMMedianShift",
    "ASCALGMMSegmentedHandoffShift",
    "ASCALGMMSegmentedMemoryShift",
    "ASCALGMMSegmentedShift",
    "ASCALGMMShift",
    "BATCLIP",
    "CLIPTTA",
    "DynaPrompt",
    "SourceOnly",
    "Tent",
    "TentLayerNorm",
    "EATA",
    "IAPL",
    "OST",
    "PoundTTA",
    "CoTTA",
    "RoTTA",
    "LAME",
    "T2A",
    "TDA",
    "SAR",
    "build_method",
]
