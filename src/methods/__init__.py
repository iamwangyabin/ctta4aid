from __future__ import annotations

from typing import Any

from .base import TTAMethod
from .batclip import BATCLIP
from .cliptta import CLIPTTA
from .cotta import CoTTA
from .dynaprompt import DynaPrompt
from .eata import EATA
from .iapl import IAPL
from .lame import LAME
from .ost import OST
from .ours import Ours
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
    """Build one registered method.

    Ours has one implementation and two adaptive readouts: ``ours`` is the
    final calibrated setting and ``ours_no_calibrated_readout`` is its retained base-readout
    ablation. ``ours_static`` is the paired frozen-source control.
    """

    normalized = name.lower().replace("²", "2").replace("_", "").replace("-", "")
    config = dict(config or {})
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
    if normalized in {"ours", "oursnocalibratedreadout", "oursstatic"}:
        expected_readout = (
            "base" if normalized == "oursnocalibratedreadout" else "calibrated"
        )
        configured_readout = str(config.get("readout_mode", expected_readout)).lower()
        if configured_readout != expected_readout:
            raise ValueError(
                f"{name} fixes readout_mode={expected_readout}, got {configured_readout}"
            )
        config["readout_mode"] = expected_readout
        if normalized == "oursstatic":
            configured_adaptation = str(
                config.get("adaptation_mode", "static")
            ).lower()
            if configured_adaptation != "static":
                raise ValueError(
                    f"{name} fixes adaptation_mode=static, got {configured_adaptation}"
                )
            config["adaptation_mode"] = "static"
        return Ours(model, device, config)
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
    "Ours",
    "CoTTA",
    "RoTTA",
    "LAME",
    "T2A",
    "TDA",
    "SAR",
    "build_method",
]
