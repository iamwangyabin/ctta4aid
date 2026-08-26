from __future__ import annotations

from typing import Any

from .ascal import ASCAL
from .ascal_gmm import (
    ASCALGMM,
    ASCALGMMSegmentedMemoryPosteriorAnalyticExpert,
    ASCALGMMSegmentedMemoryPosteriorConditionalResidual,
    ASCALGMMDensityShift,
    ASCALGMMSegmentedMemoryPosteriorEvidenceGatedRidgeExpert,
    ASCALGMMSegmentedMemoryPosteriorEqualPriorRidgeExpert,
    ASCALGMMMedianShift,
    ASCALGMMSegmentedHandoffShift,
    ASCALGMMSegmentedMemoryPosterior,
    ASCALGMMSegmentedMemoryPosteriorCurrentProjection,
    ASCALGMMSegmentedMemoryPosteriorGlobalResidual,
    ASCALGMMSegmentedMemoryPosteriorGuardedProjection,
    ASCALGMMSegmentedMemoryPosteriorJointRidge,
    ASCALGMMSegmentedMemoryPosteriorLiveRoute,
    ASCALGMMSegmentedMemoryPosteriorMixtureResidual,
    ASCALGMMSegmentedMemoryPosteriorMDLRoute,
    ASCALGMMSegmentedMemoryPosteriorOrdinalRidge,
    ASCALGMMSegmentedMemoryPosteriorOrdinalRoute,
    ASCALGMMSegmentedMemoryPosteriorPairwiseRidge,
    ASCALGMMSegmentedMemoryPosteriorPreRoute,
    ASCALGMMSegmentedMemoryPosteriorProjection,
    ASCALGMMSegmentedMemoryPosteriorRealDeviationResidual,
    ASCALGMMSegmentedMemoryPosteriorRMSRidgeExpert,
    ASCALGMMSegmentedMemoryPosteriorRoutedRidgeResidual,
    ASCALGMMSegmentedMemoryPosteriorRoutedResidual,
    ASCALGMMSegmentedMemoryPosteriorSupportProjection,
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
    if normalized in {
        "ascalgmmsegmentedmemoryposterior",
        "ascalgmmsegmentedmemoryposteriorstatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosterior(model, device, effective_config)
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorprojection",
        "ascalgmmsegmentedmemoryposteriorprojectionstatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorprojectionstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorProjection(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorpreroute",
        "ascalgmmsegmentedmemoryposteriorpreroutestatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorpreroutestatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorPreRoute(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriormdlroute",
        "ascalgmmsegmentedmemoryposteriormdlroutestatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriormdlroutestatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorMDLRoute(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorliveroute",
        "ascalgmmsegmentedmemoryposteriorliveroutestatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorliveroutestatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorLiveRoute(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorordinalroute",
        "ascalgmmsegmentedmemoryposteriorordinalroutestatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorordinalroutestatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorOrdinalRoute(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorordinalridge",
        "ascalgmmsegmentedmemoryposteriorordinalridgestatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorordinalridgestatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorOrdinalRidge(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorjointridge",
        "ascalgmmsegmentedmemoryposteriorjointridgestatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorjointridgestatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorJointRidge(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorpairwiseridge",
        "ascalgmmsegmentedmemoryposteriorpairwiseridgestatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorpairwiseridgestatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorPairwiseRidge(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposterioranalyticexpert",
        "ascalgmmsegmentedmemoryposterioranalyticexpertstatic",
    }:
        effective_config = dict(config)
        if normalized == (
            "ascalgmmsegmentedmemoryposterioranalyticexpertstatic"
        ):
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorAnalyticExpert(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorrmsridgeexpert",
        "ascalgmmsegmentedmemoryposteriorrmsridgeexpertstatic",
    }:
        effective_config = dict(config)
        if normalized == (
            "ascalgmmsegmentedmemoryposteriorrmsridgeexpertstatic"
        ):
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorRMSRidgeExpert(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorequalpriorridgeexpert",
        "ascalgmmsegmentedmemoryposteriorequalpriorridgeexpertstatic",
    }:
        effective_config = dict(config)
        if normalized == (
            "ascalgmmsegmentedmemoryposteriorequalpriorridgeexpertstatic"
        ):
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorEqualPriorRidgeExpert(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorevidencegatedridgeexpert",
        "ascalgmmsegmentedmemoryposteriorevidencegatedridgeexpertstatic",
    }:
        effective_config = dict(config)
        if normalized == (
            "ascalgmmsegmentedmemoryposteriorevidencegatedridgeexpertstatic"
        ):
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorEvidenceGatedRidgeExpert(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorroutedresidual",
        "ascalgmmsegmentedmemoryposteriorroutedresidualstatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorroutedresidualstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorRoutedResidual(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorroutedridgeresidual",
        "ascalgmmsegmentedmemoryposteriorroutedridgeresidualstatic",
    }:
        effective_config = dict(config)
        if normalized == (
            "ascalgmmsegmentedmemoryposteriorroutedridgeresidualstatic"
        ):
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorRoutedRidgeResidual(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorcurrentprojection",
        "ascalgmmsegmentedmemoryposteriorcurrentprojectionstatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorcurrentprojectionstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorCurrentProjection(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorguardedprojection",
        "ascalgmmsegmentedmemoryposteriorguardedprojectionstatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorguardedprojectionstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorGuardedProjection(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorsupportprojection",
        "ascalgmmsegmentedmemoryposteriorsupportprojectionstatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorsupportprojectionstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorSupportProjection(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorglobalresidual",
        "ascalgmmsegmentedmemoryposteriorglobalresidualstatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorglobalresidualstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorGlobalResidual(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriormixtureresidual",
        "ascalgmmsegmentedmemoryposteriormixtureresidualstatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriormixtureresidualstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorMixtureResidual(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorrealdeviationresidual",
        "ascalgmmsegmentedmemoryposteriorrealdeviationresidualstatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorrealdeviationresidualstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorRealDeviationResidual(
            model, device, effective_config
        )
    if normalized in {
        "ascalgmmsegmentedmemoryposteriorconditionalresidual",
        "ascalgmmsegmentedmemoryposteriorconditionalresidualstatic",
    }:
        effective_config = dict(config)
        if normalized == "ascalgmmsegmentedmemoryposteriorconditionalresidualstatic":
            effective_config.setdefault("adaptation_mode", "static")
        return ASCALGMMSegmentedMemoryPosteriorConditionalResidual(
            model, device, effective_config
        )
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
    "ASCALGMMSegmentedMemoryPosteriorAnalyticExpert",
    "ASCALGMMSegmentedMemoryPosteriorConditionalResidual",
    "ASCALGMMDensityShift",
    "ASCALGMMSegmentedMemoryPosteriorEvidenceGatedRidgeExpert",
    "ASCALGMMSegmentedMemoryPosteriorEqualPriorRidgeExpert",
    "ASCALGMMMedianShift",
    "ASCALGMMSegmentedHandoffShift",
    "ASCALGMMSegmentedMemoryPosterior",
    "ASCALGMMSegmentedMemoryPosteriorCurrentProjection",
    "ASCALGMMSegmentedMemoryPosteriorGlobalResidual",
    "ASCALGMMSegmentedMemoryPosteriorGuardedProjection",
    "ASCALGMMSegmentedMemoryPosteriorLiveRoute",
    "ASCALGMMSegmentedMemoryPosteriorMixtureResidual",
    "ASCALGMMSegmentedMemoryPosteriorMDLRoute",
    "ASCALGMMSegmentedMemoryPosteriorOrdinalRoute",
    "ASCALGMMSegmentedMemoryPosteriorPreRoute",
    "ASCALGMMSegmentedMemoryPosteriorProjection",
    "ASCALGMMSegmentedMemoryPosteriorRealDeviationResidual",
    "ASCALGMMSegmentedMemoryPosteriorRMSRidgeExpert",
    "ASCALGMMSegmentedMemoryPosteriorSupportProjection",
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
