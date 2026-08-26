from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


SOURCE_ANALYTIC_RIDGE_PROFILE = "clip_lora_l2_bias_two_output_ridge_v1"


def classifier_features(model: Any, images: Any) -> Any:
    features = model.forward_features(images)
    transform = getattr(model, "forward_classifier_features", None)
    return transform(features) if callable(transform) else features


def fit_source_analytic_ridge(
    model: Any,
    loader: Any,
    device: Any,
    *,
    regularization: float = 1.0,
) -> dict[str, Any]:
    """Fit a two-output Ridge head and retain its complete source statistics."""

    import torch
    import torch.nn.functional as functional

    if not math.isfinite(regularization) or regularization <= 0.0:
        raise ValueError("Source analytic Ridge regularization must be positive")
    classifier = getattr(model, "classifier", None)
    feature_dim = int(getattr(model, "feature_dim", 0))
    if (
        classifier is None
        or feature_dim < 1
        or int(getattr(classifier, "in_features", 0)) != feature_dim
        or int(getattr(classifier, "out_features", 0)) != 2
        or getattr(classifier, "bias", None) is None
    ):
        raise TypeError(
            "Source analytic Ridge requires a biased two-class linear classifier"
        )

    design_dim = feature_dim + 1
    accumulator_device = torch.device(device)
    gram = regularization * torch.eye(
        design_dim, dtype=torch.float64, device=accumulator_device
    )
    cross_covariance = torch.zeros(
        (design_dim, 2), dtype=torch.float64, device=accumulator_device
    )
    class_mass = torch.zeros(2, dtype=torch.float64, device=accumulator_device)
    samples = 0
    was_training = bool(model.training)
    model.eval()
    with torch.no_grad():
        for images, labels, _ in loader:
            labels = labels.to(accumulator_device, non_blocking=True).long()
            features = classifier_features(
                model, images.to(accumulator_device, non_blocking=True)
            ).double()
            if features.ndim != 2 or int(features.shape[1]) != feature_dim:
                raise RuntimeError(
                    "Source analytic Ridge received an invalid feature matrix"
                )
            design = torch.cat(
                (
                    features,
                    torch.ones(
                        (int(features.shape[0]), 1),
                        dtype=torch.float64,
                        device=accumulator_device,
                    ),
                ),
                dim=1,
            )
            targets = functional.one_hot(labels, num_classes=2).double()
            gram.addmm_(design.t(), design)
            cross_covariance.addmm_(design.t(), targets)
            class_mass += torch.bincount(labels, minlength=2).double()
            samples += int(labels.numel())
    model.train(was_training)
    if samples < 2 or bool(torch.any(class_mass <= 0.0)):
        raise RuntimeError(
            "Source analytic Ridge requires samples from both binary classes"
        )

    gram = 0.5 * (gram + gram.t())
    try:
        factor = torch.linalg.cholesky(gram)
        weights = torch.cholesky_solve(cross_covariance, factor)
        inverse_gram = torch.cholesky_inverse(factor)
    except RuntimeError as error:
        raise RuntimeError("Source analytic Ridge solve failed") from error
    if not (
        bool(torch.all(torch.isfinite(weights)))
        and bool(torch.all(torch.isfinite(inverse_gram)))
    ):
        raise FloatingPointError("Source analytic Ridge produced non-finite statistics")

    state = {
        "profile": SOURCE_ANALYTIC_RIDGE_PROFILE,
        "feature_coordinate": "l2_normalized_clip_lora_feature_plus_constant_bias",
        "feature_normalization": "l2",
        "bias_coordinate": True,
        "targets": "two_output_real_fake_one_hot",
        "regularization": float(regularization),
        "feature_dim": feature_dim,
        "design_dim": design_dim,
        "samples": samples,
        "class_mass": class_mass.detach().cpu(),
        "gram": gram.detach().cpu(),
        "inverse_gram": inverse_gram.detach().cpu(),
        "cross_covariance": cross_covariance.detach().cpu(),
        "weights": weights.detach().cpu(),
    }
    state["statistics_sha256"] = analytic_ridge_statistics_sha256(state)
    return state


def install_source_analytic_ridge(model: Any, state: dict[str, Any]) -> None:
    import torch

    arrays = source_analytic_ridge_arrays(
        state, expected_feature_dim=int(getattr(model, "feature_dim", 0))
    )
    classifier = getattr(model, "classifier", None)
    if classifier is None or getattr(classifier, "bias", None) is None:
        raise TypeError("Cannot install analytic Ridge without a biased classifier")
    weights = torch.from_numpy(arrays["weights"])
    with torch.no_grad():
        classifier.weight.copy_(
            weights[:-1].t().to(
                device=classifier.weight.device,
                dtype=classifier.weight.dtype,
            )
        )
        classifier.bias.copy_(
            weights[-1].to(
                device=classifier.bias.device,
                dtype=classifier.bias.dtype,
            )
        )


def source_analytic_ridge_arrays(
    state: dict[str, Any],
    *,
    expected_feature_dim: int | None = None,
) -> dict[str, Any]:
    if not isinstance(state, dict):
        raise ValueError("Source analytic Ridge checkpoint state must be a mapping")
    if state.get("profile") != SOURCE_ANALYTIC_RIDGE_PROFILE:
        raise ValueError("Unsupported source analytic Ridge profile")
    if state.get("feature_normalization") != "l2" or not bool(
        state.get("bias_coordinate")
    ):
        raise ValueError(
            "Source analytic Ridge requires the normalized feature-plus-bias coordinate"
        )
    feature_dim = int(state.get("feature_dim", 0))
    design_dim = int(state.get("design_dim", 0))
    samples = int(state.get("samples", 0))
    regularization = float(state.get("regularization", 0.0))
    if (
        feature_dim < 1
        or design_dim != feature_dim + 1
        or samples < 2
        or not math.isfinite(regularization)
        or regularization <= 0.0
    ):
        raise ValueError("Source analytic Ridge metadata is invalid")
    if expected_feature_dim is not None and feature_dim != expected_feature_dim:
        raise ValueError("Source analytic Ridge feature dimension does not match the model")

    def array(name: str, shape: tuple[int, ...]) -> np.ndarray:
        value = state.get(name)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result = np.asarray(value, dtype=np.float64)
        if result.shape != shape or not np.all(np.isfinite(result)):
            raise ValueError(f"Source analytic Ridge {name} is invalid")
        return result.copy()

    arrays = {
        "profile": SOURCE_ANALYTIC_RIDGE_PROFILE,
        "regularization": regularization,
        "feature_dim": feature_dim,
        "design_dim": design_dim,
        "samples": samples,
        "class_mass": array("class_mass", (2,)),
        "gram": array("gram", (design_dim, design_dim)),
        "inverse_gram": array("inverse_gram", (design_dim, design_dim)),
        "cross_covariance": array("cross_covariance", (design_dim, 2)),
        "weights": array("weights", (design_dim, 2)),
        "statistics_sha256": str(state.get("statistics_sha256", "")),
    }
    if np.any(arrays["class_mass"] <= 0.0):
        raise ValueError("Source analytic Ridge requires positive mass for both classes")
    if not np.allclose(
        arrays["gram"] @ arrays["weights"],
        arrays["cross_covariance"],
        rtol=1e-6,
        atol=1e-8,
    ):
        raise ValueError("Source analytic Ridge weights do not match its statistics")
    identity = arrays["gram"] @ arrays["inverse_gram"]
    if not np.allclose(identity, np.eye(design_dim), rtol=1e-5, atol=1e-7):
        raise ValueError("Source analytic Ridge inverse Gram is inconsistent")
    expected_hash = analytic_ridge_statistics_sha256(state)
    if arrays["statistics_sha256"] != expected_hash:
        raise ValueError("Source analytic Ridge statistics hash does not match")
    return arrays


def analytic_ridge_statistics_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(state.get("profile", "")).encode("utf-8"))
    for name in (
        "regularization",
        "feature_dim",
        "design_dim",
        "samples",
    ):
        digest.update(str(state.get(name)).encode("ascii"))
    for name in (
        "class_mass",
        "gram",
        "inverse_gram",
        "cross_covariance",
        "weights",
    ):
        value = state.get(name)
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
        digest.update(name.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()
