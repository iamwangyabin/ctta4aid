from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from PIL import Image, ImageOps


BIAS_CONTROL_MANIFEST = "bias_control_manifest.json"
BIAS_CONTROL_SCHEMA_VERSION = 1
BIAS_CONTROL_CAMPAIGN = "clip_vlm_bias_controlled"

_LABEL_PATH_COMPONENTS = {
    "0_real",
    "1_fake",
    "ai",
    "fake",
    "nature",
    "real",
}


@dataclass(frozen=True)
class BiasControlProfile:
    name: str
    description: str
    target_size: tuple[int, int] | None
    jpeg_qualities: tuple[int, ...]
    quality_assignment: str
    quality_seed: int
    jpeg_subsampling: int = 2
    resize_resampling: str = "bicubic"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "operations": ["exif_transpose", "rgb", "jpeg_encode"],
            "geometry": (
                {"mode": "preserve_visual_dimensions"}
                if self.target_size is None
                else {
                    "mode": "center_crop_and_resize",
                    "width": self.target_size[0],
                    "height": self.target_size[1],
                    "resampling": self.resize_resampling,
                }
            ),
            "jpeg": {
                "qualities": list(self.jpeg_qualities),
                "quality_assignment": self.quality_assignment,
                "quality_seed": self.quality_seed,
                "subsampling": self.jpeg_subsampling,
                "optimize": False,
                "progressive": False,
            },
        }


BIAS_CONTROL_PROFILES: dict[str, BiasControlProfile] = {
    "all_jpeg_q90": BiasControlProfile(
        name="all_jpeg_q90",
        description=(
            "Decode every target image, apply EXIF orientation, convert to RGB, and "
            "encode once as JPEG quality 90 while preserving visual dimensions."
        ),
        target_size=None,
        jpeg_qualities=(90,),
        quality_assignment="constant",
        quality_seed=0,
    ),
    "matched_jpeg": BiasControlProfile(
        name="matched_jpeg",
        description=(
            "Apply identical RGB and 256x256 geometry normalization to every target "
            "image, then assign JPEG quality from a fixed distribution without using "
            "the binary class label."
        ),
        target_size=(256, 256),
        jpeg_qualities=(75, 80, 85, 90, 95),
        quality_assignment="sha256_logical_path_without_label_components_v1",
        quality_seed=20260820,
    ),
}


def get_bias_control_profile(name: str) -> BiasControlProfile:
    try:
        return BIAS_CONTROL_PROFILES[name]
    except KeyError as error:
        supported = ", ".join(sorted(BIAS_CONTROL_PROFILES))
        raise ValueError(
            f"Unknown bias-control profile {name!r}; expected one of: {supported}"
        ) from error


def profile_spec_sha256(profile: BiasControlProfile | str) -> str:
    resolved = (
        get_bias_control_profile(profile) if isinstance(profile, str) else profile
    )
    encoded = json.dumps(
        resolved.as_dict(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _quality_key(image_path: str) -> str:
    components = [
        part
        for part in PurePosixPath(image_path).parts
        if part.lower() not in _LABEL_PATH_COMPONENTS
    ]
    return PurePosixPath(*components).as_posix()


def jpeg_quality_for_sample(profile: BiasControlProfile | str, image_path: str) -> int:
    resolved = (
        get_bias_control_profile(profile) if isinstance(profile, str) else profile
    )
    if len(resolved.jpeg_qualities) == 1:
        return resolved.jpeg_qualities[0]
    key = f"{resolved.quality_seed}\0{_quality_key(image_path)}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], byteorder="big")
    return resolved.jpeg_qualities[bucket % len(resolved.jpeg_qualities)]


def transform_image_bytes(
    payload: bytes,
    *,
    image_path: str,
    profile: BiasControlProfile | str,
) -> tuple[bytes, dict[str, Any]]:
    resolved = (
        get_bias_control_profile(profile) if isinstance(profile, str) else profile
    )
    try:
        with Image.open(BytesIO(payload)) as source:
            source.load()
            source_format = str(source.format or "UNKNOWN").upper()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot decode Arrow image: {image_path}") from error

    source_width, source_height = image.size
    if resolved.target_size is not None:
        image = ImageOps.fit(
            image,
            resolved.target_size,
            method=Image.Resampling.BICUBIC,
            centering=(0.5, 0.5),
        )
    quality = jpeg_quality_for_sample(resolved, image_path)
    output = BytesIO()
    image.save(
        output,
        format="JPEG",
        quality=quality,
        subsampling=resolved.jpeg_subsampling,
        optimize=False,
        progressive=False,
    )
    transformed = output.getvalue()
    width, height = image.size
    return transformed, {
        "source_format": source_format,
        "source_width": source_width,
        "source_height": source_height,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "output_format": "JPEG",
        "output_width": width,
        "output_height": height,
        "output_quality": quality,
        "output_sha256": hashlib.sha256(transformed).hexdigest(),
        "bias_control_profile": resolved.name,
    }


def validate_bias_control_bundles(
    roots: Sequence[Path], expected_profile: str | None
) -> None:
    _validate_bias_control_bundle_paths(
        tuple(str(root.resolve()) for root in roots), expected_profile
    )


@lru_cache(maxsize=64)
def _validate_bias_control_bundle_paths(
    root_paths: tuple[str, ...], expected_profile: str | None
) -> None:
    roots = [Path(path) for path in root_paths]
    manifests = [root / BIAS_CONTROL_MANIFEST for root in roots]
    present = [path.is_file() for path in manifests]
    if expected_profile is None:
        if any(present):
            first = manifests[present.index(True)]
            raise ValueError(
                "Bias-controlled Arrow data requires data.bias_control_profile; "
                f"found {first}"
            )
        return

    profile = get_bias_control_profile(expected_profile)
    expected_hash = profile_spec_sha256(profile)
    missing = [str(path) for path, exists in zip(manifests, present) if not exists]
    if missing:
        raise FileNotFoundError(
            "Bias-controlled Arrow bundle is missing its profile manifest: "
            f"{missing[0]}"
        )
    for path in manifests:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        _validate_bundle_manifest(
            manifest,
            path=path,
            expected_profile=profile.name,
            expected_spec_sha256=expected_hash,
        )


def _validate_bundle_manifest(
    manifest: Any,
    *,
    path: Path,
    expected_profile: str,
    expected_spec_sha256: str,
) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError(f"Bias-control manifest must be a JSON object: {path}")
    expected = {
        "schema_version": BIAS_CONTROL_SCHEMA_VERSION,
        "campaign": BIAS_CONTROL_CAMPAIGN,
        "profile": expected_profile,
        "profile_spec_sha256": expected_spec_sha256,
        "complete": True,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"Bias-control manifest mismatch for {key!r} in {path}: "
                f"expected {value!r}, got {manifest.get(key)!r}"
            )
