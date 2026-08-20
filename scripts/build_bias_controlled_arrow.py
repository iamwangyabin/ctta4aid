from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import PIL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.bias_control import (
    BIAS_CONTROL_CAMPAIGN,
    BIAS_CONTROL_MANIFEST,
    BIAS_CONTROL_PROFILES,
    BIAS_CONTROL_SCHEMA_VERSION,
    get_bias_control_profile,
    profile_spec_sha256,
    transform_image_bytes,
)


_GENERATED_DATASET_METADATA = {
    "bias_control_manifest.json",
    "dataset_info.json",
    "state.json",
}


def discover_arrow_bundles(root: Path) -> list[Path]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Arrow input root does not exist: {root}")
    if (root / "state.json").is_file():
        return [root]
    bundles = []
    for state_path in sorted(root.rglob("state.json")):
        relative_parts = state_path.relative_to(root).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        bundles.append(state_path.parent)
    if not bundles:
        raise FileNotFoundError(f"No save_to_disk Arrow bundles below: {root}")
    return bundles


def _binary_payload(value: Any) -> bytes:
    payload = value.get("bytes") if isinstance(value, Mapping) else value
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("Arrow image column must contain encoded binary bytes")
    return bytes(payload)


def _converted_rows(source_bundle: str, profile_name: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_from_disk
    except ImportError as error:
        raise RuntimeError("datasets is required to convert Arrow bundles") from error

    dataset = load_from_disk(source_bundle, keep_in_memory=False)
    for row_index in range(len(dataset)):
        row = dataset[row_index]
        image_path = row.get("image_path")
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(
                f"Invalid image_path at row {row_index} in {source_bundle}"
            )
        transformed, metadata = transform_image_bytes(
            _binary_payload(row.get("image")),
            image_path=image_path,
            profile=profile_name,
        )
        yield {"image": transformed, "image_path": image_path, **metadata}


def _load_mapping(bundle: Path, image_paths: list[str]) -> dict[str, int]:
    mapping_path = bundle / "mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Arrow bundle is missing mapping.json: {bundle}")
    with mapping_path.open(encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict) or len(mapping) != len(image_paths):
        raise ValueError(f"Arrow mapping does not cover every row: {mapping_path}")
    for row_index, image_path in enumerate(image_paths):
        if mapping.get(image_path) != row_index:
            raise ValueError(
                f"Arrow mapping mismatch at row {row_index}: {mapping_path}"
            )
    return {str(key): int(value) for key, value in mapping.items()}


def _copy_project_metadata(source_bundle: Path, output_bundle: Path) -> None:
    for path in sorted(source_bundle.glob("*.json")):
        if path.name in _GENERATED_DATASET_METADATA:
            continue
        shutil.copy2(path, output_bundle / path.name)


def _ordered_digest(rows: Iterable[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for image_path, value in rows:
        digest.update(image_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def convert_arrow_bundle(
    source_bundle: Path,
    output_bundle: Path,
    *,
    source_relative_path: Path,
    profile_name: str,
    cache_dir: Path,
) -> dict[str, Any]:
    try:
        from datasets import Dataset, Features, Value, load_from_disk
    except ImportError as error:
        raise RuntimeError("datasets is required to convert Arrow bundles") from error

    profile = get_bias_control_profile(profile_name)
    source = load_from_disk(str(source_bundle), keep_in_memory=False)
    required_columns = {"image", "image_path"}
    missing_columns = required_columns.difference(source.column_names)
    if missing_columns:
        raise ValueError(
            f"Arrow bundle {source_bundle} is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    image_paths = [str(value) for value in source["image_path"]]
    _load_mapping(source_bundle, image_paths)
    del source

    output_bundle.parent.mkdir(parents=True, exist_ok=True)
    features = Features(
        {
            "image": Value("binary"),
            "image_path": Value("string"),
            "source_format": Value("string"),
            "source_width": Value("int32"),
            "source_height": Value("int32"),
            "source_sha256": Value("string"),
            "output_format": Value("string"),
            "output_width": Value("int32"),
            "output_height": Value("int32"),
            "output_quality": Value("int16"),
            "output_sha256": Value("string"),
            "bias_control_profile": Value("string"),
        }
    )
    converted: Dataset = Dataset.from_generator(
        _converted_rows,
        gen_kwargs={
            "source_bundle": str(source_bundle),
            "profile_name": profile.name,
        },
        features=features,
        cache_dir=str(cache_dir),
    )
    if len(converted) != len(image_paths):
        raise RuntimeError(
            f"Converted row count changed for {source_bundle}: "
            f"{len(image_paths)} -> {len(converted)}"
        )
    if list(converted["image_path"]) != image_paths:
        raise RuntimeError(f"Converted row order changed for {source_bundle}")
    converted.save_to_disk(str(output_bundle))
    _copy_project_metadata(source_bundle, output_bundle)

    source_formats = Counter(str(value) for value in converted["source_format"])
    output_formats = Counter(str(value) for value in converted["output_format"])
    output_qualities = Counter(int(value) for value in converted["output_quality"])
    output_geometry = Counter(
        f"{int(width)}x{int(height)}"
        for width, height in zip(
            converted["output_width"], converted["output_height"]
        )
    )
    source_digest = _ordered_digest(
        zip(image_paths, (str(value) for value in converted["source_sha256"]))
    )
    output_digest = _ordered_digest(
        zip(image_paths, (str(value) for value in converted["output_sha256"]))
    )
    logical_digest = _ordered_digest(
        (image_path, str(row_index))
        for row_index, image_path in enumerate(image_paths)
    )
    manifest = {
        "schema_version": BIAS_CONTROL_SCHEMA_VERSION,
        "campaign": BIAS_CONTROL_CAMPAIGN,
        "profile": profile.name,
        "profile_spec_sha256": profile_spec_sha256(profile),
        "profile_spec": profile.as_dict(),
        "complete": True,
        "source_bundle_relative_path": source_relative_path.as_posix(),
        "sample_count": len(image_paths),
        "logical_paths_sha256": logical_digest,
        "source_bytes_sha256": source_digest,
        "output_bytes_sha256": output_digest,
        "source_format_counts": dict(sorted(source_formats.items())),
        "output_format_counts": dict(sorted(output_formats.items())),
        "output_quality_counts": {
            str(key): value for key, value in sorted(output_qualities.items())
        },
        "output_geometry_counts": dict(sorted(output_geometry.items())),
        "encoder": {"pillow_version": PIL.__version__},
    }
    (output_bundle / BIAS_CONTROL_MANIFEST).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    del converted
    return manifest


def build_bias_controlled_arrow(
    source_root: Path, output_root: Path, profile_name: str
) -> dict[str, Any]:
    source_root = source_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    profile = get_bias_control_profile(profile_name)
    if profile.name not in output_root.parts:
        raise ValueError(
            f"Output path must contain the exact profile component {profile.name!r}"
        )
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing bias-controlled Arrow root: {output_root}"
        )
    if output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root):
        raise ValueError("Input and output Arrow roots must not contain one another")

    bundles = discover_arrow_bundles(source_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.partial-", dir=str(output_root.parent)
        )
    )
    cache_dir = partial_root / ".cache"
    summaries: list[dict[str, Any]] = []
    try:
        for source_bundle in bundles:
            relative_path = (
                Path(source_root.name)
                if source_bundle == source_root
                else source_bundle.relative_to(source_root)
            )
            summary = convert_arrow_bundle(
                source_bundle,
                partial_root / relative_path,
                source_relative_path=relative_path,
                profile_name=profile.name,
                cache_dir=cache_dir / relative_path,
            )
            summaries.append(summary)
            print(
                f"profile={profile.name} bundle={relative_path.as_posix()} "
                f"samples={summary['sample_count']}"
            )
        root_manifest = {
            "schema_version": BIAS_CONTROL_SCHEMA_VERSION,
            "campaign": BIAS_CONTROL_CAMPAIGN,
            "profile": profile.name,
            "profile_spec_sha256": profile_spec_sha256(profile),
            "profile_spec": profile.as_dict(),
            "complete": True,
            "source_root_name": source_root.name,
            "bundle_count": len(summaries),
            "sample_count": sum(int(item["sample_count"]) for item in summaries),
            "bundles": summaries,
            "encoder": {"pillow_version": PIL.__version__},
        }
        (partial_root / BIAS_CONTROL_MANIFEST).write_text(
            json.dumps(root_manifest, indent=2), encoding="utf-8"
        )
        shutil.rmtree(cache_dir, ignore_errors=True)
        partial_root.rename(output_root)
        return root_manifest
    except BaseException:
        shutil.rmtree(partial_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create an immutable, separately identified JPEG bias-control copy of "
            "project-standard Arrow data"
        )
    )
    parser.add_argument(
        "--profile", required=True, choices=sorted(BIAS_CONTROL_PROFILES)
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    summary = build_bias_controlled_arrow(
        args.input_root, args.output_root, args.profile
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
