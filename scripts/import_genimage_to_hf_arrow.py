from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


IMAGE_SUFFIXES = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class SourceSpec:
    subset: str
    split: str
    label: int
    path: Path
    expected_count: int | None = None
    expected_bytes: int | None = None


@dataclass(frozen=True)
class ImageRecord:
    source_path: Path
    image_path: str
    subset: str
    split: str
    label: int
    size: int


def _safe_component(value: Any, field: str) -> str:
    component = str(value)
    if not component or component in {".", ".."} or "/" in component or "\\" in component:
        raise ValueError(f"Invalid {field}: {value!r}")
    return component


def load_plan(path: Path) -> tuple[dict[str, Any], list[SourceSpec]]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(plan, dict) or not isinstance(plan.get("sources"), list):
        raise ValueError("Conversion plan must contain a sources list")

    sources = []
    for index, item in enumerate(plan["sources"]):
        if not isinstance(item, dict):
            raise ValueError(f"Source {index} must be an object")
        subset = _safe_component(item.get("subset"), "subset")
        split = _safe_component(item.get("split"), "split")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split!r}")
        label = int(item.get("label"))
        if label not in {0, 1}:
            raise ValueError(f"Source label must be 0 or 1, got {label}")
        source_path = Path(str(item.get("path", ""))).expanduser()
        if not source_path.is_absolute():
            source_path = path.parent / source_path
        source_path = source_path.resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"Source directory does not exist: {source_path}")
        sources.append(
            SourceSpec(
                subset=subset,
                split=split,
                label=label,
                path=source_path,
                expected_count=(
                    int(item["expected_count"])
                    if item.get("expected_count") is not None
                    else None
                ),
                expected_bytes=(
                    int(item["expected_bytes"])
                    if item.get("expected_bytes") is not None
                    else None
                ),
            )
        )

    if not sources:
        raise ValueError("Conversion plan contains no sources")
    labels_by_group: dict[tuple[str, str], set[int]] = {}
    for source in sources:
        labels_by_group.setdefault((source.subset, source.split), set()).add(source.label)
    incomplete = [group for group, labels in labels_by_group.items() if labels != {0, 1}]
    if incomplete:
        raise ValueError(f"Every subset/split must include labels 0 and 1: {incomplete}")
    return plan, sources


def collect_records(
    sources: list[SourceSpec],
) -> tuple[list[ImageRecord], list[dict[str, Any]]]:
    records: list[ImageRecord] = []
    source_summaries = []
    seen_paths: set[str] = set()
    for source in sources:
        class_name = "nature" if source.label == 0 else "ai"
        files = sorted(
            candidate
            for candidate in source.path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        )
        byte_count = sum(candidate.stat().st_size for candidate in files)
        if source.expected_count is not None and len(files) != source.expected_count:
            raise ValueError(
                f"Unexpected image count under {source.path}: "
                f"expected {source.expected_count}, got {len(files)}"
            )
        if source.expected_bytes is not None and byte_count != source.expected_bytes:
            raise ValueError(
                f"Unexpected byte count under {source.path}: "
                f"expected {source.expected_bytes}, got {byte_count}"
            )
        if not files:
            raise ValueError(f"Source directory has no supported images: {source.path}")

        start_index = len(records)
        for candidate in files:
            relative = candidate.relative_to(source.path)
            image_path = PurePosixPath(
                source.subset, source.split, class_name, *relative.parts
            ).as_posix()
            if image_path in seen_paths:
                raise ValueError(f"Duplicate canonical image path: {image_path}")
            seen_paths.add(image_path)
            records.append(
                ImageRecord(
                    source_path=candidate,
                    image_path=image_path,
                    subset=source.subset,
                    split=source.split,
                    label=source.label,
                    size=candidate.stat().st_size,
                )
            )
        source_summaries.append(
            {
                "subset": source.subset,
                "split": source.split,
                "label": source.label,
                "path": str(source.path),
                "rows": len(files),
                "bytes": byte_count,
                "row_start": start_index,
                "row_end_exclusive": len(records),
            }
        )
    return records, source_summaries


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_sha256(records: Iterable[ImageRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.image_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record.size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def convert(
    *,
    plan_path: Path,
    output: Path,
    cache_dir: Path,
    max_shard_size: str,
) -> dict[str, Any]:
    plan, sources = load_plan(plan_path)
    records, source_summaries = collect_records(sources)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    partial = output.parent / f".{output.name}.partial-{os.getpid()}"
    if partial.exists():
        raise FileExistsError(f"Partial output already exists: {partial}")

    try:
        from datasets import Dataset, Features, Value, load_from_disk
    except ImportError as exc:
        raise RuntimeError("datasets is required for GenImage Arrow conversion") from exc

    def rows() -> Iterable[dict[str, Any]]:
        for record in records:
            yield {
                "image_path": record.image_path,
                "image": record.source_path.read_bytes(),
            }

    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(
        rows,
        features=Features(
            {
                "image_path": Value("string"),
                "image": Value("binary"),
            }
        ),
        cache_dir=str(cache_dir),
    )
    if len(dataset) != len(records):
        raise RuntimeError(f"Arrow row count changed: {len(dataset)} != {len(records)}")
    dataset.save_to_disk(str(partial), max_shard_size=max_shard_size)

    mapping = {record.image_path: index for index, record in enumerate(records)}
    split_metadata: dict[str, dict[str, dict[str, int]]] = {}
    for record in records:
        split_metadata.setdefault(record.split, {}).setdefault(record.subset, {})[
            record.image_path
        ] = record.label
    (partial / "mapping.json").write_text(
        json.dumps(mapping, separators=(",", ":")), encoding="utf-8"
    )
    for split, metadata in split_metadata.items():
        (partial / f"{split}.json").write_text(
            json.dumps(metadata, separators=(",", ":")), encoding="utf-8"
        )

    reloaded = load_from_disk(str(partial), keep_in_memory=False)
    sample_indices = sorted(
        {
            0,
            len(records) - 1,
            *(summary["row_start"] for summary in source_summaries),
        }
    )
    byte_checks = []
    for index in sample_indices:
        expected = records[index]
        row = reloaded[index]
        if row["image_path"] != expected.image_path:
            raise RuntimeError(f"Arrow path mismatch at row {index}")
        payload = row["image"]
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise RuntimeError(f"Arrow image is not binary at row {index}")
        source_sha = _sha256(expected.source_path)
        arrow_sha = hashlib.sha256(bytes(payload)).hexdigest()
        if source_sha != arrow_sha:
            raise RuntimeError(f"Arrow bytes changed at row {index}")
        byte_checks.append(
            {"row": index, "image_path": expected.image_path, "sha256": source_sha}
        )

    manifest = {
        "format": "ctta4aid_genimage_hf_arrow_v1",
        "plan": str(plan_path.resolve()),
        "plan_sha256": _sha256(plan_path),
        "plan_metadata": {key: value for key, value in plan.items() if key != "sources"},
        "rows": len(records),
        "image_bytes": sum(record.size for record in records),
        "inventory_sha256": _inventory_sha256(records),
        "sources": source_summaries,
        "splits": sorted(split_metadata),
        "subsets": sorted({record.subset for record in records}),
        "max_shard_size": max_shard_size,
        "sample_byte_checks": byte_checks,
    }
    (partial / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    manifest["metadata_sha256"] = {
        path.name: _sha256(path)
        for path in sorted(partial.glob("*.json"))
        if path.name != "conversion_manifest.json"
    }
    (partial / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    partial.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a verified GenImage directory plan to save_to_disk Arrow."
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-shard-size", default="1GB")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir
        else output.parent / ".datasets_cache"
    )
    manifest = convert(
        plan_path=args.plan.expanduser().resolve(),
        output=output,
        cache_dir=cache_dir,
        max_shard_size=args.max_shard_size,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
