from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from zipfile import ZipFile


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
    directory: Path | None = None
    archive: Path | None = None
    prefix: str | None = None
    expected_count: int | None = None
    expected_bytes: int | None = None


@dataclass(frozen=True)
class ImageRecord:
    source_path: Path
    archive_member: str | None
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
        directory_value = item.get("path")
        archive_value = item.get("archive")
        if (directory_value is None) == (archive_value is None):
            raise ValueError(f"Source {index} must contain exactly one of path or archive")
        directory = None
        archive = None
        prefix = None
        if directory_value is not None:
            directory = _resolve_plan_path(path, directory_value)
            if not directory.is_dir():
                raise FileNotFoundError(f"Source directory does not exist: {directory}")
        else:
            archive = _resolve_plan_path(path, archive_value)
            if not archive.is_file():
                raise FileNotFoundError(f"Source archive does not exist: {archive}")
            prefix_path = PurePosixPath(str(item.get("prefix", "")).strip("/"))
            if (
                not prefix_path.parts
                or prefix_path.is_absolute()
                or ".." in prefix_path.parts
            ):
                raise ValueError(f"Invalid archive prefix: {item.get('prefix')!r}")
            prefix = prefix_path.as_posix()
        sources.append(
            SourceSpec(
                subset=subset,
                split=split,
                label=label,
                directory=directory,
                archive=archive,
                prefix=prefix,
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


def _resolve_plan_path(plan_path: Path, value: Any) -> Path:
    resolved = Path(str(value)).expanduser()
    if not resolved.is_absolute():
        resolved = plan_path.parent / resolved
    return resolved.resolve()


def collect_records(
    sources: list[SourceSpec],
) -> tuple[list[ImageRecord], list[dict[str, Any]]]:
    records: list[ImageRecord] = []
    source_summaries = []
    seen_paths: set[str] = set()
    for source in sources:
        class_name = "nature" if source.label == 0 else "ai"
        candidates = _source_candidates(source)
        byte_count = sum(size for _, _, size in candidates)
        if source.expected_count is not None and len(candidates) != source.expected_count:
            raise ValueError(
                f"Unexpected image count under {_source_description(source)}: "
                f"expected {source.expected_count}, got {len(candidates)}"
            )
        if source.expected_bytes is not None and byte_count != source.expected_bytes:
            raise ValueError(
                f"Unexpected byte count under {_source_description(source)}: "
                f"expected {source.expected_bytes}, got {byte_count}"
            )
        if not candidates:
            raise ValueError(
                f"Source has no supported images: {_source_description(source)}"
            )

        start_index = len(records)
        for source_path, archive_member, size in candidates:
            if archive_member is None:
                relative = source_path.relative_to(source.directory)
            else:
                member_path = PurePosixPath(archive_member)
                prefix_path = PurePosixPath(source.prefix)
                relative = PurePosixPath(*member_path.parts[len(prefix_path.parts) :])
            image_path = PurePosixPath(
                source.subset, source.split, class_name, *relative.parts
            ).as_posix()
            if image_path in seen_paths:
                raise ValueError(f"Duplicate canonical image path: {image_path}")
            seen_paths.add(image_path)
            records.append(
                ImageRecord(
                    source_path=source_path,
                    archive_member=archive_member,
                    image_path=image_path,
                    subset=source.subset,
                    split=source.split,
                    label=source.label,
                    size=size,
                )
            )
        summary = {
            "subset": source.subset,
            "split": source.split,
            "label": source.label,
            "source_type": "directory" if source.directory else "zip",
            "rows": len(candidates),
            "bytes": byte_count,
            "row_start": start_index,
            "row_end_exclusive": len(records),
        }
        if source.directory:
            summary["path"] = str(source.directory)
        else:
            summary["archive"] = str(source.archive)
            summary["prefix"] = source.prefix
        source_summaries.append(summary)
    return records, source_summaries


def _source_description(source: SourceSpec) -> str:
    if source.directory:
        return str(source.directory)
    return f"{source.archive}!/{source.prefix}"


def _source_candidates(
    source: SourceSpec,
) -> list[tuple[Path, str | None, int]]:
    if source.directory:
        return [
            (candidate, None, candidate.stat().st_size)
            for candidate in sorted(source.directory.rglob("*"))
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
        ]

    prefix_path = PurePosixPath(source.prefix)
    candidates = []
    with ZipFile(source.archive) as archive:
        for info in sorted(archive.infolist(), key=lambda item: item.filename):
            member_path = PurePosixPath(info.filename)
            if info.is_dir() or member_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if member_path.is_absolute() or ".." in member_path.parts:
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            if member_path.parts[: len(prefix_path.parts)] != prefix_path.parts:
                continue
            if len(member_path.parts) == len(prefix_path.parts):
                continue
            candidates.append((source.archive, info.filename, info.file_size))
    return candidates


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


def _record_bytes(record: ImageRecord) -> bytes:
    if record.archive_member is None:
        return record.source_path.read_bytes()
    with ZipFile(record.source_path) as archive:
        return archive.read(record.archive_member)


def _iter_arrow_rows(records: Iterable[ImageRecord]) -> Iterable[dict[str, Any]]:
    archives: dict[Path, ZipFile] = {}
    try:
        for record in records:
            if record.archive_member is None:
                payload = record.source_path.read_bytes()
            else:
                archive = archives.get(record.source_path)
                if archive is None:
                    archive = ZipFile(record.source_path)
                    archives[record.source_path] = archive
                payload = archive.read(record.archive_member)
            yield {"image_path": record.image_path, "image": payload}
    finally:
        for archive in archives.values():
            archive.close()


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

    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(
        lambda: _iter_arrow_rows(records),
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
        source_sha = hashlib.sha256(_record_bytes(expected)).hexdigest()
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
