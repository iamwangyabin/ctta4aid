from __future__ import annotations

import json
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from PIL import Image


ARROW_FORMAT = "arrow"


@contextmanager
def _preserve_global_rng_state() -> Iterable[None]:
    """Keep metadata loading from changing the caller's augmentation stream."""
    python_state = random.getstate()
    numpy_module = sys.modules.get("numpy")
    numpy_state = (
        numpy_module.random.get_state() if numpy_module is not None else None
    )
    torch_module = sys.modules.get("torch")
    torch_state = torch_module.get_rng_state() if torch_module is not None else None
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            numpy_module.random.set_state(numpy_state)
        if torch_state is not None:
            torch_module.set_rng_state(torch_state)


@dataclass(frozen=True)
class ArrowRecord:
    dataset_root: str
    row_index: int
    label: int
    image_path: str
    sample_id: str


class ArrowDataset:
    """Read the project's canonical binary-image Arrow bundle.

    A bundle is a Hugging Face ``Dataset.save_to_disk`` directory containing
    ``image`` and ``image_path`` columns plus ``mapping.json``. Logical paths
    identify the generator, split, and binary label. Optional split JSON files
    can provide a faster index without changing the bundle contract.
    """

    def __init__(
        self,
        *,
        root: str | Path | Sequence[str | Path],
        generator: str,
        split: str,
        transform: Any = None,
        max_samples_per_class: int | None = None,
        sample_offset_per_class: int = 0,
        seed: int = 0,
        locked_sample_ids: Sequence[str] | None = None,
    ) -> None:
        if not generator:
            raise ValueError("Arrow format requires a generator/domain name")
        if not split:
            raise ValueError("Arrow format requires a split")

        with _preserve_global_rng_state():
            roots = _resolve_dataset_roots(root)
            records = [
                record
                for dataset_root in roots
                for record in _records_from_root(dataset_root, generator, split)
            ]
        if not records:
            joined = ", ".join(str(path) for path in roots)
            raise FileNotFoundError(
                f"No Arrow samples found for generator={generator!r} split={split!r} "
                f"under: {joined}"
            )

        sample_ids = [record.sample_id for record in records]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(
                f"Duplicate Arrow sample IDs found for generator={generator!r} split={split!r}"
            )

        locked_ids = list(locked_sample_ids) if locked_sample_ids is not None else None
        if locked_ids is not None:
            if not locked_ids:
                raise ValueError("Locked Arrow sample manifest must not be empty")
            if any(
                not isinstance(sample_id, str) or not sample_id
                for sample_id in locked_ids
            ):
                raise ValueError("Locked Arrow sample IDs must be non-empty strings")
            if len(locked_ids) != len(set(locked_ids)):
                raise ValueError("Locked Arrow sample manifest contains duplicate sample IDs")
            records_by_id = {record.sample_id: record for record in records}
            missing = [
                sample_id for sample_id in locked_ids if sample_id not in records_by_id
            ]
            if missing:
                raise FileNotFoundError(
                    "Locked Arrow sample is missing from the supplied dataset: "
                    f"{missing[0]}"
                )
            records = [records_by_id[sample_id] for sample_id in locked_ids]

        real_records = [record for record in records if record.label == 0]
        fake_records = [record for record in records if record.label == 1]
        if not real_records or not fake_records:
            raise FileNotFoundError(
                f"Arrow domain must contain both labels, got real={len(real_records)} "
                f"fake={len(fake_records)} for generator={generator!r} split={split!r}"
            )

        if locked_ids is None and (
            max_samples_per_class is not None or sample_offset_per_class > 0
        ):
            rng = random.Random(seed)
            rng.shuffle(real_records)
            rng.shuffle(fake_records)
            end = (
                sample_offset_per_class + max_samples_per_class
                if max_samples_per_class is not None
                else None
            )
            real_records = real_records[sample_offset_per_class:end]
            fake_records = fake_records[sample_offset_per_class:end]

        self.records = (
            records
            if locked_ids is not None
            else sorted(
                [*real_records, *fake_records], key=lambda record: record.sample_id
            )
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Any:
        image_bytes, label, sample_id = self.raw_item(index)
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
        except OSError as error:
            raise RuntimeError(f"Unable to decode Arrow image: {sample_id}") from error
        return image, label, sample_id

    def raw_item(self, index: int) -> tuple[bytes, int, str]:
        record = self.records[index]
        dataset = _load_hf_dataset(record.dataset_root)
        row = dataset[record.row_index]
        if "image_path" not in row:
            raise ValueError(
                f"Missing image_path column in Arrow dataset: {record.dataset_root}"
            )
        actual_path = row["image_path"]
        if str(actual_path) != record.image_path:
            raise RuntimeError(
                f"Arrow mapping mismatch at row {record.row_index}: "
                f"expected {record.image_path!r}, got {actual_path!r}"
            )
        if "image" not in row:
            raise ValueError(f"Missing image column in Arrow dataset: {record.dataset_root}")
        payload = row["image"]
        image_bytes = payload.get("bytes") if isinstance(payload, dict) else payload
        if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
            raise ValueError(f"Invalid image payload for {record.sample_id}")
        return bytes(image_bytes), record.label, record.sample_id


def build_dataset(
    *,
    data_format: str,
    root: str | Path | Sequence[str | Path],
    generator: str | None,
    split: str | None,
    transform: Any,
    max_samples_per_class: int | None = None,
    sample_offset_per_class: int = 0,
    seed: int = 0,
    locked_sample_ids: Sequence[str] | None = None,
) -> ArrowDataset:
    if data_format != ARROW_FORMAT:
        raise ValueError(
            f"Unsupported data format {data_format!r}; this project only reads "
            f"{ARROW_FORMAT!r} bundles"
        )
    if not generator:
        raise ValueError("Arrow format requires a generator/domain name")
    if not split:
        raise ValueError("Arrow format requires a split")
    return ArrowDataset(
        root=root,
        generator=generator,
        split=split,
        transform=transform,
        max_samples_per_class=max_samples_per_class,
        sample_offset_per_class=sample_offset_per_class,
        seed=seed,
        locked_sample_ids=locked_sample_ids,
    )


def _resolve_dataset_roots(
    root: str | Path | Sequence[str | Path],
) -> list[Path]:
    candidates: Iterable[str | Path]
    if isinstance(root, (str, Path)):
        candidates = [root]
    else:
        candidates = root

    resolved: list[Path] = []
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Arrow dataset root does not exist: {path}")
        if (path / "state.json").is_file():
            resolved.append(path)
            continue
        for state_path in sorted(path.rglob("state.json")):
            relative_parts = state_path.relative_to(path).parts
            if any(part.startswith(".") for part in relative_parts):
                continue
            resolved.append(state_path.parent)
    if not resolved:
        raise FileNotFoundError("No Hugging Face save_to_disk datasets found in Arrow roots")
    return list(dict.fromkeys(resolved))


def _records_from_root(root: Path, generator: str, split: str) -> list[ArrowRecord]:
    if not _bundle_matches_request(root, generator, split):
        return []
    mapping_path = root / "mapping.json"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Arrow bundle is missing mapping.json: {root}")
    mapping = _load_json(str(mapping_path))
    if not isinstance(mapping, dict):
        raise ValueError(f"Arrow mapping must be a JSON object: {mapping_path}")

    entries = _split_entries(root, generator, split)
    if entries is None:
        entries = _entries_from_paths(mapping, generator, split)
    if not entries:
        return []

    dataset = _load_hf_dataset(str(root))
    records = []
    for image_path, label_value in entries.items():
        if image_path not in mapping:
            raise ValueError(f"Split entry is missing from {mapping_path}: {image_path}")
        label = int(label_value)
        if label not in {0, 1}:
            raise ValueError(f"Arrow label must be 0/1, got {label} for {image_path}")
        row_index = int(mapping[image_path])
        if not 0 <= row_index < len(dataset):
            raise IndexError(
                f"Arrow row index {row_index} is outside dataset length {len(dataset)}"
            )
        records.append(
            ArrowRecord(
                dataset_root=str(root),
                row_index=row_index,
                label=label,
                image_path=image_path,
                sample_id=f"{root.name}/{image_path}",
            )
        )
    return records


def _bundle_matches_request(root: Path, generator: str, split: str) -> bool:
    manifest_path = root / "bundle_manifest.json"
    if not manifest_path.is_file():
        return True
    manifest = _load_json(str(manifest_path))
    if not isinstance(manifest, dict):
        raise ValueError(f"Arrow bundle manifest must be a JSON object: {manifest_path}")
    bundle_generator = manifest.get("generator")
    bundle_split = manifest.get("split")
    if not isinstance(bundle_generator, str) or not isinstance(bundle_split, str):
        raise ValueError(
            f"Arrow bundle manifest requires generator and split: {manifest_path}"
        )
    return (
        bundle_generator.lower() == generator.lower()
        and bundle_split.lower() == split.lower()
    )


def _split_entries(root: Path, generator: str, split: str) -> dict[str, int] | None:
    candidates = [root / f"{split}.json"]
    if split == "train":
        candidates.append(root / "train_binary.json")
    for path in candidates:
        if not path.is_file():
            continue
        metadata = _load_json(str(path))
        if not isinstance(metadata, dict):
            raise ValueError(f"Arrow split metadata must be a JSON object: {path}")
        entries = metadata.get(generator)
        if isinstance(entries, dict):
            return entries
        if generator.lower() == "progan" and split in {"train", "val"}:
            merged: dict[str, int] = {}
            for subset_entries in metadata.values():
                if isinstance(subset_entries, dict):
                    merged.update(subset_entries)
            if merged:
                return merged
    return None


def _entries_from_paths(
    mapping: dict[str, Any], generator: str, split: str
) -> dict[str, int]:
    entries = {}
    generator_key = generator.lower()
    split_key = split.lower()
    for image_path in mapping:
        components = tuple(part.lower() for part in PurePosixPath(image_path).parts)
        if generator_key not in components or split_key not in components:
            continue
        label = _infer_binary_label(components)
        if label is not None:
            entries[image_path] = label
    return entries


def _infer_binary_label(components: tuple[str, ...]) -> int | None:
    if {"0_real", "nature"}.intersection(components):
        return 0
    if {"1_fake", "ai"}.intersection(components):
        return 1
    return None


@lru_cache(maxsize=16)
def _load_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=8)
def _load_hf_dataset(root: str) -> Any:
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required for the Arrow data format"
        ) from exc
    return load_from_disk(root, keep_in_memory=False)
