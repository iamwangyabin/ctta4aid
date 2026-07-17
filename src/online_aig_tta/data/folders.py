from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    label: int


def _image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _label_directories(root: Path, label_names: Iterable[str]) -> list[Path]:
    names = set(label_names)
    matches = [path for path in root.rglob("*") if path.is_dir() and path.name in names]
    if root.is_dir() and root.name in names:
        matches.append(root)
    return sorted(set(matches))


class BinaryImageFolder:
    """Recursively indexes real/fake folders without assuming category depth.

    This deliberately mirrors the two common layouts:

    - UniversalFakeDetect: ``domain/**/0_real`` and ``domain/**/1_fake``
    - GenImage: ``generator/split/nature`` and ``generator/split/ai``
    """

    def __init__(
        self,
        real_dirs: Iterable[Path],
        fake_dirs: Iterable[Path],
        transform: Any = None,
        max_samples_per_class: int | None = None,
        sample_offset_per_class: int = 0,
        seed: int = 0,
    ) -> None:
        real_paths = sorted({path for directory in real_dirs for path in _image_paths(directory)})
        fake_paths = sorted({path for directory in fake_dirs for path in _image_paths(directory)})
        if not real_paths:
            raise FileNotFoundError("No real images found in configured label directories")
        if not fake_paths:
            raise FileNotFoundError("No fake images found in configured label directories")

        if max_samples_per_class is not None or sample_offset_per_class > 0:
            rng = random.Random(seed)
            rng.shuffle(real_paths)
            rng.shuffle(fake_paths)
            end = (
                sample_offset_per_class + max_samples_per_class
                if max_samples_per_class is not None
                else None
            )
            real_paths = sorted(real_paths[sample_offset_per_class:end])
            fake_paths = sorted(fake_paths[sample_offset_per_class:end])

        self.records = [SampleRecord(path, 0) for path in real_paths]
        self.records += [SampleRecord(path, 1) for path in fake_paths]
        self.records.sort(key=lambda item: str(item.path))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, int, str]:
        record = self.records[index]
        with Image.open(record.path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, record.label, str(record.path)


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
) -> Any:
    normalized = data_format.lower().replace("_", "").replace("-", "")

    if normalized in {"dfarrow", "hfarrow", "hfdiskarrow", "huggingfacearrow"}:
        if not generator:
            raise ValueError("Hugging Face Arrow format requires a generator name")
        if not split:
            raise ValueError("Hugging Face Arrow format requires a split")
        from .hf_arrow import HFDiskArrowDataset

        return HFDiskArrowDataset(
            root=root,
            generator=generator,
            split=split,
            transform=transform,
            max_samples_per_class=max_samples_per_class,
            sample_offset_per_class=sample_offset_per_class,
            seed=seed,
        )

    root = Path(root).expanduser()
    if normalized in {"caidbench", "caidbencharrow"}:
        if not generator:
            raise ValueError("CAIDBench format requires a generator name")
        if not split:
            raise ValueError("CAIDBench format requires a split")
        from .arrow import CAIDBenchArrowDataset

        return CAIDBenchArrowDataset(
            root=root,
            generator=generator,
            split=split,
            transform=transform,
            max_samples_per_class=max_samples_per_class,
            sample_offset_per_class=sample_offset_per_class,
            seed=seed,
        )
    if normalized == "genimage":
        if not generator:
            raise ValueError("GenImage format requires a generator name")
        if not split:
            raise ValueError("GenImage format requires a split")
        domain_root = root / generator / split
        real_dirs = [domain_root / "nature"]
        fake_dirs = [domain_root / "ai"]
    elif normalized in {"universalfake", "universalfakedetect", "ufd"}:
        domain_root = root / generator if generator else root
        real_dirs = _label_directories(domain_root, {"0_real"})
        fake_dirs = _label_directories(domain_root, {"1_fake"})
    else:
        raise ValueError(f"Unknown data format: {data_format}")

    return BinaryImageFolder(
        real_dirs=real_dirs,
        fake_dirs=fake_dirs,
        transform=transform,
        max_samples_per_class=max_samples_per_class,
        sample_offset_per_class=sample_offset_per_class,
        seed=seed,
    )
