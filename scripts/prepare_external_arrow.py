from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image


TREE_SUITES: dict[str, tuple[tuple[str, str], ...]] = {
    "aigc_detection_benchmark": (
        ("ProGAN", "progan"),
        ("StyleGAN", "stylegan"),
        ("BigGAN", "biggan"),
        ("CycleGAN", "cyclegan"),
        ("StarGAN", "stargan"),
        ("GauGAN", "gaugan"),
        ("StyleGAN2", "stylegan2"),
        ("WFIR", "whichfaceisreal"),
        ("ADM", "ADM"),
        ("GLIDE", "Glide"),
        ("Midjourney", "Midjourney"),
        ("SD v1.4", "stable_diffusion_v_1_4"),
        ("SD v1.5", "stable_diffusion_v_1_5"),
        ("VQDM", "VQDM"),
        ("Wukong", "wukong"),
        ("DALL-E2", "DALLE2"),
        ("SDXL", "sd_xl"),
    ),
    "aigi_holmes_p3": (
        ("Janus", "Janus"),
        ("Janus-Pro-1B", "Janus-Pro-1B"),
        ("Janus-Pro-7B", "Janus-Pro-7B"),
        ("Show-o", "Show-o"),
        ("LlamaGen", "LlamaGen"),
        ("Infinity", "Infinity"),
        ("VAR", "VAR"),
        ("PixArt-XL", "PixArt-XL"),
        ("SD3.5-L", "SD35-L"),
        ("FLUX", "FLUX"),
    ),
}

OPENSDI_DOMAINS: tuple[tuple[str, str], ...] = (
    ("SD1.5", "sd15"),
    ("SD2.1", "sd2"),
    ("SDXL", "sdxl"),
    ("SD3", "sd3"),
    ("Flux.1", "flux"),
)

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class ExternalRecord:
    image_path: str
    label: int
    source_path: Path | None = None
    payload: bytes | None = None

    def image_bytes(self) -> bytes:
        if self.payload is not None:
            return self.payload
        if self.source_path is None:
            raise RuntimeError(f"No image source for {self.image_path}")
        return self.source_path.read_bytes()


def _case_insensitive_child(root: Path, name: str) -> Path:
    exact = root / name
    if exact.is_dir():
        return exact
    matches = [path for path in root.iterdir() if path.name.lower() == name.lower()]
    if len(matches) != 1 or not matches[0].is_dir():
        raise FileNotFoundError(f"Expected directory {name!r} below {root}")
    return matches[0]


def _is_decodable_image(payload: bytes) -> bool:
    try:
        with Image.open(BytesIO(payload)) as image:
            image.load()
    except (OSError, ValueError):
        return False
    return True


def _sample_valid_paths(paths: list[Path], limit: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    candidates = list(paths)
    rng.shuffle(candidates)
    selected = []
    for path in candidates:
        if _is_decodable_image(path.read_bytes()):
            selected.append(path)
        if len(selected) == limit:
            return sorted(selected)
    raise ValueError(f"Need {limit} decodable images per class, found only {len(selected)}")


def _label_image_paths(domain_root: Path, label: int) -> list[Path]:
    aliases = ("0_real", "real") if label == 0 else ("1_fake", "fake")
    class_roots = [
        path
        for path in domain_root.rglob("*")
        if path.is_dir() and path.name.lower() in aliases
    ]
    if not class_roots:
        raise FileNotFoundError(
            f"Expected one of {aliases!r} below {domain_root} for label={label}"
        )
    return sorted(
        {
            path
            for class_root in class_roots
            for path in class_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
    )


def tree_records(
    root: Path, domain: str, raw_domain: str, samples_per_class: int, seed: int
) -> list[ExternalRecord]:
    domain_root = _case_insensitive_child(root, raw_domain)
    records: list[ExternalRecord] = []
    for label in (0, 1):
        paths = _label_image_paths(domain_root, label)
        selected = _sample_valid_paths(paths, samples_per_class, seed + label)
        for path in selected:
            relative = path.relative_to(domain_root).as_posix()
            records.append(
                ExternalRecord(
                    image_path=f"{domain}/test/{relative}",
                    label=label,
                    source_path=path,
                )
            )
    return sorted(records, key=lambda record: record.image_path)


def _binary_payload(value: object) -> bytes:
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, memoryview):
        value = value.tobytes()
    if not isinstance(value, (bytes, bytearray)):
        raise ValueError("OpenSDI parquet image column does not contain encoded bytes")
    return bytes(value)


def _binary_label(value: object) -> int:
    if isinstance(value, str):
        normalized = value.lower()
        if normalized == "real":
            return 0
        if normalized == "fake":
            return 1
    label = int(value)
    if label not in {0, 1}:
        raise ValueError(f"Expected a binary label, got {value!r}")
    return label


def opensdi_global_records(
    root: Path, domain: str, raw_prefix: str, samples_per_class: int, seed: int
) -> list[ExternalRecord]:
    try:
        from pyarrow.parquet import ParquetFile
    except ImportError as error:
        raise RuntimeError("pyarrow is required to convert OpenSDI parquet files") from error

    files = sorted((root / "data").glob(f"{raw_prefix}-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet shards for OpenSDI domain {raw_prefix!r} below {root / 'data'}"
        )

    heaps: dict[int, list[tuple[int, str, bytes]]] = {0: [], 1: []}
    for path in files:
        parquet = ParquetFile(path)
        for batch in parquet.iter_batches(columns=["key", "image", "label"], batch_size=128):
            for row in batch.to_pylist():
                key = str(row["key"])
                if not key.startswith("entire/"):
                    continue
                label = _binary_label(row["label"])
                label_dir = "0_real" if label == 0 else "1_fake"
                image_path = f"{domain}/test/{label_dir}/{key.replace('/', '__')}"
                score = int.from_bytes(
                    hashlib.sha256(f"{seed}:{domain}:{key}".encode("utf-8")).digest()[:8],
                    byteorder="big",
                )
                candidate = (-score, image_path, _binary_payload(row["image"]))
                heap = heaps[label]
                if len(heap) < samples_per_class:
                    heapq.heappush(heap, candidate)
                elif candidate[0] > heap[0][0]:
                    heapq.heapreplace(heap, candidate)

    records: list[ExternalRecord] = []
    for label, heap in heaps.items():
        if len(heap) != samples_per_class:
            raise ValueError(
                f"OpenSDI {domain} global scope has only {len(heap)} samples for label={label}; "
                f"need {samples_per_class}"
            )
        records.extend(
            ExternalRecord(image_path=image_path, label=label, payload=payload)
            for _, image_path, payload in heap
        )
    return sorted(records, key=lambda record: record.image_path)


def _rows(records: Iterable[ExternalRecord]):
    for record in records:
        payload = record.image_bytes()
        if not _is_decodable_image(payload):
            raise ValueError(f"Cannot decode selected image: {record.image_path}")
        yield {"image": payload, "image_path": record.image_path}


def write_arrow_bundle(
    output_root: Path, suite: str, domain: str, records: list[ExternalRecord]
) -> dict[str, int]:
    try:
        from datasets import Dataset, Features, Value
    except ImportError as error:
        raise RuntimeError("datasets is required to create project Arrow bundles") from error

    labels = {record.image_path: record.label for record in records}
    if len(labels) != len(records):
        raise ValueError(f"Duplicate logical image paths for {suite}/{domain}")
    counts = {label: sum(record.label == label for record in records) for label in (0, 1)}
    if not all(counts.values()):
        raise ValueError(f"Both binary labels are required for {suite}/{domain}")

    bundle = output_root / "test" / domain
    bundle.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_generator(
        _rows,
        gen_kwargs={"records": records},
        features=Features({"image": Value("binary"), "image_path": Value("string")}),
        cache_dir=str(output_root / ".cache"),
    )
    dataset.save_to_disk(str(bundle))
    (bundle / "mapping.json").write_text(
        json.dumps({record.image_path: index for index, record in enumerate(records)}, indent=2),
        encoding="utf-8",
    )
    (bundle / "test.json").write_text(
        json.dumps({domain: labels}, indent=2), encoding="utf-8"
    )
    (bundle / "bundle_manifest.json").write_text(
        json.dumps(
            {
                "suite": suite,
                "generator": domain,
                "split": "test",
                "real_samples": counts[0],
                "fake_samples": counts[1],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"real": counts[0], "fake": counts[1]}


def prepare_suite(
    suite: str, input_root: Path, output_root: Path, samples_per_class: int, seed: int
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing Arrow output: {output_root}")
    output_root.mkdir(parents=True)
    domain_summaries: dict[str, dict[str, int]] = {}
    try:
        if suite in TREE_SUITES:
            for offset, (domain, raw_domain) in enumerate(TREE_SUITES[suite]):
                records = tree_records(
                    input_root,
                    domain,
                    raw_domain,
                    samples_per_class,
                    seed + offset * 10,
                )
                domain_summaries[domain] = write_arrow_bundle(
                    output_root, suite, domain, records
                )
        elif suite == "opensdid_global":
            for offset, (domain, raw_prefix) in enumerate(OPENSDI_DOMAINS):
                records = opensdi_global_records(
                    input_root,
                    domain,
                    raw_prefix,
                    samples_per_class,
                    seed + offset * 10,
                )
                domain_summaries[domain] = write_arrow_bundle(
                    output_root, suite, domain, records
                )
        else:
            raise ValueError(f"Unsupported external suite: {suite}")
    finally:
        shutil.rmtree(output_root / ".cache", ignore_errors=True)

    summary: dict[str, object] = {
        "suite": suite,
        "split": "test",
        "samples_per_class": samples_per_class,
        "selection_seed": seed,
        "domains": domain_summaries,
    }
    (output_root / "conversion_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert approved external test data into project-standard Arrow bundles"
    )
    parser.add_argument(
        "suite", choices=[*TREE_SUITES, "opensdid_global"], help="External benchmark"
    )
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")
    summary = prepare_suite(
        args.suite,
        args.input_root.expanduser().resolve(),
        args.output_root.expanduser().resolve(),
        args.samples_per_class,
        args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
