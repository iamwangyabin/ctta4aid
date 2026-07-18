#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zlib
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Mapping


GENERATOR_DIRS = {
    "adm_imagenet": "ADM",
    "biggan_imagenet": "BigGAN",
    "glide_imagenet": "glide",
    "midjourney_imagenet": "Midjourney",
    "sdv4_imagenet": "stable_diffusion_v_1_4",
    "sdv5_imagenet": "stable_diffusion_v_1_5",
    "vqdm_imagenet": "VQDM",
    "wukong_imagenet": "wukong",
}
LABEL_DIRS = {"nature": "0_real", "ai": "1_fake"}
EXPECTED_COUNTS = {
    generator: {label: 8000 if source == "sdv5_imagenet" else 6000 for label in LABEL_DIRS}
    for source, generator in GENERATOR_DIRS.items()
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crc32_file(path: Path) -> int:
    checksum = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum & 0xFFFFFFFF


def destination_for(member: zipfile.ZipInfo, output_root: Path) -> tuple[Path, str, str] | None:
    if member.is_dir():
        return None
    parts = PurePosixPath(member.filename).parts
    if len(parts) != 4 or parts[0].lower() != "test":
        raise ValueError(f"Unexpected GenImage member path: {member.filename}")

    source_generator = parts[1].lower()
    source_label = parts[2].lower()
    if source_generator not in GENERATOR_DIRS or source_label not in LABEL_DIRS:
        raise ValueError(f"Unexpected GenImage member path: {member.filename}")
    filename = parts[3]
    if filename in {"", ".", ".."}:
        raise ValueError(f"Unsafe GenImage member path: {member.filename}")

    generator = GENERATOR_DIRS[source_generator]
    label = LABEL_DIRS[source_label]
    return output_root / "test" / generator / label / filename, generator, source_label


def extract_archive(
    archive_path: Path,
    output_root: Path,
    *,
    expected_sha256: str | None = None,
    expected_counts: Mapping[str, Mapping[str, int]] = EXPECTED_COUNTS,
    progress_every: int = 1000,
) -> dict[str, object]:
    archive_path = archive_path.resolve()
    output_root = output_root.resolve()
    archive_sha256 = sha256_file(archive_path)
    if expected_sha256 and archive_sha256 != expected_sha256.lower():
        raise ValueError(
            f"Archive SHA256 is {archive_sha256}, expected {expected_sha256.lower()}"
        )

    counts: Counter[tuple[str, str]] = Counter()
    skipped_existing = 0
    written = 0
    destinations: set[Path] = set()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            mapped = destination_for(member, output_root)
            if mapped is None:
                continue
            destination, generator, source_label = mapped
            if destination in destinations:
                raise ValueError(f"Duplicate destination: {destination}")
            destinations.add(destination)
            counts[(generator, source_label)] += 1
            destination.parent.mkdir(parents=True, exist_ok=True)

            if (
                destination.is_file()
                and destination.stat().st_size == member.file_size
                and crc32_file(destination) == member.CRC
            ):
                skipped_existing += 1
            else:
                temporary = destination.with_name(destination.name + ".part")
                try:
                    with archive.open(member) as source, temporary.open("wb") as target:
                        shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
                    if temporary.stat().st_size != member.file_size:
                        raise OSError(f"Wrong extracted size for {member.filename}")
                    os.replace(temporary, destination)
                finally:
                    temporary.unlink(missing_ok=True)
                written += 1

            processed = written + skipped_existing
            if progress_every > 0 and processed % progress_every == 0:
                print(f"processed={processed} member={member.filename}", flush=True)

    observed_counts = {
        generator: {label: counts[(generator, label)] for label in LABEL_DIRS}
        for generator in expected_counts
    }
    expected_normalized = {
        generator: dict(labels) for generator, labels in expected_counts.items()
    }
    if observed_counts != expected_normalized:
        raise ValueError(
            "Unexpected GenImage counts: "
            + json.dumps(
                {"observed": observed_counts, "expected": expected_normalized},
                sort_keys=True,
            )
        )

    summary: dict[str, object] = {
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha256,
        "output_root": str(output_root),
        "files": sum(sum(labels.values()) for labels in observed_counts.values()),
        "written": written,
        "skipped_existing": skipped_existing,
        "counts": observed_counts,
        "mapping": {
            source: {
                "generator": generator,
                "labels": LABEL_DIRS,
            }
            for source, generator in GENERATOR_DIRS.items()
        },
    }
    manifest_path = output_root / "extract_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the official GenImage test ZIP into IAPL ImageFolder layout."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_archive(
        args.archive,
        args.output_root,
        expected_sha256=args.expected_sha256,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
