from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_bytes(row: dict[str, Any]) -> bytes:
    payload = row.get("image")
    if isinstance(payload, dict):
        payload = payload.get("bytes")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("row image payload is not binary")
    return bytes(payload)


def create_view(plan_path: Path, output: Path) -> dict[str, Any]:
    from datasets import load_from_disk

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source = Path(plan["source_root"]).expanduser().resolve()
    split = str(plan["split"])
    generator = str(plan["generator"])
    exclusions = list(map(str, plan["exclude_image_paths"]))
    if not exclusions or len(exclusions) != len(set(exclusions)):
        raise ValueError("exclude_image_paths must contain unique paths")
    if not (source / "state.json").is_file():
        raise FileNotFoundError(f"Not a save_to_disk dataset: {source}")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    mapping_path = source / "mapping.json"
    split_path = source / f"{split}.json"
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    split_metadata = json.loads(split_path.read_text(encoding="utf-8"))
    entries = split_metadata.get(generator)
    if not isinstance(mapping, dict) or not isinstance(entries, dict):
        raise ValueError("Source mapping or split metadata is invalid")

    dataset = load_from_disk(str(source), keep_in_memory=False)
    excluded_rows = []
    for image_path in exclusions:
        if image_path not in mapping or image_path not in entries:
            raise ValueError(f"Exclusion is missing from source metadata: {image_path}")
        row_index = int(mapping[image_path])
        row = dataset[row_index]
        if str(row.get("image_path")) != image_path:
            raise RuntimeError(f"Source mapping mismatch for {image_path}")
        payload = _payload_bytes(row)
        if payload:
            raise ValueError(f"Refusing to exclude a non-empty image: {image_path}")
        excluded_rows.append(
            {
                "row": row_index,
                "image_path": image_path,
                "bytes": 0,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "label": int(entries[image_path]),
            }
        )

    filtered_mapping = dict(mapping)
    filtered_entries = dict(entries)
    for image_path in exclusions:
        del filtered_mapping[image_path]
        del filtered_entries[image_path]
    filtered_split = dict(split_metadata)
    filtered_split[generator] = filtered_entries

    expected_rows = int(plan["expected_rows"])
    if len(filtered_entries) != expected_rows:
        raise ValueError(
            f"Filtered row count mismatch: {len(filtered_entries)} != {expected_rows}"
        )
    label_counts = {
        str(label): sum(int(value) == label for value in filtered_entries.values())
        for label in (0, 1)
    }
    expected_label_counts = {
        str(key): int(value) for key, value in plan["expected_label_counts"].items()
    }
    if label_counts != expected_label_counts:
        raise ValueError(
            f"Filtered label counts mismatch: {label_counts} != {expected_label_counts}"
        )

    partial = output.parent / f".{output.name}.partial-{os.getpid()}"
    if partial.exists():
        raise FileExistsError(f"Partial output already exists: {partial}")
    partial.mkdir(parents=True)
    try:
        for source_file in source.iterdir():
            if not source_file.is_file() or source_file.name in {
                "mapping.json",
                f"{split}.json",
                "filter_manifest.json",
            }:
                continue
            destination = partial / source_file.name
            if source_file.suffix == ".arrow":
                os.link(source_file, destination)
            else:
                shutil.copy2(source_file, destination)
        (partial / "mapping.json").write_text(
            json.dumps(filtered_mapping, separators=(",", ":")), encoding="utf-8"
        )
        (partial / f"{split}.json").write_text(
            json.dumps(filtered_split, separators=(",", ":")), encoding="utf-8"
        )
        reloaded = load_from_disk(str(partial), keep_in_memory=False)
        if len(reloaded) != len(dataset):
            raise RuntimeError("Hard-linked Arrow storage changed physical row count")

        manifest = {
            "format": "ctta4aid_hf_arrow_filtered_view_v1",
            "plan": str(plan_path),
            "plan_sha256": _sha256(plan_path),
            "source_root": str(source),
            "source_physical_rows": len(dataset),
            "selected_rows": len(filtered_entries),
            "label_counts": label_counts,
            "excluded_rows": sorted(excluded_rows, key=lambda item: item["row"]),
            "storage": "hard-linked Arrow shards with filtered mapping/split metadata",
            "arrow_shards": len(list(partial.glob("*.arrow"))),
            "metadata_sha256": {
                "mapping.json": _sha256(partial / "mapping.json"),
                f"{split}.json": _sha256(partial / f"{split}.json"),
                "state.json": _sha256(partial / "state.json"),
                "dataset_info.json": _sha256(partial / "dataset_info.json"),
            },
        }
        (partial / "filter_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        partial.rename(output)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an atomic hard-linked Arrow view excluding audited empty rows."
    )
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manifest = create_view(
        args.plan.expanduser().resolve(), args.output.expanduser().resolve()
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
