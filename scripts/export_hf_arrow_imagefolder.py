from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Sequence

from online_aig_tta.data.hf_arrow import HFDiskArrowDataset


def _safe_relative_path(image_path: str, *, domain: str, split: str) -> Path:
    source = PurePosixPath(image_path)
    if source.is_absolute() or not source.parts or ".." in source.parts:
        raise ValueError(f"Unsafe Arrow image path: {image_path!r}")
    parts = source.parts
    if parts[0].lower() == split.lower():
        return Path(*parts)
    if parts[0].lower() == domain.lower():
        return Path(split, *parts)
    raise ValueError(
        f"Arrow image path does not begin with split {split!r} or domain {domain!r}: "
        f"{image_path!r}"
    )


def export_domains(
    *,
    roots: Sequence[str | Path],
    domains: Sequence[str],
    output_root: Path,
    split: str = "test",
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    exported_paths: dict[Path, str] = {}
    domain_summaries: dict[str, Any] = {}

    for domain in domains:
        dataset = HFDiskArrowDataset(
            root=roots,
            generator=domain,
            split=split,
            return_sample_id=True,
        )
        records = []
        label_counts = {"0": 0, "1": 0}
        for index, record in enumerate(dataset.records):
            payload, label, sample_id = dataset.raw_item(index)
            relative_path = _safe_relative_path(
                record.image_path, domain=domain, split=split
            )
            destination = (output_root / relative_path).resolve()
            if output_root not in destination.parents:
                raise ValueError(f"Export path escapes output root: {record.image_path!r}")

            digest = hashlib.sha256(payload).hexdigest()
            prior_sample = exported_paths.get(destination)
            if prior_sample is not None and prior_sample != sample_id:
                raise ValueError(
                    f"Two Arrow samples map to {destination}: {prior_sample!r}, {sample_id!r}"
                )
            exported_paths[destination] = sample_id

            if destination.exists():
                existing_digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                if existing_digest != digest:
                    raise RuntimeError(f"Existing export has different bytes: {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(payload)

            label_counts[str(label)] += 1
            records.append(
                {
                    "sample_id": sample_id,
                    "source_dataset_root": record.dataset_root,
                    "source_image_path": record.image_path,
                    "destination": destination.relative_to(output_root).as_posix(),
                    "label": label,
                    "bytes": len(payload),
                    "sha256": digest,
                }
            )

        domain_summaries[domain] = {
            "samples": len(records),
            "label_counts": label_counts,
            "records": records,
        }

    manifest = {
        "format": "hf_arrow_to_imagefolder_byte_exact_v1",
        "roots": [str(Path(root).expanduser().resolve()) for root in roots],
        "split": split,
        "output_root": str(output_root),
        "domains": domain_summaries,
        "total_samples": sum(item["samples"] for item in domain_summaries.values()),
    }
    manifest_path = output_root / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export Arrow image bytes into the ImageFolder layout expected by IAPL."
    )
    parser.add_argument("--root", action="append", required=True, help="Arrow root; repeatable")
    parser.add_argument("--domain", action="append", required=True, help="Domain; repeatable")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    manifest = export_domains(
        roots=args.root,
        domains=args.domain,
        split=args.split,
        output_root=args.output_root,
    )
    print(
        json.dumps(
            {
                "output_root": manifest["output_root"],
                "domains": {
                    name: {
                        "samples": item["samples"],
                        "label_counts": item["label_counts"],
                    }
                    for name, item in manifest["domains"].items()
                },
                "total_samples": manifest["total_samples"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
