from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image


def _image_bytes(row: dict[str, Any]) -> bytes:
    payload = row.get("image")
    if isinstance(payload, dict):
        payload = payload.get("bytes")
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError("row image payload is not binary")
    return bytes(payload)


def inspect_payload(payload: bytes) -> str:
    with Image.open(BytesIO(payload)) as image:
        image_format = image.format or "unknown"
        image.convert("RGB").load()
    return image_format


def audit(
    root: Path,
    *,
    progress_every: int,
    generator: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    if (generator is None) != (split is None):
        raise ValueError("generator and split must be provided together")
    if generator is None:
        from datasets import load_from_disk

        dataset = load_from_disk(str(root), keep_in_memory=False)
        total_rows = len(dataset)

        def get_row(index: int) -> tuple[str, bytes]:
            row = dataset[index]
            return str(row.get("image_path", "")), _image_bytes(row)

    else:
        from online_aig_tta.data.hf_arrow import HFDiskArrowDataset

        selected = HFDiskArrowDataset(
            root=root, generator=generator, split=split, return_sample_id=True
        )
        total_rows = len(selected)

        def get_row(index: int) -> tuple[str, bytes]:
            payload, _, sample_id = selected.raw_item(index)
            return sample_id, payload

    started = time.monotonic()
    formats: Counter[str] = Counter()
    failures = []
    for index in range(total_rows):
        image_path = ""
        payload = b""
        try:
            image_path, payload = get_row(index)
            formats[inspect_payload(payload)] += 1
        except Exception as error:  # Continue the audit to enumerate every bad row.
            failures.append(
                {
                    "row": index,
                    "image_path": image_path,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        completed = index + 1
        if progress_every > 0 and completed % progress_every == 0:
            elapsed = time.monotonic() - started
            print(
                f"rows={completed}/{total_rows} failures={len(failures)} "
                f"rows_per_second={completed / elapsed:.2f}",
                flush=True,
            )

    elapsed = time.monotonic() - started
    return {
        "format": "ctta4aid_hf_arrow_image_audit_v1",
        "root": str(root),
        "generator": generator,
        "split": split,
        "rows": total_rows,
        "valid_rows": total_rows - len(failures),
        "invalid_rows": len(failures),
        "formats": dict(sorted(formats.items())),
        "elapsed_seconds": elapsed,
        "rows_per_second": total_rows / elapsed if elapsed else None,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode every image payload in a Hugging Face save_to_disk dataset."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=10000)
    parser.add_argument("--generator")
    parser.add_argument("--split")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not (root / "state.json").is_file():
        raise FileNotFoundError(f"Not a save_to_disk dataset: {root}")
    result = audit(
        root,
        progress_every=args.progress_every,
        generator=args.generator,
        split=args.split,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if result["invalid_rows"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
