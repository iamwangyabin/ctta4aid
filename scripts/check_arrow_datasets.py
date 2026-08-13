from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.data.arrow import ArrowDataset

UFD_DOMAINS = [
    "crn",
    "cyclegan",
    "dalle",
    "biggan",
    "deepfake",
    "gaugan",
    "glide_50_27",
    "glide_100_10",
    "glide_100_27",
    "guided",
    "imle",
    "ldm_100",
    "ldm_200",
    "ldm_200_cfg",
    "progan",
    "san",
    "seeingdark",
    "stargan",
    "stylegan",
]
GENIMAGE_DOMAINS = [
    "ADM",
    "BigGAN",
    "glide",
    "Midjourney",
    "stable_diffusion_v_1_4",
    "stable_diffusion_v_1_5",
    "VQDM",
    "wukong",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate canonical Arrow domains, labels, row mappings and image bytes"
    )
    parser.add_argument("suite", choices=["ufd", "genimage"])
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()

    domains = UFD_DOMAINS if args.suite == "ufd" else GENIMAGE_DOMAINS
    split = "test"
    results = {}
    for domain in domains:
        dataset = ArrowDataset(
            root=args.roots,
            generator=domain,
            split=split,
        )
        real = sum(record.label == 0 for record in dataset.records)
        fake = sum(record.label == 1 for record in dataset.records)
        payload, _, sample_id = dataset.raw_item(0)
        with Image.open(BytesIO(payload)) as image:
            image.verify()
            image_format = image.format
        results[domain] = {
            "samples": len(dataset),
            "real": real,
            "fake": fake,
            "first_sample_id": sample_id,
            "first_image_format": image_format,
        }
        print(f"{domain:<24} samples={len(dataset):>6} real={real:>6} fake={fake:>6}")

    summary = {
        "suite": args.suite,
        "roots": [str(Path(root).expanduser().resolve()) for root in args.roots],
        "domains": results,
        "total_samples": sum(item["samples"] for item in results.values()),
    }
    print(f"total_samples={summary['total_samples']}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
