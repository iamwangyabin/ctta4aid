from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

# Python places the scripts directory, rather than the repository root, first
# when this checker is invoked as documented from the command line.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
EXTERNAL_DOMAINS = {
    "aigc_detection_benchmark": [
        "ProGAN",
        "StyleGAN",
        "BigGAN",
        "CycleGAN",
        "StarGAN",
        "GauGAN",
        "StyleGAN2",
        "WFIR",
        "ADM",
        "GLIDE",
        "Midjourney",
        "SD v1.4",
        "SD v1.5",
        "VQDM",
        "Wukong",
        "DALL-E2",
        "SDXL",
    ],
    "aigi_holmes_p3": [
        "Janus",
        "Janus-Pro-1B",
        "Janus-Pro-7B",
        "Show-o",
        "LlamaGen",
        "Infinity",
        "VAR",
        "PixArt-XL",
        "SD3.5-L",
        "FLUX",
    ],
    "opensdid_global": ["SD1.5", "SD2.1", "SDXL", "SD3", "Flux.1"],
}
SUITE_DOMAINS = {
    "ufd": UFD_DOMAINS,
    "genimage": GENIMAGE_DOMAINS,
    **EXTERNAL_DOMAINS,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate canonical Arrow domains, labels, row mappings and image bytes"
    )
    parser.add_argument("suite", choices=sorted(SUITE_DOMAINS))
    parser.add_argument("roots", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()

    domains = SUITE_DOMAINS[args.suite]
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
