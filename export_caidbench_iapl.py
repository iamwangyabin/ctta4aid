from __future__ import annotations

import argparse
import csv
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

from online_aig_tta.data.arrow import CAIDBenchArrowDataset


PROFILES = {
    "genimage": (
        ("ADM", "ADM"),
        ("BigGAN", "BigGAN"),
        ("GLIDE", "glide"),
        ("Midjourney v5", "Midjourney"),
        ("SD1.4", "stable_diffusion_v_1_4"),
        ("SD1.5", "stable_diffusion_v_1_5"),
        ("VQDM", "VQDM"),
        ("Wukong", "wukong"),
    ),
    "progan_proxy": (
        ("CycleGAN", "CycleGAN"),
        ("BigGAN", "BigGAN"),
        ("StyleGAN", "StyleGAN"),
        ("StarGAN", "StarGAN"),
        ("GauGAN", "GauGAN"),
        ("CRN", "CRN"),
        ("DeepFakes", "DeepFakes"),
        ("GLIDE", "GLIDE"),
        ("IMLE", "IMLE"),
        ("LDM", "LDM"),
    ),
    "genimage_sd15_local_diagnostic": (
        ("StableDiffusion-v1-5-local", "stable_diffusion_v_1_5"),
    ),
}

FORMAT_SUFFIXES = {
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}


def image_suffix(payload: bytes) -> str:
    with Image.open(BytesIO(payload)) as image:
        return FORMAT_SUFFIXES.get(str(image.format).upper(), ".png")


def export_profile(
    root: Path,
    destination: Path,
    profile: str,
    max_samples_per_class: int | None = None,
    subsets: list[str] | None = None,
) -> Path:
    rows: list[dict[str, str | int]] = []
    mappings = list(PROFILES[profile])
    if subsets:
        requested = set(subsets)
        known = {subset for _, subset in mappings}
        unknown = sorted(requested - known)
        if unknown:
            raise ValueError(f"Unknown {profile} subsets: {unknown}")
        mappings = [mapping for mapping in mappings if mapping[1] in requested]
    for generator, subset in mappings:
        dataset = CAIDBenchArrowDataset(
            root=root,
            generator=generator,
            split="test",
            max_samples_per_class=max_samples_per_class,
            seed=0,
        )
        class_positions = {0: 0, 1: 0}
        for index in range(len(dataset)):
            payload, label, sample_id = dataset.raw_item(index)
            position = class_positions[label]
            class_positions[label] += 1
            relative_path = (
                Path("test")
                / subset
                / ("0_real" if label == 0 else "1_fake")
                / f"{position:06d}{image_suffix(payload)}"
            )
            output_path = destination / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(payload).hexdigest()
            if not output_path.is_file() or hashlib.sha256(
                output_path.read_bytes()
            ).hexdigest() != digest:
                output_path.write_bytes(payload)
            rows.append(
                {
                    "relative_path": str(relative_path),
                    "source_generator": generator,
                    "destination_subset": subset,
                    "split": "test",
                    "label": label,
                    "sample_id": sample_id,
                    "sha256": digest,
                    "bytes": len(payload),
                }
            )
        if class_positions[0] != class_positions[1]:
            raise RuntimeError(
                f"Unbalanced export for {generator}: {class_positions}"
            )
        print(
            f"{generator:>16s} -> {subset:<24s} "
            f"real={class_positions[0]} fake={class_positions[1]}"
        )
        dataset.close()

    manifest_path = destination / f"caidbench_{profile}_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize lossless CAIDBench Arrow images for official IAPL"
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--max-samples-per-class", type=int)
    parser.add_argument("--subsets", nargs="+")
    args = parser.parse_args()
    manifest = export_profile(
        args.root.expanduser().resolve(),
        args.destination.expanduser().resolve(),
        args.profile,
        args.max_samples_per_class,
        args.subsets,
    )
    print(f"manifest={manifest}")


if __name__ == "__main__":
    main()
