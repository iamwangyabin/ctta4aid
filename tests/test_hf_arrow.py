from __future__ import annotations

import importlib.util
import io
import json
import random
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from online_aig_tta.data.folders import build_dataset
from online_aig_tta.data.hf_arrow import (
    HFDiskArrowDataset,
    build_iapl_arrow_dataset,
    parse_hf_arrow_uri,
)


DATASETS_AVAILABLE = importlib.util.find_spec("datasets") is not None


@unittest.skipUnless(DATASETS_AVAILABLE, "datasets and pyarrow are required")
class HFDiskArrowTests(unittest.TestCase):
    def test_construction_preserves_python_rng_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "ForenSynths"
            _save_dataset(
                root,
                {
                    "test/crn/0_real/a.png": 0,
                    "test/crn/1_fake/b.png": 1,
                },
                split_name="test",
                domain="crn",
            )
            random.seed(12345)
            before = random.getstate()

            HFDiskArrowDataset(root=root, generator="crn", split="test")

            self.assertEqual(random.getstate(), before)

    def test_combines_roots_and_uses_split_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forensynths = root / "ForenSynths"
            ojha = root / "Ojha"
            _save_dataset(
                forensynths,
                {
                    "test/crn/0_real/a.png": 0,
                    "test/crn/1_fake/b.png": 1,
                },
                split_name="test",
                domain="crn",
            )
            _save_dataset(
                ojha,
                {
                    "dalle/0_real/a.png": 0,
                    "dalle/1_fake/b.png": 1,
                },
                split_name="test",
                domain="dalle",
            )

            dataset = build_dataset(
                data_format="hf_arrow",
                root=[forensynths, ojha],
                generator="dalle",
                split="test",
                transform=lambda image: image,
            )
            self.assertEqual(len(dataset), 2)
            samples = [dataset[index] for index in range(len(dataset))]
            self.assertEqual({label for _, label, _ in samples}, {0, 1})
            self.assertTrue(all(image.mode == "RGB" for image, _, _ in samples))
            self.assertTrue(all(sample_id.startswith("Ojha/") for _, _, sample_id in samples))

    def test_falls_back_to_genimage_paths_and_balanced_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "GenImage_test"
            labels = {
                "ADM/val/nature/a.png": 0,
                "ADM/val/nature/b.png": 0,
                "ADM/val/ai/c.png": 1,
                "ADM/val/ai/d.png": 1,
            }
            _save_dataset(root, labels)

            dataset = HFDiskArrowDataset(
                root=root,
                generator="ADM",
                split="val",
                transform=lambda image: image,
                max_samples_per_class=1,
                seed=11,
            )
            self.assertEqual(len(dataset), 2)
            self.assertEqual({dataset[index][1] for index in range(2)}, {0, 1})
            payload, label, sample_id = dataset.raw_item(0)
            self.assertEqual(Image.open(io.BytesIO(payload)).format, "PNG")
            self.assertIn(label, {0, 1})
            self.assertTrue(sample_id.startswith("GenImage_test/ADM/val/"))

    def test_iapl_uri_returns_imagefolder_compatible_items(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "Ojha"
            _save_dataset(
                root,
                {"guided/0_real/a.png": 0, "guided/1_fake/b.png": 1},
                split_name="test",
                domain="guided",
            )
            uri = f"hf_arrow://{root}"
            self.assertEqual(parse_hf_arrow_uri(uri), [str(root)])
            dataset = build_iapl_arrow_dataset(
                uri,
                split="tta",
                subset="guided",
                transform=lambda image: image,
            )
            self.assertIsNotNone(dataset)
            assert dataset is not None
            self.assertEqual(len(dataset[0]), 2)

    def test_decode_error_reports_arrow_sample_id(self) -> None:
        from datasets import Dataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "balanced"
            image_path = "SDv14/train/nature/bad.jpg"
            fake_path = "SDv14/train/ai/fake.png"
            dataset = Dataset.from_dict(
                {
                    "image_path": [image_path, fake_path],
                    "image": [b"not an image", _image_bytes(0)],
                }
            )
            dataset.save_to_disk(root)
            mapping = {image_path: 0, fake_path: 1}
            (root / "mapping.json").write_text(
                json.dumps(mapping), encoding="utf-8"
            )
            (root / "train.json").write_text(
                json.dumps({"SDv14": {image_path: 0, fake_path: 1}}),
                encoding="utf-8",
            )

            arrow_dataset = HFDiskArrowDataset(
                root=root, generator="SDv14", split="train"
            )
            with self.assertRaisesRegex(RuntimeError, "SDv14/train/nature/bad.jpg"):
                arrow_dataset[1]


def _save_dataset(
    root: Path,
    labels: dict[str, int],
    *,
    split_name: str | None = None,
    domain: str | None = None,
) -> None:
    from datasets import Dataset

    root.parent.mkdir(parents=True, exist_ok=True)
    image_paths = list(labels)
    dataset = Dataset.from_dict(
        {
            "image_path": image_paths,
            "md5": [""] * len(labels),
            "width": [4] * len(labels),
            "height": [4] * len(labels),
            "image": [_image_bytes(index * 64) for index in range(len(labels))],
        }
    )
    dataset.save_to_disk(root)
    mapping = {image_path: index for index, image_path in enumerate(image_paths)}
    (root / "mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
    if split_name and domain:
        (root / f"{split_name}.json").write_text(
            json.dumps({domain: labels}), encoding="utf-8"
        )


def _image_bytes(value: int) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 4), color=(value, value, value)).save(output, format="PNG")
    return output.getvalue()


if __name__ == "__main__":
    unittest.main()
