from __future__ import annotations

import importlib.util
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.data.arrow import ArrowDataset, ArrowRecord, build_dataset


DATASETS_AVAILABLE = importlib.util.find_spec("datasets") is not None


class ArrowDatasetLockTests(unittest.TestCase):
    def test_locked_sample_ids_preserve_requested_order_without_sampling(self) -> None:
        records = [
            ArrowRecord("/bundle", 0, 0, "ADM/test/0_real/a.png", "Bundle/A"),
            ArrowRecord("/bundle", 1, 1, "ADM/test/1_fake/b.png", "Bundle/B"),
            ArrowRecord("/bundle", 2, 0, "ADM/test/0_real/c.png", "Bundle/C"),
        ]
        with patch("src.data.arrow._resolve_dataset_roots", return_value=[Path("/bundle")]), patch(
            "src.data.arrow._records_from_root", return_value=records
        ):
            dataset = ArrowDataset(
                root="unused",
                generator="ADM",
                split="test",
                max_samples_per_class=1,
                locked_sample_ids=["Bundle/B", "Bundle/A"],
            )

        self.assertEqual([record.sample_id for record in dataset.records], ["Bundle/B", "Bundle/A"])


@unittest.skipUnless(DATASETS_AVAILABLE, "datasets and pyarrow are required")
class ArrowDatasetTests(unittest.TestCase):
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

            ArrowDataset(root=root, generator="crn", split="test")

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
                data_format="arrow",
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

            dataset = ArrowDataset(
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

    def test_exclude_image_paths_removes_known_invalid_source_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "GenImage_train"
            labels = {
                "SDv14/train/nature/a.png": 0,
                "SDv14/train/nature/b.png": 0,
                "SDv14/train/ai/c.png": 1,
                "SDv14/train/ai/d.png": 1,
            }
            _save_dataset(root, labels, split_name="train", domain="SDv14")

            dataset = ArrowDataset(
                root=root,
                generator="SDv14",
                split="train",
                exclude_image_paths=["SDv14/train/ai/c.png"],
            )

            self.assertEqual(len(dataset), 3)
            self.assertNotIn(
                "SDv14/train/ai/c.png",
                [record.image_path for record in dataset.records],
            )
            self.assertEqual({record.label for record in dataset.records}, {0, 1})

    def test_discovers_nested_generator_and_split_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "GenImage-arrow"
            bundle = root / "data" / "test" / "ADM"
            _save_dataset(
                bundle,
                {
                    "ADM/test/nature/a.png": 0,
                    "ADM/test/ai/b.png": 1,
                },
            )
            (bundle / "bundle_manifest.json").write_text(
                json.dumps({"generator": "ADM", "split": "test"}),
                encoding="utf-8",
            )
            ignored = root / "data" / "train" / "ADM"
            ignored.mkdir(parents=True)
            (ignored / "state.json").write_text("{}", encoding="utf-8")
            (ignored / "bundle_manifest.json").write_text(
                json.dumps({"generator": "ADM", "split": "train"}),
                encoding="utf-8",
            )

            dataset = ArrowDataset(root=root, generator="ADM", split="test")

            self.assertEqual(len(dataset), 2)
            self.assertTrue(
                all(record.dataset_root == str(bundle) for record in dataset.records)
            )

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

            arrow_dataset = ArrowDataset(
                root=root, generator="SDv14", split="train"
            )
            with self.assertRaisesRegex(RuntimeError, "SDv14/train/nature/bad.jpg"):
                arrow_dataset[1]

    def test_sampling_offsets_are_disjoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "GenImage"
            labels = {}
            for label_name, label in (("nature", 0), ("ai", 1)):
                for index in range(4):
                    labels[f"ADM/test/{label_name}/{index}.png"] = label
            _save_dataset(root, labels)
            common = dict(
                root=root,
                generator="ADM",
                split="test",
                max_samples_per_class=2,
                seed=7,
            )

            stream = ArrowDataset(**common, sample_offset_per_class=0)
            holdout = ArrowDataset(**common, sample_offset_per_class=2)

            self.assertFalse(
                {record.sample_id for record in stream.records}
                & {record.sample_id for record in holdout.records}
            )

    def test_locked_sample_ids_preserve_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "External"
            labels = {
                "ADM/test/0_real/a.png": 0,
                "ADM/test/0_real/b.png": 0,
                "ADM/test/1_fake/c.png": 1,
                "ADM/test/1_fake/d.png": 1,
            }
            _save_dataset(root, labels, split_name="test", domain="ADM")
            locked_ids = [
                f"{root.name}/ADM/test/1_fake/d.png",
                f"{root.name}/ADM/test/0_real/a.png",
            ]

            dataset = ArrowDataset(
                root=root,
                generator="ADM",
                split="test",
                locked_sample_ids=locked_ids,
            )

            self.assertEqual([record.sample_id for record in dataset.records], locked_ids)
            with self.assertRaisesRegex(FileNotFoundError, "missing from the supplied"):
                ArrowDataset(
                    root=root,
                    generator="ADM",
                    split="test",
                    locked_sample_ids=[f"{root.name}/ADM/test/unknown.png"],
                )

    def test_dataset_factory_rejects_legacy_formats(self) -> None:
        with self.assertRaisesRegex(ValueError, "only reads 'arrow'"):
            build_dataset(
                data_format="genimage",
                root="unused",
                generator="ADM",
                split="test",
                transform=None,
            )


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
