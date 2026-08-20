from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.build_bias_controlled_arrow import build_bias_controlled_arrow
from src.data.arrow import ArrowDataset
from src.data.bias_control import (
    BIAS_CONTROL_MANIFEST,
    jpeg_quality_for_sample,
    transform_image_bytes,
)


DATASETS_AVAILABLE = importlib.util.find_spec("datasets") is not None


class BiasControlTransformTests(unittest.TestCase):
    def test_q90_profile_reencodes_every_input_without_changing_geometry(self) -> None:
        transformed, metadata = transform_image_bytes(
            _image_bytes("PNG", size=(13, 7)),
            image_path="ADM/test/1_fake/example.png",
            profile="all_jpeg_q90",
        )

        with Image.open(io.BytesIO(transformed)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (13, 7))
        self.assertEqual(metadata["source_format"], "PNG")
        self.assertEqual(metadata["output_quality"], 90)

    def test_matched_profile_normalizes_geometry_and_does_not_key_on_label(self) -> None:
        real_path = "ADM/test/0_real/shared-name.png"
        fake_path = "ADM/test/1_fake/shared-name.png"
        self.assertEqual(
            jpeg_quality_for_sample("matched_jpeg", real_path),
            jpeg_quality_for_sample("matched_jpeg", fake_path),
        )

        transformed, metadata = transform_image_bytes(
            _image_bytes("PNG", size=(19, 11)),
            image_path=fake_path,
            profile="matched_jpeg",
        )
        with Image.open(io.BytesIO(transformed)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (256, 256))
        self.assertIn(metadata["output_quality"], {75, 80, 85, 90, 95})


@unittest.skipUnless(DATASETS_AVAILABLE, "datasets and pyarrow are required")
class BiasControlledArrowBuilderTests(unittest.TestCase):
    def test_conversion_preserves_sample_identity_and_requires_profile_declaration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            source = temporary_root / "raw" / "GenImage_test"
            image_paths = [
                "ADM/test/nature/real.jpg",
                "ADM/test/ai/fake.png",
            ]
            _save_bundle(source, image_paths)
            output_root = (
                temporary_root / "bias_controlled" / "all_jpeg_q90" / "genimage"
            )

            summary = build_bias_controlled_arrow(
                source, output_root, "all_jpeg_q90"
            )

            output_bundle = output_root / source.name
            self.assertEqual(summary["profile"], "all_jpeg_q90")
            self.assertEqual(summary["sample_count"], 2)
            self.assertTrue((output_bundle / BIAS_CONTROL_MANIFEST).is_file())
            self.assertTrue(source.is_dir())

            with self.assertRaisesRegex(ValueError, "requires data.bias_control_profile"):
                ArrowDataset(root=output_root, generator="ADM", split="test")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                ArrowDataset(
                    root=output_root,
                    generator="ADM",
                    split="test",
                    bias_control_profile="matched_jpeg",
                )

            dataset = ArrowDataset(
                root=output_root,
                generator="ADM",
                split="test",
                bias_control_profile="all_jpeg_q90",
            )
            self.assertEqual(
                {record.sample_id for record in dataset.records},
                {f"GenImage_test/{path}" for path in image_paths},
            )
            for index in range(len(dataset)):
                payload, _, _ = dataset.raw_item(index)
                with Image.open(io.BytesIO(payload)) as image:
                    self.assertEqual(image.format, "JPEG")

    def test_conversion_refuses_an_unlabelled_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw" / "GenImage_test"
            _save_bundle(
                source,
                ["ADM/test/nature/a.png", "ADM/test/ai/b.png"],
            )
            with self.assertRaisesRegex(ValueError, "exact profile component"):
                build_bias_controlled_arrow(
                    source, root / "controlled" / "genimage", "all_jpeg_q90"
                )


def _save_bundle(root: Path, image_paths: list[str]) -> None:
    from datasets import Dataset

    root.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_dict(
        {
            "image": [
                _image_bytes("JPEG", size=(12, 8)),
                _image_bytes("PNG", size=(8, 12)),
            ],
            "image_path": image_paths,
        }
    )
    dataset.save_to_disk(root)
    (root / "mapping.json").write_text(
        json.dumps({path: index for index, path in enumerate(image_paths)}),
        encoding="utf-8",
    )
    (root / "test.json").write_text(
        json.dumps({"ADM": {image_paths[0]: 0, image_paths[1]: 1}}),
        encoding="utf-8",
    )


def _image_bytes(image_format: str, *, size: tuple[int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color=(96, 128, 160)).save(output, format=image_format)
    return output.getvalue()


if __name__ == "__main__":
    unittest.main()
