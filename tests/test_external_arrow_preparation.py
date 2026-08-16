from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from src.data.arrow import ArrowDataset


DATASETS_AVAILABLE = importlib.util.find_spec("datasets") is not None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "prepare_external_arrow.py"
SPEC = importlib.util.spec_from_file_location("external_arrow_script", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE_SCRIPT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PREPARE_SCRIPT
SPEC.loader.exec_module(PREPARE_SCRIPT)


class ExternalArrowPreparationTests(unittest.TestCase):
    def test_arrow_checker_runs_as_a_script(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "check_arrow_datasets.py"), "--help"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("opensdid_global", completed.stdout)

    def test_aigc_raw_directory_aliases_match_the_official_archive(self) -> None:
        aliases = dict(PREPARE_SCRIPT.TREE_SUITES["aigc_detection_benchmark"])

        self.assertEqual(aliases["GLIDE"], "Glide")
        self.assertEqual(aliases["DALL-E2"], "DALLE2")
        self.assertEqual(aliases["SDXL"], "sd_xl")
        self.assertEqual(aliases["WFIR"], "whichfaceisreal")

    def test_tree_records_support_nested_label_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_image(root / "progan" / "airplane" / "0_real" / "real.png", 0)
            _write_image(root / "progan" / "airplane" / "1_fake" / "fake.png", 128)

            records = PREPARE_SCRIPT.tree_records(root, "ProGAN", "progan", 1, 3)

            self.assertEqual({record.label for record in records}, {0, 1})
            self.assertTrue(
                all("airplane/" in record.image_path for record in records)
            )

    def test_tree_records_skip_corrupt_images_when_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_image(root / "progan" / "0_real" / "real.png", 0)
            _write_image(root / "progan" / "1_fake" / "fake.png", 128)
            _write_image(root / "progan" / "1_fake" / "fake2.png", 192)
            (root / "progan" / "0_real" / "corrupt.jpg").write_bytes(b"not an image")
            (root / "progan" / "1_fake" / "corrupt.jpg").write_bytes(b"not an image")

            records = PREPARE_SCRIPT.tree_records(root, "ProGAN", "progan", 1, 9)

            self.assertEqual({record.label for record in records}, {0, 1})
            self.assertTrue(all("corrupt.jpg" not in record.image_path for record in records))

    def test_tree_records_skip_truncated_jpegs_without_reencoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_image(root / "progan" / "0_real" / "real.png", 0)
            _write_image(root / "progan" / "0_real" / "real2.png", 64)
            _write_image(root / "progan" / "1_fake" / "fake.png", 128)
            _write_image(root / "progan" / "1_fake" / "fake2.png", 192)
            truncated = _jpeg_payload(200)[:-100]
            (root / "progan" / "0_real" / "truncated.jpg").write_bytes(truncated)

            records = PREPARE_SCRIPT.tree_records(root, "ProGAN", "progan", 2, 0)

            self.assertTrue(all(not record.repaired for record in records))
            self.assertTrue(
                all("truncated.jpg" not in record.image_path for record in records)
            )
            self.assertTrue(
                all(
                    PREPARE_SCRIPT._is_decodable_image(record.image_bytes())
                    for record in records
                )
            )

    def test_archive_tree_records_support_aigi_holmes_zip_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "TestSet.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "TestSet/JANUS/0_real/real.png", _png_payload(0)
                )
                archive.writestr(
                    "TestSet/JANUS/0_real/real2.png", _png_payload(64)
                )
                archive.writestr(
                    "TestSet/JANUS/0_real/truncated.jpg", _jpeg_payload(96)[:-100]
                )
                archive.writestr(
                    "TestSet/JANUS/1_fake/fake.png", _png_payload(128)
                )
                archive.writestr(
                    "TestSet/JANUS/1_fake/fake2.png", _png_payload(192)
                )

            records = PREPARE_SCRIPT.archive_tree_records(
                archive_path, "Janus", "Janus", 2, 0
            )

            self.assertEqual({record.label for record in records}, {0, 1})
            self.assertEqual(len(records), 4)
            self.assertTrue(
                all(record.image_path.startswith("Janus/test/") for record in records)
            )
            self.assertTrue(all(not record.repaired for record in records))
            self.assertTrue(
                all("truncated.jpg" not in record.image_path for record in records)
            )
            self.assertTrue(
                all(
                    PREPARE_SCRIPT._is_decodable_image(record.image_bytes())
                    for record in records
                )
            )

    @unittest.skipUnless(DATASETS_AVAILABLE, "datasets and pyarrow are required")
    def test_tree_records_write_a_project_arrow_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_domain = root / "JANUS"
            _write_image(raw_domain / "0_real" / "real.png", 0)
            _write_image(raw_domain / "1_fake" / "fake.png", 128)

            records = PREPARE_SCRIPT.tree_records(root, "Janus", "Janus", 1, 7)
            output = root / "arrow"
            output.mkdir()
            counts = PREPARE_SCRIPT.write_arrow_bundle(
                output, "aigi_holmes_p3", "Janus", records
            )

            self.assertEqual(counts, {"real": 1, "fake": 1})
            dataset = ArrowDataset(root=output, generator="Janus", split="test")
            self.assertEqual(len(dataset), 2)
            self.assertEqual({record.label for record in dataset.records}, {0, 1})
            self.assertTrue(
                all(record.image_path.startswith("Janus/test/") for record in dataset.records)
            )
            expected_payloads = {
                "Janus/test/0_real/real.png": (raw_domain / "0_real" / "real.png").read_bytes(),
                "Janus/test/1_fake/fake.png": (raw_domain / "1_fake" / "fake.png").read_bytes(),
            }
            actual_payloads = {
                record.image_path: dataset.raw_item(index)[0]
                for index, record in enumerate(dataset.records)
            }
            self.assertEqual(actual_payloads, expected_payloads)
            manifest = json.loads(
                (output / "test" / "Janus" / "bundle_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["recovered_images"], 0)
            self.assertEqual(manifest["content_policy"], "original_bytes_only")

    def test_write_arrow_bundle_rejects_repaired_record(self) -> None:
        record = PREPARE_SCRIPT.ExternalRecord(
            image_path="Janus/test/0_real/real.png",
            label=0,
            payload=_png_payload(0),
            repaired=True,
        )

        with self.assertRaisesRegex(ValueError, "unmodified, fully decodable original"):
            PREPARE_SCRIPT._validate_original_records([record])


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_payload(value))


def _png_payload(value: int) -> bytes:
    payload = io.BytesIO()
    Image.new("RGB", (4, 4), color=(value, value, value)).save(payload, format="PNG")
    return payload.getvalue()


def _jpeg_payload(value: int) -> bytes:
    payload = io.BytesIO()
    Image.new("RGB", (128, 128), color=(value, value, value)).save(payload, format="JPEG")
    return payload.getvalue()


if __name__ == "__main__":
    unittest.main()
