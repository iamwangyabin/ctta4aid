from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

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


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = io.BytesIO()
    Image.new("RGB", (4, 4), color=(value, value, value)).save(payload, format="PNG")
    path.write_bytes(payload.getvalue())


if __name__ == "__main__":
    unittest.main()
