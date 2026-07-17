from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.compare_ufd_raw_arrow import compare_domain, resolve_raw_layout


class CompareUFDRawArrowTest(unittest.TestCase):
    def _fixture(self, root: Path):
        raw_base = root / "CNN_synth_testset"
        real_path = "test/crn/0_real/real.png"
        fake_path = "test/crn/1_fake/fake.png"
        real_bytes = b"real-image"
        fake_bytes = b"fake-image"
        for relative, payload in (
            ("crn/0_real/real.png", real_bytes),
            ("crn/1_fake/fake.png", fake_bytes),
        ):
            destination = raw_base / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        entries = {real_path: 0, fake_path: 1}
        mapping = {real_path: 0, fake_path: 1}
        rows = [
            {
                "image_path": real_path,
                "image": real_bytes,
                "md5": hashlib.md5(real_bytes).hexdigest(),
            },
            {
                "image_path": fake_path,
                "image": fake_bytes,
                "md5": hashlib.md5(fake_bytes).hexdigest(),
            },
        ]
        return raw_base, entries, mapping, rows

    def test_exact_wrapper_layout_and_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_base, entries, mapping, rows = self._fixture(root)
            layout = resolve_raw_layout(root, list(entries))
            self.assertEqual(layout.base, raw_base.resolve())
            self.assertTrue(layout.strip_test_prefix)

            result = compare_domain(
                domain="crn",
                entries=entries,
                mapping=mapping,
                arrow_dataset=rows,
                raw_layout=layout,
            )

            self.assertTrue(result["exact"])
            self.assertTrue(result["checked_exact"])
            self.assertEqual(result["exact_byte_matches"], 2)
            self.assertEqual(
                result["arrow_manifest_sha256"], result["raw_manifest_sha256"]
            )

    def test_reports_missing_extra_and_byte_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_base, entries, mapping, rows = self._fixture(root)
            (raw_base / "crn/0_real/real.png").unlink()
            (raw_base / "crn/1_fake/fake.png").write_bytes(b"different")
            extra = raw_base / "crn/1_fake/extra.png"
            extra.write_bytes(b"extra")
            layout = resolve_raw_layout(root, ["test/crn/1_fake/fake.png"])

            result = compare_domain(
                domain="crn",
                entries=entries,
                mapping=mapping,
                arrow_dataset=rows,
                raw_layout=layout,
            )

            self.assertFalse(result["exact"])
            self.assertFalse(result["checked_exact"])
            self.assertEqual(result["missing_raw_paths"], 1)
            self.assertEqual(result["extra_raw_paths"], 1)
            self.assertEqual(result["byte_mismatches"], 1)


if __name__ == "__main__":
    unittest.main()
