from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_iapl_genimage_test import (  # noqa: E402
    GENERATOR_DIRS,
    LABEL_DIRS,
    extract_archive,
    sha256_file,
)


class ExtractIAPLGenImageTest(unittest.TestCase):
    def test_extracts_official_names_into_iapl_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "genimage_test.zip"
            expected_counts = {
                generator: {label: 1 for label in LABEL_DIRS}
                for generator in GENERATOR_DIRS.values()
            }
            with zipfile.ZipFile(archive_path, "w") as archive:
                for source, generator in GENERATOR_DIRS.items():
                    for label in LABEL_DIRS:
                        archive.writestr(
                            f"test/{source}/{label}/{generator}-{label}.png",
                            f"{generator}:{label}".encode("ascii"),
                        )

            output_root = root / "GenImage"
            summary = extract_archive(
                archive_path,
                output_root,
                expected_sha256=sha256_file(archive_path),
                expected_counts=expected_counts,
                progress_every=0,
            )

            self.assertEqual(summary["files"], 16)
            self.assertEqual(summary["written"], 16)
            for source, generator in GENERATOR_DIRS.items():
                for label, destination_label in LABEL_DIRS.items():
                    path = output_root / "test" / generator / destination_label
                    self.assertEqual(len(list(path.iterdir())), 1, source)

            resumed = extract_archive(
                archive_path,
                output_root,
                expected_counts=expected_counts,
                progress_every=0,
            )
            self.assertEqual(resumed["written"], 0)
            self.assertEqual(resumed["skipped_existing"], 16)

    def test_rejects_unexpected_archive_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "bad.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("test/unknown/nature/image.png", b"bad")

            with self.assertRaisesRegex(ValueError, "Unexpected GenImage member"):
                extract_archive(
                    archive_path,
                    root / "output",
                    expected_counts={},
                    progress_every=0,
                )


if __name__ == "__main__":
    unittest.main()
