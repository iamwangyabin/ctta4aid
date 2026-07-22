from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from import_genimage_to_hf_arrow import (  # noqa: E402
    _iter_arrow_rows,
    collect_records,
    load_plan,
)


class ImportGenImageToHFArrowTests(unittest.TestCase):
    def test_collects_balanced_sources_with_canonical_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nature = root / "source" / "nature"
            ai = root / "source" / "ai"
            nature.mkdir(parents=True)
            ai.mkdir(parents=True)
            (nature / "class-a").mkdir()
            (nature / "class-a" / "real.jpg").write_bytes(b"real")
            (nature / "ignored.txt").write_text("ignored", encoding="utf-8")
            (ai / "fake.png").write_bytes(b"fake")
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "subset": "SDv14",
                                "split": "train",
                                "label": 0,
                                "path": "source/nature",
                                "expected_count": 1,
                                "expected_bytes": 4,
                            },
                            {
                                "subset": "SDv14",
                                "split": "train",
                                "label": 1,
                                "path": "source/ai",
                                "expected_count": 1,
                                "expected_bytes": 4,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            _, sources = load_plan(plan_path)
            records, summaries = collect_records(sources)

            self.assertEqual([record.label for record in records], [0, 1])
            self.assertEqual(
                [record.image_path for record in records],
                ["SDv14/train/nature/class-a/real.jpg", "SDv14/train/ai/fake.png"],
            )
            self.assertEqual(sum(item["bytes"] for item in summaries), 8)

    def test_requires_both_labels_per_subset_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "subset": "SDv14",
                                "split": "train",
                                "label": 1,
                                "path": "images",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "labels 0 and 1"):
                load_plan(plan_path)

    def test_collects_zip_prefixes_without_extracting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "sd14.zip"
            with ZipFile(archive_path, "w") as archive:
                archive.writestr("bundle/train/nature/real.jpg", b"real")
                archive.writestr("bundle/train/ai/fake.png", b"fake")
                archive.writestr("bundle/val/ai/ignored.jpg", b"ignored")
            plan_path = root / "plan.json"
            plan_path.write_text(
                json.dumps(
                    {
                        "sources": [
                            {
                                "subset": "SDv14",
                                "split": "train",
                                "label": 0,
                                "archive": "sd14.zip",
                                "prefix": "bundle/train/nature",
                                "expected_count": 1,
                                "expected_bytes": 4,
                            },
                            {
                                "subset": "SDv14",
                                "split": "train",
                                "label": 1,
                                "archive": "sd14.zip",
                                "prefix": "bundle/train/ai",
                                "expected_count": 1,
                                "expected_bytes": 4,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            _, sources = load_plan(plan_path)
            records, summaries = collect_records(sources)

            self.assertEqual(len(records), 2)
            self.assertTrue(all(record.archive_member for record in records))
            self.assertEqual({item["source_type"] for item in summaries}, {"zip"})
            self.assertEqual(
                {record.image_path for record in records},
                {"SDv14/train/nature/real.jpg", "SDv14/train/ai/fake.png"},
            )

            with patch("import_genimage_to_hf_arrow.ZipFile", wraps=ZipFile) as opener:
                rows = list(_iter_arrow_rows(records))
            self.assertEqual(opener.call_count, 1)
            self.assertEqual([row["image"] for row in rows], [b"real", b"fake"])


if __name__ == "__main__":
    unittest.main()
