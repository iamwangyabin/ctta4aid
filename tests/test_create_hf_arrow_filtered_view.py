from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from create_hf_arrow_filtered_view import create_view  # noqa: E402


DATASETS_AVAILABLE = importlib.util.find_spec("datasets") is not None


@unittest.skipUnless(DATASETS_AVAILABLE, "datasets and pyarrow are required")
class FilteredArrowViewTests(unittest.TestCase):
    def test_excludes_only_empty_rows_and_hard_links_shards(self) -> None:
        from datasets import Dataset, load_from_disk
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "filtered"
            real_path = "SDv14/train/nature/real.png"
            empty_path = "SDv14/train/ai/empty.png"
            fake_path = "SDv14/train/ai/fake.png"
            image = io.BytesIO()
            Image.new("RGB", (2, 2), color=(12, 34, 56)).save(image, format="PNG")
            Dataset.from_dict(
                {
                    "image_path": [real_path, empty_path, fake_path],
                    "image": [image.getvalue(), b"", image.getvalue()],
                }
            ).save_to_disk(source)
            mapping = {real_path: 0, empty_path: 1, fake_path: 2}
            (source / "mapping.json").write_text(
                json.dumps(mapping), encoding="utf-8"
            )
            (source / "train.json").write_text(
                json.dumps(
                    {"SDv14": {real_path: 0, empty_path: 1, fake_path: 1}}
                ),
                encoding="utf-8",
            )
            plan = root / "plan.json"
            plan.write_text(
                json.dumps(
                    {
                        "source_root": str(source),
                        "generator": "SDv14",
                        "split": "train",
                        "exclude_image_paths": [empty_path],
                        "expected_rows": 2,
                        "expected_label_counts": {"0": 1, "1": 1},
                    }
                ),
                encoding="utf-8",
            )

            manifest = create_view(plan, output)

            self.assertEqual(manifest["selected_rows"], 2)
            self.assertEqual(len(load_from_disk(output)), 3)
            metadata = json.loads((output / "train.json").read_text(encoding="utf-8"))
            self.assertNotIn(empty_path, metadata["SDv14"])
            source_shard = next(source.glob("*.arrow"))
            output_shard = output / source_shard.name
            self.assertEqual(source_shard.stat().st_ino, output_shard.stat().st_ino)


if __name__ == "__main__":
    unittest.main()
