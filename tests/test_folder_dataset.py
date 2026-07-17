import tempfile
import unittest
from pathlib import Path

from PIL import Image

from online_aig_tta.data.folders import build_dataset


def write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=(value, value, value)).save(path)


class FolderDatasetTest(unittest.TestCase):
    def test_genimage_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_image(root / "ADM" / "val" / "nature" / "real.png", 0)
            write_image(root / "ADM" / "val" / "ai" / "fake.png", 255)
            dataset = build_dataset(
                data_format="genimage",
                root=root,
                generator="ADM",
                split="val",
                transform=lambda image: image,
            )
            self.assertEqual(len(dataset), 2)
            self.assertEqual({record.label for record in dataset.records}, {0, 1})

    def test_universal_fake_detect_recursive_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_image(root / "progan" / "airplane" / "0_real" / "r.png", 0)
            write_image(root / "progan" / "airplane" / "1_fake" / "f.png", 255)
            dataset = build_dataset(
                data_format="ufd",
                root=root,
                generator="progan",
                split=None,
                transform=lambda image: image,
            )
            self.assertEqual(len(dataset), 2)

    def test_stream_and_holdout_slices_are_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for label_name in ("nature", "ai"):
                for index in range(4):
                    write_image(root / "ADM" / "val" / label_name / f"{index}.png", index)
            common = dict(
                data_format="genimage",
                root=root,
                generator="ADM",
                split="val",
                transform=lambda image: image,
                max_samples_per_class=2,
                seed=7,
            )
            stream = build_dataset(**common, sample_offset_per_class=0)
            holdout = build_dataset(**common, sample_offset_per_class=2)
            stream_paths = {record.path for record in stream.records}
            holdout_paths = {record.path for record in holdout.records}
            self.assertFalse(stream_paths & holdout_paths)


if __name__ == "__main__":
    unittest.main()
