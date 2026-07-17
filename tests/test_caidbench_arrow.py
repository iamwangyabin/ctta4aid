from __future__ import annotations

import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from online_aig_tta.data.folders import build_dataset


PYARROW_AVAILABLE = importlib.util.find_spec("pyarrow") is not None


@unittest.skipUnless(PYARROW_AVAILABLE, "pyarrow is required")
class CAIDBenchArrowTests(unittest.TestCase):
    def test_reads_indexed_images_and_preserves_binary_labels(self) -> None:
        import pyarrow as pa
        import pyarrow.ipc as ipc
        import pyarrow.parquet as parquet

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            domain = root / "ADM"
            domain.mkdir()
            payloads = [_image_bytes(value) for value in (0, 32, 224, 255)]
            labels = [0, 0, 1, 1]
            batch = pa.record_batch(
                [pa.array(payloads, type=pa.binary()), pa.array(labels, type=pa.int8())],
                names=["image", "label"],
            )
            arrow_path = domain / "test.arrow"
            with pa.OSFile(str(arrow_path), "wb") as sink:
                with ipc.new_file(sink, batch.schema) as writer:
                    writer.write_batch(batch)

            parquet.write_table(
                pa.table(
                    {
                        "arrow_path": ["ADM/test.arrow"] * 4,
                        "batch_id": [0] * 4,
                        "row_in_batch": list(range(4)),
                        "label": labels,
                        "generator_name": ["ADM"] * 4,
                        "split": ["test"] * 4,
                    }
                ),
                root / "index.parquet",
            )

            dataset = build_dataset(
                data_format="caidbench_arrow",
                root=root,
                generator="ADM",
                split="test",
                transform=lambda image: image,
                max_samples_per_class=1,
                seed=7,
            )
            self.assertEqual(len(dataset), 2)
            samples = [dataset[index] for index in range(len(dataset))]
            self.assertEqual({label for _, label, _ in samples}, {0, 1})
            self.assertTrue(all(image.mode == "RGB" for image, _, _ in samples))
            self.assertEqual(len({sample_id for _, _, sample_id in samples}), 2)
            raw_image, raw_label, raw_sample_id = dataset.raw_item(0)
            self.assertEqual(Image.open(io.BytesIO(raw_image)).format, "PNG")
            self.assertIn(raw_label, {0, 1})
            self.assertEqual(raw_sample_id, samples[0][2])


def _image_bytes(value: int) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 4), color=(value, value, value)).save(output, format="PNG")
    return output.getvalue()


if __name__ == "__main__":
    unittest.main()
