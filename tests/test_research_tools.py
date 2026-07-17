from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.compare_iapl_ufd_runs import parse_official_summary
from scripts.export_hf_arrow_imagefolder import _safe_relative_path, export_domains


DATASETS_AVAILABLE = importlib.util.find_spec("datasets") is not None


class IAPLResearchToolTests(unittest.TestCase):
    def test_parse_numbered_official_summary(self) -> None:
        parsed = parse_official_summary(
            "(0 crn       ) acc: 92.47; ap: 99.97; racc: 90.00; facc: 94.94;\n"
            "(1 mean      ) acc: 92.47; ap: 99.97; racc: 90.00; facc: 94.94;\n"
        )
        self.assertEqual(set(parsed["by_domain"]), {"crn"})
        self.assertAlmostEqual(parsed["by_domain"]["crn"]["ap"], 0.9997)

    def test_imagefolder_path_normalizes_ojha_layout(self) -> None:
        self.assertEqual(
            _safe_relative_path(
                "guided/0_real/a.JPEG", domain="guided", split="test"
            ),
            Path("test/guided/0_real/a.JPEG"),
        )
        self.assertEqual(
            _safe_relative_path(
                "test/crn/1_fake/b.png", domain="crn", split="test"
            ),
            Path("test/crn/1_fake/b.png"),
        )

    @unittest.skipUnless(DATASETS_AVAILABLE, "datasets and pyarrow are required")
    def test_arrow_export_is_byte_exact_and_idempotent(self) -> None:
        from datasets import Dataset

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "Ojha"
            destination = root / "imagefolder"
            paths = ["guided/0_real/a.png", "guided/1_fake/b.png"]
            payloads = [_png_bytes(20), _png_bytes(220)]
            dataset = Dataset.from_dict(
                {
                    "image_path": paths,
                    "md5": ["", ""],
                    "width": [4, 4],
                    "height": [4, 4],
                    "image": payloads,
                }
            )
            dataset.save_to_disk(source)
            (source / "mapping.json").write_text(
                json.dumps({path: index for index, path in enumerate(paths)}),
                encoding="utf-8",
            )
            (source / "test.json").write_text(
                json.dumps({"guided": dict(zip(paths, [0, 1]))}), encoding="utf-8"
            )

            first = export_domains(
                roots=[source], domains=["guided"], output_root=destination
            )
            second = export_domains(
                roots=[source], domains=["guided"], output_root=destination
            )

            self.assertEqual(first["total_samples"], 2)
            self.assertEqual(second["domains"]["guided"]["label_counts"], {"0": 1, "1": 1})
            self.assertEqual(
                (destination / "test/guided/0_real/a.png").read_bytes(), payloads[0]
            )
            self.assertEqual(
                (destination / "test/guided/1_fake/b.png").read_bytes(), payloads[1]
            )


def _png_bytes(value: int) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 4), color=(value, value, value)).save(output, format="PNG")
    return output.getvalue()


if __name__ == "__main__":
    unittest.main()
