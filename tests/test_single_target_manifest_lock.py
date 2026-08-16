from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.data.streams import load_online_manifest_lock, validate_locked_sample_order


class SingleTargetManifestLockTests(unittest.TestCase):
    def test_loads_manifest_relative_to_experiment_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "configs" / "experiment.yaml"
            config_path.parent.mkdir()
            manifest_path = root / "results" / "online.csv"
            manifest_path.parent.mkdir()
            _write_manifest(
                manifest_path,
                [
                    {"batch": 0, "domain": "A", "position": 0, "sample_id": "A/a"},
                    {"batch": 1, "domain": "B", "position": 1, "sample_id": "B/b"},
                ],
            )

            lock = load_online_manifest_lock(
                {
                    "_config_path": str(config_path),
                    "data": {"locked_online_manifest": "../results/online.csv"},
                },
                ["A", "B"],
            )

            self.assertIsNotNone(lock)
            self.assertEqual(
                lock["sample_ids_by_domain"], {"A": ["A/a"], "B": ["B/b"]}
            )

    def test_rejects_a_target_result_with_different_sample_order(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "manifest order"):
            validate_locked_sample_order(
                ["A/b", "A/a"],
                ["A/a", "A/b"],
            )


def _write_manifest(path: Path, rows: list[dict[str, int | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["batch", "domain", "position", "sample_id"]
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
