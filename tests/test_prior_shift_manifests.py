from __future__ import annotations

import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from scripts.build_prior_shift_manifests import (
    binary_label_from_sample_id,
    build_prior_shift_rows,
    class_counts,
    write_manifest,
)
from src.data.streams import load_locked_manifest


def manifest_rows(per_class: int = 10) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for domain in ("first", "second"):
        domain_rows = []
        for index in range(per_class):
            domain_rows.extend(
                [
                    f"root/{domain}/test/0_real/real_{index}.jpg",
                    f"root/{domain}/test/1_fake/fake_{index}.jpg",
                ]
            )
        for sample_id in domain_rows:
            position = len(rows)
            rows.append(
                {
                    "batch": position // 4,
                    "domain": domain,
                    "position": position,
                    "sample_id": sample_id,
                }
            )
    return rows


class PriorShiftManifestTests(unittest.TestCase):
    def test_recovers_only_canonical_binary_components(self) -> None:
        self.assertEqual(binary_label_from_sample_id("x/nature/a.jpg"), 0)
        self.assertEqual(binary_label_from_sample_id("x/ai/a.jpg"), 1)
        with self.assertRaises(ValueError):
            binary_label_from_sample_id("x/unknown/a.jpg")
        with self.assertRaises(ValueError):
            binary_label_from_sample_id("x/0_real/1_fake/a.jpg")

    def test_requires_integral_nonempty_class_counts(self) -> None:
        self.assertEqual(class_counts(800, Decimal("0.1")), (720, 80))
        with self.assertRaises(ValueError):
            class_counts(7, Decimal("0.5"))
        with self.assertRaises(ValueError):
            class_counts(800, Decimal("1"))

    def test_builds_exact_deterministic_nested_prior_streams(self) -> None:
        source = manifest_rows()
        low_rows, low_summary = build_prior_shift_rows(
            source,
            samples_per_domain=8,
            fake_ratio=Decimal("0.25"),
            seed=3,
            batch_size=4,
        )
        high_rows, high_summary = build_prior_shift_rows(
            source,
            samples_per_domain=8,
            fake_ratio=Decimal("0.75"),
            seed=3,
            batch_size=4,
        )
        repeated, _ = build_prior_shift_rows(
            source,
            samples_per_domain=8,
            fake_ratio=Decimal("0.25"),
            seed=3,
            batch_size=4,
        )

        self.assertEqual(low_rows, repeated)
        self.assertEqual(
            low_summary,
            {"first": {"real": 6, "fake": 2}, "second": {"real": 6, "fake": 2}},
        )
        self.assertEqual(
            high_summary,
            {"first": {"real": 2, "fake": 6}, "second": {"real": 2, "fake": 6}},
        )
        for output in (low_rows, high_rows):
            self.assertEqual([row["position"] for row in output], list(range(16)))
            self.assertEqual(
                [row["batch"] for row in output], [i // 4 for i in range(16)]
            )
            for domain in ("first", "second"):
                positions = [
                    int(row["position"])
                    for row in output
                    if row["domain"] == domain
                ]
                self.assertEqual(positions, sorted(positions))

        for label in (0, 1):
            low_ids = {
                str(row["sample_id"])
                for row in low_rows
                if binary_label_from_sample_id(str(row["sample_id"])) == label
            }
            high_ids = {
                str(row["sample_id"])
                for row in high_rows
                if binary_label_from_sample_id(str(row["sample_id"])) == label
            }
            smaller, larger = sorted((low_ids, high_ids), key=len)
            self.assertTrue(smaller <= larger)

    def test_written_manifest_passes_project_loader(self) -> None:
        rows, _ = build_prior_shift_rows(
            manifest_rows(),
            samples_per_domain=8,
            fake_ratio=Decimal("0.5"),
            seed=0,
            batch_size=4,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prior.csv"
            write_manifest(path, rows)
            loaded = load_locked_manifest(path)
        self.assertEqual(loaded, rows)


if __name__ == "__main__":
    unittest.main()
