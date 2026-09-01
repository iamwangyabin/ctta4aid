from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.data.streams import (
    build_domain_loader,
    load_locked_manifest,
    lock_stream_to_manifest,
    locked_sample_ids_by_domain,
)
from src.types import StreamBatch


class StreamManifestLockTests(unittest.TestCase):
    def test_manifest_locks_sample_identity_order_and_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "online.csv"
            _write_manifest(
                manifest_path,
                [
                    {"batch": 0, "domain": "A", "position": 0, "sample_id": "A/a"},
                    {"batch": 0, "domain": "A", "position": 1, "sample_id": "A/b"},
                    {"batch": 1, "domain": "B", "position": 2, "sample_id": "B/a"},
                ],
            )
            manifest = load_locked_manifest(manifest_path)
            stream = [
                StreamBatch(None, None, "A", ["A/a", "A/b"]),
                StreamBatch(None, None, "B", ["B/a"]),
            ]

            observed = list(lock_stream_to_manifest(stream, manifest, name="online"))

            self.assertEqual(
                [batch.sample_ids for batch in observed], [["A/a", "A/b"], ["B/a"]]
            )
            self.assertEqual(
                locked_sample_ids_by_domain(manifest, ["A", "B"]),
                {"A": ["A/a", "A/b"], "B": ["B/a"]},
            )

    def test_manifest_lock_rejects_a_changed_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "online.csv"
            _write_manifest(
                manifest_path,
                [{"batch": 0, "domain": "A", "position": 0, "sample_id": "A/a"}],
            )
            manifest = load_locked_manifest(manifest_path)
            stream = [StreamBatch(None, None, "A", ["A/other"])]

            with self.assertRaisesRegex(RuntimeError, "mismatch at position 0"):
                list(lock_stream_to_manifest(stream, manifest, name="online"))

    def test_manifest_can_lock_identity_without_delivery_batches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "online.csv"
            _write_manifest(
                manifest_path,
                [
                    {"batch": 0, "domain": "A", "position": 0, "sample_id": "A/a"},
                    {"batch": 1, "domain": "A", "position": 1, "sample_id": "A/b"},
                ],
            )
            manifest = load_locked_manifest(manifest_path)
            stream = [StreamBatch(None, None, "A", ["A/a", "A/b"])]

            observed = list(
                lock_stream_to_manifest(
                    stream,
                    manifest,
                    name="online",
                    check_batches=False,
                )
            )

            self.assertEqual(observed[0].sample_ids, ["A/a", "A/b"])

    def test_manifest_rejects_unknown_stream_domain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "online.csv"
            _write_manifest(
                manifest_path,
                [{"batch": 0, "domain": "B", "position": 0, "sample_id": "B/a"}],
            )

            with self.assertRaisesRegex(ValueError, "outside the stream"):
                locked_sample_ids_by_domain(load_locked_manifest(manifest_path), ["A"])

    def test_manifest_rejects_a_noncanonical_domain_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "online.csv"
            _write_manifest(
                manifest_path,
                [
                    {"batch": 0, "domain": "B", "position": 0, "sample_id": "B/a"},
                    {"batch": 1, "domain": "A", "position": 1, "sample_id": "A/a"},
                ],
            )

            with self.assertRaisesRegex(ValueError, "domain order differs"):
                locked_sample_ids_by_domain(load_locked_manifest(manifest_path), ["A", "B"])


class DomainLoaderTests(unittest.TestCase):
    def test_resolves_evaluator_episode_alias_to_dataset_generator(self) -> None:
        data_config = {
            "format": "arrow",
            "root": "unused",
            "split": "test",
            "batch_size": 16,
            "num_workers": 0,
            "generator_aliases": {"A_return": "generator_a"},
        }
        with patch("src.data.streams.build_dataset", return_value=object()) as build, patch(
            "torch.utils.data.DataLoader"
        ):
            build_domain_loader(data_config, "A_return", seed=7, transform=object())

        self.assertEqual(build.call_args.kwargs["generator"], "generator_a")

    def test_uses_configured_worker_start_method(self) -> None:
        data_config = {
            "format": "arrow",
            "root": "unused",
            "split": "test",
            "batch_size": 16,
            "num_workers": 4,
            "worker_start_method": "spawn",
        }
        with patch("src.data.streams.build_dataset", return_value=object()), patch(
            "torch.utils.data.DataLoader"
        ) as loader:
            build_domain_loader(data_config, "ADM", seed=7, transform=object())

        self.assertEqual(loader.call_args.kwargs["multiprocessing_context"], "spawn")
        self.assertEqual(loader.call_args.kwargs["num_workers"], 4)

    def test_omits_worker_context_when_workers_are_disabled(self) -> None:
        data_config = {
            "format": "arrow",
            "root": "unused",
            "split": "test",
            "batch_size": 16,
            "num_workers": 0,
            "worker_start_method": "spawn",
        }
        with patch("src.data.streams.build_dataset", return_value=object()), patch(
            "torch.utils.data.DataLoader"
        ) as loader:
            build_domain_loader(data_config, "ADM", seed=7, transform=object())

        self.assertNotIn("multiprocessing_context", loader.call_args.kwargs)
        self.assertEqual(loader.call_args.kwargs["num_workers"], 0)


def _write_manifest(path: Path, rows: list[dict[str, int | str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["batch", "domain", "position", "sample_id"]
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
