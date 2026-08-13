from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.cli.common import checkpoint_sha256
from src.models.iapl import _resolve_path


class CheckpointIdentityTests(unittest.TestCase):
    def test_checkpoint_hash_is_sha256_of_exact_bytes(self) -> None:
        payload = b"shared-source-checkpoint"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.pt"
            path.write_bytes(payload)
            self.assertEqual(
                checkpoint_sha256(str(path)), hashlib.sha256(payload).hexdigest()
            )

    def test_iapl_asset_paths_resolve_from_project_root(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        self.assertEqual(_resolve_path("weights/iapl.pt"), project_root / "weights/iapl.pt")


if __name__ == "__main__":
    unittest.main()
