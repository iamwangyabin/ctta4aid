from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from online_aig_tta.cli.common import checkpoint_sha256


class CheckpointIdentityTests(unittest.TestCase):
    def test_checkpoint_hash_is_sha256_of_exact_bytes(self) -> None:
        payload = b"shared-source-checkpoint"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.pt"
            path.write_bytes(payload)
            self.assertEqual(
                checkpoint_sha256(str(path)), hashlib.sha256(payload).hexdigest()
            )


if __name__ == "__main__":
    unittest.main()
