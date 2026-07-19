from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class IAPLTrainingLauncherTests(unittest.TestCase):
    def test_ufd_launcher_preserves_authors_training_protocol(self) -> None:
        launcher = (ROOT / "scripts" / "run_iapl_ufd_train_single.sh").read_text(
            encoding="utf-8"
        )
        for argument in (
            "--nproc_per_node=1",
            "--batchsize 16",
            "--evalbatchsize 32",
            "--train_selected_subsets car cat chair horse",
            "--lr 0.00005",
            "--dataset UniversalFakeDetect",
            "--epoch 1",
            "--lr_drop 10",
            "--gate True",
            "--condition True",
            "--num_workers 0",
        ):
            self.assertIn(argument, launcher)
        self.assertNotIn("--pretrained_model", launcher)
        self.assertIn("hf_arrow://", launcher)


if __name__ == "__main__":
    unittest.main()
