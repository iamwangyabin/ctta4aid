from __future__ import annotations

import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_iapl_ufd_runs import parse_official_summary  # noqa: E402


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

    def test_parser_accepts_single_line_epoch_summary(self) -> None:
        summary = (
            "Epoch 0:(0 crn) acc: 90.00; ap: 91.00; racc: 92.00; "
            "facc: 88.00;; (1 mean) acc: 90.00; ap: 91.00; "
            "racc: 92.00; facc: 88.00;"
        )
        parsed = parse_official_summary(summary)
        self.assertEqual(parsed["by_domain"]["crn"]["acc"], 0.9)
        self.assertNotIn("mean", parsed["by_domain"])


if __name__ == "__main__":
    unittest.main()
