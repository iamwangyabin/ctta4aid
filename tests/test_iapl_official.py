from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from run_iapl_official import (
    IAPL_COMMIT,
    build_command,
    file_identity,
    parse_official_summary,
    resolve_runtime_paths,
    validate_official_metrics,
)
from merge_iapl_shards import merge_shards


class IAPLOfficialRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "repo_path": "external/IAPL",
            "dataset": "GenImage",
            "dataset_path": "data/iapl",
            "train_selected_subsets": ["SDv14"],
            "test_selected_subsets": ["ADM", "BigGAN"],
            "pretrained_model": "weights/iapl.pth",
            "clip_checkpoint": "weights/ViT-L-14.pt",
            "output_dir": "outputs/iapl",
            "nproc_per_node": 8,
            "smooth": True,
        }

    def test_build_command_preserves_released_tta_settings(self) -> None:
        command = build_command(self.config, python_executable="python")
        joined = " ".join(command)
        self.assertIn("torch.distributed.run", joined)
        self.assertIn("--nproc_per_node 8", joined)
        self.assertIn("--evalbatchsize 32", joined)
        self.assertIn("--lr 0.005", joined)
        self.assertIn("--tta_steps 2", joined)
        self.assertIn("--selection_p 0.2", joined)
        self.assertIn("--condition True", joined)
        self.assertIn("--gate True", joined)
        self.assertIn("--ois True", joined)
        self.assertIn("--smooth True", joined)
        self.assertEqual(IAPL_COMMIT, "a173e7783bbafaa00d60e6e31774a0bc14411a23")

    def test_file_identity_records_size_and_sha256(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset.bin"
            path.write_bytes(b"iapl")
            identity = file_identity(path)
            self.assertEqual(identity["bytes"], 4)
            self.assertEqual(
                identity["sha256"],
                "7389de40b5e60d6490f6ca3546d37be8d6c0cfdf1c52143cb4ff828bc9124fdc",
            )

    def test_bn_ablation_patch_is_opt_in_and_records_protocol(self) -> None:
        patch = (
            Path(__file__).resolve().parents[1]
            / "patches"
            / "iapl-a173e77-bn-buffer-ablation.patch"
        ).read_text(encoding="utf-8")

        self.assertIn('IAPL_RESET_BN_PER_SAMPLE', patch)
        self.assertIn('IAPL_DDP_BROADCAST_BUFFERS', patch)
        self.assertIn('if initial_bn_buffers is not None:', patch)
        self.assertIn('"reset_bn_per_sample": reset_bn_per_sample', patch)
        self.assertIn('"ddp_broadcast_buffers": ddp_broadcast_buffers', patch)

    def test_shared_gpu_launcher_extends_collective_timeout(self) -> None:
        root = Path(__file__).resolve().parents[1]
        patch = (
            root / "patches" / "iapl-a173e77-distributed-timeout.patch"
        ).read_text(encoding="utf-8")
        launcher = (root / "scripts" / "run_iapl_manual_ranks.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn('IAPL_DISTRIBUTED_TIMEOUT_SECONDS', patch)
        self.assertIn("datetime.timedelta(seconds=timeout_seconds)", patch)
        self.assertIn(
            'distributed_timeout_seconds=${IAPL_DISTRIBUTED_TIMEOUT_SECONDS:-7200}',
            launcher,
        )
        self.assertIn(
            'export IAPL_DISTRIBUTED_TIMEOUT_SECONDS="$distributed_timeout_seconds"',
            launcher,
        )

    def test_genimage_manual_launcher_preserves_official_protocol(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (
            root / "scripts" / "run_iapl_genimage_manual_ranks.sh"
        ).read_text(encoding="utf-8")

        for argument in (
            "--dataset GenImage",
            "--train_selected_subsets SDv14",
            "--evalbatchsize 32",
            "--tta_steps 2",
            "--selection_p 0.2",
            "--ois True",
            "--smooth True",
            "--num_workers 8",
            "--seed 100",
        ):
            self.assertIn(argument, launcher)
        self.assertIn("checkpoint_best_acc_sd14.pth", launcher)
        self.assertIn("extract_manifest.json", launcher)

    def test_false_boolean_flag_is_omitted_for_authors_argparse(self) -> None:
        config = dict(self.config, smooth=False)
        command = build_command(config, python_executable="python")
        self.assertNotIn("--smooth", command)

    def test_relative_paths_are_rooted_at_project(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = resolve_runtime_paths(self.config, root)
            self.assertEqual(resolved["repo_path"], str((root / "external/IAPL").resolve()))
            self.assertEqual(resolved["output_dir"], str((root / "outputs/iapl").resolve()))

    def test_arrow_dataset_uri_is_not_rewritten_as_a_local_path(self) -> None:
        config = dict(
            self.config,
            dataset_path="hf_arrow:///data/ForenSynths|/data/Ojha",
        )
        with tempfile.TemporaryDirectory() as temporary:
            resolved = resolve_runtime_paths(config, Path(temporary))
        self.assertEqual(resolved["dataset_path"], config["dataset_path"])

    def test_parse_official_summary(self) -> None:
        log = """
(ADM) acc: 70.00; ap: 80.00; racc: 60.00; facc: 80.00;
(0 ADM       ) acc: 70.00; ap: 80.00; racc: 60.00; facc: 80.00;
(1 BigGAN    ) acc: 75.00; ap: 85.00; racc: 65.00; facc: 85.00;
(2 mean      ) acc: 72.50; ap: 82.50; racc: 62.50; facc: 82.50;
"""
        parsed = parse_official_summary(log)
        self.assertEqual(set(parsed["by_domain"]), {"ADM", "BigGAN"})
        self.assertAlmostEqual(parsed["by_domain"]["ADM"]["ap"], 0.8)
        self.assertAlmostEqual(parsed["mean"]["acc"], 0.725)

    def test_reference_gate_accepts_metrics_within_tolerance(self) -> None:
        config = dict(
            self.config,
            reference_metrics={"acc": 0.725, "ap": 0.825},
            reference_tolerance=0.001,
        )
        parsed = {
            "by_domain": {"ADM": {}, "BigGAN": {}},
            "mean": {"acc": 0.7255, "ap": 0.8245},
        }
        check = validate_official_metrics(parsed, config)
        self.assertTrue(check["passed"])

    def test_reference_gate_rejects_metrics_outside_tolerance(self) -> None:
        config = dict(
            self.config,
            reference_metrics={"acc": 0.9, "ap": 0.9},
            reference_tolerance=0.01,
        )
        parsed = {
            "by_domain": {"ADM": {}, "BigGAN": {}},
            "mean": {"acc": 0.7, "ap": 0.8},
        }
        check = validate_official_metrics(parsed, config)
        self.assertFalse(check["passed"])

    def test_reference_gate_requires_every_configured_domain(self) -> None:
        parsed = {"by_domain": {"ADM": {}}, "mean": {"acc": 0.7, "ap": 0.8}}
        with self.assertRaisesRegex(RuntimeError, "BigGAN"):
            validate_official_metrics(parsed, self.config)

    def test_merge_shards_reconstructs_official_domain_mean(self) -> None:
        config = dict(
            self.config,
            test_selected_subsets=["ADM", "BigGAN"],
            reference_metrics={"acc": 0.75, "ap": 0.85},
            reference_tolerance=0.001,
        )
        shards = [
            {
                "by_domain": {
                    "ADM": {"acc": 0.7, "ap": 0.8, "racc": 0.6, "facc": 0.8}
                }
            },
            {
                "by_domain": {
                    "BigGAN": {
                        "acc": 0.8,
                        "ap": 0.9,
                        "racc": 0.7,
                        "facc": 0.9,
                    }
                }
            },
        ]
        merged = merge_shards(shards, config)
        self.assertAlmostEqual(merged["mean"]["acc"], 0.75)
        self.assertTrue(merged["reference_check"]["passed"])


if __name__ == "__main__":
    unittest.main()
