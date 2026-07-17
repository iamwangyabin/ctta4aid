from __future__ import annotations

import math
import unittest
from pathlib import Path

from online_aig_tta.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OfficialConfigTests(unittest.TestCase):
    def load(self, relative_path: str):
        return load_config(PROJECT_ROOT / relative_path)

    def test_cnn_configs_pin_official_method_defaults(self) -> None:
        for filename in ("configs/single_target.yaml", "configs/continual_stream.yaml"):
            with self.subTest(filename=filename):
                methods = self.load(filename)["method_configs"]
                self.assertEqual(methods["tent"]["optimizer"], "Adam")
                self.assertEqual(methods["tent"]["lr"], 0.001)
                self.assertAlmostEqual(
                    methods["eata"]["e_margin"], math.log(2.0) * 0.4
                )
                self.assertTrue(methods["eata"]["require_fisher"])
                self.assertEqual(methods["cotta"]["official_variant"], "imagenet")
                self.assertEqual(methods["cotta"]["optimizer"], "SGD")
                self.assertEqual(methods["cotta"]["lr"], 0.01)
                self.assertEqual(methods["cotta"]["anchor_confidence"], 0.1)
                self.assertEqual(methods["cotta"]["restore_probability"], 0.001)
                self.assertTrue(methods["cotta"]["symmetric_loss"])
                self.assertEqual(methods["cotta"]["augmentations"], 32)
                self.assertEqual(
                    methods["cotta"]["official_symbols"],
                    {"MT": 0.999, "RST": 0.001, "AP": 0.1, "N": 32},
                )
                self.assertEqual(methods["rotta"]["optimizer"], "Adam")
                self.assertEqual(methods["rotta"]["lr"], 0.001)
                self.assertEqual(methods["rotta"]["memory_size"], 64)
                self.assertEqual(methods["rotta"]["update_frequency"], 64)
                self.assertEqual(methods["rotta"]["num_classes"], 2)
                self.assertEqual(methods["rotta"]["nu"], 0.001)
                self.assertEqual(methods["rotta"]["alpha"], 0.05)
                self.assertEqual(
                    methods["rotta"]["official_symbols"]["UPDATE_FREQUENCY"],
                    64,
                )
                self.assertEqual(methods["lame"]["affinity"], "rbf")
                self.assertEqual(methods["lame"]["knn"], 5)
                self.assertEqual(methods["lame"]["bound_lambda"], 1.0)
                self.assertEqual(methods["lame"]["max_steps"], 100)
                self.assertEqual(methods["lame"]["parameter_update"], "none")
                self.assertEqual(
                    methods["lame"]["official_symbols"]["LAME_AFFINITY"],
                    "rbf",
                )
                self.assertEqual(methods["t2a"]["optimizer_config"]["lr"], 0.0001)
                self.assertEqual(methods["t2a"]["psi"], 0.01)
                self.assertIn("release_repairs", methods["t2a"])

    def test_eata_fisher_config_matches_official_preparation_size(self) -> None:
        training = self.load("configs/train_source.yaml")["training"]
        self.assertTrue(training["compute_fisher"])
        self.assertEqual(training["fisher_samples"], 2000)
        self.assertEqual(training["fisher_batch_size"], 64)

    def test_official_source_commits_are_fully_pinned(self) -> None:
        sources = self.load("configs/official_sources.yaml")
        expected = {
            "tent": "e9e926a668d85244c66a6d5c006efbd2b82e83e8",
            "eata": "f739b3668cc7617e9b9f1979c1a358497a3472c3",
            "cotta": "c212a204b32be4005092e4323105a24a29ad2952",
            "rotta": "67e34c900cdd355fc07e55edd4c577ea7b8ebcc9",
            "lame": "d2e5f63090bc1c8129bf7cbd781029a5955e1a67",
            "t2a": "33c8ccc64afdda260564123d6c790d030a89ff81",
            "iapl": "a173e7783bbafaa00d60e6e31774a0bc14411a23",
        }
        self.assertEqual(
            {
                name: config["commit"]
                for name, config in sources.items()
                if not name.startswith("_")
            },
            expected,
        )

    def test_every_cnn_wrapper_points_to_a_vendored_official_core(self) -> None:
        sources = self.load("configs/official_sources.yaml")
        for method in ("tent", "eata", "cotta", "rotta", "lame", "t2a"):
            with self.subTest(method=method):
                core = PROJECT_ROOT / sources[method]["official_core"]
                wrapper = PROJECT_ROOT / sources[method]["wrapper"]
                self.assertTrue(core.is_file())
                self.assertTrue(wrapper.is_file())
                wrapper_text = wrapper.read_text(encoding="utf-8")
                self.assertIn(f"online_aig_tta.official import {method}", wrapper_text)

    def test_iapl_configs_match_released_worker_settings(self) -> None:
        genimage = self.load("configs/iapl_official_genimage.yaml")
        universal = self.load("configs/iapl_official_ufd.yaml")
        self.assertEqual(genimage["num_workers"], 8)
        self.assertEqual(universal["num_workers"], 0)
        for config in (genimage, universal):
            self.assertEqual(config["nproc_per_node"], 8)
            self.assertEqual(config["evalbatchsize"], 32)
            self.assertEqual(config["tta_steps"], 2)
            self.assertEqual(config["selection_p"], 0.2)
            self.assertEqual(config["lr"], 0.005)
            self.assertEqual(config["epoch"], 1)
            self.assertEqual(config["lr_drop"], 10)
            self.assertTrue(config["gate"])
            self.assertTrue(config["condition"])
            self.assertTrue(config["tta"])
            self.assertTrue(config["eval"])
            self.assertTrue(config["require_reference_match"])

    def test_complete_dataset_setting_method_matrix_exists(self) -> None:
        methods = {"source", "tent", "eata", "cotta", "rotta", "lame", "t2a"}
        for dataset in ("genimage", "universalfake"):
            for setting in ("single_target", "continual"):
                directory = PROJECT_ROOT / "configs" / "experiments" / dataset / setting
                self.assertEqual({path.stem for path in directory.glob("*.yaml")}, methods)
                for method in methods:
                    with self.subTest(dataset=dataset, setting=setting, method=method):
                        config = load_config(directory / f"{method}.yaml")
                        self.assertEqual(config["methods"], [method])
                        self.assertEqual(config["data"]["format"], dataset)
                        self.assertEqual(config["data"]["batch_size"], 16)
                        self.assertEqual(config["protocol"]["name"], "predict_then_adapt")
                        self.assertEqual(
                            config["output_dir"], f"outputs/{dataset}/{setting}"
                        )

    def test_t2a_unreported_release_values_are_isolated(self) -> None:
        config = self.load("configs/methods/t2a.yaml")["method_configs"]["t2a"]
        self.assertEqual(config["noise_type"], "bernoulli")
        self.assertEqual(config["gamma"], 2.0)
        self.assertEqual(
            config["release_repairs"]["status"],
            "required_to_execute_not_reported_as_official_hyperparameters",
        )


if __name__ == "__main__":
    unittest.main()
