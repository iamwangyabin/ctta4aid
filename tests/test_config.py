import tempfile
import unittest
from pathlib import Path

from src.config import (
    deep_merge,
    load_config,
    method_config,
    validate_experiment_identity,
)


class ConfigTest(unittest.TestCase):
    def test_deep_merge_keeps_unmodified_nested_values(self):
        base = {"optimizer": {"lr": 1e-4, "name": "adam"}, "steps": 1}
        merged = deep_merge(base, {"optimizer": {"lr": 2e-4}})
        self.assertEqual(merged["optimizer"], {"lr": 2e-4, "name": "adam"})
        self.assertEqual(base["optimizer"]["lr"], 1e-4)

    def test_method_config_combines_defaults_and_specific_values(self):
        config = {
            "method_defaults": {"steps": 1, "learning_rate": 1e-4},
            "method_configs": {"tent": {"learning_rate": 2e-4}},
        }
        self.assertEqual(
            method_config(config, "tent"), {"steps": 1, "learning_rate": 2e-4}
        )

    def test_load_config_expands_environment_variables(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.yaml"
            path.write_text("root: ${HOME}/data\n", encoding="utf-8")
            config = load_config(path)
            self.assertNotIn("${HOME}", config["root"])

    def test_load_config_supports_relative_multiple_inheritance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base.yaml").write_text(
                "data:\n  batch_size: 16\nmethods: [source]\n",
                encoding="utf-8",
            )
            (root / "method.yaml").write_text(
                "methods: [tent]\nmethod_configs:\n  tent:\n    lr: 0.001\n",
                encoding="utf-8",
            )
            (root / "experiment.yaml").write_text(
                "extends: [base.yaml, method.yaml]\ndata:\n  shuffle: true\n",
                encoding="utf-8",
            )
            config = load_config(root / "experiment.yaml")
            self.assertEqual(config["methods"], ["tent"])
            self.assertEqual(config["data"], {"batch_size": 16, "shuffle": True})
            self.assertEqual(config["method_configs"]["tent"]["lr"], 0.001)
            self.assertEqual(len(config["_config_sources"]), 3)

    def test_bias_control_identity_requires_matching_campaign_and_output_path(self):
        config = {
            "campaign": {
                "name": "clip_vlm_bias_controlled",
                "data_profile": "all_jpeg_q90",
            },
            "data": {"bias_control_profile": "all_jpeg_q90"},
            "output_dir": "/runs/clip_vlm_bias_controlled/all_jpeg_q90/genimage/seed0",
        }
        identity = validate_experiment_identity(config)
        self.assertEqual(identity["data_profile"], "all_jpeg_q90")
        self.assertIn("profile_spec_sha256", identity)

        bad_output = deep_merge(config, {"output_dir": "/runs/clip_vlm/genimage/seed0"})
        with self.assertRaisesRegex(ValueError, "output_dir"):
            validate_experiment_identity(bad_output)

        bad_campaign = deep_merge(
            config, {"campaign": {"data_profile": "matched_jpeg"}}
        )
        with self.assertRaisesRegex(ValueError, "campaign.data_profile"):
            validate_experiment_identity(bad_campaign)


if __name__ == "__main__":
    unittest.main()
