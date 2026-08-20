from __future__ import annotations

import math
import unittest
from pathlib import Path

from src.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OfficialConfigTests(unittest.TestCase):
    def load(self, relative_path: str):
        return load_config(PROJECT_ROOT / relative_path)

    def test_cnn_configs_pin_official_method_defaults(self) -> None:
        for filename in (
            "configs/experiments/controlled_ctta/single_target.yaml",
            "configs/experiments/controlled_ctta/continual.yaml",
        ):
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
        training = self.load("configs/train/source.yaml")["training"]
        self.assertTrue(training["compute_fisher"])
        self.assertEqual(training["fisher_samples"], 2000)
        self.assertEqual(training["fisher_batch_size"], 64)
        self.assertIn("t2a", training["intended_methods"])
        self.assertEqual(
            training["checkpoint_role"], "shared_source_detector_including_t2a"
        )

    def test_t2a_has_explicit_shared_source_training_entries(self) -> None:
        expected = {
            "configs/train/t2a_ufd_source.yaml": "progan",
            "configs/train/t2a_genimage_source.yaml": "stable_diffusion_v_1_4",
        }
        for filename, generator in expected.items():
            with self.subTest(filename=filename):
                config = self.load(filename)
                self.assertEqual(config["model"]["architecture"], "resnet50")
                self.assertEqual(config["data"]["format"], "arrow")
                self.assertEqual(config["data"]["generator"], generator)
                self.assertEqual(config["training"]["requested_for"], "t2a")
                self.assertIn("t2a", config["training"]["intended_methods"])
                self.assertIn(
                    "shared_source_detector", config["training"]["checkpoint_role"]
                )

    def test_official_source_commits_are_fully_pinned(self) -> None:
        sources = self.load("configs/official_sources.yaml")
        expected = {
            "tent": "e9e926a668d85244c66a6d5c006efbd2b82e83e8",
            "eata": "f739b3668cc7617e9b9f1979c1a358497a3472c3",
            "cotta": "c212a204b32be4005092e4323105a24a29ad2952",
            "rotta": "67e34c900cdd355fc07e55edd4c577ea7b8ebcc9",
            "lame": "d2e5f63090bc1c8129bf7cbd781029a5955e1a67",
            "t2a": "33c8ccc64afdda260564123d6c790d030a89ff81",
            "ost": "1e4518b9e560baf9c5693f13a402fa5d7104190f",
            "iapl": "a173e7783bbafaa00d60e6e31774a0bc14411a23",
            "openai_clip": "d05afc436d78f1c48dc0dbf8e5980a9d471f35f6",
            "tda": "e697fb0c8078cdeff93daa56bcf8860702542069",
            "dynaprompt": "acd33cf71f5be817512f99ba3b81ec019595ad59",
            "cliptta": "ef0e6797f7618959ca85be36816a5e01299a522f",
            "batclip": "ba2e3381873ef58e76a90148ee3835864349e985",
            "sar": "20f6e24b17525f34503510afccedc0629b67b7c4",
        }
        self.assertEqual(
            {
                name: config["commit"]
                for name, config in sources.items()
                if not name.startswith("_") and "commit" in config
            },
            expected,
        )

    def test_clip_vlm_main_configs_lock_the_primary_table_protocol(self) -> None:
        expected_targets = {
            "genimage": 7,
            "aigc_detection_benchmark": 17,
            "aigi_holmes_p3": 10,
            "opensdid_global": 5,
        }
        expected_methods = [
            "source_ft",
            "tent",
            "eata",
            "sar",
            "cotta",
            "lame",
            "t2a",
            "frozen_clip",
            "tda",
            "dynaprompt",
            "cliptta",
            "batclip",
            "iapl",
            "ours_static",
            "ours",
        ]
        for dataset, target_count in expected_targets.items():
            for seed in (0, 1, 2):
                filename = f"configs/experiments/clip_vlm/{dataset}_seed{seed}.yaml"
                with self.subTest(filename=filename):
                    config = self.load(filename)
                    self.assertEqual(config["methods"], expected_methods)
                    self.assertEqual(config["seed"], seed)
                    self.assertEqual(config["model"]["family"], "clip_vlm_main")
                    self.assertEqual(
                        config["model"]["source_detector"]["architecture"],
                        "ViT-L/14",
                    )
                    self.assertEqual(
                        config["model"]["clip_native"]["architecture"], "ViT-L/14"
                    )
                    self.assertNotIn("class_prompts", config["model"])
                    self.assertEqual(config["protocol"]["name"], "method_native_online")
                    self.assertFalse(
                        config["protocol"]["target_labels_available_to_method"]
                    )
                    self.assertEqual(len(config["data"]["targets"]), target_count)
                    self.assertEqual(
                        config["method_configs"]["tent"]["data"]["batch_size"],
                        16,
                    )
                    self.assertEqual(config["data"]["num_workers"], 0)
                    for method_name in expected_methods:
                        self.assertEqual(
                            config["method_configs"][method_name]["data"][
                                "num_workers"
                            ],
                            0,
                        )
                    if dataset == "genimage":
                        self.assertEqual(
                            config["data"]["worker_start_method"], "spawn"
                        )
                    else:
                        self.assertNotIn("worker_start_method", config["data"])
                    self.assertTrue(
                        config["method_configs"]["tent"]["clip_visual_layernorm"]
                    )
                    self.assertTrue(
                        config["method_configs"]["eata"]["clip_visual_layernorm"]
                    )
                    self.assertTrue(
                        config["method_configs"]["t2a"]["clip_visual_layernorm"]
                    )
                    self.assertEqual(
                        config["method_configs"]["sar"]["data"]["batch_size"], 16
                    )
                    self.assertAlmostEqual(
                        config["method_configs"]["sar"]["margin"],
                        math.log(2.0) * 0.4,
                    )
                    self.assertEqual(
                        config["method_configs"]["dynaprompt"]["data"]["batch_size"],
                        1,
                    )
                    self.assertEqual(
                        config["method_configs"]["iapl"]["data"]["batch_size"],
                        1,
                    )
                    self.assertEqual(
                        config["method_configs"]["ours"]["adaptation_mode"],
                        "full",
                    )
                    self.assertEqual(
                        config["method_configs"]["ours_static"]["adaptation_mode"],
                        "static",
                    )
                    self.assertEqual(
                        config["method_configs"]["ours"]["memory_size"], 256
                    )
                    self.assertEqual(
                        config["method_configs"]["ours"]["checkpoint_sha256"],
                        "a3f15593bf3a46d3ce318a5e160b33372a27d0712baf66906592065890edc9d4",
                    )
                    self.assertEqual(
                        config["reporting"]["paired_static_baselines"]["ours"],
                        "ours_static",
                    )
                    self.assertTrue(
                        config["method_configs"]["batclip"]["vlm"]
                        ["dynamic_text_features"]
                    )
                    manifest = (
                        Path(config["_config_path"]).parent
                        / config["data"]["locked_online_manifest"]
                    ).resolve()
                    self.assertTrue(manifest.is_file())

        sources = self.load("configs/official_sources.yaml")
        self.assertEqual(sources["sar"]["official_core"], "src/official/sar.py")
        self.assertTrue((PROJECT_ROOT / sources["sar"]["official_core"]).is_file())

    def test_clip_vitl14_source_training_config_is_shared_and_fisher_compatible(self) -> None:
        config = self.load("configs/train/genimage_sd14_clip_vitl14.yaml")
        self.assertEqual(config["model"]["family"], "clip_source_detector")
        self.assertEqual(config["model"]["architecture"], "ViT-L/14")
        self.assertEqual(config["model"]["trainable_scope"], "full")
        self.assertEqual(config["data"]["format"], "arrow")
        self.assertEqual(config["data"]["train_generator"], "SDv14")
        self.assertEqual(
            config["data"]["val_generator"], "stable_diffusion_v_1_4"
        )
        self.assertEqual(
            config["data"]["train_exclude_image_paths"],
            [
                "SDv14/train/ai/033_sdv4_00134.png",
                "SDv14/train/ai/033_sdv4_00137.png",
                "SDv14/train/ai/033_sdv4_00152.png",
            ],
        )
        training = config["training"]
        self.assertEqual(training["epochs"], 3)
        self.assertTrue(training["compute_fisher"])
        self.assertEqual(training["fisher_parameter_scope"], "clip_visual_layernorm")
        self.assertEqual(training["fisher_samples"], 2000)
        self.assertEqual(
            training["intended_methods"],
            ["source_ft", "tent", "eata", "sar", "cotta", "lame", "t2a"],
        )

    def test_clip_vlm_bias_controlled_configs_are_isolated_from_raw_runs(self) -> None:
        expected_roots = {
            "all_jpeg_q90": {
                "genimage": "${GENIMAGE_ALL_JPEG_Q90_ARROW_ROOT}",
                "aigc_detection_benchmark": (
                    "${AIGC_DETECTION_BENCHMARK_ALL_JPEG_Q90_ARROW_ROOT}"
                ),
                "aigi_holmes_p3": "${AIGI_HOLMES_P3_ALL_JPEG_Q90_ARROW_ROOT}",
                "opensdid_global": "${OPENSDID_GLOBAL_ALL_JPEG_Q90_ARROW_ROOT}",
            },
            "matched_jpeg": {
                "genimage": "${GENIMAGE_MATCHED_JPEG_ARROW_ROOT}",
                "aigc_detection_benchmark": (
                    "${AIGC_DETECTION_BENCHMARK_MATCHED_JPEG_ARROW_ROOT}"
                ),
                "aigi_holmes_p3": "${AIGI_HOLMES_P3_MATCHED_JPEG_ARROW_ROOT}",
                "opensdid_global": "${OPENSDID_GLOBAL_MATCHED_JPEG_ARROW_ROOT}",
            },
        }
        raw_methods = self.load(
            "configs/experiments/clip_vlm/genimage_seed0.yaml"
        )["methods"]
        for profile, dataset_roots in expected_roots.items():
            for dataset, expected_root in dataset_roots.items():
                for seed in (0, 1, 2):
                    filename = (
                        "configs/experiments/clip_vlm_bias_controlled/"
                        f"{profile}_{dataset}_seed{seed}.yaml"
                    )
                    with self.subTest(filename=filename):
                        config = self.load(filename)
                        self.assertEqual(config["methods"], raw_methods)
                        self.assertEqual(config["seed"], seed)
                        self.assertEqual(
                            config["campaign"],
                            {
                                "name": "clip_vlm_bias_controlled",
                                "data_profile": profile,
                                "source_setup": "unchanged_from_clip_vlm_main",
                            },
                        )
                        self.assertEqual(
                            config["data"]["bias_control_profile"], profile
                        )
                        self.assertEqual(config["data"]["root"], expected_root)
                        self.assertEqual(
                            config["output_dir"],
                            "${CTTA4AID_EXPERIMENT_ROOT}/"
                            f"clip_vlm_bias_controlled/{profile}/{dataset}/seed{seed}",
                        )
                        manifest = (
                            Path(config["_config_path"]).parent
                            / config["data"]["locked_online_manifest"]
                        ).resolve()
                        self.assertTrue(manifest.is_file())

    def test_every_cnn_wrapper_points_to_a_vendored_official_core(self) -> None:
        sources = self.load("configs/official_sources.yaml")
        for method in ("tent", "eata", "cotta", "rotta", "lame", "t2a"):
            with self.subTest(method=method):
                core = PROJECT_ROOT / sources[method]["official_core"]
                wrapper = PROJECT_ROOT / sources[method]["wrapper"]
                self.assertTrue(core.is_file())
                self.assertTrue(wrapper.is_file())
                wrapper_text = wrapper.read_text(encoding="utf-8")
                self.assertIn(f"src.official import {method}", wrapper_text)

    def test_iapl_configs_define_independent_single_target_runs(self) -> None:
        genimage = self.load("configs/experiments/iapl/genimage.yaml")
        universal = self.load("configs/experiments/iapl/ufd.yaml")
        self.assertEqual(genimage["data"]["format"], "arrow")
        self.assertEqual(universal["data"]["format"], "arrow")
        self.assertNotIn("stable_diffusion_v_1_4", genimage["data"]["targets"])
        self.assertEqual(len(genimage["data"]["targets"]), 7)
        self.assertEqual(genimage["seed"], 0)
        self.assertEqual(genimage["data"]["num_workers"], 8)
        self.assertEqual(universal["data"]["num_workers"], 0)
        external = (
            "configs/experiments/iapl/aigc_detection_benchmark.yaml",
            "configs/experiments/iapl/aigi_holmes_p3.yaml",
            "configs/experiments/iapl/opensdid_global.yaml",
        )
        for config in (genimage, universal, *(self.load(path) for path in external)):
            method = config["method_configs"]["iapl"]
            self.assertEqual(config["methods"], ["iapl"])
            self.assertEqual(config["data"]["batch_size"], 1)
            self.assertEqual(config["protocol"]["name"], "episodic_adapt_then_predict")
            self.assertEqual(method["views"], 32)
            self.assertEqual(method["steps"], 2)
            self.assertEqual(method["selection_fraction"], 0.2)
            self.assertEqual(method["lr"], 0.005)
            self.assertTrue(method["gate"])
            self.assertTrue(method["condition"])
            self.assertTrue(method["optimal_input_selection"])
            self.assertFalse(config["protocol"]["batchnorm_buffers_accumulate_across_targets"])

        for filename in external:
            with self.subTest(filename=filename):
                config = self.load(filename)
                self.assertEqual(config["seed"], 0)
                self.assertIn("locked_online_manifest", config["data"])
                self.assertEqual(config["method_configs"]["iapl"]["adaptation_mode"], "full")

        static_external = (
            "configs/experiments/iapl/aigc_detection_benchmark_static.yaml",
            "configs/experiments/iapl/aigi_holmes_p3_static.yaml",
            "configs/experiments/iapl/opensdid_global_static.yaml",
        )
        for filename in static_external:
            with self.subTest(filename=filename):
                config = self.load(filename)
                self.assertEqual(config["seed"], 0)
                self.assertIn("locked_online_manifest", config["data"])
                self.assertEqual(config["protocol"]["name"], "predict_only")
                self.assertEqual(config["method_configs"]["iapl"]["adaptation_mode"], "static")
                self.assertFalse(
                    config["protocol"]["batchnorm_buffers_accumulate_across_targets"]
                )

        for filename, mode in (
            ("configs/experiments/iapl/genimage_static.yaml", "static"),
            ("configs/experiments/iapl/genimage_views_only.yaml", "views_only"),
        ):
            with self.subTest(filename=filename):
                config = self.load(filename)
                self.assertEqual(config["method_configs"]["iapl"]["adaptation_mode"], mode)
                self.assertEqual(config["data"]["targets"], genimage["data"]["targets"])
                self.assertNotEqual(
                    config["protocol"]["name"], "episodic_adapt_then_predict"
                )

    def test_ost_configs_define_source_template_adapt_then_predict(self) -> None:
        for filename in (
            "configs/experiments/ost/genimage.yaml",
            "configs/experiments/ost/ufd.yaml",
            "configs/experiments/ost/aigc_detection_benchmark.yaml",
            "configs/experiments/ost/aigi_holmes_p3.yaml",
            "configs/experiments/ost/opensdid_global.yaml",
        ):
            with self.subTest(filename=filename):
                config = self.load(filename)
                method = config["method_configs"]["ost"]
                self.assertEqual(config["methods"], ["ost"])
                self.assertEqual(config["data"]["format"], "arrow")
                self.assertEqual(config["data"]["batch_size"], 1)
                self.assertEqual(
                    config["protocol"]["name"], "episodic_adapt_then_predict"
                )
                self.assertFalse(config["protocol"]["source_free_during_test"])
                self.assertFalse(
                    config["protocol"]["target_labels_available_to_method"]
                )
                self.assertTrue(
                    config["protocol"]["source_labels_available_to_method"]
                )
                self.assertEqual(method["steps"], 1)
                self.assertEqual(method["task_learning_rate"], 0.0005)
                self.assertEqual(method["synthesis"], "full_frame_alpha")

                if "external" in config["output_dir"]:
                    self.assertIn("locked_online_manifest", config["data"])
                    self.assertEqual(config["data"]["source_domain"], "SDv14")

        static = self.load("configs/experiments/ost/genimage_static.yaml")
        self.assertEqual(static["method_configs"]["ost"]["adaptation_mode"], "static")
        self.assertEqual(static["seed"], 0)
        self.assertEqual(static["protocol"]["name"], "predict_only")
        self.assertEqual(
            static["data"]["source_root"],
            "${GENIMAGE_SD14_TEMPLATE_ARROW_ROOT}",
        )
        self.assertEqual(static["data"]["source_domain"], "SDv14")
        self.assertEqual(static["data"]["source_split"], "train")
        self.assertEqual(static["data"]["source_max_samples_per_class"], 1000)

    def test_ost_training_configs_preserve_the_official_meta_objective(self) -> None:
        expected_sources = {
            "configs/train/ost_ufd_meta.yaml": "progan",
            "configs/train/ost_genimage_meta.yaml": "stable_diffusion_v_1_4",
        }
        for filename, generator in expected_sources.items():
            with self.subTest(filename=filename):
                config = self.load(filename)
                self.assertEqual(config["model"]["architecture"], "meta_xception")
                self.assertEqual(config["data"]["format"], "arrow")
                self.assertEqual(config["data"]["generator"], generator)
                self.assertEqual(config["training"]["epochs"], 30)
                self.assertEqual(config["training"]["task_learning_rate"], 0.0005)
                self.assertEqual(config["training"]["outer_learning_rate"], 0.0002)
                self.assertTrue(config["training"]["second_order"])
                self.assertEqual(config["training"]["am_softmax_margin"], 0.45)

        for filename in (
            "configs/train/genimage_sd14_source.yaml",
            "configs/train/ost_genimage_meta.yaml",
        ):
            with self.subTest(filename=filename):
                config = self.load(filename)
                self.assertEqual(
                    config["data"]["train_root"],
                    "${GENIMAGE_SD14_TRAIN_ARROW_ROOT}",
                )
                self.assertEqual(config["data"]["val_root"], "${GENIMAGE_ARROW_ROOT}")
                self.assertEqual(config["data"]["train_generator"], "SDv14")

        smoke = self.load("configs/train/ost_ufd_meta_smoke.yaml")
        self.assertEqual(smoke["training"]["epochs"], 1)
        self.assertFalse(smoke["training"]["second_order"])
        self.assertEqual(smoke["training"]["max_steps_per_epoch"], 1)

    def test_ost_is_a_registered_vendored_official_core(self) -> None:
        sources = self.load("configs/official_sources.yaml")
        source = sources["ost"]
        core = PROJECT_ROOT / source["official_core"]
        self.assertTrue((core / "meta_xception.py").is_file())
        self.assertTrue((core / "inner_loop_optimizers.py").is_file())
        self.assertTrue((core / "am_softmax.py").is_file())
        self.assertTrue((core / "runtime.py").is_file())
        self.assertTrue((core / "training.py").is_file())
        self.assertTrue((PROJECT_ROOT / source["model_loader"]).is_file())
        self.assertTrue((PROJECT_ROOT / source["wrapper"]).is_file())
        self.assertEqual(source["upstream_license"], "none_declared")
        self.assertEqual(source["checkpoint_size_bytes"], 83476109)
        self.assertEqual(
            source["checkpoint_sha256"],
            "056c311b778a9e777bf3255a5a8f4e509c38190deb3bd14ba82668eadb789f8c",
        )
        self.assertEqual(
            source["numerical_validation"],
            "not_equivalent_to_official_face_benchmark",
        )

    def test_iapl_is_a_registered_framework_wrapper(self) -> None:
        sources = self.load("configs/official_sources.yaml")
        wrapper = PROJECT_ROOT / sources["iapl"]["wrapper"]
        core = PROJECT_ROOT / sources["iapl"]["official_core"]
        method = self.load("configs/methods/iapl.yaml")["method_configs"]["iapl"]
        self.assertTrue(wrapper.is_file())
        self.assertTrue((core / "clip_models.py").is_file())
        self.assertTrue((core / "clip" / "bpe_simple_vocab_16e6.txt.gz").is_file())
        self.assertNotIn("repo_path", method)
        loader_text = (PROJECT_ROOT / sources["iapl"]["model_loader"]).read_text(
            encoding="utf-8"
        )
        self.assertIn("from src.official.iapl import CLIPModel", loader_text)
        self.assertNotIn("verify_iapl_checkout", loader_text)
        self.assertNotIn("patch", sources["iapl"])
        self.assertEqual(
            sources["iapl"]["status"],
            "vendored_upstream_model_core_with_framework_wrapper",
        )

    def test_controlled_experiments_compose_dataset_protocol_and_methods(self) -> None:
        methods = {"source", "tent", "eata", "cotta", "rotta", "lame", "t2a"}
        directory = PROJECT_ROOT / "configs" / "experiments" / "controlled_ctta"
        for setting in ("single_target", "continual"):
            base = load_config(directory / f"{setting}.yaml")
            self.assertEqual(set(base["methods"]), methods)
            self.assertEqual(base["data"]["format"], "arrow")
            self.assertEqual(base["data"]["batch_size"], 16)
            self.assertEqual(base["protocol"]["name"], "predict_then_adapt")
            for seed in (0, 1, 2):
                with self.subTest(setting=setting, seed=seed):
                    config = load_config(directory / f"{setting}_seed{seed}.yaml")
                    self.assertEqual(config["seed"], seed)
                    self.assertEqual(
                        config["output_dir"],
                        f"outputs/controlled_ctta/{setting}/seed{seed}",
                    )

        motivation = self.load(
            "configs/experiments/controlled_ctta/genimage_tta_motivation_seed0.yaml"
        )
        self.assertEqual(motivation["methods"], ["source", "tent", "eata", "t2a"])
        self.assertEqual(motivation["seed"], 0)
        self.assertEqual(motivation["data"]["format"], "arrow")
        self.assertIsNone(motivation["data"]["max_samples_per_class"])
        self.assertNotIn(
            "stable_diffusion_v_1_4", motivation["data"]["targets"]
        )

    def test_genimage_controlled_experiments_use_sd14_as_source(self) -> None:
        methods = {"source", "tent", "eata", "cotta", "rotta", "lame", "t2a"}
        targets = [
            "Midjourney",
            "stable_diffusion_v_1_5",
            "ADM",
            "glide",
            "wukong",
            "VQDM",
            "BigGAN",
        ]
        stream = [
            "BigGAN",
            "ADM",
            "glide",
            "stable_diffusion_v_1_5",
            "VQDM",
            "wukong",
            "Midjourney",
        ]
        directory = PROJECT_ROOT / "configs" / "experiments" / "controlled_ctta"
        for setting in ("single_target", "continual"):
            base = load_config(directory / f"genimage_{setting}.yaml")
            self.assertEqual(set(base["methods"]), methods)
            self.assertEqual(base["data"]["source_domain"], "stable_diffusion_v_1_4")
            self.assertNotIn(base["data"]["source_domain"], base["data"]["targets"])
            self.assertEqual(base["data"]["targets"], targets)
            self.assertEqual(base["data"]["stream"], stream)
            self.assertEqual(base["data"]["batch_size"], 16)
            self.assertEqual(base["protocol"]["name"], "predict_then_adapt")
            for seed in (0, 1, 2):
                config = load_config(
                    directory / f"genimage_{setting}_seed{seed}.yaml"
                )
                self.assertEqual(config["seed"], seed)
                self.assertEqual(
                    config["output_dir"],
                    f"outputs/controlled_ctta/genimage_sd14/{setting}/seed{seed}",
                )

    def test_genimage_source_training_uses_the_unified_arrow_root(self) -> None:
        config = self.load("configs/train/genimage_sd14_source.yaml")
        self.assertEqual(config["data"]["generator"], "stable_diffusion_v_1_4")
        self.assertEqual(config["data"]["train_generator"], "SDv14")
        self.assertEqual(config["data"]["val_generator"], "stable_diffusion_v_1_4")
        self.assertEqual(config["data"]["train_split"], "train")
        self.assertEqual(config["data"]["val_split"], "test")
        self.assertTrue(config["training"]["compute_fisher"])

    def test_external_continual_configs_are_target_only_arrow_streams(self) -> None:
        expected = {
            "aigc_detection_benchmark": {
                "root": "${AIGC_DETECTION_BENCHMARK_ARROW_ROOT}",
                "domains": [
                    "ProGAN",
                    "StyleGAN",
                    "BigGAN",
                    "CycleGAN",
                    "StarGAN",
                    "GauGAN",
                    "StyleGAN2",
                    "WFIR",
                    "ADM",
                    "GLIDE",
                    "Midjourney",
                    "SD v1.4",
                    "SD v1.5",
                    "VQDM",
                    "Wukong",
                    "DALL-E2",
                    "SDXL",
                ],
            },
            "aigi_holmes_p3": {
                "root": "${AIGI_HOLMES_P3_ARROW_ROOT}",
                "domains": [
                    "Janus",
                    "Janus-Pro-1B",
                    "Janus-Pro-7B",
                    "Show-o",
                    "LlamaGen",
                    "Infinity",
                    "VAR",
                    "PixArt-XL",
                    "SD3.5-L",
                    "FLUX",
                ],
            },
            "opensdid_global": {
                "root": "${OPENSDID_GLOBAL_ARROW_ROOT}",
                "domains": ["SD1.5", "SD2.1", "SDXL", "SD3", "Flux.1"],
            },
        }
        methods = {"source", "tent", "eata", "cotta", "rotta", "lame", "t2a"}
        directory = PROJECT_ROOT / "configs" / "experiments" / "controlled_ctta"
        for setting, specification in expected.items():
            with self.subTest(setting=setting):
                base = load_config(directory / f"{setting}_continual.yaml")
                self.assertEqual(set(base["methods"]), methods)
                self.assertEqual(base["data"]["format"], "arrow")
                self.assertEqual(base["data"]["root"], specification["root"])
                self.assertEqual(base["data"]["source_domain"], "genimage_sd14")
                self.assertEqual(base["data"]["targets"], specification["domains"])
                self.assertEqual(base["data"]["stream"], specification["domains"])
                self.assertEqual(base["protocol"]["name"], "predict_then_adapt")
                self.assertFalse(base["protocol"]["target_labels_available_to_method"])
                self.assertFalse(base["protocol"]["generator_id_available_to_method"])
                for field in (
                    "locked_online_manifest",
                    "locked_final_holdout_manifest",
                ):
                    self.assertTrue((directory / base["data"][field]).resolve().is_file())
                for seed in (0, 1, 2):
                    config_path = directory / f"{setting}_continual_seed{seed}.yaml"
                    config = load_config(config_path)
                    self.assertEqual(config["seed"], seed)
                    self.assertEqual(
                        config["output_dir"],
                        f"outputs/controlled_ctta/{setting}/continual/seed{seed}",
                    )
                    for field, suffix in (
                        ("locked_online_manifest", "online"),
                        ("locked_final_holdout_manifest", "final_holdout"),
                    ):
                        manifest_path = (
                            config_path.parent / config["data"][field]
                        ).resolve()
                        self.assertTrue(manifest_path.is_file())
                        self.assertEqual(
                            manifest_path.name, f"seed{seed}_{suffix}_manifest.csv"
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
