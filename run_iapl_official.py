from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from online_aig_tta.config import load_config, require  # noqa: E402
from online_aig_tta.data.hf_arrow import arrow_uri_roots  # noqa: E402

IAPL_COMMIT = "a173e7783bbafaa00d60e6e31774a0bc14411a23"
PATH_KEYS = (
    "repo_path",
    "dataset_path",
    "pretrained_model",
    "clip_path",
    "clip_checkpoint",
    "output_dir",
)
IAPL_MODEL_PATH = "models/clip_models.py"
IAPL_HARDCODED_CLIP = "load_clip_to_cpu('/Path/to/ViT-L-14.pt'"
IAPL_CONFIGURED_CLIP = "load_clip_to_cpu(args.clip_path"
IAPL_DATASET_PATH = "utils/dataset.py"
IAPL_DATASET_IMPORT_ANCHOR = "from torch.utils.data import ConcatDataset\n"
IAPL_DATASET_IMPORT = (
    "from online_aig_tta.data.hf_arrow import build_iapl_arrow_dataset\n"
)
IAPL_DATASET_LOOP_ANCHOR = (
    "        for subset in selected_subsets:\n"
    "            if spilt_dataset == 'tta':\n"
)
IAPL_DATASET_LOOP_REPLACEMENT = (
    "        for subset in selected_subsets:\n"
    "            arrow_dataset = build_iapl_arrow_dataset(\n"
    "                self.dataset_path,\n"
    "                split=spilt_dataset,\n"
    "                subset=subset,\n"
    "                transform=self.transforms[spilt_dataset],\n"
    "            )\n"
    "            if arrow_dataset is not None:\n"
    "                sub_datasets.append(arrow_dataset)\n"
    "                continue\n"
    "            if spilt_dataset == 'tta':\n"
)
IAPL_MAIN_PATH = "main.py"
IAPL_TEST_TIME_PATH = "test_time.py"
IAPL_TORCH_LOAD = "torch.load(args.pretrained_model, map_location='cpu')"
IAPL_TORCH_LOAD_COMPAT = (
    "torch.load(args.pretrained_model, map_location='cpu', weights_only=False)"
)
IAPL_PRETRAINED_CONTEXT_LOAD = (
    "pretrained_ctx = torch.load(args.pretrained_model, "
    "map_location='cpu')['model']['prompt_learner.ctx']"
)
IAPL_PRETRAINED_CONTEXT_COMPAT = "pretrained_ctx = checkpoint['model']['prompt_learner.ctx']"


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def runtime_metadata() -> dict[str, Any]:
    import torch

    runtime = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": None,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import torchvision

        runtime["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        runtime["gpu"] = torch.cuda.get_device_name(0)
    return runtime


def arrow_dataset_identity(root: Path) -> dict[str, Any]:
    state_path = root / "state.json"
    with state_path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    shard_paths = [root / item["filename"] for item in state.get("_data_files", [])]
    missing = [str(path) for path in shard_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Arrow state references missing shards: {missing[:3]}")
    metadata_names = (
        "state.json",
        "dataset_info.json",
        "mapping.json",
        "test.json",
        "val.json",
        "train_binary.json",
    )
    return {
        "root": str(root),
        "fingerprint": state.get("_fingerprint"),
        "shard_count": len(shard_paths),
        "shard_bytes": sum(path.stat().st_size for path in shard_paths),
        "metadata": [
            file_identity(path)
            for name in metadata_names
            if (path := root / name).is_file()
        ],
    }


def flag(command: list[str], name: str, value: Any) -> None:
    command.extend([f"--{name}", str(value)])


def resolve_runtime_paths(config: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    resolved = dict(config)
    for key in PATH_KEYS:
        if key not in resolved:
            continue
        if key == "dataset_path" and arrow_uri_roots(str(resolved[key])) is not None:
            continue
        path = Path(resolved[key]).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        resolved[key] = str(path.resolve())
    return resolved


def build_command(config: dict[str, Any], python_executable: str = sys.executable) -> list[str]:
    clip_path = config.get("clip_path", config.get("clip_checkpoint"))
    require(
        config,
        "repo_path",
        "dataset",
        "dataset_path",
        "train_selected_subsets",
        "test_selected_subsets",
        "pretrained_model",
        "output_dir",
    )
    if clip_path is None:
        raise ValueError("IAPL config requires official field clip_path")
    command = [
        python_executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(config.get("nproc_per_node", 8)),
        "--master_port",
        str(config.get("master_port", 29578)),
        "main.py",
    ]
    flag(command, "batchsize", config.get("batchsize", config.get("batch_size", 32)))
    flag(
        command,
        "evalbatchsize",
        config.get("evalbatchsize", config.get("views", 32)),
    )
    flag(command, "dataset_path", config["dataset_path"])
    command.append("--train_selected_subsets")
    command.extend(map(str, config["train_selected_subsets"]))
    command.append("--test_selected_subsets")
    command.extend(map(str, config["test_selected_subsets"]))
    flag(command, "lr", config.get("lr", config.get("learning_rate", 0.005)))
    flag(command, "model_name", config.get("model_name", "tta"))
    flag(command, "dataset", config["dataset"])
    flag(command, "epoch", config.get("epoch", 1))
    flag(command, "lr_drop", config.get("lr_drop", 10))
    flag(command, "gate", config.get("gate", True))
    flag(command, "condition", config.get("condition", True))
    flag(command, "pretrained_model", config["pretrained_model"])
    flag(command, "clip_path", clip_path)
    flag(command, "tta", config.get("tta", True))
    flag(command, "tta_steps", config.get("tta_steps", 2))
    flag(
        command,
        "selection_p",
        config.get("selection_p", config.get("selection_fraction", 0.2)),
    )
    flag(command, "ois", config.get("ois", True))
    flag(command, "num_workers", config.get("num_workers", 8))
    flag(command, "seed", config.get("seed", 100))
    flag(command, "output_dir", config["output_dir"])
    if bool(config.get("smooth", False)):
        flag(command, "smooth", True)
    if bool(config.get("eval", True)):
        command.append("--eval")
    return command


def verify_checkout(repo_path: Path) -> None:
    if not (repo_path / ".git").is_dir():
        raise FileNotFoundError(
            f"IAPL checkout not found at {repo_path}. Run: python fetch_official_baselines.py iapl"
        )
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        text=True,
        capture_output=True,
    )
    if result.stdout.strip() != IAPL_COMMIT:
        raise RuntimeError(f"IAPL must be pinned to commit {IAPL_COMMIT}")
    committed_source = subprocess.run(
        ["git", "show", f"HEAD:{IAPL_MODEL_PATH}"],
        cwd=repo_path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    expected_source = committed_source.replace(IAPL_HARDCODED_CLIP, IAPL_CONFIGURED_CLIP)
    model_source = (repo_path / IAPL_MODEL_PATH).read_text(encoding="utf-8")
    committed_dataset = subprocess.run(
        ["git", "show", f"HEAD:{IAPL_DATASET_PATH}"],
        cwd=repo_path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if IAPL_DATASET_IMPORT_ANCHOR not in committed_dataset:
        raise RuntimeError("Pinned IAPL dataset source no longer contains the import anchor")
    if IAPL_DATASET_LOOP_ANCHOR not in committed_dataset:
        raise RuntimeError("Pinned IAPL dataset source no longer contains the loop anchor")
    expected_dataset = committed_dataset.replace(
        IAPL_DATASET_IMPORT_ANCHOR,
        IAPL_DATASET_IMPORT_ANCHOR + IAPL_DATASET_IMPORT,
        1,
    ).replace(IAPL_DATASET_LOOP_ANCHOR, IAPL_DATASET_LOOP_REPLACEMENT, 1)
    dataset_source = (repo_path / IAPL_DATASET_PATH).read_text(encoding="utf-8")
    committed_main = subprocess.run(
        ["git", "show", f"HEAD:{IAPL_MAIN_PATH}"],
        cwd=repo_path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    expected_main = committed_main.replace(IAPL_TORCH_LOAD, IAPL_TORCH_LOAD_COMPAT)
    main_source = (repo_path / IAPL_MAIN_PATH).read_text(encoding="utf-8")
    committed_test_time = subprocess.run(
        ["git", "show", f"HEAD:{IAPL_TEST_TIME_PATH}"],
        cwd=repo_path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    expected_test_time = committed_test_time.replace(
        IAPL_TORCH_LOAD, IAPL_TORCH_LOAD_COMPAT, 1
    ).replace(IAPL_PRETRAINED_CONTEXT_LOAD, IAPL_PRETRAINED_CONTEXT_COMPAT, 1)
    test_time_source = (repo_path / IAPL_TEST_TIME_PATH).read_text(encoding="utf-8")
    changed_files = subprocess.run(
        ["git", "diff", "HEAD", "--name-only"],
        cwd=repo_path,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    if committed_source == expected_source:
        raise RuntimeError("Pinned IAPL source no longer contains the expected CLIP path")
    if committed_main == expected_main or committed_test_time == expected_test_time:
        raise RuntimeError("Pinned IAPL source no longer contains the expected torch.load calls")
    changed_python_files = sorted(path for path in changed_files if path.endswith(".py"))
    expected_changed_files = sorted(
        [IAPL_MODEL_PATH, IAPL_DATASET_PATH, IAPL_MAIN_PATH, IAPL_TEST_TIME_PATH]
    )
    if (
        model_source != expected_source
        or dataset_source != expected_dataset
        or main_source != expected_main
        or test_time_source != expected_test_time
        or changed_python_files != expected_changed_files
    ):
        raise RuntimeError(
            "IAPL checkout differs from the pinned commit by more than the approved "
            "CLIP-path, torch.load and Arrow-loader compatibility patch"
        )


def parse_official_summary(text: str) -> dict[str, Any]:
    pattern = re.compile(
        r"^\((?P<index>\d+)\s+(?P<name>.+?)\)\s+acc:\s*(?P<acc>[\d.]+);\s*"
        r"ap:\s*(?P<ap>[\d.]+);\s*racc:\s*(?P<racc>[\d.]+);\s*"
        r"facc:\s*(?P<facc>[\d.]+);",
        flags=re.MULTILINE,
    )
    rows = {}
    for match in pattern.finditer(text):
        name = match.group("name").strip()
        rows[name] = {
            key: float(match.group(key)) / 100.0 for key in ("acc", "ap", "racc", "facc")
        }
    return {"by_domain": {k: v for k, v in rows.items() if k != "mean"}, "mean": rows.get("mean")}


def validate_official_metrics(
    parsed: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    expected_domains = set(map(str, config["test_selected_subsets"]))
    actual_domains = set(parsed["by_domain"])
    missing_domains = sorted(expected_domains - actual_domains)
    if missing_domains or parsed.get("mean") is None:
        raise RuntimeError(
            "Official IAPL output is incomplete; missing domains="
            f"{missing_domains}, mean_present={parsed.get('mean') is not None}"
        )

    reference = config.get("reference_metrics")
    if not reference:
        return {"enabled": False, "passed": None}
    tolerance = float(config.get("reference_tolerance", 0.01))
    differences = {
        metric: abs(float(parsed["mean"][metric]) - float(reference[metric]))
        for metric in ("acc", "ap")
    }
    check = {
        "enabled": True,
        "passed": all(difference <= tolerance for difference in differences.values()),
        "tolerance": tolerance,
        "reference": reference,
        "absolute_differences": differences,
    }
    return check


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pinned official IAPL evaluation")
    parser.add_argument("--config", default="configs/iapl_official_genimage.yaml")
    parser.add_argument(
        "--domains",
        nargs="+",
        help="Run independent domain shards; disables the full-benchmark reference gate",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = resolve_runtime_paths(
        load_config(args.config), PROJECT_ROOT
    )
    if args.domains:
        configured_domains = set(map(str, config["test_selected_subsets"]))
        unknown_domains = sorted(set(args.domains) - configured_domains)
        if unknown_domains:
            raise ValueError(f"Unknown IAPL domains for this config: {unknown_domains}")
        shard_name = "-".join(args.domains)
        config["test_selected_subsets"] = list(args.domains)
        config["output_dir"] = str(Path(config["output_dir"]) / "shards" / shard_name)
        config["reference_metrics"] = None
        config["require_reference_match"] = False
    command = build_command(config)
    repo_path = Path(config["repo_path"])
    if args.dry_run:
        print(" ".join(command))
        return

    verify_checkout(repo_path)
    required_paths = ["pretrained_model"]
    required_paths.append("clip_path" if "clip_path" in config else "clip_checkpoint")
    for key in required_paths:
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"Configured {key} does not exist: {config[key]}")

    arrow_roots = arrow_uri_roots(str(config["dataset_path"]))
    if arrow_roots is None:
        if not Path(config["dataset_path"]).exists():
            raise FileNotFoundError(
                f"Configured dataset_path does not exist: {config['dataset_path']}"
            )
    else:
        for root in arrow_roots:
            if not root.is_dir():
                raise FileNotFoundError(f"Configured Arrow root does not exist: {root}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_key = "clip_path" if "clip_path" in config else "clip_checkpoint"
    dataset_manifests = sorted(
        Path(config["dataset_path"]).glob("caidbench_*_manifest.csv")
    )
    asset_identities = {
        "pretrained_model": file_identity(config["pretrained_model"]),
        "clip_checkpoint": file_identity(config[clip_key]),
        "dataset_manifests": [file_identity(path) for path in dataset_manifests],
        "arrow_datasets": (
            [arrow_dataset_identity(root) for root in arrow_roots]
            if arrow_roots is not None
            else []
        ),
    }
    manifest = {
        "method": "IAPL",
        "implementation": "official",
        "official_commit": IAPL_COMMIT,
        "compatibility_patch": "patches/iapl-a173e77-compat.patch",
        "protocol": "per_image_reset_adapt_then_predict",
        "command": command,
        "config": config,
        "assets": asset_identities,
        "runtime": runtime_metadata(),
    }
    (output_dir / "official_iapl_run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log_lines = []
    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{SRC_ROOT}{os.pathsep}{existing_python_path}"
        if existing_python_path
        else str(SRC_ROOT)
    )
    process = subprocess.Popen(
        command,
        cwd=repo_path,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        log_lines.append(line)
    return_code = process.wait()
    log_text = "".join(log_lines)
    (output_dir / "official_iapl.log").write_text(log_text, encoding="utf-8")
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    parsed = parse_official_summary(log_text)
    reference_check = validate_official_metrics(parsed, config)
    metrics = {
        "method": "IAPL",
        "implementation": "official",
        "official_commit": IAPL_COMMIT,
        "protocol": "per_image_reset_adapt_then_predict",
        "reported_metrics": ["accuracy", "average_precision", "real_accuracy", "fake_accuracy"],
        "dataset": config["dataset"],
        "numerical_validation": (
            "reference_gate_passed"
            if reference_check.get("passed")
            else (
                "reference_gate_failed"
                if reference_check.get("enabled")
                else "completed_without_reference_gate"
            )
        ),
        "reference_check": reference_check,
        "assets": asset_identities,
        **parsed,
    }
    (output_dir / "official_iapl_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if (
        bool(config.get("require_reference_match", True))
        and reference_check.get("enabled")
        and not reference_check.get("passed")
    ):
        raise RuntimeError(
            "IAPL completed, but the reported mean differs from the authors' reference "
            f"beyond tolerance: {reference_check}"
        )


if __name__ == "__main__":
    main()
