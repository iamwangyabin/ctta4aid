from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _digest(value: Any) -> str:
    return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()


def rng_snapshot() -> dict[str, str]:
    return {
        "python_random": _digest(random.getstate()),
        "numpy_random": _digest(np.random.get_state()),
        "torch_cpu": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether IAPL dataset construction changes global RNG state."
    )
    parser.add_argument("--iapl-repo", required=True, type=Path)
    parser.add_argument("--project-src", required=True, type=Path)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["crn", "guided", "imle", "san", "seeingdark"],
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    iapl_repo = args.iapl_repo.expanduser().resolve()
    project_src = args.project_src.expanduser().resolve()
    sys.path.insert(0, str(project_src))
    sys.path.insert(0, str(iapl_repo))
    os.chdir(iapl_repo)

    from utils.dataset import Dataset_Creator

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    snapshots = {"seeded": rng_snapshot()}

    creator = Dataset_Creator(
        dataset_path=args.dataset_path,
        batch_size=32,
        num_workers=0,
        img_resolution=256,
        crop_resolution=224,
    )
    snapshots["after_creator"] = rng_snapshot()
    datasets, names = creator.build_dataset("tta", selected_subsets=args.domains)
    snapshots["after_build"] = rng_snapshot()

    report = {
        "dataset_path": args.dataset_path,
        "domains": list(names),
        "dataset_lengths": {
            name: len(dataset) for name, dataset in zip(names, datasets)
        },
        "seed": args.seed,
        "snapshots": snapshots,
        "changed_after_creator": {
            key: snapshots["seeded"][key] != snapshots["after_creator"][key]
            for key in snapshots["seeded"]
        },
        "changed_after_build": {
            key: snapshots["after_creator"][key] != snapshots["after_build"][key]
            for key in snapshots["seeded"]
        },
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
