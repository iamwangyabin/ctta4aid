from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _sample(value: str) -> tuple[str, int]:
    try:
        domain, index = value.split(":", 1)
        return domain, int(index)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sample must be DOMAIN:INDEX") from exc


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def _compare_item(left: Any, right: Any, *, index: int, seed: int) -> dict[str, Any]:
    _seed(seed)
    left_views, left_label = left[index]
    _seed(seed)
    right_views, right_label = right[index]
    if int(left_label) != int(right_label):
        raise ValueError(f"Label mismatch at index {index}: {left_label} != {right_label}")
    if len(left_views) != len(right_views):
        raise ValueError(f"View-count mismatch at index {index}")

    absolute = [
        (left_view.to(torch.float32) - right_view.to(torch.float32)).abs()
        for left_view, right_view in zip(left_views, right_views)
    ]
    total_elements = sum(delta.numel() for delta in absolute)
    return {
        "index": index,
        "label": int(left_label),
        "seed": seed,
        "views": len(absolute),
        "shapes_equal": all(
            left_view.shape == right_view.shape
            for left_view, right_view in zip(left_views, right_views)
        ),
        "all_values_equal": all(torch.count_nonzero(delta).item() == 0 for delta in absolute),
        "different_elements": sum(
            torch.count_nonzero(delta).item() for delta in absolute
        ),
        "total_elements": total_elements,
        "mean_absolute_delta": (
            sum(delta.sum().item() for delta in absolute) / total_elements
            if total_elements
            else 0.0
        ),
        "max_absolute_delta": max(
            (delta.max().item() for delta in absolute), default=0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare IAPL's decoded and augmented views across two data backends."
    )
    parser.add_argument("--iapl-repo", required=True, type=Path)
    parser.add_argument("--project-src", required=True, type=Path)
    parser.add_argument("--left-dataset-path", required=True)
    parser.add_argument("--right-dataset-path", required=True)
    parser.add_argument("--sample", action="append", required=True, type=_sample)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    iapl_repo = args.iapl_repo.expanduser().resolve()
    sys.path.insert(0, str(args.project_src.expanduser().resolve()))
    sys.path.insert(0, str(iapl_repo))
    os.chdir(iapl_repo)

    from utils.dataset import Dataset_Creator

    domains = list(dict.fromkeys(domain for domain, _ in args.sample))

    def build(dataset_path: str) -> dict[str, Any]:
        creator = Dataset_Creator(
            dataset_path=dataset_path,
            batch_size=32,
            num_workers=0,
            img_resolution=256,
            crop_resolution=224,
        )
        datasets, names = creator.build_dataset("tta", selected_subsets=domains)
        return dict(zip(names, datasets))

    left = build(args.left_dataset_path)
    right = build(args.right_dataset_path)
    comparisons = []
    for offset, (domain, index) in enumerate(args.sample):
        if not 0 <= index < len(left[domain]) or not 0 <= index < len(right[domain]):
            raise IndexError(f"Index {index} is outside domain {domain}")
        comparison = _compare_item(
            left[domain], right[domain], index=index, seed=args.seed + offset
        )
        comparison["domain"] = domain
        comparisons.append(comparison)

    report = {
        "left_dataset_path": args.left_dataset_path,
        "right_dataset_path": args.right_dataset_path,
        "comparisons": comparisons,
        "all_views_equal": all(item["all_values_equal"] for item in comparisons),
    }
    output = json.dumps(report, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
