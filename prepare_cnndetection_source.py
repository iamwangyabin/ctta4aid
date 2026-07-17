from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from online_aig_tta.cli.common import resolve_device, seed_everything
from online_aig_tta.data import build_dataset, build_eval_transform
from online_aig_tta.evaluation.metrics import MetricAccumulator
from online_aig_tta.models import build_detector, save_checkpoint
from train_source import build_loader, estimate_bn_fishers


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(32 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_state_dict(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    import torch

    source = {key.removeprefix("module."): value for key, value in source.items()}
    converted = {
        f"backbone.{key}": value
        for key, value in source.items()
        if key not in {"fc.weight", "fc.bias"}
    }
    if tuple(source["fc.weight"].shape) != (1, target["classifier.weight"].shape[1]):
        raise ValueError(f"Unexpected CNNDetection fc.weight shape: {source['fc.weight'].shape}")
    classifier_weight = torch.zeros_like(target["classifier.weight"])
    classifier_bias = torch.zeros_like(target["classifier.bias"])
    classifier_weight[1].copy_(source["fc.weight"][0])
    classifier_bias[1].copy_(source["fc.bias"][0])
    converted["classifier.weight"] = classifier_weight
    converted["classifier.bias"] = classifier_bias

    missing = sorted(set(target) - set(converted))
    unexpected = sorted(set(converted) - set(target))
    mismatched = sorted(
        key for key in target.keys() & converted.keys() if target[key].shape != converted[key].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Incompatible CNNDetection checkpoint: missing={missing}, "
            f"unexpected={unexpected}, mismatched={mismatched}"
        )
    return converted


def evaluate(model: Any, loader: Any, device: Any) -> dict[str, Any]:
    import torch

    model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for images, labels, _ in loader:
            probabilities = model(images.to(device, non_blocking=True)).softmax(dim=1)[:, 1]
            accumulator.update(labels, probabilities, "ProGAN")
    return accumulator.summary()["overall"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert official CNNDetection weights into the common detector checkpoint"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--caidbench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fisher-samples", type=int, default=2000)
    parser.add_argument("--fisher-batch-size", type=int, default=64)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    model = build_detector("resnet50", pretrained=False, num_classes=2)

    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source_state = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(convert_state_dict(source_state, model.state_dict()), strict=True)
    model.to(device)

    dataset = build_dataset(
        data_format="caidbench_arrow",
        root=args.caidbench_root,
        generator="ProGAN",
        split="test",
        transform=build_eval_transform(224, resize_before_crop=False),
        seed=args.seed,
    )
    data_config = {
        "batch_size": args.fisher_batch_size,
        "num_workers": args.workers,
    }
    eval_loader = build_loader(dataset, data_config, shuffle=False)
    validation_metrics = evaluate(model, eval_loader, device)
    fisher_loader = build_loader(
        dataset,
        data_config,
        shuffle=True,
        batch_size=args.fisher_batch_size,
        drop_last=False,
        seed=args.seed,
    )
    fishers = estimate_bn_fishers(model, fisher_loader, device, args.fisher_samples)
    save_checkpoint(
        model,
        args.output,
        architecture="resnet50",
        source_generator="ProGAN",
        validation_metrics=validation_metrics,
        fishers=fishers,
        seed=args.seed,
        imported_from="CNNDetection blur_jpg_prob0.5.pth",
        imported_checkpoint_sha256=sha256(args.checkpoint),
        classifier_conversion="sigmoid(z_fake) == softmax([0, z_fake])[1]",
        evaluation_transform="center_crop_224_imagenet_normalization",
    )
    print(
        f"saved={args.output} auc={validation_metrics['auc']:.5f} "
        f"accuracy={validation_metrics['accuracy']:.5f} fishers={len(fishers)}"
    )


if __name__ == "__main__":
    main()
