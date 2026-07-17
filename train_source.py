from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from online_aig_tta.cli.common import resolve_device, seed_everything, write_json
from online_aig_tta.config import load_config, require
from online_aig_tta.data import build_dataset, build_eval_transform, build_train_transform
from online_aig_tta.evaluation.metrics import MetricAccumulator
from online_aig_tta.models import build_detector, save_checkpoint


def build_loader(
    dataset: Any,
    data_config: dict[str, Any],
    *,
    shuffle: bool,
    batch_size: int | None = None,
    drop_last: bool | None = None,
    seed: int | None = None,
) -> Any:
    import torch
    from torch.utils.data import DataLoader

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size or int(data_config.get("batch_size", 32)),
        shuffle=shuffle,
        num_workers=int(data_config.get("num_workers", 4)),
        pin_memory=True,
        drop_last=shuffle if drop_last is None else drop_last,
        generator=generator,
    )


def evaluate(model: Any, loader: Any, device: Any) -> dict[str, Any]:
    import torch

    model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad():
        for images, labels, _ in loader:
            probabilities = model(images.to(device, non_blocking=True)).softmax(dim=1)[:, 1]
            accumulator.update(labels, probabilities, "validation")
    return accumulator.summary()["overall"]


def estimate_bn_fishers(model: Any, loader: Any, device: Any, max_samples: int) -> dict[str, Any]:
    """Reproduce the official EATA pseudo-label Fisher preparation."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    fisher_model = deepcopy(model).to(device)
    fisher_model.train()
    fisher_model.requires_grad_(False)
    for module in fisher_model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    selected = {
        name: parameter
        for name, parameter in fisher_model.named_parameters()
        if parameter.requires_grad
    }
    fisher_sums = {
        name: torch.zeros_like(parameter, device="cpu") for name, parameter in selected.items()
    }
    batches = 0
    samples = 0
    for images, _, _ in loader:
        if samples >= max_samples:
            break
        remaining = max_samples - samples
        if images.shape[0] > remaining:
            images = images[:remaining]
        images = images.to(device, non_blocking=True)
        logits = fisher_model(images)
        pseudo_labels = logits.detach().argmax(dim=1)
        loss = functional.cross_entropy(logits, pseudo_labels)
        fisher_model.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in selected.items():
            if parameter.grad is not None:
                fisher_sums[name] += parameter.grad.detach().cpu().square()
        batches += 1
        samples += int(images.shape[0])
    if not batches:
        return {}
    return {
        name: [
            fisher_sums[name] / batches,
            parameter.detach().cpu().clone(),
        ]
        for name, parameter in selected.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the common binary source detector")
    parser.add_argument("--config", default="configs/train_source.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    require(config, "model", "data", "training", "output_dir")
    data_config = config["data"]
    require(data_config, "format", "generator", "train_split", "val_split")
    if "root" not in data_config and not {
        "train_root",
        "val_root",
    }.issubset(data_config):
        raise ValueError("data requires root or both train_root and val_root")
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))

    image_size = int(data_config.get("image_size", 224))
    train_dataset = build_dataset(
        data_format=data_config["format"],
        root=data_config.get("train_root", data_config.get("root")),
        generator=data_config["generator"],
        split=data_config["train_split"],
        transform=build_train_transform(image_size),
        max_samples_per_class=data_config.get("max_train_samples_per_class"),
        seed=seed,
    )
    val_dataset = build_dataset(
        data_format=data_config["format"],
        root=data_config.get("val_root", data_config.get("root")),
        generator=data_config["generator"],
        split=data_config["val_split"],
        transform=build_eval_transform(
            image_size,
            resize_before_crop=bool(data_config.get("resize_before_crop", True)),
        ),
        max_samples_per_class=data_config.get("max_val_samples_per_class"),
        seed=seed,
    )
    train_loader = build_loader(train_dataset, data_config, shuffle=True, seed=seed)
    val_loader = build_loader(val_dataset, data_config, shuffle=False)

    model_config = config["model"]
    model = build_detector(
        model_config.get("architecture", "resnet50"),
        pretrained=bool(model_config.get("pretrained", True)),
        pretrained_weights=str(model_config.get("pretrained_weights", "default")),
        num_classes=int(model_config.get("num_classes", 2)),
    ).to(device)

    import torch
    import torch.nn.functional as functional

    training = config["training"]
    optimizer_name = str(training.get("optimizer", "adamw")).lower()
    optimizer_class = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optimizer_class(
        model.parameters(),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )

    best_auc = float("-inf")
    best_state = None
    best_metrics: dict[str, Any] = {}
    epochs = int(training.get("epochs", 10))
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss = functional.cross_entropy(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        metrics = evaluate(model, val_loader, device)
        print(
            f"epoch={epoch:03d} loss={running_loss / max(len(train_loader), 1):.5f} "
            f"val_auc={metrics['auc']:.5f} val_acc={metrics['accuracy']:.5f}"
        )
        if float(metrics["auc"]) > best_auc:
            best_auc = float(metrics["auc"])
            best_state = deepcopy(model.state_dict())
            best_metrics = dict(metrics)

    if best_state is None:
        raise RuntimeError("Training finished without a valid checkpoint")
    model.load_state_dict(best_state)
    fishers = None
    if bool(training.get("compute_fisher", True)):
        # Official EATA uses clean source validation images with evaluation
        # preprocessing, pseudo labels and a shuffled loader.
        fisher_loader = build_loader(
            val_dataset,
            data_config,
            shuffle=True,
            batch_size=int(training.get("fisher_batch_size", 64)),
            drop_last=False,
            seed=seed,
        )
        fishers = estimate_bn_fishers(
            model, fisher_loader, device, int(training.get("fisher_samples", 2000))
        )

    output_root = Path(config["output_dir"])
    write_json(output_root / "effective_config.json", config)
    output_path = output_root / "source.pt"
    save_checkpoint(
        model,
        output_path,
        architecture=model_config.get("architecture", "resnet50"),
        pretrained_weights=model_config.get("pretrained_weights", "default"),
        source_generator=data_config["generator"],
        validation_metrics=best_metrics,
        fishers=fishers,
        seed=seed,
    )
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
