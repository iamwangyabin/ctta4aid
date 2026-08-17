from __future__ import annotations

import argparse
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from src.cli.common import resolve_device, seed_everything, write_json
from src.config import load_config, require
from src.data import (
    build_clip_eval_transform,
    build_clip_train_transform,
    build_dataset,
    build_eval_transform,
    build_train_transform,
)
from src.evaluation.metrics import MetricAccumulator
from src.models import (
    build_clip_source_detector,
    build_detector,
    build_ost_training_detector,
    save_checkpoint,
)


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


def data_generator(data_config: dict[str, Any], role: str) -> str:
    generator = data_config.get(f"{role}_generator", data_config.get("generator"))
    if not generator:
        raise ValueError(f"data requires generator or {role}_generator")
    return str(generator)


def excluded_image_paths(data_config: dict[str, Any], role: str) -> list[str] | None:
    configured = data_config.get(
        f"{role}_exclude_image_paths", data_config.get("exclude_image_paths")
    )
    if configured is None:
        return None
    if not isinstance(configured, list) or not all(
        isinstance(path, str) and path.strip() for path in configured
    ):
        raise ValueError(f"data.{role}_exclude_image_paths must be a list of paths")
    return list(configured)


def _autocast(device: Any, enabled: bool) -> Any:
    if enabled and str(device).startswith("cuda"):
        import torch

        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def evaluate(
    model: Any, loader: Any, device: Any, *, amp_enabled: bool = False
) -> dict[str, Any]:
    import torch

    model.eval()
    accumulator = MetricAccumulator()
    with torch.no_grad(), _autocast(device, amp_enabled):
        for images, labels, _ in loader:
            probabilities = model(images.to(device, non_blocking=True)).softmax(dim=1)[:, 1]
            accumulator.update(labels, probabilities, "validation")
    return accumulator.summary()["overall"]


def evaluate_ost(
    model: Any,
    loader: Any,
    template_dataset: Any,
    trainer: Any,
    device: Any,
    *,
    alpha: float,
    max_batches: int | None,
) -> dict[str, Any]:
    import torch

    model.eval()
    accumulator = MetricAccumulator()
    template_index = 0
    for batch_index, (query, query_labels, _) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        template, template_labels = _ost_template_batch(
            template_dataset, template_index, int(query.shape[0])
        )
        template_index = (template_index + int(query.shape[0])) % len(template_dataset)
        query = query.to(device, non_blocking=True)
        template = template.to(device, non_blocking=True)
        template_labels = template_labels.to(device, non_blocking=True)
        pseudo = alpha * query + (1.0 - alpha) * template
        support = torch.cat((pseudo, template), dim=0)
        support_labels = torch.cat(
            (
                torch.ones(query.shape[0], dtype=torch.long, device=device),
                template_labels,
            ),
            dim=0,
        )
        logits, _ = trainer.adapt_and_predict(support, support_labels, query)
        accumulator.update(query_labels, logits.softmax(dim=1)[:, 1], "validation")
    return accumulator.summary()["overall"]


def _ost_template_batch(dataset: Any, offset: int, batch_size: int) -> tuple[Any, Any]:
    import torch

    samples = [dataset[(offset + index) % len(dataset)] for index in range(batch_size)]
    return (
        torch.stack([sample[0] for sample in samples]),
        torch.tensor([int(sample[1]) for sample in samples], dtype=torch.long),
    )


def _ost_episode(
    query: Any,
    query_labels: Any,
    template: Any,
    template_labels: Any,
    *,
    alpha_min: float,
    alpha_max: float,
) -> tuple[Any, Any, Any, Any]:
    import torch

    alpha_shape = (int(query.shape[0]),) + (1,) * (query.ndim - 1)
    first_alphas = torch.empty(alpha_shape, device=query.device).uniform_(
        alpha_min, alpha_max
    )
    second_alphas = torch.empty(alpha_shape, device=query.device).uniform_(
        alpha_min, alpha_max
    )
    first_pseudo = first_alphas * query + (1.0 - first_alphas) * template
    second_pseudo = second_alphas * template + (1.0 - second_alphas) * query
    pseudo_labels = torch.ones(
        query.shape[0], dtype=torch.long, device=query.device
    )
    if bool(torch.rand((), device=query.device) > 0.5):
        support = torch.cat((template, first_pseudo), dim=0)
        support_labels = torch.cat((template_labels, pseudo_labels), dim=0)
        meta_query = torch.cat((query, second_pseudo), dim=0)
    else:
        support = torch.cat((template, second_pseudo), dim=0)
        support_labels = torch.cat((template_labels, pseudo_labels), dim=0)
        meta_query = torch.cat((query, first_pseudo), dim=0)
    meta_query_labels = torch.cat((query_labels, pseudo_labels), dim=0)
    return support, support_labels, meta_query, meta_query_labels


def train_ost(config: dict[str, Any], device: Any, seed: int) -> None:
    from src.data.ost import build_ost_transform
    from src.official.ost import OSTMetaTrainingCore

    data_config = config["data"]
    training = config["training"]
    model_config = config["model"]
    image_size = int(data_config.get("image_size", 256))
    transform = build_ost_transform(image_size)
    train_generator = data_generator(data_config, "train")
    val_generator = data_generator(data_config, "val")
    train_dataset = build_dataset(
        data_format=data_config["format"],
        root=data_config.get("train_root", data_config.get("root")),
        generator=train_generator,
        split=data_config["train_split"],
        transform=transform,
        max_samples_per_class=data_config.get("max_train_samples_per_class"),
        seed=seed,
    )
    val_dataset = build_dataset(
        data_format=data_config["format"],
        root=data_config.get("val_root", data_config.get("root")),
        generator=val_generator,
        split=data_config["val_split"],
        transform=transform,
        max_samples_per_class=data_config.get("max_val_samples_per_class"),
        seed=seed,
    )
    batch_size = int(data_config.get("batch_size", 4))
    train_loader = build_loader(
        train_dataset,
        data_config,
        shuffle=True,
        batch_size=batch_size,
        drop_last=True,
        seed=seed,
    )
    val_loader = build_loader(
        val_dataset,
        data_config,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
    )
    model, initialization_metadata = build_ost_training_detector(
        model_config, device=device
    )
    trainer = OSTMetaTrainingCore(
        model,
        device,
        task_learning_rate=float(training.get("task_learning_rate", 0.0005)),
        outer_learning_rate=float(training.get("outer_learning_rate", 0.0002)),
        second_order=bool(training.get("second_order", True)),
        enable_inner_loop_optimizable_bn_params=bool(
            model_config.get("enable_inner_loop_optimizable_bn_params", True)
        ),
        margin=float(training.get("am_softmax_margin", 0.45)),
        scale=float(training.get("am_softmax_scale", 30.0)),
    )

    alpha_min = float(training.get("alpha_min", 0.65))
    alpha_max = float(training.get("alpha_max", 0.90))
    if not 0.0 < alpha_min <= alpha_max < 1.0:
        raise ValueError("OST training alpha range must satisfy 0 < min <= max < 1")
    best_auc = float("-inf")
    best_state = None
    best_metrics: dict[str, Any] = {}
    epochs = int(training.get("epochs", 30))
    max_steps = training.get("max_steps_per_epoch")
    max_steps = int(max_steps) if max_steps is not None else None
    template_index = 0
    for epoch in range(1, epochs + 1):
        running_query_loss = 0.0
        steps = 0
        for query, query_labels, _ in train_loader:
            if max_steps is not None and steps >= max_steps:
                break
            template, template_labels = _ost_template_batch(
                train_dataset, template_index, int(query.shape[0])
            )
            template_index = (template_index + int(query.shape[0])) % len(train_dataset)
            query = query.to(device, non_blocking=True)
            query_labels = query_labels.to(device, non_blocking=True)
            template = template.to(device, non_blocking=True)
            template_labels = template_labels.to(device, non_blocking=True)
            episode = _ost_episode(
                query,
                query_labels,
                template,
                template_labels,
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )
            result = trainer.train_step(*episode)
            running_query_loss += float(result["query_loss"].cpu())
            steps += 1

        metrics = evaluate_ost(
            model,
            val_loader,
            train_dataset,
            trainer,
            device,
            alpha=(alpha_min + alpha_max) / 2.0,
            max_batches=(
                int(training["validation_batches"])
                if training.get("validation_batches") is not None
                else None
            ),
        )
        print(
            f"epoch={epoch:03d} query_loss={running_query_loss / max(steps, 1):.5f} "
            f"val_auc={metrics['auc']:.5f} val_acc={metrics['accuracy']:.5f}"
        )
        if float(metrics["auc"]) > best_auc:
            best_auc = float(metrics["auc"])
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_metrics = dict(metrics)

    if best_state is None:
        raise RuntimeError("OST meta-training finished without a valid checkpoint")
    model.load_state_dict(best_state)
    output_root = Path(config["output_dir"])
    write_json(output_root / "effective_config.json", config)
    output_path = output_root / "ost_meta.pt"
    save_checkpoint(
        model,
        output_path,
        architecture="meta_xception",
        method="ost",
        official_commit="1e4518b9e560baf9c5693f13a402fa5d7104190f",
        training_profile=str(training.get("profile", "ost_meta_training")),
        source_generator=train_generator,
        validation_generator=val_generator,
        validation_metrics=best_metrics,
        initialization=initialization_metadata,
        seed=seed,
    )
    print(f"saved={output_path}")


def estimate_bn_fishers(
    model: Any,
    loader: Any,
    device: Any,
    max_samples: int,
    *,
    parameter_scope: str = "batchnorm",
) -> dict[str, Any]:
    """Reproduce EATA's pseudo-label Fisher preparation for the chosen norm scope."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    fisher_model = deepcopy(model).to(device)
    normalized_scope = parameter_scope.lower().replace("-", "_")
    fisher_model.train()
    fisher_model.requires_grad_(False)
    if normalized_scope == "clip_visual_layernorm":
        from src.methods.utils import select_clip_visual_norm_parameters

        _parameters, names = select_clip_visual_norm_parameters(fisher_model)
        selected = dict(fisher_model.named_parameters())
        selected = {name: selected[name] for name in names}
    elif normalized_scope == "batchnorm":
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
    else:
        raise ValueError(
            "Fisher parameter_scope must be batchnorm or clip_visual_layernorm, "
            f"got {parameter_scope!r}"
        )
    if not selected:
        raise RuntimeError("Fisher preparation did not select any adaptation parameters")
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
    parser = argparse.ArgumentParser(description="Train a source detector or OST meta-model")
    parser.add_argument("--config", default="configs/train/source.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    require(config, "model", "data", "training", "output_dir")
    data_config = config["data"]
    require(data_config, "format", "train_split", "val_split")
    if "root" not in data_config and not {
        "train_root",
        "val_root",
    }.issubset(data_config):
        raise ValueError("data requires root or both train_root and val_root")
    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    model_config = config["model"]
    model_family = str(model_config.get("family", "detector")).lower().replace("-", "_")
    if str(model_config.get("architecture", "resnet50")).lower() in {
        "meta_xception",
        "metaxception",
    }:
        train_ost(config, device, seed)
        return
    train_generator = data_generator(data_config, "train")
    val_generator = data_generator(data_config, "val")

    image_size = int(data_config.get("image_size", 224))
    clip_source_detector = model_family == "clip_source_detector"
    train_transform = (
        build_clip_train_transform(image_size)
        if clip_source_detector
        else build_train_transform(image_size)
    )
    val_transform = (
        build_clip_eval_transform(
            image_size,
            resize_size=int(data_config.get("resize_size", round(image_size / 0.875))),
        )
        if clip_source_detector
        else build_eval_transform(
            image_size,
            resize_before_crop=bool(data_config.get("resize_before_crop", True)),
        )
    )
    train_dataset = build_dataset(
        data_format=data_config["format"],
        root=data_config.get("train_root", data_config.get("root")),
        generator=train_generator,
        split=data_config["train_split"],
        transform=train_transform,
        max_samples_per_class=data_config.get("max_train_samples_per_class"),
        seed=seed,
        exclude_image_paths=excluded_image_paths(data_config, "train"),
    )
    val_dataset = build_dataset(
        data_format=data_config["format"],
        root=data_config.get("val_root", data_config.get("root")),
        generator=val_generator,
        split=data_config["val_split"],
        transform=val_transform,
        max_samples_per_class=data_config.get("max_val_samples_per_class"),
        seed=seed,
        exclude_image_paths=excluded_image_paths(data_config, "val"),
    )
    train_loader = build_loader(train_dataset, data_config, shuffle=True, seed=seed)
    val_loader = build_loader(val_dataset, data_config, shuffle=False)

    initialization_metadata: dict[str, Any] | None = None
    if clip_source_detector:
        model, initialization_metadata = build_clip_source_detector(
            model_config, device=device
        )
    else:
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
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training.get("learning_rate", 1e-4)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    amp_enabled = str(device).startswith("cuda") and bool(training.get("amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_auc = float("-inf")
    best_state = None
    best_metrics: dict[str, Any] = {}
    epochs = int(training.get("epochs", 10))
    max_steps = training.get("max_steps_per_epoch")
    max_steps = None if max_steps is None else int(max_steps)
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        steps = 0
        for images, labels, _ in train_loader:
            if max_steps is not None and steps >= max_steps:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with _autocast(device, amp_enabled):
                loss = functional.cross_entropy(model(images), labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            steps += 1

        metrics = evaluate(model, val_loader, device, amp_enabled=amp_enabled)
        print(
            f"epoch={epoch:03d} loss={running_loss / max(steps, 1):.5f} "
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
            model,
            fisher_loader,
            device,
            int(training.get("fisher_samples", 2000)),
            parameter_scope=str(training.get("fisher_parameter_scope", "batchnorm")),
        )

    output_root = Path(config["output_dir"])
    write_json(output_root / "effective_config.json", config)
    output_path = output_root / "source.pt"
    save_checkpoint(
        model,
        output_path,
        architecture=model_config.get("architecture", "resnet50"),
        model_family=model_family,
        pretrained_weights=model_config.get("pretrained_weights", "default"),
        source_generator=str(data_config.get("generator", train_generator)),
        train_generator=train_generator,
        validation_generator=val_generator,
        validation_metrics=best_metrics,
        fishers=fishers,
        training_profile=str(training.get("profile", "common_source_detector")),
        checkpoint_role=str(training.get("checkpoint_role", "source_detector")),
        intended_methods=list(training.get("intended_methods", [])),
        requested_for=training.get("requested_for"),
        initialization=initialization_metadata,
        fisher_parameter_scope=str(training.get("fisher_parameter_scope", "batchnorm")),
        seed=seed,
    )
    print(f"saved={output_path}")


if __name__ == "__main__":
    main()
