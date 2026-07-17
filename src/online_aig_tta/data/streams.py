from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from online_aig_tta.types import StreamBatch

from .folders import build_dataset
from .transforms import build_eval_transform


def build_domain_loader(
    data_config: dict[str, Any],
    domain: str,
    *,
    seed: int,
    sample_seed: int | None = None,
    loader_seed: int | None = None,
    max_samples_per_class: int | None = None,
    sample_offset_per_class: int = 0,
    shuffle: bool | None = None,
) -> Any:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to build data loaders") from exc

    dataset = build_dataset(
        data_format=data_config["format"],
        root=data_config["root"],
        generator=domain,
        split=data_config.get("split"),
        transform=build_eval_transform(
            int(data_config.get("image_size", 224)),
            resize_before_crop=bool(data_config.get("resize_before_crop", True)),
        ),
        max_samples_per_class=(
            max_samples_per_class
            if max_samples_per_class is not None
            else data_config.get("max_samples_per_class")
        ),
        sample_offset_per_class=sample_offset_per_class,
        seed=seed if sample_seed is None else sample_seed,
    )
    effective_shuffle = (
        bool(data_config.get("shuffle", True)) if shuffle is None else shuffle
    )
    generator_state = torch.Generator()
    generator_state.manual_seed(seed if loader_seed is None else loader_seed)
    return DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 16)),
        shuffle=effective_shuffle,
        num_workers=int(data_config.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
        generator=generator_state,
    )


def as_stream(loader: Iterable[Any], domain: str) -> Iterator[StreamBatch]:
    for images, labels, paths in loader:
        yield StreamBatch(
            images=images,
            hidden_labels=labels,
            domain=domain,
            sample_ids=list(paths),
        )


def concatenate_domain_streams(
    data_config: dict[str, Any], domains: list[str], *, seed: int
) -> Iterator[StreamBatch]:
    for offset, domain in enumerate(domains):
        loader = build_domain_loader(data_config, domain, seed=seed + offset)
        yield from as_stream(loader, domain)
