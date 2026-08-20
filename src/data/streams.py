from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.types import StreamBatch

from .arrow import build_dataset
from .transforms import build_eval_transform


_MANIFEST_FIELDS = ("batch", "domain", "position", "sample_id")


def load_online_manifest_lock(
    config: Mapping[str, Any], domains: Sequence[str]
) -> dict[str, Any] | None:
    configured = config["data"].get("locked_online_manifest")
    if configured is None:
        return None
    path = Path(str(configured)).expanduser()
    if not path.is_absolute():
        path = Path(str(config["_config_path"])).parent / path
    manifest = load_locked_manifest(path.resolve())
    return {
        "sample_ids_by_domain": locked_sample_ids_by_domain(manifest, domains),
        "config": {"online_manifest": str(configured)},
    }


def validate_locked_sample_order(
    observed_sample_ids: Sequence[str], expected_sample_ids: Sequence[str]
) -> None:
    if list(observed_sample_ids) != list(expected_sample_ids):
        raise RuntimeError(
            "Single-target evaluation did not consume the configured locked samples "
            "in manifest order"
        )


def load_locked_manifest(path: str | Path) -> list[dict[str, int | str]]:
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Locked sample manifest does not exist: {manifest_path}")
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != _MANIFEST_FIELDS:
            raise ValueError(
                f"Locked sample manifest has unexpected columns: {manifest_path}"
            )
        rows: list[dict[str, int | str]] = []
        previous_batch = -1
        for expected_position, row in enumerate(reader):
            try:
                batch = int(row["batch"])
                position = int(row["position"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Locked sample manifest has invalid batch or position: {manifest_path}"
                ) from error
            domain = row["domain"]
            sample_id = row["sample_id"]
            if (
                position != expected_position
                or batch < 0
                or batch > previous_batch + 1
                or not domain
                or not sample_id
            ):
                raise ValueError(
                    f"Locked sample manifest is not a complete ordered stream: {manifest_path}"
                )
            rows.append(
                {
                    "batch": batch,
                    "domain": domain,
                    "position": position,
                    "sample_id": sample_id,
                }
            )
            previous_batch = batch
    if not rows:
        raise ValueError(f"Locked sample manifest must not be empty: {manifest_path}")
    return rows


def locked_sample_ids_by_domain(
    manifest: Sequence[Mapping[str, int | str]], domains: Sequence[str]
) -> dict[str, list[str]]:
    if len(domains) != len(set(domains)):
        raise ValueError("Locked stream domains must be unique")
    grouped = {domain: [] for domain in domains}
    observed_domains: list[str] = []
    completed_domains: set[str] = set()
    active_domain: str | None = None
    for row in manifest:
        domain = str(row["domain"])
        if domain not in grouped:
            raise ValueError(
                f"Locked sample manifest references a domain outside the stream: {domain}"
            )
        if domain != active_domain:
            if domain in completed_domains:
                raise ValueError(
                    f"Locked sample manifest revisits a completed domain: {domain}"
                )
            if active_domain is not None:
                completed_domains.add(active_domain)
            observed_domains.append(domain)
            active_domain = domain
        grouped[domain].append(str(row["sample_id"]))
    empty_domains = [domain for domain, sample_ids in grouped.items() if not sample_ids]
    if empty_domains:
        raise ValueError(
            "Locked sample manifest is missing stream domains: "
            + ", ".join(empty_domains)
        )
    if observed_domains != list(domains):
        raise ValueError(
            "Locked sample manifest domain order differs from the configured stream"
        )
    return grouped


def lock_stream_to_manifest(
    stream: Iterable[StreamBatch],
    manifest: Sequence[Mapping[str, int | str]],
    *,
    name: str,
) -> Iterator[StreamBatch]:
    position = 0
    for batch_index, batch in enumerate(stream):
        for sample_id in batch.sample_ids:
            if position >= len(manifest):
                raise RuntimeError(f"Locked {name} stream contains unexpected samples")
            expected = manifest[position]
            actual = {
                "batch": batch_index,
                "domain": batch.domain,
                "position": position,
                "sample_id": sample_id,
            }
            if actual != expected:
                raise RuntimeError(
                    f"Locked {name} stream mismatch at position {position}: "
                    f"expected {expected}, got {actual}"
                )
            position += 1
        yield batch
    if position != len(manifest):
        raise RuntimeError(
            f"Locked {name} stream ended after {position} samples; "
            f"expected {len(manifest)}"
        )


def build_domain_loader(
    data_config: dict[str, Any],
    domain: str,
    *,
    seed: int,
    transform: Any = None,
    sample_seed: int | None = None,
    loader_seed: int | None = None,
    max_samples_per_class: int | None = None,
    sample_offset_per_class: int = 0,
    shuffle: bool | None = None,
    locked_sample_ids: Sequence[str] | None = None,
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
        transform=(
            transform
            if transform is not None
            else build_eval_transform(
                int(data_config.get("image_size", 224)),
                resize_before_crop=bool(data_config.get("resize_before_crop", True)),
            )
        ),
        max_samples_per_class=(
            None
            if locked_sample_ids is not None
            else (
                max_samples_per_class
                if max_samples_per_class is not None
                else data_config.get("max_samples_per_class")
            )
        ),
        sample_offset_per_class=(
            0 if locked_sample_ids is not None else sample_offset_per_class
        ),
        seed=seed if sample_seed is None else sample_seed,
        locked_sample_ids=locked_sample_ids,
        bias_control_profile=data_config.get("bias_control_profile"),
    )
    effective_shuffle = False if locked_sample_ids is not None else (
        bool(data_config.get("shuffle", True)) if shuffle is None else shuffle
    )
    num_workers = int(data_config.get("num_workers", 4))
    worker_start_method = data_config.get("worker_start_method")
    loader_options = {}
    if worker_start_method is not None and num_workers > 0:
        # CLIP methods construct a CUDA model before its model-native transform
        # is available. Spawn avoids forking that initialized CUDA process.
        loader_options["multiprocessing_context"] = str(worker_start_method)
    generator_state = torch.Generator()
    generator_state.manual_seed(seed if loader_seed is None else loader_seed)
    return DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 16)),
        shuffle=effective_shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        generator=generator_state,
        **loader_options,
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
    data_config: dict[str, Any],
    domains: list[str],
    *,
    seed: int,
    transform: Any = None,
    locked_samples_by_domain: Mapping[str, Sequence[str]] | None = None,
) -> Iterator[StreamBatch]:
    for offset, domain in enumerate(domains):
        locked_sample_ids = None
        if locked_samples_by_domain is not None:
            if domain not in locked_samples_by_domain:
                raise ValueError(f"No locked samples for stream domain: {domain}")
            locked_sample_ids = locked_samples_by_domain[domain]
        loader = build_domain_loader(
            data_config,
            domain,
            seed=seed + offset,
            transform=transform,
            locked_sample_ids=locked_sample_ids,
        )
        yield from as_stream(loader, domain)
