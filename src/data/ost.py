from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

from .arrow import build_dataset


def build_ost_transform(image_size: int = 256) -> Any:
    """Match the resize and [-1, 1] normalization in the OST release."""
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for OST image transforms") from exc
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


class OSTTemplateSampler:
    """Draw labeled source templates from the canonical Arrow dataset."""

    def __init__(
        self,
        *,
        root: str | Path | Sequence[str | Path],
        domain: str,
        split: str,
        transform: Any,
        seed: int,
        max_samples_per_class: int | None = None,
    ) -> None:
        self.seed = seed
        self._rng = random.Random(seed)
        self.dataset = build_dataset(
            data_format="arrow",
            root=root,
            generator=domain,
            split=split,
            transform=transform,
            max_samples_per_class=max_samples_per_class,
            seed=seed,
        )

    @classmethod
    def from_data_config(
        cls, data_config: dict[str, Any], *, transform: Any, seed: int
    ) -> "OSTTemplateSampler":
        if data_config.get("format") != "arrow":
            raise ValueError("OST source templates require data.format: arrow")
        domain = data_config.get("source_domain")
        if not domain:
            raise ValueError("OST data config requires source_domain")
        return cls(
            root=data_config.get("source_root", data_config["root"]),
            domain=str(domain),
            split=str(data_config.get("source_split", "train")),
            transform=transform,
            seed=seed,
            max_samples_per_class=data_config.get("source_max_samples_per_class"),
        )

    def sample(self) -> tuple[Any, int, str]:
        return self.dataset[self._rng.randrange(len(self.dataset))]

    def reset(self) -> None:
        self._rng.seed(self.seed)
