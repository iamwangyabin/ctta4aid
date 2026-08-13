from __future__ import annotations

from typing import Any


class GlobalLocalViewTransform:
    """Build one global view and repeated local views for image-level adaptation."""

    def __init__(
        self,
        *,
        views: int = 32,
        image_size: int = 224,
        resize_size: int = 256,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        if views < 2:
            raise ValueError("At least two views are required")
        if image_size <= 0 or resize_size < image_size:
            raise ValueError("resize_size must be at least image_size")

        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self.views = views
        self.image_size = image_size
        self.resize_size = resize_size
        self._to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        self._global = transforms.Compose(
            [
                transforms.Resize(
                    (resize_size, resize_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(image_size),
            ]
        )
        self._resized_local = transforms.RandomResizedCrop(
            image_size,
            scale=((image_size / resize_size) ** 2,) * 2,
            ratio=(1.0, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        )
        self._native_local = transforms.RandomCrop(image_size)
        self._flip = transforms.RandomHorizontalFlip()

    def __call__(self, image: Any) -> Any:
        import torch

        image = image.convert("RGB")
        global_view = self._to_tensor(self._global(image))
        can_crop_without_resize = (
            max(image.size) >= self.resize_size and min(image.size) >= self.image_size
        )
        local_crop = self._native_local if can_crop_without_resize else self._resized_local
        local_views = [
            self._to_tensor(self._flip(local_crop(image)))
            for _ in range(self.views - 1)
        ]
        return torch.stack([global_view, *local_views], dim=0)
