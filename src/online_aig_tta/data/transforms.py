from __future__ import annotations

from typing import Any


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _torchvision_transforms() -> Any:
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required for image transforms") from exc
    return transforms


def build_train_transform(image_size: int = 224) -> Any:
    transforms = _torchvision_transforms()
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform(image_size: int = 224, *, resize_before_crop: bool = True) -> Any:
    transforms = _torchvision_transforms()
    resize_size = int(round(image_size / 0.875))
    operations = []
    if resize_before_crop:
        operations.append(transforms.Resize(resize_size))
    operations.extend(
        [
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(
        operations
    )
