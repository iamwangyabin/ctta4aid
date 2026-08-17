from __future__ import annotations

from typing import Any


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


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


def build_clip_eval_transform(image_size: int = 224, *, resize_size: int | None = None) -> Any:
    """Match OpenAI CLIP's deterministic resize, crop, and normalization."""

    transforms = _torchvision_transforms()
    if resize_size is None:
        resize_size = int(round(image_size / 0.875))
    try:
        interpolation = transforms.InterpolationMode.BICUBIC
    except AttributeError:
        from PIL import Image

        interpolation = Image.BICUBIC
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
