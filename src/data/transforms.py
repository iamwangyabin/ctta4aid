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


def build_clip_train_transform(image_size: int = 224) -> Any:
    """Augment source images while preserving OpenAI CLIP normalization."""

    transforms = _torchvision_transforms()
    try:
        interpolation = transforms.InterpolationMode.BICUBIC
    except AttributeError:
        from PIL import Image

        interpolation = Image.BICUBIC
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size, scale=(0.8, 1.0), interpolation=interpolation
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def jpeg_reencode(image: Any, quality: int) -> Any:
    """Re-encode a PIL image as JPEG in memory and decode it back to RGB."""

    from io import BytesIO

    from PIL import Image

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


class _RandomJPEGCompression:
    """Torchvision-compatible JPEG recompression with a torch-seeded quality."""

    def __init__(self, quality_range: tuple[int, int] = (70, 95)) -> None:
        low, high = int(quality_range[0]), int(quality_range[1])
        if not 1 <= low <= high <= 100:
            raise ValueError("JPEG quality range must satisfy 1 <= min <= max <= 100")
        self.low = low
        self.high = high

    def __call__(self, image: Any) -> Any:
        import torch

        quality = int(torch.randint(self.low, self.high + 1, (1,)).item())
        return jpeg_reencode(image, quality)


def build_clip_lora_train_transform(
    image_size: int = 224,
    *,
    jpeg_prob: float = 0.1,
    blur_prob: float = 0.1,
    jpeg_quality_range: tuple[int, int] = (70, 95),
) -> Any:
    """CLIP train augmentation with light JPEG/blur degradation for Ours.

    The degradation branch compresses deployment-time perturbation jitter into
    the training phase so the calibrated score anchors stay compact.
    """

    if not 0.0 <= jpeg_prob <= 1.0 or not 0.0 <= blur_prob <= 1.0:
        raise ValueError("jpeg_prob and blur_prob must be in [0, 1]")
    transforms = _torchvision_transforms()
    try:
        interpolation = transforms.InterpolationMode.BICUBIC
    except AttributeError:
        from PIL import Image

        interpolation = Image.BICUBIC
    operations: list[Any] = [
        transforms.RandomResizedCrop(
            image_size, scale=(0.8, 1.0), interpolation=interpolation
        ),
        transforms.RandomHorizontalFlip(),
    ]
    if jpeg_prob > 0.0:
        operations.append(
            transforms.RandomApply(
                [_RandomJPEGCompression(jpeg_quality_range)], p=jpeg_prob
            )
        )
    if blur_prob > 0.0:
        operations.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))],
                p=blur_prob,
            )
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    return transforms.Compose(operations)


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
