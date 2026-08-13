"""RoTTA authors' strong test-time augmentation with torchvision API patches.

Source commit: 67e34c900cdd355fc07e55edd4c577ea7b8ebcc9
Source file: core/utils/custom_transforms.py
License: MIT, see THIRD_PARTY_NOTICES.md
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter


def get_tta_transforms(image_size: int, gaussian_std: float = 0.005, soft=False):
    n_pixels = image_size
    return transforms.Compose(
        [
            Clip(0.0, 1.0),
            ColorJitterPro(
                brightness=[0.8, 1.2] if soft else [0.6, 1.4],
                contrast=[0.85, 1.15] if soft else [0.7, 1.3],
                saturation=[0.75, 1.25] if soft else [0.5, 1.5],
                hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
                gamma=[0.85, 1.15] if soft else [0.7, 1.3],
            ),
            transforms.Pad(padding=int(n_pixels / 2), padding_mode="edge"),
            transforms.RandomAffine(
                degrees=[-8, 8] if soft else [-15, 15],
                translate=(1 / 16, 1 / 16),
                scale=(0.95, 1.05) if soft else (0.9, 1.1),
                shear=None,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=None,
            ),
            transforms.GaussianBlur(
                kernel_size=5,
                sigma=[0.001, 0.25] if soft else [0.001, 0.5],
            ),
            transforms.CenterCrop(size=n_pixels),
            transforms.RandomHorizontalFlip(p=0.5),
            GaussianNoise(0, gaussian_std),
            Clip(0.0, 1.0),
        ]
    )


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size(), device=img.device) * self.std + self.mean
        return img + noise


class Clip(torch.nn.Module):
    def __init__(self, min_val=0.0, max_val=1.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)


class ColorJitterPro(ColorJitter):
    """Authors' ColorJitter extension with gamma correction."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, "gamma")

    def forward(self, img):
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                factor = torch.tensor(1.0).uniform_(*self.brightness).item()
                img = F.adjust_brightness(img, factor)
            if fn_id == 1 and self.contrast is not None:
                factor = torch.tensor(1.0).uniform_(*self.contrast).item()
                img = F.adjust_contrast(img, factor)
            if fn_id == 2 and self.saturation is not None:
                factor = torch.tensor(1.0).uniform_(*self.saturation).item()
                img = F.adjust_saturation(img, factor)
            if fn_id == 3 and self.hue is not None:
                factor = torch.tensor(1.0).uniform_(*self.hue).item()
                img = F.adjust_hue(img, factor)
            if fn_id == 4 and self.gamma is not None:
                factor = torch.tensor(1.0).uniform_(*self.gamma).item()
                img = F.adjust_gamma(img.clamp(1e-8, 1.0), factor)
        return img
