"""Vendored CoTTA transforms from the official ImageNet implementation.

Source commit: c212a204b32be4005092e4323105a24a29ad2952
Source file: imagenet/my_transforms.py
License: MIT, see THIRD_PARTY_NOTICES.md
"""

import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter, Compose, Lambda
from numpy import random


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Clip(torch.nn.Module):
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)


class ColorJitterPro(ColorJitter):
    """Randomly change brightness, contrast, saturation, hue, and gamma."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, 'gamma')

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        transforms = []
        if brightness is not None:
            factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, factor)))
        if contrast is not None:
            factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, factor)))
        if saturation is not None:
            factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, factor)))
        if hue is not None:
            factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, factor)))
        if gamma is not None:
            factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, factor)))
        random.shuffle(transforms)
        return Compose(transforms)

    def forward(self, img):
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                factor = torch.tensor(1.0).uniform_(self.brightness[0], self.brightness[1]).item()
                img = F.adjust_brightness(img, factor)
            if fn_id == 1 and self.contrast is not None:
                factor = torch.tensor(1.0).uniform_(self.contrast[0], self.contrast[1]).item()
                img = F.adjust_contrast(img, factor)
            if fn_id == 2 and self.saturation is not None:
                factor = torch.tensor(1.0).uniform_(self.saturation[0], self.saturation[1]).item()
                img = F.adjust_saturation(img, factor)
            if fn_id == 3 and self.hue is not None:
                factor = torch.tensor(1.0).uniform_(self.hue[0], self.hue[1]).item()
                img = F.adjust_hue(img, factor)
            if fn_id == 4 and self.gamma is not None:
                factor = torch.tensor(1.0).uniform_(self.gamma[0], self.gamma[1]).item()
                img = img.clamp(1e-8, 1.0)
                img = F.adjust_gamma(img, factor)
        return img
