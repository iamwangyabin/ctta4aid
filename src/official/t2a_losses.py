"""Patched authors' T²A loss core.

Source commit: 33c8ccc64afdda260564123d6c790d030a89ff81
Source file: losses/__init__.py
License: MIT, see THIRD_PARTY_NOTICES.md

Required patches: logits are passed to log-softmax losses, and the released
BxC Bernoulli target is converted to one non-pseudo complementary class index
per sample.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def __call__(self, logits):
        return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class NormalizedLoss(torch.nn.Module):
    def __init__(self, num_classes: int = 2, gamma: float = 0.0):
        super(NormalizedLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        norm = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1 - pt) ** self.gamma * logpt
        loss = loss / norm.clamp_min(1e-8)
        return loss.mean()


class NormalizedNegativeLoss(torch.nn.Module):
    def __init__(self, num_classes: int = 2, gamma: float = 0.0, p0: float = 1e-8) -> None:
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        p0_tensor = torch.as_tensor(p0).detach().clamp(1e-8, 1 - 1e-8)
        self.logmp = p0_tensor.log()
        self.p0 = -((1 - p0_tensor) ** self.gamma) * self.logmp

    def forward(self, input, target):
        logmp = self.logmp.to(input.device)
        p0 = self.p0.to(input.device)
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1).clamp(min=logmp)
        norm = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1 - pt) ** self.gamma * logpt
        denominator = (self.num_classes * p0 - norm).clamp_min(1e-8)
        loss = 1 - (p0 - loss) / denominator
        return loss.mean()


def complementary_labels(logits, noise_type: str = "uniform"):
    probs = logits.softmax(-1)
    pseudo_labels = logits.argmax(dim=-1)
    if noise_type == "bernoulli":
        if logits.shape[1] < 2:
            raise ValueError("Complementary labels require at least two classes")
        if logits.shape[1] == 2:
            return 1 - pseudo_labels
        # Keep 1-p as the released sampling weight, but exclude the predicted
        # class so the repaired target is actually complementary.
        weights = (1 - probs).detach().clone()
        weights.scatter_(1, pseudo_labels[:, None], 0.0)
        return torch.multinomial(weights, num_samples=1).squeeze(1)
    if noise_type == "uniform":
        return (pseudo_labels == 0).long()
    return pseudo_labels


def compute_noise_tolerant_negative_loss(
    x, noise_type: str = "uniform", gamma: float = 2.0,
    alpha: float = 1.0, beta: float = 1.0
):
    num_classes = x.shape[1]
    probs = x.softmax(-1)
    flipped_labels = complementary_labels(x, noise_type)
    # Patched: the released code passed `probs` here, causing a second
    # softmax/log-softmax. The published normalized losses require logits.
    Lnorm = NormalizedLoss(num_classes=num_classes, gamma=gamma)(
        x, flipped_labels
    )
    Lnn = NormalizedNegativeLoss(
        num_classes=num_classes, gamma=gamma, p0=probs.min()
    )(x, flipped_labels)
    return alpha * Lnorm + beta * Lnn
