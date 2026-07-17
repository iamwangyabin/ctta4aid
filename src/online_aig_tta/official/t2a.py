"""Patched authors' public T²A adapter core.

Source commit: 33c8ccc64afdda260564123d6c790d030a89ff81
Source files: adapters/base_adapter.py, adapters/T2A.py, utils.py
License: MIT, see THIRD_PARTY_NOTICES.md

The release is incomplete. Every executable repair is called out inline and
in REPRODUCIBILITY.md; the framework wrapper only supplies tensors and invokes
``adapt``/``reset`` on this core.
"""

import logging
from abc import ABC, abstractmethod
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .t2a_losses import Entropy, compute_noise_tolerant_negative_loss


logger = logging.getLogger(__name__)


def compute_cosine_similarity(a, b, strategy: str):
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    if a.shape[1] > b.shape[1]:
        b = F.pad(b, (0, a.shape[1] - b.shape[1]))
    elif b.shape[1] > a.shape[1]:
        a = F.pad(a, (0, b.shape[1] - a.shape[1]))
    return F.cosine_similarity(a, b, dim=1).item()


class BaseAdapter(nn.Module, ABC):
    """Base class retained from the authors' public adapter."""

    def __init__(self, model: nn.Module, device, **kwargs):
        super().__init__()
        self.config = kwargs
        self.device = device
        self.model = model
        self.setup_adapter()
        self.setup_model()

    @abstractmethod
    def setup_adapter(self):
        pass

    @abstractmethod
    def setup_model(self):
        pass

    def setup_optimizer(self, params):
        if self.config["optimizer"] == "Adam":
            return optim.Adam(
                params,
                lr=self.config["optimizer_config"]["lr"],
                betas=(
                    self.config["optimizer_config"]["beta1"],
                    self.config["optimizer_config"]["beta2"],
                ),
                weight_decay=self.config["optimizer_config"]["weight_decay"],
            )
        elif self.config["optimizer"] == "AdamW":
            return optim.AdamW(
                params,
                lr=self.config["optimizer_config"]["lr"],
                betas=(
                    self.config["optimizer_config"]["beta1"],
                    self.config["optimizer_config"]["beta2"],
                ),
                weight_decay=self.config["optimizer_config"]["weight_decay"],
            )
        elif self.config["optimizer"] == "SGD":
            return optim.SGD(
                params,
                lr=self.config["optimizer_config"]["lr"],
                momentum=self.config["optimizer_config"]["momentum"],
                dampening=self.config["optimizer_config"]["dampening"],
                weight_decay=self.config["optimizer_config"]["weight_decay"],
                nesterov=self.config["optimizer_config"]["nesterov"],
            )
        raise NotImplementedError(
            f"Optimizer {self.config['optimizer']} not implemented"
        )

    @torch.enable_grad()
    def adapt_and_predict(self, data_dict: dict):
        return self.forward(data_dict)


class T2AAdapter(BaseAdapter):
    def setup_adapter(self):
        self.steps = self.config.get("steps")
        self.episodic = self.config.get("episodic")
        self.noise_type = self.config.get("noise_type")
        self.gamma = self.config.get("gamma")
        self.l1_lambda = self.config.get("l1_lambda")
        self.psi = self.config.get("psi")
        self.alpha = self.config.get("alpha")
        self.beta = self.config.get("beta")

        # Required release repairs: these attributes are used by the public
        # adapter but never initialized there.
        self.entropy_fn = self.config.get("entropy_fn", Entropy())
        self.e_margin = self.config.get("e_margin")
        self.filter_grad = self.config.get("filter_grad", True)
        self.cosine_strategy = self.config.get("cosine_strategy", "zero_pad")
        self.last_loss = None
        self.last_masked = 0

    def setup_model(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = self.setup_optimizer(params)
        # Required repair: the release stores aliased state dictionaries.
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        self.model.load_state_dict(deepcopy(self.model_state))
        self.optimizer.load_state_dict(deepcopy(self.optimizer_state))

    @torch.enable_grad()
    def entropy_minimization(self, data_dict: dict):
        logits = self.model(data_dict)["cls"]
        entropy = self.entropy_fn(logits)
        coeff = 1 / (torch.exp(entropy - self.e_margin))
        return (entropy * coeff).mean(0)

    @torch.enable_grad()
    def noise_tolerant_negative_loss(self, data_dict: dict):
        outputs = self.model(data_dict)["cls"]
        return compute_noise_tolerant_negative_loss(
            outputs,
            noise_type=self.noise_type,
            gamma=self.gamma,
            alpha=self.alpha,
            beta=self.beta,
        )

    def perform_gradient_masking(self):
        bn_grads = []
        bn_parameter_ids = set()
        for _, module in self.model.named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                for parameter in (module.weight, module.bias):
                    if parameter is not None:
                        bn_parameter_ids.add(id(parameter))
                        if parameter.grad is not None:
                            bn_grads.append(parameter.grad.flatten())
        if not bn_grads:
            return 0

        bn_grad_vector = torch.cat(bn_grads)
        masked = 0
        for _, param in self.model.named_parameters():
            # Required repair: the released name-containment test classifies
            # BN parameters incorrectly. Identity is unambiguous.
            if param.grad is not None and id(param) not in bn_parameter_ids:
                param_grad_flat = param.grad.flatten().unsqueeze(0)
                cos_sim = compute_cosine_similarity(
                    param_grad_flat,
                    bn_grad_vector,
                    strategy=self.cosine_strategy,
                )
                if cos_sim < self.psi:
                    param.grad.zero_()
                    masked += 1
        return masked

    @torch.enable_grad()
    def adapt(self, data_dict: dict):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            loss = self.entropy_minimization(
                data_dict
            ) + self.noise_tolerant_negative_loss(data_dict)
            for param in self.model.parameters():
                if param.requires_grad:
                    loss += self.l1_lambda * torch.norm(param, p=1)
            self.optimizer.zero_grad()
            loss.backward()
            self.last_masked = (
                self.perform_gradient_masking() if self.filter_grad else 0
            )
            self.optimizer.step()
            self.last_loss = float(loss.detach().cpu())
        return self.last_loss

    def predict(self, data_dict: dict):
        with torch.no_grad():
            logits = self.model(data_dict)["cls"]
            prob = logits.softmax(1)[:, 1]
        return {"cls": logits, "prob": prob}

    @torch.enable_grad()
    def forward(self, data_dict: dict) -> dict:
        self.adapt(data_dict)
        return self.predict(data_dict)
