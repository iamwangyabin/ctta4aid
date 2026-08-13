"""Vendored RoTTA core with minimal device/protocol compatibility patches.

Source commit: 67e34c900cdd355fc07e55edd4c577ea7b8ebcc9
Source files: core/adapter/rotta.py, core/utils/memory.py,
core/utils/bn_layers.py, core/adapter/base_adapter.py
License: MIT, see THIRD_PARTY_NOTICES.md
"""

import math
from copy import deepcopy

import torch
import torch.nn as nn

from .rotta_transforms import get_tta_transforms


class MemoryItem:
    def __init__(self, data=None, uncertainty=0, age=0):
        self.data = data
        self.uncertainty = uncertainty
        self.age = age

    def increase_age(self):
        self.age += 1


class CSTU:
    """Category-balanced sampling with timeliness and uncertainty."""

    def __init__(self, capacity, num_class, lambda_t=1.0, lambda_u=1.0):
        self.capacity = capacity
        self.num_class = num_class
        self.per_class = self.capacity / self.num_class
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.data = [[] for _ in range(self.num_class)]

    def get_occupancy(self):
        return sum(len(data_per_cls) for data_per_cls in self.data)

    def per_class_dist(self):
        return [len(class_list) for class_list in self.data]

    def add_instance(self, instance):
        assert len(instance) == 3
        x, prediction, uncertainty = instance
        new_item = MemoryItem(data=x, uncertainty=uncertainty, age=0)
        new_score = self.heuristic_score(0, uncertainty)
        if self.remove_instance(prediction, new_score):
            self.data[prediction].append(new_item)
        self.add_age()

    def remove_instance(self, cls, score):
        class_occupied = len(self.data[cls])
        all_occupancy = self.get_occupancy()
        if class_occupied < self.per_class:
            if all_occupancy < self.capacity:
                return True
            return self.remove_from_classes(self.get_majority_classes(), score)
        return self.remove_from_classes([cls], score)

    def remove_from_classes(self, classes, score_base):
        max_class = None
        max_index = None
        max_score = None
        for cls in classes:
            for idx, item in enumerate(self.data[cls]):
                score = self.heuristic_score(item.age, item.uncertainty)
                if max_score is None or score >= max_score:
                    max_score = score
                    max_index = idx
                    max_class = cls
        if max_class is None:
            return True
        if max_score > score_base:
            self.data[max_class].pop(max_index)
            return True
        return False

    def get_majority_classes(self):
        distribution = self.per_class_dist()
        maximum = max(distribution)
        return [index for index, occupied in enumerate(distribution) if occupied == maximum]

    def heuristic_score(self, age, uncertainty):
        return self.lambda_t / (1 + math.exp(-age / self.capacity)) + (
            self.lambda_u * uncertainty / math.log(self.num_class)
        )

    def add_age(self):
        for class_list in self.data:
            for item in class_list:
                item.increase_age()

    def get_memory(self):
        memory_data = []
        memory_age = []
        for class_list in self.data:
            for item in class_list:
                memory_data.append(item.data)
                memory_age.append(item.age)
        return memory_data, [age / self.capacity for age in memory_age]


class MomentumBN(nn.Module):
    def __init__(self, bn_layer, momentum):
        super().__init__()
        self.num_features = bn_layer.num_features
        self.momentum = momentum
        self.register_buffer("source_mean", deepcopy(bn_layer.running_mean))
        self.register_buffer("source_var", deepcopy(bn_layer.running_var))
        self.source_num = bn_layer.num_batches_tracked
        self.weight = deepcopy(bn_layer.weight)
        self.bias = deepcopy(bn_layer.bias)
        self.register_buffer("target_mean", torch.zeros_like(self.source_mean))
        self.register_buffer("target_var", torch.ones_like(self.source_var))
        self.eps = bn_layer.eps


class RobustBN1d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=0, unbiased=False)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1), var.view(1, -1)
        else:
            mean, var = self.source_mean.view(1, -1), self.source_var.view(1, -1)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1) + self.bias.view(1, -1)


class RobustBN2d(MomentumBN):
    def forward(self, x):
        if self.training:
            b_var, b_mean = torch.var_mean(x, dim=[0, 2, 3], unbiased=False)
            mean = (1 - self.momentum) * self.source_mean + self.momentum * b_mean
            var = (1 - self.momentum) * self.source_var + self.momentum * b_var
            self.source_mean, self.source_var = deepcopy(mean.detach()), deepcopy(var.detach())
            mean, var = mean.view(1, -1, 1, 1), var.view(1, -1, 1, 1)
        else:
            mean = self.source_mean.view(1, -1, 1, 1)
            var = self.source_var.view(1, -1, 1, 1)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def _set_named_submodule(model, name, module):
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, module)


def configure_model(model, alpha=0.05):
    model.requires_grad_(False)
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d))
    ]
    for name in names:
        bn_layer = model.get_submodule(name)
        if bn_layer.running_mean is None or bn_layer.running_var is None:
            raise RuntimeError("RoTTA RobustBN requires source BatchNorm running statistics")
        new_class = RobustBN1d if isinstance(bn_layer, nn.BatchNorm1d) else RobustBN2d
        new_bn = new_class(bn_layer, alpha)
        new_bn.requires_grad_(True)
        _set_named_submodule(model, name, new_bn)
    if not names:
        raise RuntimeError("Official RoTTA requires BatchNorm1d or BatchNorm2d layers")
    return model


def collect_params(model):
    names = []
    params = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            names.append(name)
            params.append(parameter)
    return params, names


@torch.jit.script
def softmax_entropy(x, x_ema):
    return -(x_ema.softmax(1) * x.log_softmax(1)).sum(1)


def timeliness_reweighting(ages, device):
    if isinstance(ages, list):
        ages = torch.tensor(ages, dtype=torch.float32, device=device)
    return torch.exp(-ages) / (1 + torch.exp(-ages))


class RoTTA(nn.Module):
    def __init__(
        self,
        model,
        optimizer,
        *,
        num_classes=2,
        memory_size=64,
        lambda_t=1.0,
        lambda_u=1.0,
        nu=0.001,
        update_frequency=64,
        image_size=224,
        gaussian_std=0.005,
        soft=False,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.memory_size = memory_size
        self.lambda_t = lambda_t
        self.lambda_u = lambda_u
        self.nu = nu
        self.update_frequency = update_frequency
        self.image_size = image_size
        self.gaussian_std = gaussian_std
        self.soft = soft
        self.transform = get_tta_transforms(image_size, gaussian_std, soft)
        self.model_state = deepcopy(model.state_dict())
        self.optimizer_state = deepcopy(optimizer.state_dict())
        self.last_loss = None
        self.update_count = 0
        self._reset_online_state()

    def _reset_online_state(self):
        self.mem = CSTU(
            capacity=self.memory_size,
            num_class=self.num_classes,
            lambda_t=self.lambda_t,
            lambda_u=self.lambda_u,
        )
        self.model_ema = deepcopy(self.model)
        for parameter in self.model_ema.parameters():
            parameter.detach_()
        self.current_instance = 0
        self.update_count = 0
        self.last_loss = None

    def reset(self):
        self.model.load_state_dict(deepcopy(self.model_state), strict=True)
        self.optimizer.load_state_dict(deepcopy(self.optimizer_state))
        self.transform = get_tta_transforms(
            self.image_size, self.gaussian_std, self.soft
        )
        self._reset_online_state()

    @torch.no_grad()
    def teacher_prediction(self, batch_data):
        self.model.eval()
        self.model_ema.eval()
        return self.model_ema(batch_data)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, ema_out=None):
        if ema_out is None:
            ema_out = self.teacher_prediction(batch_data)
        with torch.no_grad():
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(-predict * torch.log(predict + 1e-6), dim=1)
        for index, data in enumerate(batch_data):
            instance = (data, pseudo_label[index].item(), entropy[index].item())
            self.mem.add_instance(instance)
            self.current_instance += 1
            if self.current_instance % self.update_frequency == 0:
                self.update_model()
        return ema_out

    def update_model(self):
        self.model.train()
        self.model_ema.train()
        memory_data, ages = self.mem.get_memory()
        loss = None
        if memory_data:
            memory_data = torch.stack(memory_data)
            strong_augmented = self.transform(memory_data)
            ema_output = self.model_ema(memory_data)
            student_output = self.model(strong_augmented)
            weights = timeliness_reweighting(ages, student_output.device)
            loss = (softmax_entropy(student_output, ema_output) * weights).mean()
        if loss is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.last_loss = float(loss.detach().cpu())
            self.update_count += 1
        self.update_ema_variables(self.model_ema, self.model, self.nu)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model
