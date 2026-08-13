"""Single-device extraction of the authors' released OST inference loop.

Derived from ``trainer/whole_model.py`` at OST commit
``1e4518b9e560baf9c5693f13a402fa5d7104190f``. The framework supplies the
pseudo sample and template, replacing the unavailable SimSwap runtime while
preserving Algorithm 1's AM-Softmax support loss and one-step fast weights.
The upstream OST repository declares no software license.
"""

from __future__ import annotations

from typing import Any

from .am_softmax import AMSoftmaxLoss
from .inner_loop_optimizers import LSLRGradientDescentLearningRule


class OSTInferenceCore:
    """Apply OST fast weights without mutating the detector parameters."""

    def __init__(
        self,
        model: Any,
        device: Any,
        *,
        learning_rate: float = 0.0005,
        steps: int = 1,
        second_order: bool = True,
        enable_inner_loop_optimizable_bn_params: bool = True,
    ) -> None:
        if steps != 1:
            raise ValueError("The released OST inference protocol uses exactly one step")
        self.model = model
        self.device = device
        self.steps = steps
        self.second_order = second_order
        self.enable_inner_loop_optimizable_bn_params = (
            enable_inner_loop_optimizable_bn_params
        )
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(
            device=device,
            init_learning_rate=learning_rate,
            total_num_inner_loop_steps=steps,
            use_learnable_learning_rates=True,
        )
        self.inner_loop_optimizer.initialise(
            names_weights_dict=self._inner_loop_parameters()
        )
        self.inner_loop_optimizer.to(device)
        self.criterion = AMSoftmaxLoss(gamma=0.0, m=0.45, s=30, t=1.0)

    def _inner_loop_parameters(self) -> dict[str, Any]:
        parameters = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if (
                not self.enable_inner_loop_optimizable_bn_params
                and "norm_layer" in name
            ):
                continue
            parameters[name] = parameter
        if not parameters:
            raise RuntimeError("OST found no inner-loop parameters")
        return parameters

    def _updated_parameters(self, support: Any, labels: Any) -> tuple[dict[str, Any], Any]:
        import torch

        parameters = self._inner_loop_parameters()
        scores, _ = self.model.forward(
            x=support,
            params=parameters,
            training=True,
            backup_running_statistics=True,
            num_step=0,
        )
        loss = self.criterion(scores, labels).mean()
        gradients = torch.autograd.grad(
            loss,
            tuple(parameters.values()),
            create_graph=self.second_order,
            allow_unused=True,
        )
        usable_parameters = {
            name: parameter
            for (name, parameter), gradient in zip(parameters.items(), gradients)
            if gradient is not None
        }
        usable_gradients = {
            name: gradient
            for name, gradient in zip(parameters, gradients)
            if gradient is not None
        }
        updated = self.inner_loop_optimizer.update_params(
            names_weights_dict=usable_parameters,
            names_grads_wrt_params_dict=usable_gradients,
            num_step=0,
        )
        for name, parameter in parameters.items():
            updated.setdefault(name, parameter)
        return updated, loss

    def infer(
        self,
        target: Any,
        pseudo_sample: Any,
        template: Any,
        template_label: Any,
    ) -> tuple[Any, Any]:
        import torch

        support = torch.cat((pseudo_sample, template), dim=0)
        pseudo_label = torch.ones(
            pseudo_sample.shape[0], dtype=torch.long, device=self.device
        )
        labels = torch.cat((pseudo_label, template_label), dim=0)
        fast_parameters, support_loss = self._updated_parameters(support, labels)
        with torch.no_grad():
            scores, _ = self.model.forward(
                x=target,
                params=fast_parameters,
                training=False,
                backup_running_statistics=False,
                num_step=0,
            )
        if hasattr(self.model, "restore_backup_stats"):
            self.model.restore_backup_stats()
        return scores, support_loss.detach()
