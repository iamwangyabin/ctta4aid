"""Single-device OST meta-training core.

Derived from ``Model.optimize`` in ``trainer/whole_model.py`` at OST commit
``1e4518b9e560baf9c5693f13a402fa5d7104190f``. The upstream repository has no
repository-wide software license. This extraction removes DataParallel tensor
replication and accepts framework-created support/query episodes; it preserves
the released one-step second-order update, AM-Softmax objective, and Adam outer
update.
"""

from __future__ import annotations

from typing import Any

from .am_softmax import AMSoftmaxLoss
from .inner_loop_optimizers import LSLRGradientDescentLearningRule


class OSTMetaTrainingCore:
    """Optimize a MetaXception initialization through one-step OST episodes."""

    def __init__(
        self,
        model: Any,
        device: Any,
        *,
        task_learning_rate: float = 0.0005,
        outer_learning_rate: float = 0.0002,
        second_order: bool = True,
        enable_inner_loop_optimizable_bn_params: bool = True,
        margin: float = 0.45,
        scale: float = 30.0,
    ) -> None:
        import torch

        self.model = model
        self.device = device
        self.second_order = second_order
        self.enable_inner_loop_optimizable_bn_params = (
            enable_inner_loop_optimizable_bn_params
        )
        self.inner_loop_optimizer = LSLRGradientDescentLearningRule(
            device=device,
            init_learning_rate=task_learning_rate,
            total_num_inner_loop_steps=1,
            use_learnable_learning_rates=True,
        )
        self.inner_loop_optimizer.initialise(self._inner_loop_parameters())
        self.inner_loop_optimizer.to(device)
        self.criterion = AMSoftmaxLoss(gamma=0.0, m=margin, s=scale, t=1.0)
        # The released optimizer updates only MetaXception, not the nominally
        # learnable inner-loop rates.
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=outer_learning_rate, betas=(0.9, 0.999)
        )

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

    def _updated_parameters(
        self, support: Any, support_labels: Any, *, create_graph: bool
    ) -> tuple[dict[str, Any], Any]:
        import torch

        parameters = self._inner_loop_parameters()
        scores, _ = self.model.forward(
            x=support,
            params=parameters,
            training=True,
            backup_running_statistics=True,
            num_step=0,
        )
        support_loss = self.criterion(scores, support_labels).mean()
        gradients = torch.autograd.grad(
            support_loss,
            tuple(parameters.values()),
            create_graph=create_graph,
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
        return updated, support_loss

    def train_step(
        self,
        support: Any,
        support_labels: Any,
        query: Any,
        query_labels: Any,
    ) -> dict[str, Any]:
        self.model.train()
        fast_parameters, support_loss = self._updated_parameters(
            support, support_labels, create_graph=self.second_order
        )
        query_scores, _ = self.model.forward(
            x=query,
            params=fast_parameters,
            training=True,
            backup_running_statistics=True,
            num_step=0,
        )
        query_loss = self.criterion(query_scores, query_labels).mean()
        self.optimizer.zero_grad(set_to_none=True)
        query_loss.backward()
        self.optimizer.step()
        return {
            "support_loss": support_loss.detach(),
            "query_loss": query_loss.detach(),
            "query_scores": query_scores.detach(),
        }

    def adapt_and_predict(
        self,
        support: Any,
        support_labels: Any,
        query: Any,
    ) -> tuple[Any, Any]:
        import torch

        fast_parameters, support_loss = self._updated_parameters(
            support, support_labels, create_graph=False
        )
        with torch.no_grad():
            query_scores, _ = self.model.forward(
                x=query,
                params=fast_parameters,
                training=False,
                backup_running_statistics=False,
                num_step=0,
            )
        if hasattr(self.model, "restore_backup_stats"):
            self.model.restore_backup_stats()
        return query_scores, support_loss.detach()
