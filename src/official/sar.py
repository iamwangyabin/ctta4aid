"""SAR primitives retained from the authors' public implementation.

Vendored from mr-eggplant/SAR commit
20f6e24b17525f34503510afccedc0629b67b7c4 (``sar.py`` and ``sam.py``).
The upstream repository is distributed under the BSD 3-Clause License.

The framework wrapper owns the CLIP visual-tower parameter selection, Arrow
streaming, and result serialization.  The empty-reliable-set branch restores
SAM parameters rather than propagating a NaN update; this is recorded by the
wrapper as an intentional runtime guard.
"""

from __future__ import annotations

from typing import Any

import torch


def update_ema(ema: float | None, new_data: float) -> float:
    """Update SAR's exponential moving average of the reliable entropy."""

    if ema is None:
        return new_data
    return 0.9 * ema + 0.1 * new_data


def softmax_entropy(logits: Any) -> Any:
    """Entropy of a categorical prediction, as used by the SAR release."""

    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class SAM(torch.optim.Optimizer):
    """The SAM optimizer shipped in the pinned SAR repository."""

    def __init__(
        self,
        params: Any,
        base_optimizer: type[torch.optim.Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs: Any,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> bool:
        grad_norm = self._grad_norm()
        if grad_norm is None:
            return False
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                self.state[parameter]["old_p"] = parameter.data.clone()
                perturbation = (
                    (torch.pow(parameter, 2) if group["adaptive"] else 1.0)
                    * parameter.grad
                    * scale.to(parameter)
                )
                parameter.add_(perturbation)
        if zero_grad:
            self.zero_grad()
        return True

    @torch.no_grad()
    def restore(self, zero_grad: bool = False) -> None:
        """Restore the pre-perturbation state when SAR rejects its second pass."""

        for group in self.param_groups:
            for parameter in group["params"]:
                previous = self.state[parameter].pop("old_p", None)
                if previous is not None:
                    parameter.copy_(previous)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        self.restore()
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self) -> Any | None:
        gradients = []
        device = None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                device = parameter.device if device is None else device
                factor = torch.abs(parameter) if group["adaptive"] else 1.0
                gradients.append((factor * parameter.grad).norm(p=2).to(device))
        if not gradients:
            return None
        return torch.norm(torch.stack(gradients), p=2)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


@torch.enable_grad()
def forward_and_adapt_sar(
    images: Any,
    model: Any,
    optimizer: SAM,
    *,
    margin: float,
    reset_constant: float,
    ema: float | None,
) -> tuple[Any, float | None, bool, int, int, float | None]:
    """Run one SAR update and return pre-update logits plus update statistics."""

    optimizer.zero_grad()
    outputs = model(images)
    first_entropy = softmax_entropy(outputs)
    first_reliable = torch.where(first_entropy < margin)[0]
    if first_reliable.numel() == 0:
        return outputs, ema, False, 0, 0, None

    first_loss = first_entropy[first_reliable].mean()
    if not torch.isfinite(first_loss):
        return outputs, ema, False, int(first_reliable.numel()), 0, None
    first_loss.backward()
    if not optimizer.first_step(zero_grad=True):
        return outputs, ema, False, int(first_reliable.numel()), 0, None

    second_entropy = softmax_entropy(model(images))[first_reliable]
    second_reliable = torch.where(second_entropy < margin)[0]
    if second_reliable.numel() == 0:
        optimizer.restore(zero_grad=True)
        return outputs, ema, False, int(first_reliable.numel()), 0, None

    second_loss = second_entropy[second_reliable].mean()
    if not torch.isfinite(second_loss):
        optimizer.restore(zero_grad=True)
        return outputs, ema, False, int(first_reliable.numel()), 0, None
    updated_ema = update_ema(ema, float(second_loss.detach().item()))
    second_loss.backward()
    optimizer.second_step(zero_grad=True)
    reset = updated_ema < reset_constant
    return (
        outputs,
        updated_ema,
        reset,
        int(first_reliable.numel()),
        int(second_reliable.numel()),
        float(second_loss.detach().item()),
    )
