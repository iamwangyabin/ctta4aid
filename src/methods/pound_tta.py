"""PoundTTA: semantic-conditioned residual memory adaptation for PoundNet."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from src.models.poundnet import POUNDNET_COMMIT
from src.types import AdaptationStats, PredictionBatch

from .base import TTAMethod


@dataclass
class _MemoryItem:
    semantic_key: Any
    residual: Any
    pseudo_label: int
    reliability: float
    timestamp: int
    source_margin: float
    conditional_margin: float
    role: str = "train"


def _residual_adapter_class() -> Any:
    import torch.nn as nn

    class ResidualAdapter(nn.Module):
        def __init__(self, feature_dim: int, rank: int) -> None:
            super().__init__()
            self.down = nn.Linear(feature_dim, rank, bias=False)
            self.activation = nn.GELU()
            self.up = nn.Linear(rank, 1, bias=False)
            nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
            nn.init.zeros_(self.up.weight)

        def forward(self, residuals: Any) -> Any:
            return self.up(self.activation(self.down(residuals))).squeeze(-1)

    return ResidualAdapter


class PoundTTA(TTAMethod):
    """Adapt only PoundNet authenticity residuals with a causal guarded memory."""

    protocol_name = "predict_then_adapt"

    def __init__(self, model: Any, device: Any, config: dict[str, Any] | None = None) -> None:
        config = dict(config or {})
        config.setdefault("capture_initial_state", False)
        super().__init__(model, device, config)
        if not hasattr(self.model, "forward_pound_features"):
            raise TypeError("PoundTTA requires a PoundNet detector with feature decomposition")

        import torch

        self.model.eval()
        self.model.requires_grad_(False)
        self.adaptation_mode = str(self.config.get("adaptation_mode", "full")).lower()
        if self.adaptation_mode not in {"static", "full"}:
            raise ValueError("PoundTTA adaptation_mode must be static or full")
        self.semantic_temperature = float(self.config.get("semantic_temperature", 0.07))
        self.prototype_temperature = float(self.config.get("prototype_temperature", 0.07))
        self.prototype_logit_scale = float(self.config.get("prototype_logit_scale", 3.0))
        self.prototype_weight = float(self.config.get("prototype_weight", 1.0))
        self.adapter_weight = float(self.config.get("adapter_weight", 1.0))
        self.max_adaptation_weight = float(self.config.get("max_adaptation_weight", 0.5))
        self.gate_threshold = float(self.config.get("gate_threshold", 0.1))
        self.evidence_temperature = float(self.config.get("evidence_temperature", 1.0))
        self.max_logit_correction = float(self.config.get("max_logit_correction", 4.0))

        self.memory_size = int(self.config.get("memory_size", 256))
        self.class_capacity = self.memory_size // 2
        self.min_memory_per_class = int(self.config.get("min_memory_per_class", 8))
        self.support_saturation = int(self.config.get("support_saturation", 32))
        self.candidate_queue_size = int(self.config.get("candidate_queue_size", 512))
        self.promotion_delay = int(self.config.get("promotion_delay", 1))
        self.min_vote_ratio = float(self.config.get("min_vote_ratio", 2.0 / 3.0))
        self.candidate_reliability = float(self.config.get("candidate_reliability", 0.55))
        self.promotion_reliability = float(self.config.get("promotion_reliability", 0.5))
        self.guard_fraction = float(self.config.get("guard_fraction", 0.2))
        self.use_horizontal_flip = bool(self.config.get("use_horizontal_flip", True))
        self.use_scale_view = bool(self.config.get("use_scale_view", True))
        self.scale_ratio = float(self.config.get("scale_ratio", 0.875))
        self.age_weight = float(self.config.get("eviction_age_weight", 1.0))
        self.unreliability_weight = float(
            self.config.get("eviction_unreliability_weight", 1.0)
        )
        self.redundancy_weight = float(self.config.get("eviction_redundancy_weight", 1.0))

        if self.semantic_temperature <= 0.0 or self.prototype_temperature <= 0.0:
            raise ValueError("PoundTTA semantic and prototype temperatures must be positive")
        if self.memory_size < 4 or self.class_capacity < self.min_memory_per_class:
            raise ValueError("PoundTTA memory must reserve min_memory_per_class for both labels")
        if self.candidate_queue_size < 1 or self.promotion_delay < 1:
            raise ValueError("PoundTTA candidate queue and promotion delay must be positive")
        if not 0.5 <= self.min_vote_ratio <= 1.0:
            raise ValueError("PoundTTA min_vote_ratio must be in [0.5, 1]")
        if not 0.0 <= self.guard_fraction < 0.5:
            raise ValueError("PoundTTA guard_fraction must be in [0, 0.5)")
        if not 0.0 < self.scale_ratio <= 1.0:
            raise ValueError("PoundTTA scale_ratio must be in (0, 1]")

        self.memory: dict[int, list[_MemoryItem]] = {0: [], 1: []}
        self.candidate_queue: list[_MemoryItem] = []
        self._promotions = {0: 0, 1: 0}
        self._clock = 0
        self._pending: dict[str, Any] | None = None
        self._last_gate_mean = 0.0
        self._adapter_updates = 0
        self._adapter_rollbacks = 0

        self.residual_adapter: Any = None
        self._adapter_optimizer: Any = None
        self._initial_adapter_state: dict[str, Any] | None = None
        self._initial_adapter_optimizer_state: dict[str, Any] | None = None
        if self.adaptation_mode == "full":
            feature_dim = int(getattr(self.model, "feature_dim"))
            rank = int(self.config.get("adapter_rank", 16))
            if rank < 1:
                raise ValueError("PoundTTA adapter_rank must be positive")
            self.residual_adapter = _residual_adapter_class()(feature_dim, rank).to(device)
            self._adapter_optimizer = torch.optim.AdamW(
                self.residual_adapter.parameters(),
                lr=float(self.config.get("adapter_learning_rate", 1e-3)),
                weight_decay=float(self.config.get("adapter_weight_decay", 1e-4)),
            )
            self._initial_adapter_state = deepcopy(self.residual_adapter.state_dict())
            self._initial_adapter_optimizer_state = deepcopy(
                self._adapter_optimizer.state_dict()
            )
        self.adapter_update_frequency = int(
            self.config.get("adapter_update_frequency", 8)
        )
        self.adapter_steps = int(self.config.get("adapter_steps", 1))
        self.adapter_batch_size = int(self.config.get("adapter_batch_size", 64))
        self.adapter_min_per_class = int(self.config.get("adapter_min_per_class", 4))
        self.adapter_regularization = float(
            self.config.get("adapter_regularization", 1e-3)
        )
        self.adapter_guard_tolerance = float(
            self.config.get("adapter_guard_tolerance", 1e-4)
        )
        self.adapter_deviation_budget = float(
            self.config.get("adapter_deviation_budget", 2.0)
        )
        if min(self.adapter_update_frequency, self.adapter_steps, self.adapter_batch_size) < 1:
            raise ValueError("PoundTTA adapter update values must be positive")

    @property
    def trainable_parameters(self) -> int:
        if self.residual_adapter is None:
            return 0
        return sum(parameter.numel() for parameter in self.residual_adapter.parameters())

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "project_method_over_published_poundnet_source_detector",
            "source_method_commit": POUNDNET_COMMIT,
            "protocol_wrapper": self.protocol_name,
            "adaptation_mode": self.adaptation_mode,
            "semantic_role": "frozen_memory_index",
            "adaptive_role": "authenticity_residual_only",
            "intentional_changes": [
                "PoundNet paired prompts are decomposed into semantic midpoints and "
                "real/fake difference directions",
                "target samples enter candidate memory only after their pre-update prediction",
                "the CLIP backbone and PoundNet prompts remain frozen during deployment",
            ],
        }

    def _model_outputs(self, images: Any) -> dict[str, Any]:
        import torch

        with torch.no_grad():
            outputs = self.model.forward_pound_features(
                images.to(self.device, non_blocking=True),
                semantic_temperature=self.semantic_temperature,
            )
        required = {
            "source_logits",
            "semantic_keys",
            "residuals",
            "conditional_margin",
        }
        if not required.issubset(outputs):
            raise RuntimeError("PoundNet feature decomposition returned incomplete outputs")
        return {name: value.detach() for name, value in outputs.items()}

    def _memory_ready(self, minimum: int | None = None) -> bool:
        required = self.min_memory_per_class if minimum is None else minimum
        return all(len(self.memory[label]) >= required for label in (0, 1))

    def _class_prototype(self, semantic_keys: Any, label: int) -> tuple[Any, Any, float]:
        import torch
        import torch.nn.functional as functional

        entries = self.memory[label]
        keys = torch.stack([item.semantic_key for item in entries]).to(
            device=semantic_keys.device, dtype=semantic_keys.dtype
        )
        residuals = torch.stack([item.residual for item in entries]).to(
            device=semantic_keys.device, dtype=semantic_keys.dtype
        )
        reliability = torch.tensor(
            [item.reliability for item in entries],
            device=semantic_keys.device,
            dtype=semantic_keys.dtype,
        )
        similarity = semantic_keys @ keys.t()
        scaled = similarity / self.prototype_temperature
        scaled = scaled - scaled.max(dim=1, keepdim=True).values
        weights = scaled.exp() * reliability[None]
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        prototype = weights @ residuals
        coverage = ((similarity.max(dim=1).values + 1.0) * 0.5).clamp(0.0, 1.0)
        mean_reliability = float(reliability.mean().item())
        return functional.normalize(prototype, dim=-1), coverage, mean_reliability

    def _prototype_signal(
        self, semantic_keys: Any, residuals: Any, *, minimum: int | None = None
    ) -> tuple[Any, Any, bool]:
        import torch
        import torch.nn.functional as functional

        if not self._memory_ready(minimum):
            zeros = residuals.new_zeros(residuals.shape[0])
            return zeros, zeros, False
        real_prototype, real_coverage, real_reliability = self._class_prototype(
            semantic_keys, 0
        )
        fake_prototype, fake_coverage, fake_reliability = self._class_prototype(
            semantic_keys, 1
        )
        normalized_residuals = functional.normalize(residuals, dim=-1)
        margin = self.prototype_logit_scale * (
            (normalized_residuals * fake_prototype).sum(dim=1)
            - (normalized_residuals * real_prototype).sum(dim=1)
        )
        required = self.min_memory_per_class if minimum is None else minimum
        saturation = (
            max(required, self.support_saturation) if minimum is None else required
        )
        support = min(min(len(self.memory[0]), len(self.memory[1])) / saturation, 1.0)
        reliability = (real_reliability * fake_reliability) ** 0.5
        coverage = torch.minimum(real_coverage, fake_coverage)
        trust = coverage * support * reliability
        return margin, trust, True

    def predict(self, images: Any) -> PredictionBatch:
        import torch

        outputs = self._model_outputs(images)
        source_logits = outputs["source_logits"]
        source_margin = source_logits[:, 1].float() - source_logits[:, 0].float()
        correction = source_margin.new_zeros(source_margin.shape)
        gate = source_margin.new_zeros(source_margin.shape)
        prototype_margin = source_margin.new_zeros(source_margin.shape)
        trust = source_margin.new_zeros(source_margin.shape)
        if self.adaptation_mode == "full":
            prototype_margin, trust, ready = self._prototype_signal(
                outputs["semantic_keys"].float(), outputs["residuals"].float()
            )
            if ready:
                with torch.no_grad():
                    adapter_delta = self.residual_adapter(outputs["residuals"].float())
                source_probability = source_margin.sigmoid()
                prototype_probability = prototype_margin.sigmoid()
                source_uncertainty = 1.0 - (2.0 * source_probability - 1.0).abs()
                disagreement = (prototype_probability - source_probability).abs() * 2.0
                need = torch.maximum(source_uncertainty, disagreement).clamp(0.0, 1.0)
                evidence = torch.tanh(
                    prototype_margin.abs() / max(self.evidence_temperature, 1e-8)
                )
                raw_gate = (trust * need * evidence).clamp(0.0, 1.0)
                gate = torch.where(
                    raw_gate >= self.gate_threshold,
                    self.max_adaptation_weight * raw_gate,
                    torch.zeros_like(raw_gate),
                )
                correction = gate * (
                    self.prototype_weight * prototype_margin
                    + self.adapter_weight * adapter_delta
                )
                correction = correction.clamp(
                    -self.max_logit_correction, self.max_logit_correction
                )

        final_logits = source_logits.clone()
        final_logits[:, 0] -= (0.5 * correction).to(dtype=final_logits.dtype)
        final_logits[:, 1] += (0.5 * correction).to(dtype=final_logits.dtype)
        probabilities = final_logits.softmax(dim=1)
        self._last_gate_mean = float(gate.mean().item()) if gate.numel() else 0.0
        self._pending = {
            "source_logits": source_logits.detach(),
            "semantic_keys": outputs["semantic_keys"].detach(),
            "residuals": outputs["residuals"].detach(),
            "conditional_margin": outputs["conditional_margin"].detach(),
            "prototype_margin_mean": float(prototype_margin.mean().item()),
            "trust_mean": float(trust.mean().item()),
            "gate_mean": self._last_gate_mean,
        }
        return PredictionBatch(
            logits=final_logits.detach(),
            prob_fake=probabilities[:, 1].detach(),
            pred_label=probabilities.argmax(dim=1).detach(),
        )

    @staticmethod
    def _scaled_view(images: Any, ratio: float) -> Any:
        import torch.nn.functional as functional

        height, width = int(images.shape[-2]), int(images.shape[-1])
        crop_height = max(1, round(height * ratio))
        crop_width = max(1, round(width * ratio))
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        cropped = images[..., top : top + crop_height, left : left + crop_width]
        return functional.interpolate(
            cropped, size=(height, width), mode="bilinear", align_corners=False
        )

    @staticmethod
    def _margin_confidence(margin: Any) -> Any:
        return (2.0 * margin.float().sigmoid() - 1.0).abs()

    def _candidate_items(self, images: Any) -> list[_MemoryItem]:
        import torch

        if self._pending is None:
            raise RuntimeError("PoundTTA adapt requires a matching pre-update predict call")
        source_margin = (
            self._pending["source_logits"][:, 1].float()
            - self._pending["source_logits"][:, 0].float()
        )
        conditional_margin = self._pending["conditional_margin"].float()
        margins = [source_margin, conditional_margin]
        conditional_views = [conditional_margin]
        if self.use_horizontal_flip:
            flipped = self._model_outputs(torch.flip(images, dims=(-1,)))
            flipped_conditional = flipped["conditional_margin"].float()
            margins.append(flipped_conditional)
            conditional_views.append(flipped_conditional)
        if self.use_scale_view:
            scaled = self._model_outputs(self._scaled_view(images, self.scale_ratio))
            scaled_conditional = scaled["conditional_margin"].float()
            margins.append(scaled_conditional)
            conditional_views.append(scaled_conditional)

        stacked = torch.stack(margins, dim=1)
        votes = (stacked >= 0.0).float()
        fake_vote_ratio = votes.mean(dim=1)
        vote_consistency = torch.maximum(fake_vote_ratio, 1.0 - fake_vote_ratio)
        tie_label = stacked.mean(dim=1) >= 0.0
        pseudo_labels = torch.where(
            fake_vote_ratio == 0.5, tie_label, fake_vote_ratio > 0.5
        ).long()
        confidence = self._margin_confidence(stacked).mean(dim=1)
        base_conditional_probability = conditional_views[0].sigmoid()
        view_consistency = torch.ones_like(confidence)
        scale_consistency = torch.ones_like(confidence)
        conditional_index = 1
        if self.use_horizontal_flip:
            view_consistency = 1.0 - (
                base_conditional_probability
                - conditional_views[conditional_index].sigmoid()
            ).abs()
            conditional_index += 1
        if self.use_scale_view:
            scale_consistency = 1.0 - (
                base_conditional_probability
                - conditional_views[conditional_index].sigmoid()
            ).abs()
        reliability = (
            vote_consistency * confidence * view_consistency * scale_consistency
        ).clamp(0.0, 1.0)

        candidates = []
        for index in range(int(stacked.shape[0])):
            if (
                float(vote_consistency[index].item()) < self.min_vote_ratio
                or float(reliability[index].item()) < self.candidate_reliability
            ):
                continue
            candidates.append(
                _MemoryItem(
                    semantic_key=self._pending["semantic_keys"][index].detach().float(),
                    residual=self._pending["residuals"][index].detach().float(),
                    pseudo_label=int(pseudo_labels[index].item()),
                    reliability=float(reliability[index].item()),
                    timestamp=self._clock,
                    source_margin=float(source_margin[index].item()),
                    conditional_margin=float(conditional_margin[index].item()),
                )
            )
        return candidates

    def _promotion_role(self, label: int) -> str:
        if self.guard_fraction <= 0.0:
            return "train"
        interval = max(2, round(1.0 / self.guard_fraction))
        return "guard" if (self._promotions[label] + 1) % interval == 0 else "train"

    def _eviction_index(self, label: int) -> int:
        import torch
        import torch.nn.functional as functional

        entries = self.memory[label]
        keys = functional.normalize(torch.stack([item.semantic_key for item in entries]), dim=-1)
        similarity = keys @ keys.t()
        similarity.fill_diagonal_(-1.0)
        redundancy = ((similarity.max(dim=1).values + 1.0) * 0.5).clamp(0.0, 1.0)
        maximum_age = max(1, max(self._clock - item.timestamp for item in entries))
        scores = []
        role_counts = {
            role: sum(item.role == role for item in entries) for role in ("train", "guard")
        }
        for index, item in enumerate(entries):
            age = (self._clock - item.timestamp) / maximum_age
            score = (
                self.age_weight * age
                + self.unreliability_weight * (1.0 - item.reliability)
                + self.redundancy_weight * float(redundancy[index].item())
            )
            if role_counts[item.role] <= 1 and role_counts["train"] and role_counts["guard"]:
                score = float("-inf")
            scores.append(score)
        return max(range(len(scores)), key=scores.__getitem__)

    def _insert_memory(self, item: _MemoryItem) -> None:
        entries = self.memory[item.pseudo_label]
        entries.append(item)
        while len(entries) > self.class_capacity:
            entries.pop(self._eviction_index(item.pseudo_label))

    def _promote_candidates(self) -> int:
        import torch

        promoted = 0
        retained = []
        for item in self.candidate_queue:
            if self._clock - item.timestamp < self.promotion_delay:
                retained.append(item)
                continue
            temporal_consistency = 1.0
            if self._memory_ready(minimum=1):
                margin, trust, _ready = self._prototype_signal(
                    item.semantic_key[None], item.residual[None], minimum=1
                )
                probability = float(torch.sigmoid(margin[0]).item())
                memory_label = int(probability >= 0.5)
                if memory_label != item.pseudo_label:
                    continue
                temporal_consistency = max(probability, 1.0 - probability) * float(
                    trust[0].item()
                )
            reliability = item.reliability * temporal_consistency**0.5
            if reliability < self.promotion_reliability:
                continue
            item.reliability = reliability
            item.role = self._promotion_role(item.pseudo_label)
            self._insert_memory(item)
            self._promotions[item.pseudo_label] += 1
            promoted += 1
        self.candidate_queue = retained
        return promoted

    @staticmethod
    def _weighted_binary_loss(logits: Any, labels: Any, weights: Any) -> Any:
        import torch.nn.functional as functional

        losses = functional.binary_cross_entropy_with_logits(
            logits, labels.float(), reduction="none"
        )
        return (losses * weights).sum() / weights.sum().clamp_min(1e-8)

    def _adapter_entries(self, role: str) -> list[_MemoryItem]:
        per_class = max(1, self.adapter_batch_size // 2)
        selected = []
        for label in (0, 1):
            entries = [item for item in self.memory[label] if item.role == role]
            entries.sort(key=lambda item: (item.reliability, item.timestamp), reverse=True)
            selected.extend(entries[:per_class])
        return selected

    def _stack_adapter_entries(self, entries: list[_MemoryItem]) -> tuple[Any, Any, Any, Any]:
        import torch

        residuals = torch.stack([item.residual for item in entries]).to(self.device)
        source_margins = torch.tensor(
            [item.source_margin for item in entries], device=self.device
        )
        labels = torch.tensor(
            [item.pseudo_label for item in entries], device=self.device, dtype=torch.long
        )
        weights = torch.tensor(
            [item.reliability for item in entries], device=self.device
        )
        return residuals, source_margins, labels, weights

    def _guard_loss(self, entries: list[_MemoryItem]) -> tuple[float, float]:
        import torch

        residuals, source_margins, labels, weights = self._stack_adapter_entries(entries)
        with torch.no_grad():
            delta = self.residual_adapter(residuals)
            loss = self._weighted_binary_loss(source_margins + delta, labels, weights)
        return float(loss.item()), float(delta.abs().mean().item())

    def _update_adapter(self) -> tuple[float | None, bool | None]:
        if self.residual_adapter is None or self._clock % self.adapter_update_frequency:
            return None, None
        train_entries = self._adapter_entries("train")
        guard_entries = self._adapter_entries("guard")
        if any(
            sum(item.pseudo_label == label for item in train_entries)
            < self.adapter_min_per_class
            for label in (0, 1)
        ) or any(not any(item.pseudo_label == label for item in guard_entries) for label in (0, 1)):
            return None, None

        residuals, source_margins, labels, weights = self._stack_adapter_entries(
            train_entries
        )
        before_guard, _before_deviation = self._guard_loss(guard_entries)
        adapter_state = deepcopy(self.residual_adapter.state_dict())
        optimizer_state = deepcopy(self._adapter_optimizer.state_dict())
        losses = []
        self.residual_adapter.train()
        for _ in range(self.adapter_steps):
            delta = self.residual_adapter(residuals)
            loss = self._weighted_binary_loss(source_margins + delta, labels, weights)
            loss = loss + self.adapter_regularization * delta.square().mean()
            self._adapter_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self._adapter_optimizer.step()
            losses.append(float(loss.detach().item()))
        self.residual_adapter.eval()
        after_guard, after_deviation = self._guard_loss(guard_entries)
        accepted = (
            after_guard <= before_guard + self.adapter_guard_tolerance
            and after_deviation <= self.adapter_deviation_budget
        )
        if accepted:
            self._adapter_updates += 1
        else:
            self.residual_adapter.load_state_dict(adapter_state)
            self._adapter_optimizer.load_state_dict(optimizer_state)
            self._adapter_rollbacks += 1
        return sum(losses) / len(losses), accepted

    def adapt(self, images: Any) -> AdaptationStats:
        if self.adaptation_mode == "static":
            self._pending = None
            return AdaptationStats(
                selected=0,
                extra={
                    "adaptation_mode": "static",
                    "candidate_entries": 0,
                    "memory_entries": 0,
                    "gate_mean": 0.0,
                },
            )
        if self._pending is None:
            raise RuntimeError("PoundTTA adapt requires a matching predict call")
        self._clock += 1
        promoted = self._promote_candidates()
        candidates = self._candidate_items(images)
        self.candidate_queue.extend(candidates)
        if len(self.candidate_queue) > self.candidate_queue_size:
            self.candidate_queue = self.candidate_queue[-self.candidate_queue_size :]
        adapter_loss, adapter_accepted = self._update_adapter()
        pending = self._pending
        self._pending = None
        train_entries = sum(
            item.role == "train" for entries in self.memory.values() for item in entries
        )
        guard_entries = sum(
            item.role == "guard" for entries in self.memory.values() for item in entries
        )
        return AdaptationStats(
            loss=adapter_loss,
            selected=len(candidates),
            extra={
                "adaptation_mode": "full",
                "candidate_entries": len(self.candidate_queue),
                "candidates_selected": len(candidates),
                "candidates_promoted": promoted,
                "memory_entries": train_entries + guard_entries,
                "memory_real": len(self.memory[0]),
                "memory_fake": len(self.memory[1]),
                "memory_train": train_entries,
                "memory_guard": guard_entries,
                "memory_ready": self._memory_ready(),
                "prototype_margin_mean": pending["prototype_margin_mean"],
                "trust_mean": pending["trust_mean"],
                "gate_mean": pending["gate_mean"],
                "adapter_update_accepted": adapter_accepted,
                "adapter_updates": self._adapter_updates,
                "adapter_rollbacks": self._adapter_rollbacks,
            },
        )

    def discard_pending_prediction(self) -> None:
        self._pending = None

    def reset(self) -> None:
        self.memory = {0: [], 1: []}
        self.candidate_queue.clear()
        self._promotions = {0: 0, 1: 0}
        self._clock = 0
        self._pending = None
        self._last_gate_mean = 0.0
        self._adapter_updates = 0
        self._adapter_rollbacks = 0
        if self.residual_adapter is not None:
            self.residual_adapter.load_state_dict(deepcopy(self._initial_adapter_state))
            self._adapter_optimizer.load_state_dict(
                deepcopy(self._initial_adapter_optimizer_state)
            )


__all__ = ["PoundTTA"]
