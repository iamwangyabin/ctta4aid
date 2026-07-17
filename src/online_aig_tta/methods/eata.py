"""Framework wrapper around the pinned authors' EATA core."""

from __future__ import annotations

import math
from typing import Any

from online_aig_tta.types import AdaptationStats

from .base import TTAMethod
from .utils import build_optimizer


class EATA(TTAMethod):
    """Run official EATA adaptation while the framework owns protocol I/O."""

    def __init__(
        self,
        model: Any,
        device: Any,
        config: dict[str, Any] | None = None,
        fishers: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model, device, config)
        from online_aig_tta.official import eata as official_eata

        self.official_module = official_eata
        official_eata.configure_model(self.model)
        parameters, self.official_parameter_names = official_eata.collect_params(self.model)
        if not parameters:
            raise RuntimeError("Official EATA requires at least one BatchNorm2d layer")
        normalized_fishers = self._normalize_fishers(fishers)
        if bool(self.config.get("require_fisher", True)) and normalized_fishers is None:
            raise RuntimeError(
                "Full EATA requires source Fisher information. Train with "
                "training.compute_fisher=true; require_fisher=false is only ETA."
            )

        self.optimizer = build_optimizer(parameters, self.config)
        self.core = official_eata.EATA(
            self.model,
            self.optimizer,
            fishers=normalized_fishers,
            fisher_alpha=float(self.config.get("fisher_alpha", 2000.0)),
            steps=int(self.config.get("steps", 1)),
            episodic=bool(self.config.get("episodic", False)),
            e_margin=float(
                self.config.get(
                    "e_margin",
                    self.config.get("entropy_margin", math.log(2.0) * 0.4),
                )
            ),
            d_margin=float(
                self.config.get(
                    "d_margin", self.config.get("redundancy_margin", 0.05)
                )
            ),
        )

    def _normalize_fishers(
        self, fishers: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if not fishers:
            return None
        normalized = {}
        current_state = self.model.state_dict()
        for name, value in fishers.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                fisher, source_parameter = value
            else:
                fisher = value
                source_parameter = current_state[name]
            normalized[name] = [
                fisher.to(self.device),
                source_parameter.to(self.device),
            ]
        return normalized

    @property
    def reproduction_metadata(self) -> dict[str, Any]:
        return {
            "level": "vendored_official_core_with_binary_protocol_wrapper",
            "official_commit": "f739b3668cc7617e9b9f1979c1a358497a3472c3",
            "official_core": "online_aig_tta.official.eata",
            "numerical_validation": "not_run_requires_official_data_and_weights",
            "protocol_wrapper": "predict_then_adapt",
            "intentional_changes": [
                "framework separates prediction from official adaptation",
                "entropy margin scales from 1000 to 2 classes",
                "source-validation Fisher is loaded from the common checkpoint",
                "framework reset also clears probability EMA and counters",
            ],
        }

    def adapt(self, images: Any) -> AdaptationStats:
        images = images.to(self.device, non_blocking=True)
        reliable_before = int(self.core.num_samples_update_1)
        selected_before = int(self.core.num_samples_update_2)
        self.core(images)
        reliable = int(self.core.num_samples_update_1) - reliable_before
        selected = int(self.core.num_samples_update_2) - selected_before
        return AdaptationStats(
            loss=None,
            selected=selected,
            extra={
                "reliable": reliable,
                "reliable_total": int(self.core.num_samples_update_1),
                "selected_total": int(self.core.num_samples_update_2),
                "fisher_enabled": self.core.fishers is not None,
            },
        )

    def reset(self) -> None:
        self.core.reset()
        self.core.current_model_probs = None
        self.core.num_samples_update_1 = 0
        self.core.num_samples_update_2 = 0
