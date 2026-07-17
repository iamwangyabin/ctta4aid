from __future__ import annotations

import csv
import json
import random
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from online_aig_tta.types import StreamBatch

from .metrics import MetricAccumulator


def _sync_if_cuda(device: Any) -> None:
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(device)
    except ImportError:
        return


def _reset_peak_memory(device: Any) -> None:
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except ImportError:
        return


def _peak_memory_mb(device: Any) -> float | None:
    try:
        import torch

        if str(device).startswith("cuda") and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated(device) / (1024**2))
    except ImportError:
        pass
    return None


class OnlineEvaluator:
    """Strict Predict-Then-Adapt evaluator.

    The method receives only ``batch.images``. Hidden labels are routed directly
    to ``MetricAccumulator`` after prediction and never enter ``adapt``.
    """

    def __init__(self, *, threshold: float = 0.5, curve_window_batches: int = 20) -> None:
        self.threshold = threshold
        self.curve_window_batches = curve_window_batches

    def run(
        self,
        method: Any,
        stream: Iterable[StreamBatch],
        *,
        on_domain_end: Any = None,
    ) -> dict[str, Any]:
        accumulator = MetricAccumulator(threshold=self.threshold)
        batch_stats: list[dict[str, Any]] = []
        sample_manifest: list[dict[str, Any]] = []
        domain_end_evaluations: list[dict[str, Any]] = []
        active_domain: str | None = None
        _reset_peak_memory(method.device)

        for batch_index, batch in enumerate(stream):
            if active_domain is None:
                active_domain = batch.domain
            elif batch.domain != active_domain:
                if on_domain_end is not None:
                    domain_end_evaluations.append(
                        {
                            "after_domain": active_domain,
                            "evaluation": on_domain_end(method, active_domain),
                        }
                    )
                active_domain = batch.domain
            _sync_if_cuda(method.device)
            predict_start = time.perf_counter()
            predictions = method.predict(batch.images)
            _sync_if_cuda(method.device)
            predict_elapsed_ms = (time.perf_counter() - predict_start) * 1000.0
            accumulator.update(batch.hidden_labels, predictions.prob_fake, batch.domain)
            manifest_start = len(sample_manifest)
            sample_manifest.extend(
                {
                    "position": manifest_start + offset,
                    "batch": batch_index,
                    "domain": batch.domain,
                    "sample_id": sample_id,
                }
                for offset, sample_id in enumerate(batch.sample_ids)
            )

            _sync_if_cuda(method.device)
            start = time.perf_counter()
            adaptation = method.adapt(batch.images)
            _sync_if_cuda(method.device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            batch_stats.append(
                {
                    "batch": batch_index,
                    "domain": batch.domain,
                    "samples": int(np.asarray(batch.hidden_labels.cpu()).size)
                    if hasattr(batch.hidden_labels, "cpu")
                    else int(np.asarray(batch.hidden_labels).size),
                    "predict_ms": predict_elapsed_ms,
                    "adapt_ms": elapsed_ms,
                    "total_ms": predict_elapsed_ms + elapsed_ms,
                    "loss": adaptation.loss,
                    "selected": adaptation.selected,
                    **adaptation.extra,
                }
            )

        if active_domain is not None and on_domain_end is not None:
            domain_end_evaluations.append(
                {
                    "after_domain": active_domain,
                    "evaluation": on_domain_end(method, active_domain),
                }
            )

        summary = accumulator.summary()
        predict_times = [item["predict_ms"] for item in batch_stats]
        adapt_times = [item["adapt_ms"] for item in batch_stats]
        total_times = [item["total_ms"] for item in batch_stats]
        summary["efficiency"] = {
            "trainable_parameters": int(method.trainable_parameters),
            "mean_predict_ms_per_batch": (
                float(np.mean(predict_times)) if predict_times else 0.0
            ),
            "mean_adapt_ms_per_batch": float(np.mean(adapt_times)) if adapt_times else 0.0,
            "mean_total_ms_per_batch": (
                float(np.mean(total_times)) if total_times else 0.0
            ),
            "peak_memory_mb": _peak_memory_mb(method.device),
            "batches": len(batch_stats),
        }
        reproduction = dict(
            getattr(
                method,
                "reproduction_metadata",
                {
                    "level": "external_or_test_method",
                    "protocol_wrapper": "predict_then_adapt",
                    "intentional_changes": [],
                },
            )
        )
        if hasattr(method, "source_checkpoint_identity"):
            reproduction["source_checkpoint"] = method.source_checkpoint_identity
        if hasattr(method, "config"):
            reproduction["effective_method_config"] = method.config
        return {
            "summary": summary,
            "curve": accumulator.sliding_curve(self.curve_window_batches),
            "batch_stats": batch_stats,
            "sample_manifest": sample_manifest,
            "domain_end_evaluations": domain_end_evaluations,
            "reproduction": reproduction,
        }


def evaluate_without_adaptation(
    method: Any,
    stream: Iterable[StreamBatch],
    *,
    threshold: float = 0.5,
    include_manifest: bool = False,
    evaluation_seed: int | None = None,
) -> dict[str, Any]:
    accumulator = MetricAccumulator(threshold=threshold)
    sample_manifest: list[dict[str, Any]] = []
    with _preserve_random_state(evaluation_seed):
        for batch_index, batch in enumerate(stream):
            predictions = method.predict(batch.images)
            accumulator.update(batch.hidden_labels, predictions.prob_fake, batch.domain)
            if hasattr(method, "discard_pending_prediction"):
                method.discard_pending_prediction()
            if include_manifest:
                manifest_start = len(sample_manifest)
                sample_manifest.extend(
                    {
                        "position": manifest_start + offset,
                        "batch": batch_index,
                        "domain": batch.domain,
                        "sample_id": sample_id,
                    }
                    for offset, sample_id in enumerate(batch.sample_ids)
                )
    summary = accumulator.summary()
    if include_manifest:
        summary["sample_manifest"] = sample_manifest
    return summary


@contextmanager
def _preserve_random_state(seed: int | None):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_module = None
    torch_state = None
    cuda_states = None
    try:
        import torch

        torch_module = torch
        torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            cuda_states = torch.cuda.get_rng_state_all()
    except ImportError:
        pass

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        if torch_module is not None:
            torch_module.manual_seed(seed)
            if torch_module.cuda.is_available():
                torch_module.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch_module is not None and torch_state is not None:
            torch_module.set_rng_state(torch_state)
            if cuda_states is not None:
                torch_module.cuda.set_rng_state_all(cuda_states)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def save_evaluation(result: dict[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        compact_result = {
            key: value
            for key, value in result.items()
            if key not in {"sample_manifest", "final_holdout_manifest"}
        }
        json.dump(compact_result, handle, indent=2, ensure_ascii=False, default=_json_default)

    for key, filename in (
        ("curve", "online_curve.csv"),
        ("batch_stats", "batch_stats.csv"),
        ("sample_manifest", "sample_manifest.csv"),
        ("holdout_matrix", "holdout_matrix.csv"),
        ("final_holdout_manifest", "final_holdout_manifest.csv"),
    ):
        rows = result.get(key, [])
        if not rows:
            continue
        fieldnames = sorted({field for row in rows for field in row})
        with (output / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
