"""Randomized-block latency measurement with raw repetition preservation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable

import numpy as np
import torch

from .latency import synchronize


@dataclass(frozen=True)
class InterleavedSpec:
    variant: str
    component: str
    device: torch.device
    function: Callable[[], object]


def _summary(
    spec: InterleavedSpec, values: list[float], *, warmup: int,
    cold_values: list[float] | None = None,
) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    cold = np.asarray(cold_values or [], dtype=np.float64)
    mean = float(array.mean())
    return {
        "variant": spec.variant,
        "component": spec.component,
        "device": str(spec.device),
        "warmup": warmup,
        "repeats": len(values),
        "mean_ms": mean,
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "p99_ms": float(np.percentile(array, 99)),
        "decisions_per_second": 1000.0 / mean if mean > 0 else float("inf"),
        "cold_start_repeats": int(cold.size),
        "cold_start_mean_ms": float(cold.mean()) if cold.size else None,
        "cold_start_p50_ms": float(np.percentile(cold, 50)) if cold.size else None,
        "cold_start_p95_ms": float(np.percentile(cold, 95)) if cold.size else None,
        "cold_start_p99_ms": float(np.percentile(cold, 99)) if cold.size else None,
    }


def benchmark_interleaved(
    specs: list[InterleavedSpec], *, warmup: int, repeats: int,
    order_seed: int,
    cold_samples: dict[tuple[str, str, str], list[float]] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Benchmark every spec once per block in deterministic shuffled order."""

    if not specs:
        raise ValueError("at least one latency spec is required")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")
    devices = {str(spec.device) for spec in specs}
    if len(devices) != 1:
        raise ValueError("one interleaved benchmark may contain only one device")
    identities = [(spec.variant, spec.component, str(spec.device)) for spec in specs]
    if len(identities) != len(set(identities)):
        raise ValueError("latency spec identities must be unique")
    rng = np.random.default_rng(order_seed)
    with torch.inference_mode():
        for _ in range(warmup):
            for index in rng.permutation(len(specs)):
                specs[int(index)].function()
        synchronize(specs[0].device)
        values: list[list[float]] = [[] for _ in specs]
        raw: list[dict[str, object]] = []
        for block in range(repeats):
            for position, index_value in enumerate(rng.permutation(len(specs))):
                index = int(index_value)
                spec = specs[index]
                synchronize(spec.device)
                started = perf_counter_ns()
                spec.function()
                synchronize(spec.device)
                elapsed = (perf_counter_ns() - started) / 1_000_000.0
                values[index].append(elapsed)
                raw.append({
                    "block": block,
                    "order_position": position,
                    "variant": spec.variant,
                    "component": spec.component,
                    "device": str(spec.device),
                    "latency_ms": elapsed,
                })
    cold_samples = cold_samples or {}
    summaries = [
        _summary(
            spec, values[index], warmup=warmup,
            cold_values=cold_samples.get(
                (spec.variant, spec.component, str(spec.device))
            ),
        )
        for index, spec in enumerate(specs)
    ]
    return summaries, raw
