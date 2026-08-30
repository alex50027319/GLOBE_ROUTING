"""Lightweight deployment-cost measurements for routing policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from pathlib import Path
from time import perf_counter_ns
import tracemalloc
from typing import Any, Callable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from .evaluator import RoutingPolicy


@dataclass(frozen=True)
class PolicyCost:
    parameter_count: int
    input_bytes: int
    mean_latency_ms: float
    peak_python_memory_bytes: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    serialized_model_bytes: int

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def observation_bytes(
    observation: dict[str, NDArray[np.generic]],
) -> int:
    return int(sum(value.nbytes for value in observation.values()))


def parameter_count(model: nn.Module | None) -> int:
    if model is None:
        return 0
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _synchronize(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    elif resolved.type == "mps":
        torch.mps.synchronize()


def measure_policy_cost(
    policy: RoutingPolicy,
    observation: dict[str, NDArray[np.generic]],
    *,
    model: nn.Module | None = None,
    input_observation: dict[str, NDArray[np.generic]] | None = None,
    device: torch.device | str = "cpu",
    warmup: int = 10,
    repeats: int = 100,
    serialized_model_path: Path | None = None,
    prepare: Callable[[], None] | None = None,
) -> PolicyCost:
    """Measure steady-state action latency and Python allocation peak."""

    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")
    policy.reset(0)
    if prepare is not None:
        prepare()
    for _ in range(warmup):
        policy.act(observation)
    _synchronize(device)

    tracemalloc.start()
    timings: list[float] = []
    for _ in range(repeats):
        started = perf_counter_ns()
        policy.act(observation)
        _synchronize(device)
        timings.append((perf_counter_ns() - started) / 1_000_000.0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return PolicyCost(
        parameter_count=parameter_count(model),
        input_bytes=observation_bytes(input_observation or observation),
        mean_latency_ms=float(mean(timings)),
        peak_python_memory_bytes=int(peak),
        latency_p50_ms=float(np.percentile(timings, 50)),
        latency_p95_ms=float(np.percentile(timings, 95)),
        latency_p99_ms=float(np.percentile(timings, 99)),
        serialized_model_bytes=(
            int(serialized_model_path.stat().st_size)
            if serialized_model_path is not None and serialized_model_path.is_file()
            else 0
        ),
    )
