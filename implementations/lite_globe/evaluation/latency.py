"""Reproducible cold, component, and end-to-end policy latency benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter_ns
from typing import Callable

import numpy as np
import torch
from numpy.typing import NDArray

from ..models.policy_adapter import StudentPolicyAdapter
from ..models.student_policy import RiskSwitchLiteGlobePStudentPolicy
from ..models.tensor_observation import observation_to_tensors
from ..models.masking import masked_logits, masked_softmax


def synchronize(device: torch.device | str) -> None:
    resolved = torch.device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    elif resolved.type == "mps":
        torch.mps.synchronize()


@dataclass(frozen=True)
class LatencyBenchmark:
    variant: str
    component: str
    device: str
    warmup: int
    repeats: int
    cold_start_ms: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    decisions_per_second: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def benchmark_callable(
    function: Callable[[], object], *, variant: str, component: str,
    device: torch.device | str, warmup: int = 50, repeats: int = 500,
) -> LatencyBenchmark:
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")
    with torch.inference_mode():
        synchronize(device)
        started = perf_counter_ns()
        function()
        synchronize(device)
        cold = (perf_counter_ns() - started) / 1_000_000.0
        for _ in range(warmup):
            function()
        synchronize(device)
        values: list[float] = []
        for _ in range(repeats):
            synchronize(device)
            started = perf_counter_ns()
            function()
            synchronize(device)
            values.append((perf_counter_ns() - started) / 1_000_000.0)
    mean = float(np.mean(values))
    return LatencyBenchmark(
        variant=variant, component=component, device=str(torch.device(device)),
        warmup=warmup, repeats=repeats, cold_start_ms=float(cold), mean_ms=mean,
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        decisions_per_second=1000.0 / mean if mean > 0 else float("inf"),
    )


def profile_student_policy(
    policy: StudentPolicyAdapter,
    observation: dict[str, NDArray[np.generic]], *, variant: str,
    warmup: int = 50, repeats: int = 500,
) -> list[LatencyBenchmark]:
    """Benchmark independently reproducible stages and total decision latency."""

    device = policy.device
    tensors = observation_to_tensors(observation, device=device)
    model = policy.model
    if isinstance(model, RiskSwitchLiteGlobePStudentPolicy):
        model_call = lambda: model.decide(tensors)
    else:
        model_call = lambda: model(tensors)
    with torch.inference_mode():
        cached = model_call()
    probabilities = cached.output.probabilities if hasattr(cached, "output") else cached.probabilities
    return [
        benchmark_callable(
            lambda: observation_to_tensors(observation, device=device),
            variant=variant, component="preprocess", device=device,
            warmup=warmup, repeats=repeats,
        ),
        benchmark_callable(
            model_call, variant=variant, component="model", device=device,
            warmup=warmup, repeats=repeats,
        ),
        benchmark_callable(
            lambda: int(torch.argmax(probabilities).item()),
            variant=variant, component="action_extraction", device=device,
            warmup=warmup, repeats=repeats,
        ),
        benchmark_callable(
            lambda: policy.act_with_metadata(observation),
            variant=variant, component="end_to_end_policy", device=device,
            warmup=warmup, repeats=repeats,
        ),
    ]


@torch.inference_mode()
def legacy_repeated_switchglobe_action(
    policy: StudentPolicyAdapter,
    observation: dict[str, NDArray[np.generic]],
) -> int:
    """Reproduce the pre-fusion four-forward timed action for comparison."""

    model = policy.model
    if not isinstance(model, RiskSwitchLiteGlobePStudentPolicy):
        raise TypeError("legacy comparison requires SwitchGLOBE")
    tensors = observation_to_tensors(observation, device=policy.device)
    for value in model.diagnostics(tensors).values():
        float(value)
    normal = model.normal_policy(tensors)
    predictive = model.predictive_policy(tensors)
    mask = tensors["action_mask"].unsqueeze(0)
    normal_logits = normal.logits.unsqueeze(0)
    predictive_logits = predictive.logits.unsqueeze(0)
    switch = model._switch_mask(
        tensors, action_mask=mask, normal_logits=normal_logits,
        predictive_logits=predictive_logits, unbatched=True,
    )
    probabilities = masked_softmax(
        torch.where(switch.unsqueeze(-1), predictive_logits, normal_logits),
        mask,
    ).squeeze(0)
    action = int(torch.argmax(probabilities).item())
    if policy.force_forward_if_available and action == model.drop_action:
        candidate_mask = tensors["action_mask"][: model.max_nodes].bool()
        if torch.any(candidate_mask):
            action = int(torch.argmax(
                probabilities[: model.max_nodes].masked_fill(
                    ~candidate_mask, -1.0
                )
            ).item())
    return action
