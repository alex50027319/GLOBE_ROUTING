"""Deterministic episode runner and baseline metric aggregation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from ..env.fanet_env import FanetRoutingEnv


class RoutingPolicy(Protocol):
    def reset(self, seed: int | None = None) -> None: ...

    def act(self, observation: dict[str, NDArray[np.generic]]) -> int: ...


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    delivered: bool
    dropped: bool
    drop_reason: str | None
    steps: int
    hop_count: int
    total_reward: float
    initially_connected: bool
    initial_shortest_hops: int | None
    path_stretch: float | None
    transmission_attempts: int
    cumulative_link_distance: float
    expected_transmissions_proxy: float
    transmission_energy_proxy: float
    minimum_link_lifetime_steps: float | None
    minimum_link_margin: float | None
    cumulative_queue_delay_proxy: float
    deadline_steps: int | None
    deadline_met: bool
    local_observation_bytes: int
    policy_input_bytes: int
    switch_steps: float
    safe_forward_candidates: float
    mean_selected_danger: float


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    delivered: int
    dropped: int
    packet_delivery_ratio: float
    mean_delay_steps: float
    mean_hop_count: float
    throughput_packets_per_step: float
    loop_drop_rate: float
    mean_episode_reward: float
    p95_success_delay_steps: float
    mean_path_stretch: float
    mean_expected_transmissions_proxy: float
    mean_transmission_energy_proxy: float
    mean_minimum_link_lifetime_steps: float
    mean_minimum_link_margin: float
    mean_queue_delay_proxy: float
    deadline_delivery_ratio: float
    delivery_energy_efficiency_proxy: float
    delivery_transmission_efficiency_proxy: float
    mean_local_observation_bytes: float
    mean_policy_input_bytes: float
    mean_switch_steps: float
    mean_safe_forward_candidates: float
    mean_selected_danger: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def run_episode(
    env: FanetRoutingEnv,
    policy: RoutingPolicy,
    *,
    seed: int,
    reset_options: dict[str, Any] | None = None,
) -> EpisodeResult:
    policy.reset(seed)
    observation, initial_info = env.reset(seed=seed, options=reset_options)
    info = initial_info
    total_reward = 0.0
    local_observation_bytes = 0
    policy_input_bytes = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        local_observation_bytes += sum(
            int(value.nbytes) for value in observation.values()
        )
        byte_counter = getattr(policy, "observation_bytes", None)
        policy_input_bytes += (
            int(byte_counter(observation))
            if byte_counter is not None
            else sum(int(value.nbytes) for value in observation.values())
        )
        action = policy.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    diagnostic_fn = getattr(policy, "episode_diagnostics", None)
    diagnostics = diagnostic_fn() if diagnostic_fn is not None else {}
    diagnostic_steps = max(float(diagnostics.get("diagnostic_steps", 0.0)), 1.0)
    shortest_hops = initial_info["initial_shortest_hops"]
    deadline_steps = (
        int(np.ceil(1.5 * int(shortest_hops))) + 1
        if shortest_hops is not None
        else None
    )
    return EpisodeResult(
        seed=seed,
        delivered=bool(info["delivered"]),
        dropped=bool(info["dropped"]),
        drop_reason=info["drop_reason"],
        steps=int(info["episode_step"]),
        hop_count=int(info["hop_count"]),
        total_reward=float(total_reward),
        initially_connected=bool(initial_info["initially_connected"]),
        initial_shortest_hops=shortest_hops,
        path_stretch=(
            int(info["hop_count"]) / int(shortest_hops)
            if info["delivered"] and shortest_hops
            else None
        ),
        transmission_attempts=int(info["transmission_attempts"]),
        cumulative_link_distance=float(info["cumulative_link_distance"]),
        expected_transmissions_proxy=(
            int(info["transmission_attempts"])
            / max(1.0 - env.config.stochastic_link_loss, 1e-8)
        ),
        transmission_energy_proxy=float(
            info["transmission_energy_proxy"]
        ),
        minimum_link_lifetime_steps=(
            float(info["minimum_link_lifetime_steps"])
            if info["minimum_link_lifetime_steps"] is not None
            else None
        ),
        minimum_link_margin=(
            float(info["minimum_link_margin"])
            if info["minimum_link_margin"] is not None
            else None
        ),
        cumulative_queue_delay_proxy=float(
            info["cumulative_queue_delay_proxy"]
        ),
        deadline_steps=deadline_steps,
        deadline_met=bool(
            info["delivered"]
            and deadline_steps is not None
            and int(info["episode_step"]) <= deadline_steps
        ),
        local_observation_bytes=local_observation_bytes,
        policy_input_bytes=policy_input_bytes,
        switch_steps=float(diagnostics.get("switch_steps", 0.0)),
        safe_forward_candidates=float(
            diagnostics.get("safe_forward_candidates", 0.0)
        ),
        mean_selected_danger=float(
            diagnostics.get("mean_selected_danger", 0.0) / diagnostic_steps
        ),
    )


def evaluate_policy(
    env: FanetRoutingEnv,
    policy: RoutingPolicy,
    seeds: list[int],
    *,
    reset_options: dict[str, Any] | None = None,
) -> EvaluationSummary:
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    results = evaluate_policy_results(
        env,
        policy,
        seeds,
        reset_options=reset_options,
    )
    return summarize_episode_results(results)


def evaluate_policy_results(
    env: FanetRoutingEnv,
    policy: RoutingPolicy,
    seeds: list[int],
    *,
    reset_options: dict[str, Any] | None = None,
) -> list[EpisodeResult]:
    """Return one immutable result per evaluation episode."""

    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    return [
        run_episode(
            env,
            policy,
            seed=seed,
            reset_options=reset_options,
        )
        for seed in seeds
    ]


def summarize_episode_results(
    results: list[EpisodeResult],
) -> EvaluationSummary:
    """Aggregate episode records without discarding the raw observations."""

    if not results:
        raise ValueError("at least one episode result is required")
    delivered_results = [result for result in results if result.delivered]
    stretches = [
        result.path_stretch
        for result in delivered_results
        if result.path_stretch is not None
    ]
    link_lifetimes = [
        result.minimum_link_lifetime_steps
        for result in results
        if result.minimum_link_lifetime_steps is not None
    ]
    link_margins = [
        result.minimum_link_margin
        for result in results
        if result.minimum_link_margin is not None
    ]
    delivered = len(delivered_results)
    total_steps = sum(result.steps for result in results)
    return EvaluationSummary(
        episodes=len(results),
        delivered=delivered,
        dropped=sum(result.dropped for result in results),
        packet_delivery_ratio=delivered / len(results),
        mean_delay_steps=(
            float(np.mean([result.steps for result in delivered_results]))
            if delivered_results
            else 0.0
        ),
        mean_hop_count=float(np.mean([result.hop_count for result in results])),
        throughput_packets_per_step=delivered / max(total_steps, 1),
        loop_drop_rate=(
            sum(result.drop_reason == "routing_loop" for result in results)
            / len(results)
        ),
        mean_episode_reward=float(
            np.mean([result.total_reward for result in results])
        ),
        p95_success_delay_steps=(
            float(
                np.percentile(
                    [result.steps for result in delivered_results], 95
                )
            )
            if delivered_results
            else 0.0
        ),
        mean_path_stretch=(
            float(np.mean(stretches)) if stretches else 0.0
        ),
        mean_expected_transmissions_proxy=float(
            np.mean(
                [result.expected_transmissions_proxy for result in results]
            )
        ),
        mean_transmission_energy_proxy=float(
            np.mean(
                [result.transmission_energy_proxy for result in results]
            )
        ),
        mean_minimum_link_lifetime_steps=(
            float(np.mean(link_lifetimes)) if link_lifetimes else 0.0
        ),
        mean_minimum_link_margin=(
            float(np.mean(link_margins)) if link_margins else 0.0
        ),
        mean_queue_delay_proxy=float(
            np.mean(
                [result.cumulative_queue_delay_proxy for result in results]
            )
        ),
        deadline_delivery_ratio=(
            sum(result.deadline_met for result in results) / len(results)
        ),
        delivery_energy_efficiency_proxy=(
            delivered
            / max(
                sum(
                    result.transmission_energy_proxy
                    for result in results
                ),
                1e-8,
            )
        ),
        delivery_transmission_efficiency_proxy=(
            delivered
            / max(
                sum(
                    result.expected_transmissions_proxy
                    for result in results
                ),
                1e-8,
            )
        ),
        mean_local_observation_bytes=float(
            np.mean([result.local_observation_bytes for result in results])
        ),
        mean_policy_input_bytes=float(
            np.mean([result.policy_input_bytes for result in results])
        ),
        mean_switch_steps=float(
            np.mean([result.switch_steps for result in results])
        ),
        mean_safe_forward_candidates=float(
            np.mean([result.safe_forward_candidates for result in results])
        ),
        mean_selected_danger=float(
            np.mean([result.mean_selected_danger for result in results])
        ),
    )
