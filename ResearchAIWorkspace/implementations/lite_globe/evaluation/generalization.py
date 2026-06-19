"""Connectivity-aware Phase 7 evaluation summaries."""

from __future__ import annotations

from typing import Any

import numpy as np

from .evaluator import EpisodeResult


GENERALIZATION_METRICS = (
    "endpoint_availability",
    "overall_pdr",
    "connected_pair_pdr",
    "mean_success_delay",
    "p95_success_delay",
    "mean_path_stretch",
    "loop_drop_rate",
    "invalid_action_drop_rate",
    "ttl_drop_rate",
    "agent_drop_rate",
    "mean_expected_transmissions_proxy",
    "mean_transmission_energy_proxy",
    "mean_minimum_link_lifetime_steps",
    "mean_minimum_link_margin",
    "mean_queue_delay_proxy",
    "deadline_delivery_ratio",
    "delivery_energy_efficiency_proxy",
    "delivery_transmission_efficiency_proxy",
    "mean_local_observation_bytes",
    "mean_policy_input_bytes",
    "mean_switch_steps",
    "mean_safe_forward_candidates",
    "mean_selected_danger",
    "mean_episode_reward",
)


def generalization_summary(
    results: list[EpisodeResult],
    *,
    method: str,
    scenario: str,
    training_seed: int,
) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one episode result is required")
    connected = [result for result in results if result.initially_connected]
    connected_delivered = [result for result in connected if result.delivered]
    delivered = [result for result in results if result.delivered]
    stretches = [
        result.path_stretch
        for result in delivered
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
    success_delays = [result.steps for result in delivered]
    return {
        "method": method,
        "scenario": scenario,
        "training_seed": training_seed,
        "episodes": len(results),
        "connected_episodes": len(connected),
        "delivered": len(delivered),
        "endpoint_availability": len(connected) / len(results),
        "overall_pdr": len(delivered) / len(results),
        "connected_pair_pdr": (
            len(connected_delivered) / len(connected) if connected else 0.0
        ),
        "mean_success_delay": (
            float(np.mean(success_delays))
            if delivered
            else 0.0
        ),
        "p95_success_delay": (
            float(np.percentile(success_delays, 95)) if delivered else 0.0
        ),
        "mean_path_stretch": (
            float(np.mean(stretches)) if stretches else 0.0
        ),
        "loop_drop_rate": (
            sum(result.drop_reason == "routing_loop" for result in results)
            / len(results)
        ),
        "invalid_action_drop_rate": (
            sum(result.drop_reason == "invalid_action" for result in results)
            / len(results)
        ),
        "ttl_drop_rate": (
            sum(result.drop_reason == "ttl_expired" for result in results)
            / len(results)
        ),
        "agent_drop_rate": (
            sum(result.drop_reason == "agent_drop" for result in results)
            / len(results)
        ),
        "mean_expected_transmissions_proxy": float(
            np.mean(
                [result.expected_transmissions_proxy for result in results]
            )
        ),
        "mean_transmission_energy_proxy": float(
            np.mean(
                [result.transmission_energy_proxy for result in results]
            )
        ),
        "mean_minimum_link_lifetime_steps": (
            float(np.mean(link_lifetimes)) if link_lifetimes else 0.0
        ),
        "mean_minimum_link_margin": (
            float(np.mean(link_margins)) if link_margins else 0.0
        ),
        "mean_queue_delay_proxy": float(
            np.mean(
                [result.cumulative_queue_delay_proxy for result in results]
            )
        ),
        "deadline_delivery_ratio": (
            sum(result.deadline_met for result in results) / len(results)
        ),
        "delivery_energy_efficiency_proxy": (
            len(delivered)
            / max(
                sum(
                    result.transmission_energy_proxy
                    for result in results
                ),
                1e-8,
            )
        ),
        "delivery_transmission_efficiency_proxy": (
            len(delivered)
            / max(
                sum(
                    result.expected_transmissions_proxy
                    for result in results
                ),
                1e-8,
            )
        ),
        "mean_local_observation_bytes": float(
            np.mean([result.local_observation_bytes for result in results])
        ),
        "mean_policy_input_bytes": float(
            np.mean([result.policy_input_bytes for result in results])
        ),
        "mean_switch_steps": float(
            np.mean([result.switch_steps for result in results])
        ),
        "mean_safe_forward_candidates": float(
            np.mean([result.safe_forward_candidates for result in results])
        ),
        "mean_selected_danger": float(
            np.mean([result.mean_selected_danger for result in results])
        ),
        "mean_episode_reward": float(
            np.mean([result.total_reward for result in results])
        ),
    }
