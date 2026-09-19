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
    "late_delivery_ratio_all",
    "late_delivery_ratio_delivered",
    "total_drop_rate",
    "time_limit_drop_rate",
    "mean_per_hop_delay",
    "mean_deadline_slack_steps",
    "energy_per_generated_packet",
    "energy_per_delivered_packet",
    "energy_per_on_time_delivery",
    "decision_latency_p50_ms",
    "decision_latency_p95_ms",
    "decision_latency_p99_ms",
    "mean_decision_latency_total_ms",
    "mean_environment_step_time_total_ms",
    "mean_effective_end_to_end_delay_ms",
    "latency_aware_deadline_delivery_ratio",
    "mean_control_messages",
    "mean_control_bytes",
    "switch_activation_rate",
    "branch_disagreement_rate",
    "mean_switch_danger_reduction",
    "false_switch_rate",
    "missed_risk_rate",
    "backup_availability_rate",
    "fast_failover_success_rate",
    "fast_failover_miss_rate",
    "freshness_cache_hit_rate",
    "freshness_cache_stale_eviction_rate",
    "freshness_cache_state_eviction_rate",
    "freshness_cache_capacity_eviction_rate",
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
    on_time = [result for result in results if result.deadline_met]
    energy = sum(result.transmission_energy_proxy for result in results)
    decision_steps = max(sum(result.steps for result in results), 1)
    per_hop = [result.mean_per_hop_delay for result in delivered if result.mean_per_hop_delay is not None]
    slack = [result.deadline_slack_steps for result in delivered if result.deadline_slack_steps is not None]
    effective_delays = [
        result.effective_end_to_end_delay_ms
        for result in delivered
        if result.effective_end_to_end_delay_ms is not None
    ]
    latency_aware = [
        result for result in results
        if result.deadline_met_latency_aware is not None
    ]
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
            else None
        ),
        "p95_success_delay": (
            float(np.percentile(success_delays, 95)) if delivered else None
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
        "late_delivery_ratio_all": sum(result.late_delivery for result in results) / len(results),
        "late_delivery_ratio_delivered": sum(result.late_delivery for result in results) / max(len(delivered), 1),
        "total_drop_rate": sum(result.dropped for result in results) / len(results),
        "time_limit_drop_rate": sum(result.drop_reason == "time_limit" for result in results) / len(results),
        "mean_per_hop_delay": float(np.mean(per_hop)) if per_hop else 0.0,
        "mean_deadline_slack_steps": float(np.mean(slack)) if slack else 0.0,
        "energy_per_generated_packet": energy / len(results),
        "energy_per_delivered_packet": (
            energy / len(delivered) if delivered else None
        ),
        "energy_per_on_time_delivery": (
            energy / len(on_time) if on_time else None
        ),
        "decision_latency_p50_ms": float(np.percentile([result.decision_latency_p50_ms for result in results], 50)),
        "decision_latency_p95_ms": float(np.percentile([result.decision_latency_p95_ms for result in results], 95)),
        "decision_latency_p99_ms": float(np.percentile([result.decision_latency_p99_ms for result in results], 99)),
        "mean_decision_latency_total_ms": float(np.mean([result.decision_latency_total_ms for result in results])),
        "mean_environment_step_time_total_ms": float(np.mean([result.environment_step_time_total_ms for result in results])),
        "mean_effective_end_to_end_delay_ms": (
            float(np.mean(effective_delays)) if effective_delays else 0.0
        ),
        "latency_aware_deadline_delivery_ratio": (
            sum(bool(result.deadline_met_latency_aware) for result in latency_aware)
            / len(latency_aware)
            if latency_aware else 0.0
        ),
        "mean_control_messages": float(np.mean([result.control_messages for result in results])),
        "mean_control_bytes": float(np.mean([result.control_bytes for result in results])),
        "switch_activation_rate": sum(result.switch_steps for result in results) / decision_steps,
        "branch_disagreement_rate": sum(result.branch_disagreement_steps for result in results) / decision_steps,
        "mean_switch_danger_reduction": sum(result.switch_danger_reduction for result in results) / max(sum(result.switch_steps for result in results), 1.0),
        "false_switch_rate": sum(result.false_switch_steps for result in results) / max(sum(result.switch_steps for result in results), 1.0),
        "missed_risk_rate": sum(result.missed_risk_steps for result in results) / decision_steps,
        "backup_availability_rate": sum(
            result.backup_available_steps for result in results
        ) / decision_steps,
        "fast_failover_success_rate": sum(
            result.fast_failover_steps for result in results
        ) / max(
            sum(
                result.fast_failover_steps + result.fast_failover_miss_steps
                for result in results
            ),
            1.0,
        ),
        "fast_failover_miss_rate": sum(
            result.fast_failover_miss_steps for result in results
        ) / decision_steps,
        "freshness_cache_hit_rate": sum(
            result.freshness_cache_hit_steps for result in results
        ) / max(
            sum(
                result.freshness_cache_hit_steps
                + result.freshness_cache_miss_steps
                for result in results
            ),
            1.0,
        ),
        "freshness_cache_stale_eviction_rate": sum(
            result.freshness_cache_stale_evictions for result in results
        ) / decision_steps,
        "freshness_cache_state_eviction_rate": sum(
            result.freshness_cache_state_evictions for result in results
        ) / decision_steps,
        "freshness_cache_capacity_eviction_rate": sum(
            result.freshness_cache_capacity_evictions for result in results
        ) / decision_steps,
    }
