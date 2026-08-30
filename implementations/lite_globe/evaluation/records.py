"""Conversion between episode results and paper-ready raw rows."""

from __future__ import annotations

from typing import Any

from .evaluator import EpisodeResult, EvaluationSummary


SUMMARY_METRICS = (
    "packet_delivery_ratio",
    "mean_delay_steps",
    "mean_hop_count",
    "throughput_packets_per_step",
    "loop_drop_rate",
    "mean_episode_reward",
)


def episode_row(
    result: EpisodeResult,
    *,
    method: str,
    scenario: str,
    training_seed: int,
) -> dict[str, Any]:
    """Create a flat raw-result row without imputing failed-packet delay."""

    return {
        "method": method,
        "scenario": scenario,
        "training_seed": training_seed,
        "evaluation_seed": result.seed,
        "delivered": int(result.delivered),
        "dropped": int(result.dropped),
        "drop_reason": result.drop_reason or "",
        "delay_steps": result.steps if result.delivered else "",
        "steps": result.steps,
        "hop_count": result.hop_count,
        "throughput": int(result.delivered) / max(result.steps, 1),
        "loop": int(result.drop_reason == "routing_loop"),
        "total_reward": result.total_reward,
        "initially_connected": int(result.initially_connected),
        "initial_shortest_hops": (
            result.initial_shortest_hops
            if result.initial_shortest_hops is not None
            else ""
        ),
        "path_stretch": (
            result.path_stretch if result.path_stretch is not None else ""
        ),
        "transmission_attempts": result.transmission_attempts,
        "cumulative_link_distance": result.cumulative_link_distance,
        "expected_transmissions_proxy": result.expected_transmissions_proxy,
        "transmission_energy_proxy": result.transmission_energy_proxy,
        "minimum_link_lifetime_steps": (
            result.minimum_link_lifetime_steps
            if result.minimum_link_lifetime_steps is not None
            else ""
        ),
        "minimum_link_margin": (
            result.minimum_link_margin
            if result.minimum_link_margin is not None
            else ""
        ),
        "queue_delay_proxy": result.cumulative_queue_delay_proxy,
        "deadline_steps": (
            result.deadline_steps
            if result.deadline_steps is not None
            else ""
        ),
        "deadline_met": int(result.deadline_met),
        "local_observation_bytes": result.local_observation_bytes,
        "policy_input_bytes": result.policy_input_bytes,
        "switch_steps": result.switch_steps,
        "safe_forward_candidates": result.safe_forward_candidates,
        "mean_selected_danger": result.mean_selected_danger,
        "late_delivery": int(result.late_delivery),
        "deadline_slack_steps": result.deadline_slack_steps if result.deadline_slack_steps is not None else "",
        "mean_per_hop_delay": result.mean_per_hop_delay if result.mean_per_hop_delay is not None else "",
        "decision_latency_p50_ms": result.decision_latency_p50_ms,
        "decision_latency_p95_ms": result.decision_latency_p95_ms,
        "decision_latency_p99_ms": result.decision_latency_p99_ms,
        "control_messages": result.control_messages,
        "control_bytes": result.control_bytes,
        "branch_disagreement_steps": result.branch_disagreement_steps,
        "switch_danger_reduction": result.switch_danger_reduction,
        "false_switch_steps": result.false_switch_steps,
        "missed_risk_steps": result.missed_risk_steps,
    }


def summary_row(
    summary: EvaluationSummary,
    *,
    method: str,
    scenario: str,
    training_seed: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "method": method,
        "scenario": scenario,
        "training_seed": training_seed,
    }
    row.update(summary.to_dict())
    return row
