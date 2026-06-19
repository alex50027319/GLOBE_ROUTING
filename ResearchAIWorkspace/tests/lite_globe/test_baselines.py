"""Behavior of non-learning policies and evaluation metrics."""

from __future__ import annotations

from implementations.lite_globe.baselines import GpsrPolicy, RandomPolicy
from implementations.lite_globe.evaluation import evaluate_policy, run_episode


def test_gpsr_selects_neighbor_closest_to_destination(
    line_env, line_positions
) -> None:
    observation, _ = line_env.reset(
        seed=11,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    policy = GpsrPolicy(line_env.drop_action)
    assert policy.act(observation) == 1


def test_random_policy_is_seed_reproducible(line_env, line_positions) -> None:
    observation, _ = line_env.reset(
        seed=11,
        options={"positions": line_positions, "source": 1, "destination": 2},
    )
    left = RandomPolicy(line_env.drop_action)
    right = RandomPolicy(line_env.drop_action)
    left.reset(19)
    right.reset(19)
    assert [left.act(observation) for _ in range(8)] == [
        right.act(observation) for _ in range(8)
    ]


def test_gpsr_delivers_on_line_topology(line_env, line_positions) -> None:
    result = run_episode(
        line_env,
        GpsrPolicy(line_env.drop_action),
        seed=4,
        reset_options={
            "positions": line_positions,
            "source": 0,
            "destination": 2,
        },
    )
    assert result.delivered
    assert result.steps == 2
    assert result.hop_count == 2


def test_evaluation_summary_counts_episodes(line_env) -> None:
    summary = evaluate_policy(
        line_env,
        GpsrPolicy(line_env.drop_action),
        seeds=[1, 2, 3],
    )
    assert summary.episodes == 3
    assert summary.delivered + summary.dropped == 3
    assert 0.0 <= summary.packet_delivery_ratio <= 1.0
    assert 0.0 <= summary.loop_drop_rate <= 1.0
