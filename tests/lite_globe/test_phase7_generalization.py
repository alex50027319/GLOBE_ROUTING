"""Phase 7 connectivity, shaping, oracle, and reporting guarantees."""

from __future__ import annotations

import numpy as np

from implementations.lite_globe.baselines import ShortestPathOraclePolicy
from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.env.graph_utils import (
    connected_pairs,
    shortest_path,
)
from implementations.lite_globe.evaluation import run_episode
from implementations.lite_globe.scenarios import (
    phase7_curriculum,
    phase7_evaluation_scenarios,
)


def test_shortest_path_and_connected_pair_minimum_hops() -> None:
    adjacency = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.bool_,
    )
    assert shortest_path(adjacency, 0, 3) == [0, 1, 2, 3]
    assert shortest_path(adjacency, 0, 0) == [0]
    pairs = connected_pairs(adjacency, min_hops=3)
    assert pairs == [(0, 3, 3), (3, 0, 3)]


def test_connected_endpoint_sampling_and_node_id_removal() -> None:
    stage = phase7_curriculum(42)[1]
    env = FanetRoutingEnv(stage.config)
    observation, info = env.reset(seed=99, options=stage.reset_options)
    assert info["initially_connected"]
    assert info["initial_shortest_hops"] >= 2
    assert observation["packet_features"][-2:].tolist() == [0.0, 0.0]


def test_progress_shaping_rewards_forward_motion(line_positions) -> None:
    config = FanetConfig(
        num_nodes=3,
        max_nodes=3,
        area_size=10.0,
        communication_radius=1.1,
        packet_ttl=4,
        max_episode_steps=4,
        min_speed=0.0,
        max_speed=0.0,
        reward_progress=2.0,
    )
    env = FanetRoutingEnv(config)
    env.reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    _, reward, terminated, _, _ = env.step(1)
    assert not terminated
    expected_progress = (
        np.linalg.norm(line_positions[2] - line_positions[0])
        - np.linalg.norm(line_positions[2] - line_positions[1])
    ) / config.area_size
    assert np.isclose(
        reward,
        -config.reward_delay + config.reward_progress * expected_progress,
    )


def test_shortest_path_oracle_delivers_on_static_line(
    line_env, line_positions
) -> None:
    result = run_episode(
        line_env,
        ShortestPathOraclePolicy(line_env),
        seed=4,
        reset_options={
            "positions": line_positions,
            "source": 0,
            "destination": 2,
        },
    )
    assert result.delivered
    assert result.initially_connected
    assert result.initial_shortest_hops == 2
    assert result.path_stretch == 1.0


def test_latency_aware_delay_is_separate_from_step_delay(
    line_env, line_positions
) -> None:
    result = run_episode(
        line_env,
        ShortestPathOraclePolicy(line_env),
        seed=4,
        reset_options={
            "positions": line_positions,
            "source": 0,
            "destination": 2,
        },
        routing_step_duration_ms=5.0,
    )
    assert result.steps == 2
    assert result.routing_step_duration_ms == 5.0
    assert result.effective_end_to_end_delay_ms is not None
    assert result.effective_end_to_end_delay_ms >= 10.0
    assert result.deadline_ms == result.deadline_steps * 5.0


def test_phase7_families_are_held_out_and_shape_compatible() -> None:
    curriculum = phase7_curriculum(42)
    evaluation = phase7_evaluation_scenarios(42)
    assert [stage.name for stage in curriculum] == [
        "train_easy",
        "train_medium",
        "train_hard",
    ]
    assert len({scenario.name for scenario in evaluation}) == 6
    assert all(stage.config.max_nodes == 12 for stage in curriculum)
    assert all(scenario.config.max_nodes == 12 for scenario in evaluation)
    assert evaluation[4].config.num_nodes == 10
    assert evaluation[-1].reset_options is None
