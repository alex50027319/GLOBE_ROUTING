"""Core packet-routing environment behavior."""

from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv


def test_two_hop_delivery(line_env, line_positions) -> None:
    observation, _ = line_env.reset(
        seed=3,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    assert observation["action_mask"].tolist() == [0, 1, 0, 0, 1]

    _, first_reward, terminated, truncated, info = line_env.step(1)
    assert first_reward == -line_env.config.reward_delay
    assert not terminated
    assert not truncated
    assert info["path"] == (0, 1)

    _, final_reward, terminated, truncated, info = line_env.step(2)
    assert final_reward == (
        line_env.config.reward_delivery - line_env.config.reward_delay
    )
    assert terminated
    assert not truncated
    assert info["delivered"]
    assert info["hop_count"] == 2


def test_invalid_disconnected_action_is_dropped(line_env, line_positions) -> None:
    line_env.reset(
        seed=3,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    _, reward, terminated, _, info = line_env.step(2)
    assert terminated
    assert reward == (
        -line_env.config.reward_failure - line_env.config.reward_delay
    )
    assert info["drop_reason"] == "invalid_action"


def test_revisiting_node_is_loop_drop(line_env, line_positions) -> None:
    line_env.reset(
        seed=3,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    line_env.step(1)
    _, _, terminated, _, info = line_env.step(0)
    assert terminated
    assert info["drop_reason"] == "routing_loop"
    assert info["path"] == (0, 1, 0)


def test_ttl_expiration(line_positions) -> None:
    config = FanetConfig(
        num_nodes=3,
        max_nodes=3,
        area_size=10.0,
        communication_radius=1.1,
        packet_ttl=1,
        min_speed=0.0,
        max_speed=0.0,
    )
    env = FanetRoutingEnv(config)
    env.reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    _, _, terminated, _, info = env.step(1)
    assert terminated
    assert info["drop_reason"] == "ttl_expired"


def test_seed_reproduces_reset_and_trajectory() -> None:
    config = FanetConfig(
        num_nodes=5,
        max_nodes=6,
        communication_radius=2000.0,
        min_speed=1.0,
        max_speed=3.0,
    )
    left = FanetRoutingEnv(config)
    right = FanetRoutingEnv(config)
    left_obs, left_info = left.reset(seed=123)
    right_obs, right_info = right.reset(seed=123)
    assert left_info == right_info
    for key in left_obs:
        np.testing.assert_array_equal(left_obs[key], right_obs[key])

    for action in [1, 2]:
        if left.packet.delivered or left.packet.dropped:
            break
        if left_obs["action_mask"][action] == 0:
            action = int(np.flatnonzero(left_obs["action_mask"][:-1])[0])
        left_obs, left_reward, left_done, left_truncated, left_info = left.step(action)
        right_obs, right_reward, right_done, right_truncated, right_info = right.step(action)
        assert (left_reward, left_done, left_truncated, left_info) == (
            right_reward,
            right_done,
            right_truncated,
            right_info,
        )
        for key in left_obs:
            np.testing.assert_array_equal(left_obs[key], right_obs[key])


def test_gymnasium_contract(line_config) -> None:
    check_env(FanetRoutingEnv(line_config), skip_render_check=True)
