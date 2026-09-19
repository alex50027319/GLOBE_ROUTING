"""Phase 8 loop safety, geographic prior, and metric guarantees."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from implementations.lite_globe.baselines import GpsrPolicy
from implementations.lite_globe.data import generate_teacher_dataset
from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.evaluation import run_episode
from implementations.lite_globe.models import (
    GeographicResidualStudentPolicy,
    GlobalTeacherActorCritic,
)
from implementations.lite_globe.models.policy_adapter import (
    StudentPolicyAdapter,
)
from implementations.lite_globe.scenarios import (
    phase8_hole_evaluation_scenarios,
    routing_hole_config,
    routing_hole_options,
)


def test_visited_actions_can_be_structurally_masked(
    line_positions,
) -> None:
    config = FanetConfig(
        num_nodes=3,
        max_nodes=3,
        area_size=10.0,
        communication_radius=1.1,
        packet_ttl=4,
        min_speed=0.0,
        max_speed=0.0,
        mask_visited_actions=True,
    )
    env = FanetRoutingEnv(config)
    env.reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    observation, _, _, _, _ = env.step(1)
    assert observation["action_mask"][0] == 0
    assert observation["action_mask"][2] == 1
    assert env.global_observation()["action_mask"][0] == 0


def test_untrained_geographic_residual_matches_gpsr_action() -> None:
    config = replace(
        routing_hole_config(),
        mask_visited_actions=True,
        include_node_ids=False,
    )
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(seed=4, options=routing_hole_options())
    gpsr_action = GpsrPolicy(env.drop_action).act(observation)
    torch.manual_seed(8)
    model = GeographicResidualStudentPolicy(
        env.config.max_nodes,
        hidden_dim=32,
    )
    residual_action = StudentPolicyAdapter(model).act(observation)
    assert residual_action == gpsr_action


def test_teacher_dataset_contains_legal_oracle_targets() -> None:
    config = replace(
        routing_hole_config(),
        mask_visited_actions=True,
        include_node_ids=False,
    )
    env = FanetRoutingEnv(config)
    teacher = GlobalTeacherActorCritic(config.max_nodes, hidden_dim=32)
    dataset = generate_teacher_dataset(
        env,
        teacher,
        episode_seeds=list(range(6)),
        scenario_id="phase8_test",
        reset_options=routing_hole_options(),
    )
    assert "oracle_actions" in dataset.arrays
    indices = np.arange(len(dataset))
    masks = dataset.arrays["action_mask"].astype(bool)
    actions = dataset.arrays["oracle_actions"]
    assert np.all(masks[indices, actions])


def test_episode_reports_literature_aligned_proxies(
    line_env,
    line_positions,
) -> None:
    result = run_episode(
        line_env,
        GpsrPolicy(line_env.drop_action),
        seed=2,
        reset_options={
            "positions": line_positions,
            "source": 0,
            "destination": 2,
        },
    )
    assert result.delivered
    assert result.transmission_attempts == 2
    assert result.expected_transmissions_proxy == 2.0
    assert result.transmission_energy_proxy > 0.0
    assert result.minimum_link_lifetime_steps is not None
    assert result.local_observation_bytes > 0


def test_forwardability_exposes_greedy_dead_end() -> None:
    scenario = phase8_hole_evaluation_scenarios(42)[0]
    env = FanetRoutingEnv(scenario.config)
    observation, _ = env.reset(
        seed=42,
        options=scenario.reset_options,
    )
    forwardability = observation["candidate_forwardability"]
    assert observation["action_mask"][1] == 1
    assert observation["action_mask"][2] == 1
    assert forwardability[1, 0] == 0.0
    assert forwardability[2, 0] == 1.0


def test_untrained_forwardability_prior_does_not_solve_hole() -> None:
    scenario = phase8_hole_evaluation_scenarios(42)[0]
    env = FanetRoutingEnv(scenario.config)
    observation, _ = env.reset(seed=42, options=scenario.reset_options)
    gpsr_action = GpsrPolicy(env.drop_action).act(observation)
    model = GeographicResidualStudentPolicy(
        env.config.max_nodes,
        hidden_dim=32,
    )
    assert StudentPolicyAdapter(model).act(observation) == gpsr_action
