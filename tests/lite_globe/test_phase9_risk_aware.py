"""Phase 9 predictive features, oracle, policy, and metric guarantees."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from implementations.lite_globe.baselines import (
    RiskAwareOraclePolicy,
    risk_aware_shortest_path,
)
from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.evaluation import run_episode
from implementations.lite_globe.models import (
    GeographicResidualStudentPolicy,
    RiskAwareGeographicResidualStudentPolicy,
)
from implementations.lite_globe.models.policy_adapter import (
    StudentPolicyAdapter,
)
from implementations.lite_globe.scenarios import (
    phase9_curriculum,
    phase9_evaluation_scenarios,
    predictive_break_config,
    predictive_break_options,
)


def test_risk_features_are_local_normalized_and_mask_aligned(
    line_positions,
) -> None:
    config = FanetConfig(
        num_nodes=3,
        max_nodes=4,
        area_size=10.0,
        communication_radius=1.1,
        max_queue_size=8,
        min_speed=0.0,
        max_speed=0.0,
        include_risk_features=True,
    )
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    risk = observation["candidate_risk_features"]
    assert risk.shape == (config.max_nodes, 4)
    assert np.all((risk >= 0.0) & (risk <= 1.0))
    invalid = observation["action_mask"][: config.max_nodes] == 0
    assert np.all(risk[invalid] == 0.0)
    assert risk[1, 1] == 1.0


def test_risk_oracle_returns_legal_path_on_static_line(
    line_positions,
) -> None:
    config = FanetConfig(
        num_nodes=3,
        max_nodes=3,
        area_size=10.0,
        communication_radius=1.1,
        max_queue_size=8,
        min_speed=0.0,
        max_speed=0.0,
        include_risk_features=True,
    )
    env = FanetRoutingEnv(config)
    env.reset(
        seed=2,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    assert risk_aware_shortest_path(env) == [0, 1, 2]
    result = run_episode(
        env,
        RiskAwareOraclePolicy(env),
        seed=2,
        reset_options={
            "positions": line_positions,
            "source": 0,
            "destination": 2,
        },
    )
    assert result.delivered
    assert result.deadline_met
    assert result.cumulative_queue_delay_proxy >= 2.0
    assert result.minimum_link_margin is not None


def test_phase9_policy_is_phase8_equivalent_when_risk_disabled() -> None:
    stage = phase9_curriculum(42)[0]
    env = FanetRoutingEnv(stage.config)
    observation, _ = env.reset(
        seed=8,
        options=stage.reset_options,
    )
    torch.manual_seed(9)
    phase8 = GeographicResidualStudentPolicy(
        stage.config.max_nodes,
        hidden_dim=32,
    )
    phase9 = RiskAwareGeographicResidualStudentPolicy(
        stage.config.max_nodes,
        hidden_dim=32,
    )
    incompatible = phase9.load_state_dict(phase8.state_dict(), strict=False)
    assert set(incompatible.missing_keys) == {
        "risk_weight",
        "log_risk_strength",
    }
    assert not incompatible.unexpected_keys
    phase9.set_risk_weight(0.0)
    action8 = StudentPolicyAdapter(phase8).act(observation)
    action9 = StudentPolicyAdapter(phase9).act(observation)
    assert action9 == action8


def test_phase9_scenarios_include_stress_and_scalability() -> None:
    scenarios = phase9_evaluation_scenarios(42)
    names = {scenario.name for scenario in scenarios}
    assert {
        "ood_link_loss_30",
        "ood_extreme_mobility",
        "ood_nodes_16",
        "ood_nodes_24",
    }.issubset(names)
    assert all(scenario.config.max_nodes == 32 for scenario in scenarios)
    assert all(
        scenario.config.include_risk_features for scenario in scenarios
    )


def test_queue_delay_proxy_increases_with_queue(
    line_positions,
) -> None:
    config = FanetConfig(
        num_nodes=3,
        max_nodes=3,
        area_size=10.0,
        communication_radius=1.1,
        max_queue_size=8,
        min_speed=0.0,
        max_speed=0.0,
        include_risk_features=True,
    )
    low = FanetRoutingEnv(config)
    high = FanetRoutingEnv(replace(config, seed=3))
    options = {
        "positions": line_positions,
        "source": 0,
        "destination": 2,
    }
    low.reset(seed=3, options=options)
    high.reset(seed=3, options=options)
    low.queues[:] = 0.0
    high.queues[:] = config.max_queue_size
    low.step(1)
    high.step(1)
    assert (
        high.cumulative_queue_delay_proxy
        > low.cumulative_queue_delay_proxy
    )


def test_predictive_break_trap_exposes_future_link_failure() -> None:
    config = predictive_break_config(42)
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(
        seed=42,
        options=predictive_break_options(0.0),
    )
    assert observation["candidate_forwardability"][1, 0] == 1.0
    assert observation["candidate_forwardability"][3, 0] == 1.0
    assert (
        observation["candidate_risk_features"][1, 3]
        < observation["candidate_risk_features"][3, 3]
    )


def test_phase9_safe_execution_avoids_premature_drop() -> None:
    config = predictive_break_config(42)
    model = RiskAwareGeographicResidualStudentPolicy(
        config.max_nodes,
        hidden_dim=32,
        initial_risk_strength=(0.5, 0.75, 0.2, 20.0),
    )
    model.drop_scorer[-1].bias.data.fill_(100.0)
    adapter = StudentPolicyAdapter(
        model,
        force_forward_if_available=True,
    )
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(
        seed=42,
        options=predictive_break_options(0.0),
    )
    assert adapter.act(observation) != env.drop_action
