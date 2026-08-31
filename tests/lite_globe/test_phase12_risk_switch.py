"""Phase 12 Risk-Switch Lite-GLOBE-P policy tests."""

from __future__ import annotations

from copy import deepcopy

from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    RiskSwitchLiteGlobePStudentPolicy,
)
from implementations.lite_globe.models.policy_adapter import (
    StudentPolicyAdapter,
)
from implementations.lite_globe.scenarios import (
    predictive_break_config,
    predictive_break_options,
)


def _policy(max_nodes: int) -> RiskSwitchLiteGlobePStudentPolicy:
    normal = GeographicResidualStudentPolicy(max_nodes, hidden_dim=32)
    predictive = LiteGlobePStudentPolicy(
        max_nodes,
        hidden_dim=32,
        initial_predictive_strength=(0.75, 3.0, 0.25, 6.0),
        initial_break_penalty=18.0,
        initial_residual_bound=1.5,
    )
    predictive.set_residual_weight(0.0)
    return RiskSwitchLiteGlobePStudentPolicy(
        normal,
        predictive,
        switch_threshold=0.05,
        margin_gate=0.04,
        lifetime_gate=0.20,
        onward_gate=0.20,
    )


def test_risk_switch_uses_predictive_branch_on_break_trap() -> None:
    config = predictive_break_config(42)
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(
        seed=42,
        options=predictive_break_options(0.0),
    )
    policy = _policy(config.max_nodes)
    action = StudentPolicyAdapter(
        policy,
        force_forward_if_available=True,
    ).act(observation)
    assert action == 3


def test_risk_switch_can_be_disabled_to_match_phase8() -> None:
    config = predictive_break_config(42)
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(
        seed=42,
        options=predictive_break_options(0.0),
    )
    policy = _policy(config.max_nodes)
    phase8_action = StudentPolicyAdapter(
        policy.normal_policy,
        force_forward_if_available=True,
    ).act(observation)
    policy.set_switch_parameters(
        switch_threshold=3.0,
        margin_gate=0.0,
        lifetime_gate=0.0,
        onward_gate=0.0,
    )
    switched_action = StudentPolicyAdapter(
        policy,
        force_forward_if_available=True,
    ).act(observation)
    assert switched_action == phase8_action


def test_risk_switch_observation_bytes_are_below_full_predictive_when_safe(
    line_positions,
) -> None:
    from implementations.lite_globe.env.config import FanetConfig

    config = FanetConfig(
        num_nodes=3,
        max_nodes=4,
        area_size=10.0,
        communication_radius=1.1,
        max_queue_size=8,
        min_speed=0.0,
        max_speed=0.0,
        include_forwardability=True,
        include_risk_features=True,
    )
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    policy = _policy(config.max_nodes)
    policy.set_switch_parameters(
        switch_threshold=3.0,
        margin_gate=0.0,
        lifetime_gate=0.0,
        onward_gate=0.0,
    )
    switch_bytes = StudentPolicyAdapter(policy).observation_bytes(observation)
    predictive = deepcopy(policy.predictive_policy)
    predictive.set_residual_weight(0.0)
    predictive_bytes = StudentPolicyAdapter(predictive).observation_bytes(
        observation
    )
    assert switch_bytes < predictive_bytes
