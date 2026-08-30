"""SwitchGLOBE policy tests (historically developed as Phase 12)."""

from __future__ import annotations

from copy import deepcopy

import torch

from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from implementations.lite_globe.models.policy_adapter import (
    StudentPolicyAdapter,
)
from implementations.lite_globe.scenarios import (
    predictive_break_config,
    predictive_break_options,
)


def _policy(max_nodes: int) -> SwitchGlobePolicy:
    normal = GeographicResidualStudentPolicy(max_nodes, hidden_dim=32)
    predictive = LiteGlobePStudentPolicy(
        max_nodes,
        hidden_dim=32,
        initial_predictive_strength=(0.75, 3.0, 0.25, 6.0),
        initial_break_penalty=18.0,
        initial_residual_bound=1.5,
    )
    predictive.set_residual_weight(0.0)
    return SwitchGlobePolicy(
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


def test_fused_decision_runs_each_branch_once(line_positions) -> None:
    from implementations.lite_globe.env.config import FanetConfig

    config = FanetConfig(
        num_nodes=3, max_nodes=4, area_size=10.0,
        communication_radius=1.1, max_queue_size=8,
        min_speed=0.0, max_speed=0.0,
        include_forwardability=True, include_risk_features=True,
    )
    observation, _ = FanetRoutingEnv(config).reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    policy = _policy(config.max_nodes)
    adapter = StudentPolicyAdapter(policy, force_forward_if_available=True)
    counts = {"normal": 0, "predictive": 0}
    normal_hook = policy.normal_policy.register_forward_hook(
        lambda *_: counts.__setitem__("normal", counts["normal"] + 1)
    )
    predictive_hook = policy.predictive_policy.register_forward_hook(
        lambda *_: counts.__setitem__(
            "predictive", counts["predictive"] + 1
        )
    )
    try:
        decision = adapter.act_with_metadata(observation)
    finally:
        normal_hook.remove()
        predictive_hook.remove()
    assert decision.action < config.max_nodes
    assert counts == {"normal": 1, "predictive": 1}


def test_fused_metadata_matches_legacy_byte_accounting(line_positions) -> None:
    from implementations.lite_globe.env.config import FanetConfig

    config = FanetConfig(
        num_nodes=3, max_nodes=4, area_size=10.0,
        communication_radius=1.1, max_queue_size=8,
        min_speed=0.0, max_speed=0.0,
        include_forwardability=True, include_risk_features=True,
    )
    observation, _ = FanetRoutingEnv(config).reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    policy = _policy(config.max_nodes)
    adapter = StudentPolicyAdapter(policy, force_forward_if_available=True)
    legacy_bytes = adapter.observation_bytes(observation)
    decision = adapter.act_with_metadata(observation)
    assert decision.input_bytes == legacy_bytes
    assert all(
        torch.isfinite(parameter).all() for parameter in policy.parameters()
    )


def test_buffered_adapter_matches_eager_action(line_positions) -> None:
    from implementations.lite_globe.env.config import FanetConfig

    config = FanetConfig(
        num_nodes=3, max_nodes=4, area_size=10.0,
        communication_radius=1.1, max_queue_size=8,
        min_speed=0.0, max_speed=0.0,
        include_forwardability=True, include_risk_features=True,
    )
    observation, _ = FanetRoutingEnv(config).reset(
        seed=1,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    policy = _policy(config.max_nodes)
    eager = StudentPolicyAdapter(
        deepcopy(policy), force_forward_if_available=True
    )
    buffered = StudentPolicyAdapter(
        deepcopy(policy), force_forward_if_available=True,
        reuse_tensor_buffer=True,
    )
    assert (
        eager.act_with_metadata(observation)
        == buffered.act_with_metadata(observation)
    )
