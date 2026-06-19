"""Phase 11 Lite-GLOBE-P predictive prior and integration tests."""

from __future__ import annotations

import torch

from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
)
from implementations.lite_globe.models.policy_adapter import (
    StudentPolicyAdapter,
)
from implementations.lite_globe.scenarios import (
    predictive_break_config,
    predictive_break_options,
)


def test_lite_globe_p_avoids_predictive_break_trap() -> None:
    config = predictive_break_config(42)
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(
        seed=42,
        options=predictive_break_options(0.0),
    )
    policy = LiteGlobePStudentPolicy(
        config.max_nodes,
        hidden_dim=32,
        initial_predictive_strength=(0.75, 3.0, 0.25, 6.0),
        initial_break_penalty=18.0,
        initial_residual_bound=1.5,
    )
    action = StudentPolicyAdapter(
        policy,
        force_forward_if_available=True,
    ).act(observation)
    assert action != 1
    assert action == 3


def test_lite_globe_p_loads_phase8_state_with_predictive_keys_missing() -> None:
    phase8 = GeographicResidualStudentPolicy(max_nodes=8, hidden_dim=32)
    phase11 = LiteGlobePStudentPolicy(max_nodes=8, hidden_dim=32)
    incompatible = phase11.load_state_dict(phase8.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert {
        "log_predictive_strength",
        "log_break_penalty",
        "log_residual_bound",
        "predictive_weight",
        "lifetime_gate",
        "onward_gate",
        "margin_gate",
    } == set(incompatible.missing_keys)


def test_lite_globe_p_observation_bytes_include_predictive_features() -> None:
    config = predictive_break_config(7)
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(
        seed=7,
        options=predictive_break_options(45.0),
    )
    phase8 = GeographicResidualStudentPolicy(config.max_nodes, hidden_dim=32)
    phase11 = LiteGlobePStudentPolicy(config.max_nodes, hidden_dim=32)
    phase8_bytes = StudentPolicyAdapter(phase8).observation_bytes(
        observation
    )
    phase11_bytes = StudentPolicyAdapter(phase11).observation_bytes(
        observation
    )
    assert phase11_bytes == (
        phase8_bytes + observation["candidate_risk_features"].nbytes
    )


def test_lite_globe_p_weight_setters_validate_range() -> None:
    policy = LiteGlobePStudentPolicy(max_nodes=4, hidden_dim=32)
    policy.set_predictive_weight(0.25)
    policy.set_residual_weight(0.5)
    assert torch.isclose(policy.predictive_weight, torch.tensor(0.25))
    assert torch.isclose(policy.residual_weight, torch.tensor(0.5))
