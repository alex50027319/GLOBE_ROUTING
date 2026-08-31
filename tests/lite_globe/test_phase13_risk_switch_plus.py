"""Phase 13 Risk-Switch Lite-GLOBE-P+ policy tests."""

from __future__ import annotations

from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.evaluation import evaluate_policy_results
from implementations.lite_globe.models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    RiskSwitchLiteGlobePPlusStudentPolicy,
)
from implementations.lite_globe.models.policy_adapter import (
    StudentPolicyAdapter,
)


def _policy(max_nodes: int) -> RiskSwitchLiteGlobePPlusStudentPolicy:
    normal = GeographicResidualStudentPolicy(max_nodes, hidden_dim=32)
    predictive = LiteGlobePStudentPolicy(max_nodes, hidden_dim=32)
    predictive.set_residual_weight(0.0)
    return RiskSwitchLiteGlobePPlusStudentPolicy(
        normal,
        predictive,
        switch_threshold=0.05,
        margin_gate=0.04,
        lifetime_gate=0.20,
        onward_gate=0.20,
        topk_onward_gate=0.15,
        redundancy_gate=0.0,
        loss_keep_gate=0.70,
        predictive_margin=0.05,
        energy_tie_weight=0.35,
        drop_suppression_bonus=8.0,
    )


def test_plus_observation_features_are_available(line_positions) -> None:
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
    assert observation["candidate_switch_features"].shape == (4, 4)
    assert observation["candidate_switch_features"][1, 2] == 1.0


def test_plus_episode_records_switch_diagnostics(line_positions) -> None:
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
    policy = _policy(config.max_nodes)
    results = evaluate_policy_results(
        env,
        StudentPolicyAdapter(policy, force_forward_if_available=True),
        [1],
        reset_options={
            "positions": line_positions,
            "source": 0,
            "destination": 2,
        },
    )
    result = results[0]
    assert result.safe_forward_candidates > 0
    assert result.mean_selected_danger >= 0.0
