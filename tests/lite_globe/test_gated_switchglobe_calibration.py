"""Gate-calibration instrumentation must never change SwitchGLOBE Exact's actions."""

from __future__ import annotations

import torch

from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.evaluation.evaluator import run_episode
from implementations.lite_globe.evaluation.gated_switchglobe_calibration import (
    GateCalibrationAdapter,
    aggregate_gate_calibration,
)
from implementations.lite_globe.models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from implementations.lite_globe.models.policy_adapter import StudentPolicyAdapter
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


def test_gate_calibration_adapter_does_not_change_actions() -> None:
    """GateCalibrationAdapter is a passive observer, not a different policy."""

    config = predictive_break_config(42)
    model = _policy(max_nodes=config.max_nodes)
    reset_options = predictive_break_options(0.0)

    baseline = StudentPolicyAdapter(
        model, deterministic=True, force_forward_if_available=True
    )
    gated = GateCalibrationAdapter(model, gate_margins=(0.0, 0.05, 0.5))

    baseline_result = run_episode(
        FanetRoutingEnv(config), baseline, seed=7, reset_options=reset_options
    )
    gated_result = run_episode(
        FanetRoutingEnv(config), gated, seed=7, reset_options=reset_options
    )

    assert gated_result.delivered == baseline_result.delivered
    assert gated_result.dropped == baseline_result.dropped
    assert gated_result.drop_reason == baseline_result.drop_reason
    assert gated_result.steps == baseline_result.steps
    assert gated_result.hop_count == baseline_result.hop_count
    assert gated_result.switch_steps == baseline_result.switch_steps
    assert (
        gated_result.branch_disagreement_steps
        == baseline_result.branch_disagreement_steps
    )


def test_gate_skip_rate_is_monotonic_in_margin() -> None:
    """A larger absolute danger-score cutoff can only skip more often.

    The gate requires ``normal_danger <= margin``, so a bigger margin makes
    the condition strictly easier to satisfy.
    """

    config = predictive_break_config(42)
    model = _policy(max_nodes=config.max_nodes)
    reset_options = predictive_break_options(0.0)
    margins = (0.0, 0.05, 0.2, 1.0)
    gated = GateCalibrationAdapter(model, gate_margins=margins)

    run_episode(FanetRoutingEnv(config), gated, seed=11, reset_options=reset_options)
    diagnostics = gated.episode_diagnostics()

    skip_counts = [
        diagnostics[f"gate_skip_steps__{margin:.4f}"] for margin in margins
    ]
    assert skip_counts == sorted(skip_counts)


def test_gate_outcome_divergence_bounded_by_switch_and_disagreement() -> None:
    """A skip can only diverge from Exact on steps that actually switched."""

    config = predictive_break_config(42)
    model = _policy(max_nodes=config.max_nodes)
    reset_options = predictive_break_options(0.0)
    margins = (0.0, 0.05, 1.0)
    gated = GateCalibrationAdapter(model, gate_margins=margins)

    run_episode(FanetRoutingEnv(config), gated, seed=13, reset_options=reset_options)
    diagnostics = gated.episode_diagnostics()

    upper_bound = min(
        diagnostics["switch_steps"], diagnostics["branch_disagreement_steps"]
    )
    for margin in margins:
        divergence = diagnostics[f"gate_outcome_divergence_steps__{margin:.4f}"]
        assert 0.0 <= divergence <= upper_bound


def test_gate_never_counts_drop_as_early_exit() -> None:
    """The passive counter must mirror the deployed guard, including DROP."""

    config = predictive_break_config(42)
    model = _policy(max_nodes=config.max_nodes)
    observation, _ = FanetRoutingEnv(config).reset(
        seed=17, options=predictive_break_options(0.0)
    )
    with torch.no_grad():
        # Make DROP the unique normal-branch choice while keeping risk benign.
        for parameter in model.normal_policy.parameters():
            parameter.zero_()
        final_linear = model.normal_policy.drop_scorer[-1]
        final_linear.bias.fill_(100.0)
    gated = GateCalibrationAdapter(model, gate_margins=(1_000.0,))
    gated.act_with_metadata(observation)
    diagnostics = gated.episode_diagnostics()

    assert diagnostics["gate_skip_steps__1000.0000"] == 0.0


def test_aggregate_gate_calibration_weights_by_exposure() -> None:
    rows = [
        {
            "scenario": "a",
            "training_seed": 1,
            "gate_decision_steps": 10.0,
            "gate_skip_steps__0.0000": 5.0,
            "gate_outcome_divergence_steps__0.0000": 1.0,
        },
        {
            "scenario": "a",
            "training_seed": 1,
            "gate_decision_steps": 20.0,
            "gate_skip_steps__0.0000": 10.0,
            "gate_outcome_divergence_steps__0.0000": 0.0,
        },
    ]
    aggregates = aggregate_gate_calibration(rows, gate_margins=(0.0,))
    overall = next(
        row
        for row in aggregates
        if row["scope"] == "overall" and row["gate_margin"] == "0.0000"
    )
    assert overall["decision_steps"] == 30.0
    assert overall["skip_rate"] == 0.5
    assert overall["outcome_divergence_rate_of_skipped"] == 1.0 / 15.0
