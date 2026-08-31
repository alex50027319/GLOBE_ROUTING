"""Evo-inspired cost-to-go distillation guarantees."""

from __future__ import annotations

import numpy as np
import torch

from implementations.lite_globe.experiments.evo_globe_campaign import (
    CompositionalCurriculumConfig,
    CostToGoDistillationConfig,
    evaluate_cost_to_go_candidate,
    train_cost_to_go_switchglobe,
    train_compositional_switchglobe,
)
from implementations.lite_globe.models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from implementations.lite_globe.scenarios import (
    phase9_compositional_predictive_calibration_scenarios,
    phase9_compositional_predictive_training_scenarios,
    phase9_predictive_evaluation_scenarios,
    phase9_predictive_calibration_scenarios,
)


def _reference() -> SwitchGlobePolicy:
    return SwitchGlobePolicy(
        GeographicResidualStudentPolicy(32, hidden_dim=32),
        LiteGlobePStudentPolicy(32, hidden_dim=32),
    )


def test_cost_to_go_training_preserves_deployment_architecture() -> None:
    torch.manual_seed(29)
    reference = _reference()
    keys = tuple(reference.state_dict())
    parameters = sum(parameter.numel() for parameter in reference.parameters())
    candidate, metrics = train_cost_to_go_switchglobe(
        reference,
        CostToGoDistillationConfig(
            dataset_episodes_per_scenario=9,
            epochs=1,
            batch_size=64,
            return_action_coefficient=0.05,
            early_stopping_patience=0,
        ),
        seed=29,
    )

    assert tuple(candidate.state_dict()) == keys
    assert sum(parameter.numel() for parameter in candidate.parameters()) == parameters
    assert metrics["parameter_count"] == parameters
    assert metrics["dataset_samples"] > 0
    assert metrics["validation_rollout_agreement"] is not None


def test_cost_to_go_candidate_evaluation_returns_scenario_rows() -> None:
    rows = evaluate_cost_to_go_candidate(
        _reference(),
        phase9_predictive_calibration_scenarios(31),
        episode_seed_base=1_900_000,
        episodes_per_scenario=2,
    )
    assert 0.0 <= rows["connected_pair_pdr"] <= 1.0
    assert 0.0 <= rows["deadline_delivery_ratio"] <= 1.0
    assert len(rows["scenario_rows"]) == 1


def test_compositional_scenarios_keep_rotations_disjoint() -> None:
    training = phase9_compositional_predictive_training_scenarios(42)
    calibration = phase9_compositional_predictive_calibration_scenarios(42)
    evaluation = phase9_predictive_evaluation_scenarios(42)

    assert len(training) == 3
    assert len(calibration) == 2
    assert not ({item.name for item in training} & {item.name for item in evaluation})
    assert not ({item.name for item in calibration} & {item.name for item in evaluation})
    assert all(item.config.stochastic_link_loss > 0 for item in training)
    speeds = [
        float(np.linalg.norm(item.reset_options["velocities"][2]))
        for item in training
    ]
    assert len(set(speeds)) == 3
    assert max(speeds) <= 1.5


def test_compositional_training_changes_only_predictive_prior_scalars() -> None:
    torch.manual_seed(41)
    reference = _reference()
    normal_before = {
        key: value.detach().clone()
        for key, value in reference.normal_policy.state_dict().items()
    }
    keys = tuple(reference.state_dict())
    parameters = sum(parameter.numel() for parameter in reference.parameters())
    candidate, metrics = train_compositional_switchglobe(
        reference,
        CompositionalCurriculumConfig(
            dataset_episodes_per_scenario=9,
            epochs=1,
            batch_size=64,
            learning_rate=1e-2,
            early_stopping_patience=0,
        ),
        seed=41,
    )

    assert tuple(candidate.state_dict()) == keys
    assert sum(parameter.numel() for parameter in candidate.parameters()) == parameters
    assert metrics["parameter_count"] == parameters
    assert set(metrics["changed_state_keys"]).issubset(
        {
            "predictive_policy.log_predictive_strength",
            "predictive_policy.log_break_penalty",
        }
    )
    assert metrics["validation_risk_oracle_agreement"] is not None
    assert float(candidate.predictive_policy.residual_weight.item()) == 0.0
    for key, before in normal_before.items():
        torch.testing.assert_close(candidate.normal_policy.state_dict()[key], before)
