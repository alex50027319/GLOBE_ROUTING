"""Evo-inspired cost-to-go distillation guarantees."""

from __future__ import annotations

import torch

from implementations.lite_globe.experiments.evo_globe_campaign import (
    CostToGoDistillationConfig,
    evaluate_cost_to_go_candidate,
    train_cost_to_go_switchglobe,
)
from implementations.lite_globe.models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from implementations.lite_globe.scenarios import (
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
