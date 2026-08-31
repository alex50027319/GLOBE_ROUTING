"""Phase 10 external RL baselines and reporting guarantees."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from implementations.lite_globe.baselines import (
    DramaPolicy,
    EvoQGeoPolicy,
    IqmrPolicy,
)
from implementations.lite_globe.baselines.external_rl import (
    candidate_feature_matrix,
    train_drama_baseline,
    train_value_baseline,
)
from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.evaluation.phase10_reporting import (
    write_phase10_artifacts,
)
from implementations.lite_globe.scenarios.evaluation_suite import (
    EvaluationScenario,
)


def _line_scenario(config: FanetConfig, positions: np.ndarray) -> EvaluationScenario:
    return EvaluationScenario(
        "line",
        config,
        {
            "positions": positions,
            "source": 0,
            "destination": 2,
        },
        "test",
    )


def test_candidate_features_are_mask_aligned(line_positions) -> None:
    config = FanetConfig(
        num_nodes=3,
        max_nodes=4,
        communication_radius=1.1,
        min_speed=0.0,
        max_speed=0.0,
        include_forwardability=True,
        include_risk_features=True,
    )
    env = FanetRoutingEnv(config)
    observation, _ = env.reset(
        seed=1,
        options={
            "positions": line_positions,
            "source": 0,
            "destination": 2,
        },
    )
    features, mask, prior = candidate_feature_matrix(
        observation,
        env.drop_action,
    )
    assert features.shape == (config.max_nodes, 13)
    assert mask.tolist() == [False, True, False, False]
    assert prior[1] > -1e5
    assert np.all(features[~mask] == 0.0)


def test_evo_qgeo_and_iqmr_learn_finite_updates(line_positions) -> None:
    config = FanetConfig(
        num_nodes=3,
        max_nodes=4,
        communication_radius=1.1,
        min_speed=0.0,
        max_speed=0.0,
        include_forwardability=True,
        include_risk_features=True,
    )
    scenario = _line_scenario(config, line_positions)
    for policy in (EvoQGeoPolicy(config.max_nodes), IqmrPolicy(config.max_nodes)):
        result = train_value_baseline(
            FanetRoutingEnv,
            policy,
            [scenario],
            training_seed=5,
            episodes_per_stage=3,
        )
        assert result.episodes == 3
        assert result.updates > 0
        assert np.isfinite(result.mean_training_td_error)
        observation, _ = FanetRoutingEnv(config).reset(
            seed=3,
            options=scenario.reset_options,
        )
        action = policy.act(observation)
        assert observation["action_mask"][action] == 1


def test_drama_network_trains_and_returns_legal_action(line_positions) -> None:
    torch.manual_seed(4)
    config = FanetConfig(
        num_nodes=3,
        max_nodes=4,
        communication_radius=1.1,
        min_speed=0.0,
        max_speed=0.0,
        include_forwardability=True,
        include_risk_features=True,
    )
    scenario = _line_scenario(config, line_positions)
    policy = DramaPolicy(config.max_nodes, hidden_dim=16)
    result = train_drama_baseline(
        FanetRoutingEnv,
        policy,
        [scenario],
        training_seed=4,
        episodes_per_stage=5,
        batch_size=4,
        replay_capacity=64,
    )
    assert result.episodes == 5
    assert result.updates > 0
    observation, _ = FanetRoutingEnv(config).reset(
        seed=8,
        options=scenario.reset_options,
    )
    action = policy.act(observation)
    assert observation["action_mask"][action] == 1


def test_phase10_artifacts_include_tables_and_figures(tmp_path: Path) -> None:
    summary_rows = []
    for method_index, method in enumerate(
        (
            "GPSR",
            "Predictive Geographic",
            "Evo-QGeo",
            "IQMR Q(lambda)",
            "DRAMA",
            "Phase 8 Geo-Residual KD",
        )
    ):
        for seed in (1, 2):
            for scenario in (
                "heldout_medium",
                "ood_link_loss",
                "ood_fast_mobility",
                "ood_sparse",
                "ood_nodes_10",
                "unconditional_sparse",
                "structural_hole_45",
                "structural_hole_225_link_loss",
                "predictive_break_45",
                "predictive_break_225_link_loss",
                "ood_link_loss_30",
                "ood_extreme_mobility",
                "ood_nodes_16",
                "ood_nodes_24",
            ):
                value = 0.05 * method_index + 0.01 * seed
                summary_rows.append(
                    {
                        "method": method,
                        "scenario": scenario,
                        "training_seed": seed,
                        "episodes": 5,
                        "connected_episodes": 5,
                        "delivered": 4,
                        "endpoint_availability": 1.0,
                        "overall_pdr": value,
                        "connected_pair_pdr": value,
                        "mean_success_delay": 3.0,
                        "p95_success_delay": 4.0 + value,
                        "mean_path_stretch": 1.2,
                        "loop_drop_rate": 0.0,
                        "invalid_action_drop_rate": 0.0,
                        "ttl_drop_rate": 0.0,
                        "agent_drop_rate": 0.0,
                        "mean_expected_transmissions_proxy": 2.0,
                        "mean_transmission_energy_proxy": 1.0 + value,
                        "mean_minimum_link_lifetime_steps": 2.0,
                        "mean_minimum_link_margin": 0.5,
                        "mean_queue_delay_proxy": 2.0,
                        "deadline_delivery_ratio": value,
                        "delivery_energy_efficiency_proxy": 0.5,
                        "delivery_transmission_efficiency_proxy": 0.5,
                        "mean_local_observation_bytes": 100.0,
                        "mean_policy_input_bytes": 100.0 + method_index,
                        "mean_switch_steps": 1.0,
                        "mean_safe_forward_candidates": 2.0,
                        "mean_selected_danger": 0.1,
                        "mean_episode_reward": 1.0,
                    }
                )
    manifest = write_phase10_artifacts(
        tmp_path,
        episode_rows=[{"method": "GPSR", "scenario": "heldout_medium"}],
        summary_rows=summary_rows,
        training_rows=[{"training_seed": 1}],
        metadata={"phase": 10, "mode": "test"},
    )
    assert manifest["statistics_rows"] > 0
    for relative in (
        "raw/episodes.csv",
        "summaries/statistics.json",
        "summaries/paired_effects.csv",
        "tables/external_rl_main_results.md",
        "tables/phase8_improvement_over_external_rl.md",
        "figures/external_rl_pdr.png",
        "figures/external_rl_delay_p95.pdf",
        "manifest.json",
    ):
        assert (tmp_path / relative).is_file()
