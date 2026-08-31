"""Contract tests for the causal SwitchGLOBE novelty ablation."""

from __future__ import annotations

from copy import deepcopy

import torch

from implementations.lite_globe.evaluation.novelty_ablation_reporting import (
    write_novelty_ablation_artifacts,
)
from implementations.lite_globe.experiments.novelty_ablation_campaign import (
    FULL,
    NO_DISTILLATION,
    NO_GEO_RESIDUAL,
    NO_RISK_SWITCH,
    NoveltyAblationConfig,
    build_novelty_ablation_policies,
    run_novelty_ablation_campaign,
)
from implementations.lite_globe.models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    LocalStudentPolicy,
    SwitchGlobePolicy,
)
from implementations.lite_globe.utils import save_checkpoint


def _checkpoints(tmp_path):
    phase7 = tmp_path / "phase7"
    phase8 = tmp_path / "phase8"
    phase12 = tmp_path / "phase12"
    torch.manual_seed(42)
    kd_only = LocalStudentPolicy(32, hidden_dim=64)
    normal = GeographicResidualStudentPolicy(32, hidden_dim=64)
    predictive = LiteGlobePStudentPolicy(32, hidden_dim=64)
    full = SwitchGlobePolicy(deepcopy(normal), predictive)
    save_checkpoint(phase7 / "seed_42" / "kd_only_student.pt", kd_only)
    save_checkpoint(phase8 / "seed_42" / "geo_residual_kd.pt", normal)
    save_checkpoint(
        phase12 / "seed_42" / "risk_switch_lite_globe_p.pt",
        full,
    )
    return phase7, phase8, phase12


def test_leave_one_out_variants_preserve_requested_components(tmp_path) -> None:
    phase7, phase8, phase12 = _checkpoints(tmp_path)
    policies = build_novelty_ablation_policies(
        NoveltyAblationConfig(training_seeds=(42,), evaluation_episodes=1),
        seed=42,
        max_nodes=32,
        phase7_checkpoint_dir=phase7,
        phase8_checkpoint_dir=phase8,
        switchglobe_checkpoint_dir=phase12,
        device=torch.device("cpu"),
    )

    assert isinstance(policies[FULL].model, SwitchGlobePolicy)
    assert isinstance(
        policies[NO_RISK_SWITCH].model,
        GeographicResidualStudentPolicy,
    )
    no_geo = policies[NO_GEO_RESIDUAL].model
    assert isinstance(no_geo, SwitchGlobePolicy)
    assert type(no_geo.normal_policy) is LocalStudentPolicy
    no_distillation = policies[NO_DISTILLATION].model
    assert isinstance(no_distillation, SwitchGlobePolicy)
    assert isinstance(
        no_distillation.normal_policy,
        GeographicResidualStudentPolicy,
    )
    assert torch.count_nonzero(
        no_distillation.normal_policy.candidate_scorer[-1].weight
    ).item() == 0
    torch.testing.assert_close(
        no_geo.predictive_policy.log_predictive_strength,
        policies[FULL].model.predictive_policy.log_predictive_strength,
    )


def test_novelty_campaign_and_reporting_contract(tmp_path) -> None:
    phase7, phase8, phase12 = _checkpoints(tmp_path)
    config = NoveltyAblationConfig(
        training_seeds=(42,),
        evaluation_episodes=1,
    )
    rows = run_novelty_ablation_campaign(
        config,
        phase7_checkpoint_dir=phase7,
        phase8_checkpoint_dir=phase8,
        switchglobe_checkpoint_dir=phase12,
    )
    assert len(rows["episodes"]) == 4 * 14
    assert len(rows["seed_summaries"]) == 4 * 14

    manifest = write_novelty_ablation_artifacts(
        tmp_path / "output",
        episode_rows=rows["episodes"],
        summary_rows=rows["seed_summaries"],
        metadata={
            "mode": "smoke",
            "config": {
                "training_seeds": [42],
                "evaluation_episodes": 1,
            },
        },
    )
    assert manifest["complete"] is True
    assert manifest["episode_rows"] == 56
    assert manifest["seed_overall_rows"] == 4
    assert manifest["paired_effect_rows"] == 12
