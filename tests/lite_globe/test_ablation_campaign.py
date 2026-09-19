"""SwitchGLOBE ablation harness: schema, checkpoint reuse, Fast/Top-2 parity."""

from __future__ import annotations

import torch

from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.experiments import ablation_campaign as ablation
from implementations.lite_globe.evaluation import ablation_reporting
from implementations.lite_globe.evaluation.ablation_reporting import (
    ABLATION_METHODS,
    FAST_SWITCHGLOBE,
    FAST_SWITCHGLOBE_TOP2,
    SWITCHGLOBE_EXACT,
    paired_effects_vs_exact,
    validate_fast_top2_outcome_equality,
    validate_summary_rows,
)
from implementations.lite_globe.models import (
    FastSwitchGlobePolicy,
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from implementations.lite_globe.scenarios.evaluation_suite import EvaluationScenario
from implementations.lite_globe.utils import save_checkpoint


def _tiny_scenarios(seed: int):
    config = FanetConfig(
        num_nodes=3, max_nodes=4, area_size=10.0, communication_radius=4.5,
        max_episode_steps=4, packet_ttl=4, max_queue_size=4,
        stochastic_link_loss=0.0, min_speed=0.0, max_speed=0.0,
        include_forwardability=True, include_risk_features=True, seed=seed,
    )
    return [
        EvaluationScenario(
            name="tiny", config=config, reset_options={}, distribution="in_distribution"
        )
    ]


def _write_checkpoints(root, *, seed: int, max_nodes: int, hidden_dim: int, fast_hidden_dim: int):
    phase8_dir, phase11_dir = root / "phase8", root / "phase11"
    switchglobe_dir, fast_dir = root / "switchglobe", root / "fast"
    save_checkpoint(
        phase8_dir / f"seed_{seed}" / "geo_residual_kd.pt",
        GeographicResidualStudentPolicy(max_nodes, hidden_dim=hidden_dim),
    )
    save_checkpoint(
        phase11_dir / f"seed_{seed}" / "lite_globe_p.pt",
        LiteGlobePStudentPolicy(max_nodes, hidden_dim=hidden_dim),
    )
    save_checkpoint(
        switchglobe_dir / f"seed_{seed}" / "switchglobe.pt",
        SwitchGlobePolicy(
            GeographicResidualStudentPolicy(max_nodes, hidden_dim=hidden_dim),
            LiteGlobePStudentPolicy(max_nodes, hidden_dim=hidden_dim),
        ),
    )
    fast_path = fast_dir / f"seed_{seed}" / "fast_switchglobe.pt"
    fast_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "complete": True, "training_seed": seed,
            "model_state": FastSwitchGlobePolicy(max_nodes, hidden_dim=fast_hidden_dim).state_dict(),
        },
        fast_path,
    )
    return phase8_dir, phase11_dir, switchglobe_dir, fast_dir


def test_ablation_campaign_schema_and_fast_top2_parity(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ablation, "phase9_evaluation_scenarios", _tiny_scenarios)
    monkeypatch.setattr(ablation_reporting, "SCENARIOS", ("tiny",))
    seed = 1
    phase8_dir, phase11_dir, switchglobe_dir, fast_dir = _write_checkpoints(
        tmp_path, seed=seed, max_nodes=4, hidden_dim=32, fast_hidden_dim=32,
    )
    config = ablation.AblationConfig(
        training_seeds=(seed,), evaluation_episodes=2, hidden_dim=32, fast_hidden_dim=32,
    )
    rows = ablation.run_ablation_campaign(
        config,
        phase8_checkpoint_dir=phase8_dir, phase11_checkpoint_dir=phase11_dir,
        switchglobe_checkpoint_dir=switchglobe_dir, fast_checkpoint_dir=fast_dir,
        device="cpu",
    )
    assert len(rows["episodes"]) == len(ABLATION_METHODS) * 1 * config.evaluation_episodes
    assert {row["method"] for row in rows["seed_summaries"]} == set(ABLATION_METHODS)
    validate_summary_rows(rows["seed_summaries"], training_seeds=(seed,))
    assert validate_fast_top2_outcome_equality(rows["episodes"]) == 0


def test_fast_top2_outcome_mismatch_is_detected() -> None:
    good = {
        "method": FAST_SWITCHGLOBE, "scenario": "s", "training_seed": 1,
        "evaluation_seed": 1, "delivered": 1, "dropped": 0, "drop_reason": "",
        "steps": 3, "hop_count": 2, "transmission_attempts": 3,
    }
    tampered = {**good, "method": FAST_SWITCHGLOBE_TOP2, "steps": 4}
    try:
        validate_fast_top2_outcome_equality([good, tampered])
    except ValueError:
        pass
    else:
        raise AssertionError("expected a Fast/Top-2 outcome mismatch to be detected")


def test_paired_effects_contrast_every_variant_against_exact_only(monkeypatch) -> None:
    monkeypatch.setattr(ablation_reporting, "SCENARIOS", ("s",))
    rows = []
    for method in ABLATION_METHODS:
        for seed in (1, 2):
            rows.append({
                "method": method, "scenario": "s", "training_seed": seed,
                "connected_pair_pdr": 0.8, "deadline_delivery_ratio": 0.7,
                "p95_success_delay": 4.0, "energy_per_delivered_packet": 1.2,
                "decision_latency_p95_ms": 0.2, "mean_policy_input_bytes": 128.0,
                "switch_activation_rate": 0.1, "backup_availability_rate": 0.0,
                "fast_failover_success_rate": 0.0,
            })
    effects = paired_effects_vs_exact(rows)
    assert {row["variant"] for row in effects} == set(ABLATION_METHODS) - {SWITCHGLOBE_EXACT}
    assert all(row["baseline"] == SWITCHGLOBE_EXACT for row in effects)
