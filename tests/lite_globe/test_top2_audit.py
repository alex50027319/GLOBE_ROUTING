"""Top-2 synthetic stale-primary failover audit: single forward, correct resolve."""

from __future__ import annotations

import torch

from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.evaluation import top2_audit
from implementations.lite_globe.models import FastSwitchGlobePolicy
from implementations.lite_globe.scenarios.evaluation_suite import EvaluationScenario


def _tiny_scenarios(seed: int):
    config = FanetConfig(
        num_nodes=4, max_nodes=5, area_size=8.0, communication_radius=20.0,
        max_episode_steps=6, packet_ttl=6, max_queue_size=4,
        stochastic_link_loss=0.0, min_speed=0.0, max_speed=0.0,
        include_forwardability=True, include_risk_features=True, seed=seed,
    )
    return [
        EvaluationScenario(
            name="tiny", config=config, reset_options={}, distribution="in_distribution"
        )
    ]


def test_top2_synthetic_audit_reports_single_forward_and_correct_resolution(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(top2_audit, "phase9_evaluation_scenarios", _tiny_scenarios)
    seed = 1
    max_nodes = 5
    fast_dir = tmp_path / "fast"
    path = fast_dir / f"seed_{seed}" / "fast_switchglobe.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "complete": True, "training_seed": seed,
            "model_state": FastSwitchGlobePolicy(max_nodes, hidden_dim=32).state_dict(),
        },
        path,
    )
    config = top2_audit.Top2AuditConfig(
        training_seeds=(seed,), episodes_per_scenario=3, fast_hidden_dim=32,
        resolver_warmup=2, resolver_repeats=5,
    )
    result = top2_audit.run_top2_synthetic_failover_audit(
        config, fast_checkpoint_dir=fast_dir, device="cpu",
    )
    metrics = result["metrics"]
    assert metrics["total_decisions"] > 0
    assert metrics["decision_forward_count_anomalies"] == 0
    assert metrics["additional_neural_forwards_during_resolution"] == 0
    if metrics["eligible_failover_events"] > 0:
        assert metrics["failover_success_rate"] == 1.0
        assert metrics["both_invalid_drop_confirmation_rate"] == 1.0
    assert len(result["resolver_latency"]) == 1
    assert result["resolver_latency"][0]["training_seed"] == seed
