"""FastSwitchGLOBE training CLI contract: config hashing and resume safety."""

from __future__ import annotations

import torch

from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.experiments import latency_optimization_campaign as loc
from implementations.lite_globe.experiments.latency_optimization_campaign import (
    LatencyOptimizationConfig,
    config_sha256,
    train_or_load_fast,
)
from implementations.lite_globe.models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from implementations.lite_globe.models.policy_adapter import StudentPolicyAdapter
from implementations.lite_globe.scenarios.generalization_suite import CurriculumStage


def _tiny_scenarios(seed: int):
    config = FanetConfig(
        num_nodes=3, max_nodes=4, area_size=10.0, communication_radius=4.5,
        max_episode_steps=4, packet_ttl=4, max_queue_size=4,
        stochastic_link_loss=0.0, min_speed=0.0, max_speed=0.0,
        include_forwardability=True, include_risk_features=True, seed=seed,
    )
    return [CurriculumStage(name="tiny", config=config, reset_options={})]


def _teacher(max_nodes: int) -> StudentPolicyAdapter:
    model = SwitchGlobePolicy(
        GeographicResidualStudentPolicy(max_nodes, hidden_dim=32),
        LiteGlobePStudentPolicy(max_nodes, hidden_dim=32),
    )
    model.eval()
    return StudentPolicyAdapter(model, device="cpu", force_forward_if_available=True)


def _tiny_config(**overrides) -> LatencyOptimizationConfig:
    defaults = dict(
        dataset_episodes_per_scenario=2, epochs=2, batch_size=4, hidden_dim=32,
    )
    defaults.update(overrides)
    return LatencyOptimizationConfig(**defaults)


def test_config_sha256_is_deterministic_and_sensitive_to_changes() -> None:
    base = LatencyOptimizationConfig()
    same = LatencyOptimizationConfig()
    changed = LatencyOptimizationConfig(epochs=base.epochs + 1)
    assert config_sha256(base) == config_sha256(same)
    assert config_sha256(base) != config_sha256(changed)


def test_train_or_load_fast_tracks_best_epoch_and_sample_counts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loc, "training_scenarios", _tiny_scenarios)
    config = _tiny_config()
    teacher = _teacher(4)
    _, training = train_or_load_fast(
        teacher, config=config, seed=1, checkpoint_dir=tmp_path,
        device=torch.device("cpu"), resume=False,
    )
    assert training["epochs"] == config.epochs
    assert 1 <= training["best_epoch"] <= training["epochs"]
    assert training["training_samples"] > 0
    assert training["validation_samples"] >= 0
    assert training["test_samples"] >= 0
    assert training["resumed"] == 0


def test_resume_accepts_matching_config_and_rejects_drift(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(loc, "training_scenarios", _tiny_scenarios)
    teacher = _teacher(4)
    config = _tiny_config()
    train_or_load_fast(
        teacher, config=config, seed=1, checkpoint_dir=tmp_path,
        device=torch.device("cpu"), resume=False,
    )
    _, resumed_training = train_or_load_fast(
        teacher, config=config, seed=1, checkpoint_dir=tmp_path,
        device=torch.device("cpu"), resume=True,
    )
    assert resumed_training["resumed"] == 1

    drifted_config = _tiny_config(epochs=config.epochs + 1)
    try:
        train_or_load_fast(
            teacher, config=drifted_config, seed=1, checkpoint_dir=tmp_path,
            device=torch.device("cpu"), resume=True,
        )
    except ValueError as error:
        assert "different config" in str(error)
    else:
        raise AssertionError("expected resume to refuse a config-hash mismatch")
