"""Each new run_*.py CLI's manifest must carry git/config/checkpoint provenance.

These call the real ``main()`` entry points (not just their library
functions) against tiny scenarios and tiny checkpoints, so a regression in
how the CLI wires provenance into its manifest is caught even though the
underlying campaign logic is already covered elsewhere.
"""

from __future__ import annotations

import sys

import torch

from implementations.lite_globe import run_ablation, run_fast_switchglobe, run_latency_benchmark, run_top2_audit
from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.experiments import ablation_campaign, latency_optimization_campaign as loc
from implementations.lite_globe.evaluation import top2_audit
from implementations.lite_globe.models import (
    FastSwitchGlobePolicy,
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from implementations.lite_globe.scenarios.evaluation_suite import EvaluationScenario
from implementations.lite_globe.utils import save_checkpoint


def _tiny_scenarios_factory(*, max_nodes: int, num_nodes: int = 3):
    def _factory(seed: int):
        config = FanetConfig(
            num_nodes=num_nodes, max_nodes=max_nodes, area_size=10.0,
            communication_radius=20.0, max_episode_steps=4, packet_ttl=4,
            max_queue_size=4, stochastic_link_loss=0.0, min_speed=0.0,
            max_speed=0.0, include_forwardability=True,
            include_risk_features=True, seed=seed,
        )
        return [
            EvaluationScenario(
                name="tiny", config=config, reset_options={}, distribution="in_distribution"
            )
        ]
    return _factory


def _assert_provenance_shape(fields: dict) -> None:
    commit = fields.get("git_commit_hash")
    assert commit is None or (isinstance(commit, str) and len(commit) == 40)
    assert isinstance(fields.get("dirty_files"), list)
    config_hash = fields.get("config_sha256")
    assert isinstance(config_hash, str) and len(config_hash) == 64
    checkpoint_hashes = fields.get("checkpoint_sha256")
    assert isinstance(checkpoint_hashes, dict) and checkpoint_hashes
    for value in checkpoint_hashes.values():
        assert isinstance(value, str) and len(value) == 64


def test_run_fast_switchglobe_manifest_has_provenance(tmp_path, monkeypatch) -> None:
    seed = 1
    max_nodes = 4
    tiny = _tiny_scenarios_factory(max_nodes=max_nodes)
    monkeypatch.setattr(run_fast_switchglobe, "phase9_evaluation_scenarios", tiny)
    monkeypatch.setattr(loc, "training_scenarios", tiny)

    switchglobe_dir = tmp_path / "switchglobe"
    save_checkpoint(
        switchglobe_dir / f"seed_{seed}" / "switchglobe.pt",
        SwitchGlobePolicy(
            GeographicResidualStudentPolicy(max_nodes, hidden_dim=64),
            LiteGlobePStudentPolicy(max_nodes, hidden_dim=64),
        ),
    )
    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "run_fast_switchglobe", "--smoke", "--seed", str(seed), "--device", "cpu",
        "--switchglobe-checkpoint-dir", str(switchglobe_dir),
        "--output-dir", str(output_dir),
    ])
    assert run_fast_switchglobe.main() == 0
    manifest = __import__("json").loads((output_dir / "manifest.json").read_text())
    _assert_provenance_shape(manifest)
    assert f"switchglobe_exact_seed_{seed}" in manifest["checkpoint_sha256"]
    assert f"fast_switchglobe_seed_{seed}" in manifest["checkpoint_sha256"]


def test_run_ablation_manifest_has_provenance(tmp_path, monkeypatch) -> None:
    seed = 1
    max_nodes = 4
    tiny = _tiny_scenarios_factory(max_nodes=max_nodes)
    monkeypatch.setattr(ablation_campaign, "phase9_evaluation_scenarios", tiny)
    monkeypatch.setattr(
        "implementations.lite_globe.evaluation.ablation_reporting.SCENARIOS", ("tiny",)
    )

    phase8_dir, phase11_dir = tmp_path / "phase8", tmp_path / "phase11"
    switchglobe_dir, fast_dir = tmp_path / "switchglobe", tmp_path / "fast"
    save_checkpoint(
        phase8_dir / f"seed_{seed}" / "geo_residual_kd.pt",
        GeographicResidualStudentPolicy(max_nodes, hidden_dim=64),
    )
    save_checkpoint(
        phase11_dir / f"seed_{seed}" / "lite_globe_p.pt",
        LiteGlobePStudentPolicy(max_nodes, hidden_dim=64),
    )
    save_checkpoint(
        switchglobe_dir / f"seed_{seed}" / "switchglobe.pt",
        SwitchGlobePolicy(
            GeographicResidualStudentPolicy(max_nodes, hidden_dim=64),
            LiteGlobePStudentPolicy(max_nodes, hidden_dim=64),
        ),
    )
    fast_path = fast_dir / f"seed_{seed}" / "fast_switchglobe.pt"
    fast_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "complete": True, "training_seed": seed,
            "model_state": FastSwitchGlobePolicy(max_nodes, hidden_dim=32).state_dict(),
        },
        fast_path,
    )
    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "run_ablation", "--smoke", "--seed", str(seed), "--device", "cpu",
        "--phase8-checkpoint-dir", str(phase8_dir),
        "--phase11-checkpoint-dir", str(phase11_dir),
        "--switchglobe-checkpoint-dir", str(switchglobe_dir),
        "--fast-checkpoint-dir", str(fast_dir),
        "--output-dir", str(output_dir),
    ])
    assert run_ablation.main() == 0
    manifest = __import__("json").loads((output_dir / "manifest.json").read_text())
    _assert_provenance_shape(manifest["metadata"])
    checkpoint_hashes = manifest["metadata"]["checkpoint_sha256"]
    for label in (
        f"geo_residual_seed_{seed}", f"predictive_seed_{seed}",
        f"switchglobe_exact_seed_{seed}", f"fast_switchglobe_seed_{seed}",
    ):
        assert label in checkpoint_hashes


def test_run_top2_audit_manifest_has_provenance(tmp_path, monkeypatch) -> None:
    seed = 1
    max_nodes = 5
    tiny = _tiny_scenarios_factory(max_nodes=max_nodes, num_nodes=4)
    monkeypatch.setattr(top2_audit, "phase9_evaluation_scenarios", tiny)

    fast_dir = tmp_path / "fast"
    fast_path = fast_dir / f"seed_{seed}" / "fast_switchglobe.pt"
    fast_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "complete": True, "training_seed": seed,
            "model_state": FastSwitchGlobePolicy(max_nodes, hidden_dim=32).state_dict(),
        },
        fast_path,
    )
    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "run_top2_audit", "--smoke", "--seed", str(seed), "--device", "cpu",
        "--fast-checkpoint-dir", str(fast_dir),
        "--output-dir", str(output_dir),
    ])
    assert run_top2_audit.main() == 0
    manifest = __import__("json").loads((output_dir / "manifest.json").read_text())
    _assert_provenance_shape(manifest["metadata"])
    assert f"fast_switchglobe_seed_{seed}" in manifest["metadata"]["checkpoint_sha256"]


def test_run_latency_benchmark_manifest_has_provenance(tmp_path, monkeypatch) -> None:
    seed = 1
    max_nodes = 5
    tiny = _tiny_scenarios_factory(max_nodes=max_nodes, num_nodes=4)
    monkeypatch.setattr(run_latency_benchmark, "phase9_evaluation_scenarios", tiny)

    checkpoint_dir, fast_dir = tmp_path / "exact", tmp_path / "fast"
    save_checkpoint(
        checkpoint_dir / f"seed_{seed}" / "switchglobe.pt",
        SwitchGlobePolicy(
            GeographicResidualStudentPolicy(max_nodes, hidden_dim=64),
            LiteGlobePStudentPolicy(max_nodes, hidden_dim=64),
        ),
    )
    fast_path = fast_dir / f"seed_{seed}" / "fast_switchglobe.pt"
    fast_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "complete": True, "training_seed": seed,
            "model_state": FastSwitchGlobePolicy(max_nodes, hidden_dim=32).state_dict(),
        },
        fast_path,
    )
    output_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "run_latency_benchmark", "--smoke", "--seed", str(seed), "--include-fast",
        "--checkpoint-dir", str(checkpoint_dir),
        "--fast-checkpoint-dir", str(fast_dir),
        "--output-dir", str(output_dir),
    ])
    assert run_latency_benchmark.main() == 0
    manifest = __import__("json").loads((output_dir / "manifest.json").read_text())
    _assert_provenance_shape(manifest)
    assert f"switchglobe_exact_seed_{seed}" in manifest["checkpoint_sha256"]
    assert f"fast_switchglobe_seed_{seed}" in manifest["checkpoint_sha256"]
