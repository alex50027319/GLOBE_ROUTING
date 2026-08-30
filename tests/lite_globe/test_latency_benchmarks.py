"""Unified latency benchmark additions: resolver-only timing, Fast loader."""

from __future__ import annotations

import torch

from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.evaluation.latency import benchmark_resolver
from implementations.lite_globe.models import FastSwitchGlobePolicy
from implementations.lite_globe.models.policy_adapter import StudentPolicyAdapter
from implementations.lite_globe.run_latency_benchmark import _load_fast


def _observation(max_nodes: int) -> dict:
    config = FanetConfig(
        num_nodes=4, max_nodes=max_nodes, area_size=8.0, communication_radius=20.0,
        max_episode_steps=6, min_speed=0.0, max_speed=0.0,
        include_forwardability=True, include_risk_features=True, seed=1,
    )
    observation, _ = FanetRoutingEnv(config).reset(seed=1, options={})
    return observation


def test_benchmark_resolver_measures_resolution_without_forward() -> None:
    max_nodes = 5
    model = FastSwitchGlobePolicy(max_nodes, hidden_dim=32)
    adapter = StudentPolicyAdapter(
        model, device="cpu", force_forward_if_available=True, enable_fast_failover=True,
    )
    observation = _observation(max_nodes)
    result = benchmark_resolver(adapter, observation, variant="test", warmup=2, repeats=5)
    assert result.component == "resolver_only"
    assert result.repeats == 5
    assert result.mean_ms >= 0.0


def test_load_fast_round_trips_checkpoint_and_top2_flag(tmp_path) -> None:
    max_nodes = 5
    seed = 7
    root = tmp_path / "fast" / f"seed_{seed}"
    root.mkdir(parents=True)
    model = FastSwitchGlobePolicy(max_nodes, hidden_dim=32)
    torch.save(
        {"complete": True, "training_seed": seed, "model_state": model.state_dict()},
        root / "fast_switchglobe.pt",
    )
    plain = _load_fast(
        tmp_path / "fast", seed=seed, max_nodes=max_nodes, device=torch.device("cpu"),
        enable_top2=False,
    )
    top2 = _load_fast(
        tmp_path / "fast", seed=seed, max_nodes=max_nodes, device=torch.device("cpu"),
        enable_top2=True,
    )
    assert plain.enable_fast_failover is False
    assert top2.enable_fast_failover is True
    observation = _observation(max_nodes)
    assert plain.act(observation) == top2.act(observation)
