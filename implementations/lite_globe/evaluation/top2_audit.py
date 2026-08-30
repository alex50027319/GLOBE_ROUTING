"""Synthetic stale-primary failover audit for FastSwitchGLOBE + Top-2.

This is NOT a wireless link-failure test. The simulator has no natural race
between policy inference and ``env.step()``, so this audit synthetically
invalidates the primary action in a copy of the live action mask *after* the
model has already committed to a primary/backup pair, and checks that
``resolve_decision`` (which performs no model inference) recovers the backup
without an extra forward pass. Report these numbers as a synthetic audit,
never as evidence about real wireless failures.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..env.fanet_env import FanetRoutingEnv
from ..experiments.latency_optimization_campaign import (
    checkpoint_path as fast_checkpoint_path,
)
from ..models.student_policy import FastSwitchGlobePolicy
from ..models.policy_adapter import StudentPolicyAdapter
from ..scenarios import phase9_evaluation_scenarios
from .latency import benchmark_resolver
from .reporting import write_csv


@dataclass(frozen=True)
class Top2AuditConfig:
    training_seeds: tuple[int, ...] = (42, 77, 123, 314, 2718)
    episodes_per_scenario: int = 20
    fast_hidden_dim: int = 32
    resolver_warmup: int = 50
    resolver_repeats: int = 500


def _load_fast_top2(
    fast_checkpoint_dir: Path, *, seed: int, max_nodes: int, hidden_dim: int,
    device: torch.device,
) -> StudentPolicyAdapter:
    from .. import experiments  # local import to avoid a package cycle
    path = experiments.latency_optimization_campaign.checkpoint_path(fast_checkpoint_dir, seed)
    if not path.is_file():
        raise FileNotFoundError(
            f"FastSwitchGLOBE checkpoint not found for seed {seed} at {path}; "
            "run run_fast_switchglobe.py first"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = FastSwitchGlobePolicy(max_nodes, hidden_dim=hidden_dim)
    model.load_state_dict(payload["model_state"])
    return StudentPolicyAdapter(
        model, device=device, force_forward_if_available=True,
        enable_fast_failover=True,
    )


def run_top2_synthetic_failover_audit(
    config: Top2AuditConfig, *, fast_checkpoint_dir: Path,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    device = torch.device(device)
    total_decisions = 0
    eligible_events = 0
    successful_backup_resolutions = 0
    both_invalid_drop_confirmations = 0
    decision_forward_anomalies = 0
    additional_neural_forwards = 0
    resolver_rows: list[dict[str, Any]] = []
    for seed in config.training_seeds:
        scenarios = phase9_evaluation_scenarios(seed)
        max_nodes = scenarios[0].config.max_nodes
        adapter = _load_fast_top2(
            fast_checkpoint_dir, seed=seed, max_nodes=max_nodes,
            hidden_dim=config.fast_hidden_dim, device=device,
        )
        forward_count = {"n": 0}
        handle = adapter.model.register_forward_hook(
            lambda module, args, output: forward_count.__setitem__(
                "n", forward_count["n"] + 1
            )
        )
        try:
            for scenario_index, scenario in enumerate(scenarios):
                env = FanetRoutingEnv(scenario.config)
                for episode_index in range(config.episodes_per_scenario):
                    episode_seed = (
                        1_900_000 + scenario_index * 10_000 + episode_index
                    )
                    observation, _ = env.reset(
                        seed=episode_seed, options=scenario.reset_options
                    )
                    adapter.reset(episode_seed)
                    done = False
                    while not done:
                        before = forward_count["n"]
                        decision = adapter.act_with_metadata(observation)
                        after_decision = forward_count["n"]
                        total_decisions += 1
                        if after_decision - before != 1:
                            decision_forward_anomalies += 1
                        if decision.backup_action is not None:
                            eligible_events += 1
                            live_mask = np.array(
                                observation["action_mask"], copy=True
                            )
                            primary = decision.action
                            if 0 <= primary < adapter.model.max_nodes:
                                live_mask[primary] = False
                            resolved = adapter.resolve_decision(
                                decision, live_mask
                            )
                            after_single_invalid = forward_count["n"]
                            additional_neural_forwards += max(
                                after_single_invalid - after_decision, 0
                            )
                            if resolved == decision.backup_action:
                                successful_backup_resolutions += 1
                            both_invalid_mask = np.array(live_mask, copy=True)
                            backup = decision.backup_action
                            if 0 <= backup < adapter.model.max_nodes:
                                both_invalid_mask[backup] = False
                            resolved_both = adapter.resolve_decision(
                                decision, both_invalid_mask
                            )
                            after_both_invalid = forward_count["n"]
                            additional_neural_forwards += max(
                                after_both_invalid - after_single_invalid, 0
                            )
                            if resolved_both == adapter.model.drop_action:
                                both_invalid_drop_confirmations += 1
                        observation, _, terminated, truncated, _ = env.step(
                            decision.action
                        )
                        done = bool(terminated or truncated)
        finally:
            handle.remove()
        bench_scenario = scenarios[0]
        bench_observation, _ = FanetRoutingEnv(bench_scenario.config).reset(
            seed=1_099_999, options=bench_scenario.reset_options
        )
        resolver_benchmark = benchmark_resolver(
            adapter, bench_observation, variant=f"seed_{seed}",
            warmup=config.resolver_warmup, repeats=config.resolver_repeats,
        )
        resolver_rows.append({"training_seed": seed, **resolver_benchmark.to_dict()})
    metrics = {
        "note": "synthetic stale-primary audit; not a real wireless link-failure test",
        "total_decisions": total_decisions,
        "eligible_failover_events": eligible_events,
        "backup_availability_rate": (
            eligible_events / total_decisions if total_decisions else None
        ),
        "successful_backup_resolutions": successful_backup_resolutions,
        "failover_success_rate": (
            successful_backup_resolutions / eligible_events
            if eligible_events else None
        ),
        "failover_miss_rate": (
            (eligible_events - successful_backup_resolutions) / eligible_events
            if eligible_events else None
        ),
        "both_invalid_drop_confirmations": both_invalid_drop_confirmations,
        "both_invalid_drop_confirmation_rate": (
            both_invalid_drop_confirmations / eligible_events
            if eligible_events else None
        ),
        "decision_forward_count_anomalies": decision_forward_anomalies,
        "additional_neural_forwards_during_resolution": additional_neural_forwards,
    }
    return {"metrics": metrics, "resolver_latency": resolver_rows}


def write_top2_audit_report(
    output_dir: Path, *, metrics: dict[str, Any],
    resolver_rows: list[dict[str, Any]], metadata: dict[str, Any],
) -> dict[str, Any]:
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "resolver_latency.csv", resolver_rows)
    lines = [
        "# Top-2 synthetic stale-primary failover audit",
        "",
        "This audits the Top-2 backup-resolution mechanism under a "
        "synthetically invalidated primary action. It is NOT a real "
        "wireless link-failure test.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in metrics.items():
        if key == "note":
            continue
        lines.append(f"| {key} | {value} |")
    (output_dir / "top2_synthetic_failover_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "complete": True,
        "suite": "top2_synthetic_stale_primary_audit",
        "metrics": metrics,
        "resolver_latency_rows": len(resolver_rows),
        "metadata": metadata,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
