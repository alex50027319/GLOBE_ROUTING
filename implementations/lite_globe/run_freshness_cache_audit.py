"""Supplementary repeated-query freshness-cache audit.

This is a controlled policy-query benchmark, not a network-deployment test.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter_ns

os.environ.setdefault("MPLCONFIGDIR", "/tmp/switchglobe-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .env.fanet_env import FanetRoutingEnv
from .evaluation.reporting import write_csv
from .evaluation.statistics import summarize_values
from .experiments.latency_optimization_campaign import checkpoint_path as fast_checkpoint_path
from .models.policy_adapter import StudentPolicyAdapter
from .models.student_policy import FastSwitchGlobePolicy
from .provenance import checkpoint_sha256_map, config_sha256, git_provenance
from .run_latency_benchmark import _load_fast
from .scenarios import phase9_evaluation_scenarios


SEEDS = (42, 77, 123, 314, 2718)
DISABLED = "FastSwitchGLOBE + Top-2, cache disabled"
ENABLED = "FastSwitchGLOBE + Top-2, freshness cache enabled"
WORKLOADS = ("changing_observation", "identical_fresh_observation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--changing-sequence-length", type=int, default=256)
    parser.add_argument("--cache-capacity", type=int, default=128)
    parser.add_argument("--benchmark-ttl-ms", type=float, default=60_000.0)
    parser.add_argument("--zip-results", action="store_true")
    return parser.parse_args()


def _copy_observation(observation):
    return {key: np.array(value, copy=True) for key, value in observation.items()}


def _load_cached(
    root: Path, *, seed: int, max_nodes: int, ttl_ms: float, capacity: int,
) -> StudentPolicyAdapter:
    path = fast_checkpoint_path(root, seed)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = FastSwitchGlobePolicy(max_nodes, hidden_dim=32)
    model.load_state_dict(payload["model_state"])
    return StudentPolicyAdapter(
        model, device="cpu", force_forward_if_available=True,
        enable_fast_failover=True, enable_freshness_cache=True,
        freshness_cache_ttl_ms=ttl_ms, freshness_cache_capacity=capacity,
    )


def _changing_observations(policy, *, seed: int, count: int):
    scenario = phase9_evaluation_scenarios(seed)[0]
    env = FanetRoutingEnv(scenario.config)
    output = []
    episode = 0
    while len(output) < count:
        episode_seed = 2_100_000 + episode
        observation, _ = env.reset(
            seed=episode_seed, options=scenario.reset_options
        )
        policy.reset(episode_seed)
        done = False
        while not done and len(output) < count:
            output.append(_copy_observation(observation))
            action = policy.act(observation)
            observation, _, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
        episode += 1
    return output


def _delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return float(after.get(key, 0.0) - before.get(key, 0.0))


def _benchmark_workload(
    *, seed: int, workload: str, disabled: StudentPolicyAdapter,
    enabled: StudentPolicyAdapter, observations: list[dict],
    warmup: int, repeats: int,
) -> tuple[list[dict], list[dict], int]:
    policies = {DISABLED: disabled, ENABLED: enabled}
    for policy in policies.values():
        policy.reset(seed)
    for index in range(warmup):
        observation = observations[0] if workload == WORKLOADS[1] else observations[index % len(observations)]
        for policy in policies.values():
            policy.act_with_metadata(observation)
    diagnostics_before = {
        name: policy.episode_diagnostics() for name, policy in policies.items()
    }
    rng = np.random.default_rng(seed + (0 if workload == WORKLOADS[0] else 300_000))
    raw_rows = []
    actions: dict[tuple[int, str], int] = {}
    with torch.inference_mode():
        for block in range(repeats):
            observation = observations[0] if workload == WORKLOADS[1] else observations[block % len(observations)]
            for position, variant in enumerate(rng.permutation((DISABLED, ENABLED))):
                started = perf_counter_ns()
                decision = policies[str(variant)].act_with_metadata(observation)
                latency_ms = (perf_counter_ns() - started) / 1_000_000.0
                actions[(block, str(variant))] = decision.action
                raw_rows.append({
                    "training_seed": seed, "workload": workload,
                    "block": block, "order_position": position,
                    "variant": str(variant), "latency_ms": latency_ms,
                    "action": decision.action,
                    "backup_action": decision.backup_action,
                })
    mismatches = sum(
        actions[(block, DISABLED)] != actions[(block, ENABLED)]
        for block in range(repeats)
    )
    diagnostic_rows = []
    for variant, policy in policies.items():
        before = diagnostics_before[variant]
        after = policy.episode_diagnostics()
        hits = _delta(after, before, "freshness_cache_hit_steps")
        misses = _delta(after, before, "freshness_cache_miss_steps")
        total = hits + misses
        diagnostic_rows.append({
            "training_seed": seed, "workload": workload, "variant": variant,
            "queries": repeats,
            "cache_hits": hits, "cache_misses": misses,
            "cache_hit_rate": hits / total if total else 0.0,
            "cache_miss_rate": misses / total if total else 0.0,
            "stale_evictions": _delta(after, before, "freshness_cache_stale_evictions"),
            "state_evictions": _delta(after, before, "freshness_cache_state_evictions"),
            "capacity_evictions": _delta(after, before, "freshness_cache_capacity_evictions"),
            "action_mismatches_vs_disabled": mismatches if variant == ENABLED else 0,
        })
    return raw_rows, diagnostic_rows, mismatches


def _correctness_audit(
    root: Path, *, seed: int, max_nodes: int, observation: dict,
) -> dict[str, object]:
    adapter = _load_cached(root, seed=seed, max_nodes=max_nodes, ttl_ms=5.0, capacity=4)
    clock = [0]
    adapter._cache_clock_ns = lambda: clock[0]
    forward_count = {"n": 0}
    handle = adapter.model.register_forward_hook(
        lambda module, args, output: forward_count.__setitem__("n", forward_count["n"] + 1)
    )
    try:
        first = adapter.act_with_metadata(observation)
        after_first = forward_count["n"]
        clock[0] += 1_000_000
        hit = adapter.act_with_metadata(observation)
        after_hit = forward_count["n"]
        clock[0] += 6_000_000
        expired = adapter.act_with_metadata(observation)
        after_expired = forward_count["n"]

        changed_mask = _copy_observation(observation)
        valid = np.flatnonzero(changed_mask["action_mask"][:max_nodes])
        if valid.size:
            changed_mask["action_mask"][int(valid[0])] = False
        clock[0] += 1_000_000
        adapter.act_with_metadata(changed_mask)
        after_mask = forward_count["n"]

        clock[0] += 1_000_000
        adapter.act_with_metadata(observation)
        baseline_neighbor = forward_count["n"]
        changed_neighbor = _copy_observation(observation)
        changed_neighbor["neighbor_features"].flat[0] += 1e-3
        clock[0] += 1_000_000
        adapter.act_with_metadata(changed_neighbor)
        after_neighbor = forward_count["n"]
    finally:
        handle.remove()
    diagnostics = adapter.episode_diagnostics()
    return {
        "training_seed": seed,
        "cache_hit_action_equal": first == hit,
        "cache_hit_avoids_forward": after_hit == after_first,
        "ttl_expiry_triggers_one_forward": after_expired == after_hit + 1,
        "action_mask_change_triggers_one_forward": after_mask == after_expired + 1,
        "neighbor_state_change_triggers_one_forward": after_neighbor == baseline_neighbor + 1,
        "stale_evictions": diagnostics.get("freshness_cache_stale_evictions", 0.0),
        "state_evictions": diagnostics.get("freshness_cache_state_evictions", 0.0),
    }


def _summaries(raw_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(list)
    for row in raw_rows:
        grouped[(row["training_seed"], row["workload"], row["variant"])].append(float(row["latency_ms"]))
    rows = []
    for (seed, workload, variant), values in grouped.items():
        rows.append({
            "training_seed": seed, "workload": workload, "variant": variant,
            "repeats": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
        })
    lookup = {(row["training_seed"], row["workload"], row["variant"]): row for row in rows}
    aggregate = []
    for workload in WORKLOADS:
        for variant in (DISABLED, ENABLED):
            for metric in ("mean_ms", "p50_ms", "p95_ms", "p99_ms"):
                stats = summarize_values([float(lookup[(seed, workload, variant)][metric]) for seed in SEEDS]).to_dict()
                aggregate.append({"workload": workload, "variant": variant, "metric": metric, **stats})
        reductions = []
        for seed in SEEDS:
            disabled = float(lookup[(seed, workload, DISABLED)]["p95_ms"])
            enabled = float(lookup[(seed, workload, ENABLED)]["p95_ms"])
            reductions.append(100.0 * (disabled - enabled) / disabled)
        aggregate.append({
            "workload": workload, "variant": ENABLED,
            "metric": "p95_reduction_vs_disabled_percent",
            **summarize_values(reductions).to_dict(),
        })
    return rows, aggregate


def _plot(aggregate: list[dict], output_dir: Path) -> None:
    lookup = {(row["workload"], row["variant"], row["metric"]): row for row in aggregate}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, workload in zip(axes, WORKLOADS, strict=True):
        means = [lookup[(workload, variant, "p95_ms")]["mean"] for variant in (DISABLED, ENABLED)]
        errors = [
            (lookup[(workload, variant, "p95_ms")]["ci95_high"] - lookup[(workload, variant, "p95_ms")]["ci95_low"]) / 2
            for variant in (DISABLED, ENABLED)
        ]
        axis.bar((0, 1), means, yerr=errors, capsize=4)
        axis.set_xticks((0, 1), ("disabled", "enabled"))
        axis.set_title(workload)
        axis.set_ylabel("p95 policy-query latency (ms)")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Freshness cache supplementary controlled workloads")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"freshness_cache_latency.{suffix}", dpi=220)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows, diagnostic_rows, correctness_rows = [], [], []
    for seed in SEEDS:
        scenario = phase9_evaluation_scenarios(seed)[0]
        collector = _load_fast(
            args.fast_checkpoint_dir, seed=seed,
            max_nodes=scenario.config.max_nodes, device=torch.device("cpu"),
            enable_top2=True,
        )
        changing = _changing_observations(
            collector, seed=seed, count=args.changing_sequence_length
        )
        for workload, observations in (
            (WORKLOADS[0], changing),
            (WORKLOADS[1], [changing[0]]),
        ):
            disabled = _load_fast(
                args.fast_checkpoint_dir, seed=seed,
                max_nodes=scenario.config.max_nodes, device=torch.device("cpu"),
                enable_top2=True,
            )
            enabled = _load_cached(
                args.fast_checkpoint_dir, seed=seed,
                max_nodes=scenario.config.max_nodes,
                ttl_ms=args.benchmark_ttl_ms, capacity=args.cache_capacity,
            )
            raw, diagnostics, mismatches = _benchmark_workload(
                seed=seed, workload=workload, disabled=disabled, enabled=enabled,
                observations=observations, warmup=args.warmup, repeats=args.repeats,
            )
            assert mismatches == 0
            raw_rows.extend(raw)
            diagnostic_rows.extend(diagnostics)
        correctness_rows.append(_correctness_audit(
            args.fast_checkpoint_dir, seed=seed,
            max_nodes=scenario.config.max_nodes, observation=changing[0],
        ))
    summary_rows, aggregate_rows = _summaries(raw_rows)
    assert all(
        all(bool(row[key]) for key in (
            "cache_hit_action_equal", "cache_hit_avoids_forward",
            "ttl_expiry_triggers_one_forward",
            "action_mask_change_triggers_one_forward",
            "neighbor_state_change_triggers_one_forward",
        )) for row in correctness_rows
    )
    write_csv(args.output_dir / "raw_timings.csv", raw_rows)
    write_csv(args.output_dir / "cache_diagnostics.csv", diagnostic_rows)
    write_csv(args.output_dir / "latency_summaries.csv", summary_rows)
    write_csv(args.output_dir / "aggregate_statistics.csv", aggregate_rows)
    write_csv(args.output_dir / "correctness_audit.csv", correctness_rows)
    _plot(aggregate_rows, args.output_dir)
    effective_config = {
        "seeds": list(SEEDS), "warmup": args.warmup, "repeats": args.repeats,
        "changing_sequence_length": args.changing_sequence_length,
        "cache_capacity": args.cache_capacity,
        "benchmark_ttl_ms": args.benchmark_ttl_ms,
        "device": "cpu", "batch_size": 1,
        "order": "deterministic_randomized_block",
    }
    checkpoint_paths = {
        f"fast_switchglobe_seed_{seed}": fast_checkpoint_path(args.fast_checkpoint_dir, seed)
        for seed in SEEDS
    }
    manifest = {
        "schema_version": 1, "complete": True,
        "suite": "freshness_cache_supplementary_controlled_query_audit",
        "note": "controlled repeated-query benchmark; not a network-deployment test",
        "workloads": list(WORKLOADS), "variants": [DISABLED, ENABLED],
        "raw_timing_rows": len(raw_rows),
        "diagnostic_rows": len(diagnostic_rows),
        "summary_rows": len(summary_rows),
        "aggregate_rows": len(aggregate_rows),
        "correctness_rows": len(correctness_rows),
        "action_mismatches": sum(int(row["action_mismatches_vs_disabled"]) for row in diagnostic_rows),
        "config": effective_config, "config_sha256": config_sha256(effective_config),
        "checkpoint_sha256": checkpoint_sha256_map(checkpoint_paths),
        "python": sys.version, "platform": sys.platform, "torch": torch.__version__,
        **git_provenance(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.zip_results:
        archive = Path(shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir))
        manifest["result_zip"] = str(archive)
        manifest["result_zip_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
