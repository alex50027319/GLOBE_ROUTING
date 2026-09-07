"""Same-session randomized-block CPU/A100 batch-1 latency ablation."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import platform
import sys

import torch

from .env.fanet_env import FanetRoutingEnv
from .evaluation.interleaved_latency import InterleavedSpec, benchmark_interleaved
from .evaluation.reporting import write_csv
from .evaluation.statistics import summarize_values
from .experiments.ablation_campaign import (
    AblationConfig, build_variant_policies, FAST_SWITCHGLOBE, GEO_RESIDUAL,
    PREDICTIVE_NO_SWITCH, PREDICTIVE_PRIOR_ONLY, SWITCHGLOBE_EXACT,
)
from .models.policy_adapter import StudentPolicyAdapter
from .models.teacher_adapter import TeacherPolicyAdapter
from .provenance import config_sha256, git_provenance
from .run_model_selection_ablation import (
    GLOBAL_TEACHER, HISTORICAL_KD_PPO, HISTORICAL_NO_KD, SEEDS,
    _phase7_models,
)
from .scenarios import phase9_evaluation_scenarios


METHODS = (
    HISTORICAL_NO_KD, HISTORICAL_KD_PPO, GLOBAL_TEACHER,
    GEO_RESIDUAL, PREDICTIVE_PRIOR_ONLY, PREDICTIVE_NO_SWITCH,
    SWITCHGLOBE_EXACT, FAST_SWITCHGLOBE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase7-dir", type=Path, required=True)
    parser.add_argument("--phase8-dir", type=Path, required=True)
    parser.add_argument("--phase11-dir", type=Path, required=True)
    parser.add_argument("--phase12-dir", type=Path, required=True)
    parser.add_argument("--fast-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=2000)
    return parser.parse_args()


def _policies(args: argparse.Namespace, seed: int, device: torch.device):
    scenario = phase9_evaluation_scenarios(seed)[0]
    env = FanetRoutingEnv(scenario.config)
    observation, _ = env.reset(seed=1_099_999, options=scenario.reset_options)
    final = build_variant_policies(
        AblationConfig(training_seeds=(seed,), evaluation_episodes=200),
        seed=seed, max_nodes=scenario.config.max_nodes,
        phase8_checkpoint_dir=args.phase8_dir,
        phase11_checkpoint_dir=args.phase11_dir,
        switchglobe_checkpoint_dir=args.phase12_dir,
        fast_checkpoint_dir=args.fast_dir, device=device,
    )
    locals_, teacher = _phase7_models(
        args.phase7_dir, seed=seed, max_nodes=scenario.config.max_nodes,
        device=device,
    )
    policies = {
        HISTORICAL_NO_KD: StudentPolicyAdapter(
            locals_[HISTORICAL_NO_KD], device=device,
            force_forward_if_available=True,
        ),
        HISTORICAL_KD_PPO: StudentPolicyAdapter(
            locals_[HISTORICAL_KD_PPO], device=device,
            force_forward_if_available=True,
        ),
        GLOBAL_TEACHER: TeacherPolicyAdapter(env, teacher, device=device),
        **{name: final[name] for name in (
            GEO_RESIDUAL, PREDICTIVE_PRIOR_ONLY, PREDICTIVE_NO_SWITCH,
            SWITCHGLOBE_EXACT, FAST_SWITCHGLOBE,
        )},
    }
    return observation, policies


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This confirmatory benchmark requires an A100 CUDA session")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for device in (torch.device("cpu"), torch.device("cuda")):
            observation, policies = _policies(args, seed, device)
            specs = []
            for method in METHODS:
                policy = policies[method]
                decide = getattr(policy, "act_with_metadata", None)
                function = (
                    (lambda decide=decide: decide(observation))
                    if decide is not None
                    else (lambda policy=policy: policy.act(observation))
                )
                specs.append(InterleavedSpec(
                    method, "end_to_end_policy", device, function,
                ))
            summaries, timings = benchmark_interleaved(
                specs, warmup=args.warmup, repeats=args.repeats,
                order_seed=seed + (0 if device.type == "cpu" else 500_000),
            )
            summary_rows.extend({"training_seed": seed, **row} for row in summaries)
            raw_rows.extend({"training_seed": seed, **row} for row in timings)
    aggregate_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in summary_rows:
        grouped[(str(row["device"]), str(row["variant"]))].append(float(row["p95_ms"]))
    for (device, method), values in sorted(grouped.items()):
        aggregate_rows.append({
            "device": device, "method": method, "metric": "seed_p95_ms",
            **summarize_values(values).to_dict(),
        })
    paired_rows: list[dict[str, object]] = []
    lookup = {
        (str(row["device"]), str(row["variant"]), int(row["training_seed"])): float(row["p95_ms"])
        for row in summary_rows
    }
    for device in ("cpu", "cuda"):
        for method in METHODS:
            if method == SWITCHGLOBE_EXACT:
                continue
            deltas = [
                lookup[(device, method, seed)] - lookup[(device, SWITCHGLOBE_EXACT, seed)]
                for seed in SEEDS
            ]
            reductions = [
                100.0 * (lookup[(device, SWITCHGLOBE_EXACT, seed)] - lookup[(device, method, seed)])
                / lookup[(device, SWITCHGLOBE_EXACT, seed)]
                for seed in SEEDS
            ]
            paired_rows.append({
                "device": device, "variant": method, "baseline": SWITCHGLOBE_EXACT,
                "metric": "p95_ms_variant_minus_exact", **summarize_values(deltas).to_dict(),
                "mean_reduction_percent": summarize_values(reductions).mean,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw_timings.csv", raw_rows)
    write_csv(args.output_dir / "seed_runtime_benchmarks.csv", summary_rows)
    write_csv(args.output_dir / "aggregate_latency_ci.csv", aggregate_rows)
    write_csv(args.output_dir / "paired_latency_vs_exact.csv", paired_rows)
    config = {
        "seeds": list(SEEDS), "methods": list(METHODS), "devices": ["cpu", "cuda"],
        "batch_size": 1, "warmup": args.warmup, "repeats": args.repeats,
        "order": "deterministic randomized block", "torch_num_threads": 1,
    }
    manifest = {
        "schema_version": 1, "complete": True,
        "suite": "switchglobe_model_selection_batch1_latency",
        "config": config, "config_sha256": config_sha256(config),
        "summary_rows": len(summary_rows), "raw_timing_rows": len(raw_rows),
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0), **git_provenance(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
