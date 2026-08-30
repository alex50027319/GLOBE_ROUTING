"""Benchmark exact SwitchGLOBE latency variants without retraining."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/switchglobe-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .env.fanet_env import FanetRoutingEnv
from .evaluation import (
    benchmark_callable, benchmark_resolver, legacy_repeated_switchglobe_action,
    measure_policy_cost, profile_student_policy,
)
from .evaluation.reporting import write_csv
from .experiments.external_comparison_campaign import (
    _switchglobe_path,
    load_switchglobe,
)
from .experiments.latency_optimization_campaign import checkpoint_path as fast_checkpoint_path
from .models.policy_adapter import StudentPolicyAdapter
from .models.student_policy import FastSwitchGlobePolicy
from .provenance import checkpoint_sha256_map, config_sha256, git_provenance
from .scenarios import phase9_evaluation_scenarios


SEEDS = (42, 77, 123, 314, 2718)
FAST_HIDDEN_DIM = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("ResearchAIWorkspace/artifacts/lite_globe/phase12/checkpoints"),
    )
    parser.add_argument(
        "--fast-checkpoint-dir", type=Path,
        default=Path("artifacts/switchglobe_latency_optimization/fast_switchglobe/checkpoints"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/switchglobe_latency_optimization/exact_runtime"),
    )
    parser.add_argument(
        "--include-fast", action="store_true",
        help="Also benchmark FastSwitchGLOBE and FastSwitchGLOBE + Top-2 "
        "(requires checkpoints from run_fast_switchglobe.py) in the same "
        "session as Legacy/Exact, per the master prompt's same-session rule.",
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=500)
    parser.add_argument("--include-compile", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--zip-results", action="store_true")
    return parser.parse_args()


def _load(
    root: Path, *, seed: int, max_nodes: int, device: torch.device,
    buffered: bool,
) -> StudentPolicyAdapter:
    loaded = load_switchglobe(
        root, seed=seed, max_nodes=max_nodes, hidden_dim=64, device=device
    )
    return StudentPolicyAdapter(
        loaded.model, device=device, force_forward_if_available=True,
        reuse_tensor_buffer=buffered,
    )


def _compiled(
    root: Path, *, seed: int, max_nodes: int, device: torch.device,
) -> StudentPolicyAdapter:
    policy = _load(
        root, seed=seed, max_nodes=max_nodes, device=device, buffered=False
    )
    mode = "default" if device.type == "cuda" else "reduce-overhead"
    policy.model.normal_policy = torch.compile(
        policy.model.normal_policy, mode=mode, fullgraph=False
    )
    policy.model.predictive_policy = torch.compile(
        policy.model.predictive_policy, mode=mode, fullgraph=False
    )
    return policy


def _load_fast(
    root: Path, *, seed: int, max_nodes: int, device: torch.device,
    enable_top2: bool,
) -> StudentPolicyAdapter:
    path = fast_checkpoint_path(root, seed)
    if not path.is_file():
        raise FileNotFoundError(
            f"FastSwitchGLOBE checkpoint not found for seed {seed} at {path}; "
            "run run_fast_switchglobe.py first or omit --include-fast"
        )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = FastSwitchGlobePolicy(max_nodes, hidden_dim=FAST_HIDDEN_DIM)
    model.load_state_dict(payload["model_state"])
    return StudentPolicyAdapter(
        model, device=device, force_forward_if_available=True,
        enable_fast_failover=enable_top2,
    )


def _plot(rows: list[dict], output_dir: Path) -> None:
    selected = [row for row in rows if row["component"] == "end_to_end_policy"]
    variants = sorted({str(row["variant"]) for row in selected})
    means, errors = [], []
    for variant in variants:
        values = np.asarray([
            float(row["p95_ms"]) for row in selected
            if row["variant"] == variant
        ])
        means.append(float(values.mean()))
        errors.append(
            float(2.776445 * values.std(ddof=1) / np.sqrt(values.size))
            if values.size > 1 else 0.0
        )
    figure, axis = plt.subplots(figsize=(11, 5))
    colors = ["#1f77b4" if "eager" in item else "#ff7f0e" for item in variants]
    axis.bar(np.arange(len(variants)), means, yerr=errors, capsize=4, color=colors)
    axis.set_xticks(np.arange(len(variants)))
    axis.set_xticklabels(variants, rotation=25, ha="right")
    axis.set_ylabel("Synchronized end-to-end policy p95 (ms)")
    axis.set_title("SwitchGLOBE exact runtime variants")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "latency_p95_variants.png", dpi=240)
    figure.savefig(output_dir / "latency_p95_variants.svg")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    seeds = tuple(args.seed) if args.seed else SEEDS
    warmup = 5 if args.smoke else args.warmup
    repeats = 20 if args.smoke else args.repeats
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    cost_rows: list[dict] = []
    for seed in seeds:
        scenario = phase9_evaluation_scenarios(seed)[0]
        observation, _ = FanetRoutingEnv(scenario.config).reset(
            seed=1_099_999, options=scenario.reset_options
        )
        specs: list[tuple[str, StudentPolicyAdapter]] = [
            ("exact_eager_cpu", _load(
                args.checkpoint_dir, seed=seed,
                max_nodes=scenario.config.max_nodes,
                device=torch.device("cpu"), buffered=False,
            )),
            ("exact_buffered_cpu", _load(
                args.checkpoint_dir, seed=seed,
                max_nodes=scenario.config.max_nodes,
                device=torch.device("cpu"), buffered=True,
            )),
        ]
        legacy_specs: list[tuple[str, StudentPolicyAdapter]] = [
            ("legacy_repeated_cpu", _load(
                args.checkpoint_dir, seed=seed,
                max_nodes=scenario.config.max_nodes,
                device=torch.device("cpu"), buffered=False,
            ))
        ]
        if torch.cuda.is_available():
            specs.extend([
                ("exact_eager_cuda", _load(
                    args.checkpoint_dir, seed=seed,
                    max_nodes=scenario.config.max_nodes,
                    device=torch.device("cuda"), buffered=False,
                )),
                ("exact_buffered_cuda", _load(
                    args.checkpoint_dir, seed=seed,
                    max_nodes=scenario.config.max_nodes,
                    device=torch.device("cuda"), buffered=True,
                )),
            ])
            legacy_specs.append(("legacy_repeated_cuda", _load(
                args.checkpoint_dir, seed=seed,
                max_nodes=scenario.config.max_nodes,
                device=torch.device("cuda"), buffered=False,
            )))
        if args.include_compile:
            specs.append(("exact_compiled_cpu", _compiled(
                args.checkpoint_dir, seed=seed,
                max_nodes=scenario.config.max_nodes,
                device=torch.device("cpu"),
            )))
            if torch.cuda.is_available():
                specs.append(("exact_compiled_cuda", _compiled(
                    args.checkpoint_dir, seed=seed,
                    max_nodes=scenario.config.max_nodes,
                    device=torch.device("cuda"),
                )))
        fast_specs: list[tuple[str, StudentPolicyAdapter]] = []
        resolver_specs: list[tuple[str, StudentPolicyAdapter]] = []
        if args.include_fast:
            fast_specs.extend([
                ("fast_eager_cpu", _load_fast(
                    args.fast_checkpoint_dir, seed=seed,
                    max_nodes=scenario.config.max_nodes,
                    device=torch.device("cpu"), enable_top2=False,
                )),
                ("fast_top2_eager_cpu", _load_fast(
                    args.fast_checkpoint_dir, seed=seed,
                    max_nodes=scenario.config.max_nodes,
                    device=torch.device("cpu"), enable_top2=True,
                )),
            ])
            resolver_specs.append(("fast_top2_eager_cpu", fast_specs[-1][1]))
            if torch.cuda.is_available():
                fast_specs.extend([
                    ("fast_eager_cuda", _load_fast(
                        args.fast_checkpoint_dir, seed=seed,
                        max_nodes=scenario.config.max_nodes,
                        device=torch.device("cuda"), enable_top2=False,
                    )),
                    ("fast_top2_eager_cuda", _load_fast(
                        args.fast_checkpoint_dir, seed=seed,
                        max_nodes=scenario.config.max_nodes,
                        device=torch.device("cuda"), enable_top2=True,
                    )),
                ])
                resolver_specs.append(("fast_top2_eager_cuda", fast_specs[-1][1]))
        fast_variant_names = {name for name, _ in fast_specs}
        for variant, policy in specs + fast_specs:
            for result in profile_student_policy(
                policy, observation, variant=variant,
                warmup=warmup, repeats=repeats,
            ):
                rows.append({"training_seed": seed, **result.to_dict()})
            serialized_model_path = (
                fast_checkpoint_path(args.fast_checkpoint_dir, seed)
                if variant in fast_variant_names
                else args.checkpoint_dir / f"seed_{seed}" / "risk_switch_lite_globe_p.pt"
            )
            cost = measure_policy_cost(
                policy, observation, model=policy.model,
                input_observation=observation, device=policy.device,
                warmup=warmup, repeats=min(repeats, 100),
                serialized_model_path=serialized_model_path,
            )
            cost_rows.append({
                "training_seed": seed, "variant": variant, **cost.to_dict()
            })
        for variant, policy in resolver_specs:
            result = benchmark_resolver(
                policy, observation, variant=variant,
                warmup=warmup, repeats=repeats,
            )
            rows.append({"training_seed": seed, **result.to_dict()})
        for variant, policy in legacy_specs:
            result = benchmark_callable(
                lambda policy=policy: legacy_repeated_switchglobe_action(
                    policy, observation
                ),
                variant=variant, component="end_to_end_policy",
                device=policy.device, warmup=warmup, repeats=repeats,
            )
            rows.append({"training_seed": seed, **result.to_dict()})
    write_csv(args.output_dir / "runtime_benchmarks.csv", rows)
    write_csv(args.output_dir / "deployment_costs.csv", cost_rows)
    _plot(rows, args.output_dir)
    effective_config = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "fast_checkpoint_dir": str(args.fast_checkpoint_dir),
        "seeds": list(seeds), "warmup": warmup, "repeats": repeats,
        "include_compile": args.include_compile, "include_fast": args.include_fast,
        "smoke": args.smoke,
    }
    checkpoint_paths: dict[str, Path] = {}
    for seed in seeds:
        try:
            checkpoint_paths[f"switchglobe_exact_seed_{seed}"] = _switchglobe_path(
                args.checkpoint_dir, seed
            )
        except FileNotFoundError:
            pass
        if args.include_fast:
            checkpoint_paths[f"fast_switchglobe_seed_{seed}"] = fast_checkpoint_path(
                args.fast_checkpoint_dir, seed
            )
    manifest = {
        "schema_version": 1, "complete": True,
        "suite": (
            "switchglobe_unified_latency" if args.include_fast
            else "switchglobe_exact_latency"
        ),
        "variants": (
            ["legacy_repeated", "exact", "fast", "fast_top2"] if args.include_fast
            else ["legacy_repeated", "exact"]
        ),
        "training_seeds": list(seeds),
        "warmup": warmup, "repeats": repeats,
        "runtime_rows": len(rows), "deployment_cost_rows": len(cost_rows),
        "torch": torch.__version__, "python": sys.version,
        "platform": platform.platform(), "cuda_available": torch.cuda.is_available(),
        "candidate_3_executed": False,
        "fast_checkpoint_dir": str(args.fast_checkpoint_dir) if args.include_fast else None,
        "config_sha256": config_sha256(effective_config),
        "checkpoint_sha256": checkpoint_sha256_map(checkpoint_paths),
        **git_provenance(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.zip_results:
        manifest["result_zip"] = shutil.make_archive(
            str(args.output_dir), "zip", root_dir=args.output_dir
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
