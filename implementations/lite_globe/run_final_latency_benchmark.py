"""Final same-session randomized-block CPU/GPU latency campaign."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tracemalloc

os.environ.setdefault("MPLCONFIGDIR", "/tmp/switchglobe-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .env.fanet_env import FanetRoutingEnv
from .evaluation.interleaved_latency import InterleavedSpec, benchmark_interleaved
from .evaluation.latency import legacy_repeated_switchglobe_action, synchronize
from .evaluation.reporting import write_csv
from .evaluation.statistics import summarize_values
from .experiments.external_comparison_campaign import _switchglobe_path
from .experiments.latency_optimization_campaign import checkpoint_path as fast_checkpoint_path
from .models.student_policy import RiskSwitchLiteGlobePStudentPolicy
from .models.tensor_observation import observation_to_tensors
from .provenance import checkpoint_sha256_map, config_sha256, git_provenance
from .run_latency_benchmark import _load, _load_fast
from .scenarios import phase9_evaluation_scenarios


SEEDS = (42, 77, 123, 314, 2718)
VARIANTS = (
    "Legacy repeated SwitchGLOBE",
    "SwitchGLOBE Exact",
    "FastSwitchGLOBE",
    "FastSwitchGLOBE + Top-2",
)
WORKER_NAMES = {
    VARIANTS[0]: "legacy_repeated",
    VARIANTS[1]: "switchglobe_exact",
    VARIANTS[2]: "fast",
    VARIANTS[3]: "fast_top2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fast-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--cold-repeats", type=int, default=5)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--zip-results", action="store_true")
    return parser.parse_args()


def _action_extraction(policy, probabilities, tensors):
    action = int(torch.argmax(probabilities).item())
    if policy.force_forward_if_available and action == policy.model.drop_action:
        mask = tensors["action_mask"][: policy.model.max_nodes].bool()
        if torch.any(mask):
            action = int(torch.argmax(
                probabilities[: policy.model.max_nodes].masked_fill(~mask, -1.0)
            ).item())
    backup = policy._backup_action(
        probabilities, tensors["action_mask"], primary_action=action
    )
    return action, backup


def _component_specs(variant: str, policy, observation) -> list[InterleavedSpec]:
    device = policy.device
    tensors = observation_to_tensors(observation, device=device)
    model = policy.model
    if isinstance(model, RiskSwitchLiteGlobePStudentPolicy):
        model_call = lambda: model.decide(tensors)
    else:
        model_call = lambda: model(tensors)
    with torch.inference_mode():
        cached = model_call()
    probabilities = (
        cached.output.probabilities
        if hasattr(cached, "output") else cached.probabilities
    )
    specs = [
        InterleavedSpec(
            variant, "preprocess", device,
            lambda: observation_to_tensors(observation, device=device),
        ),
        InterleavedSpec(variant, "model", device, model_call),
        InterleavedSpec(
            variant, "action_extraction", device,
            lambda: _action_extraction(policy, probabilities, tensors),
        ),
        InterleavedSpec(
            variant, "end_to_end_policy", device,
            lambda: policy.act_with_metadata(observation),
        ),
    ]
    if variant == VARIANTS[3]:
        decision = policy.act_with_metadata(observation)
        live_mask = torch.as_tensor(observation["action_mask"], device=device)
        specs.append(InterleavedSpec(
            variant, "resolver_only", device,
            lambda: policy.resolve_decision(decision, live_mask),
        ))
    return specs


def _cold_samples(
    *, seed: int, device: torch.device, checkpoint_dir: Path,
    fast_checkpoint_dir: Path, repeats: int,
) -> tuple[dict[tuple[str, str, str], list[float]], list[dict[str, object]]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        for variant in VARIANTS:
            command = [
                sys.executable, "-m", "implementations.lite_globe.latency_cold_worker",
                "--checkpoint-dir", str(checkpoint_dir),
                "--fast-checkpoint-dir", str(fast_checkpoint_dir),
                "--seed", str(seed), "--variant", WORKER_NAMES[variant],
                "--device", str(device),
            ]
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            value = float(payload["latency_ms"])
            key = (variant, "end_to_end_policy", str(device))
            grouped[key].append(value)
            rows.append({
                "training_seed": seed, "fresh_process_repeat": repeat,
                "variant": variant, "component": "end_to_end_policy",
                "device": str(device), "latency_ms": value,
            })
    return dict(grouped), rows


def _cost_row(
    *, seed: int, variant: str, policy, observation, model_path: Path,
    end_to_end,
) -> dict[str, object]:
    synchronize(policy.device)
    if policy.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(policy.device)
        baseline = torch.cuda.memory_allocated(policy.device)
        with torch.inference_mode():
            for _ in range(10):
                end_to_end()
        synchronize(policy.device)
        peak_device = max(torch.cuda.max_memory_allocated(policy.device) - baseline, 0)
    else:
        peak_device = None
    tracemalloc.start()
    with torch.inference_mode():
        for _ in range(10):
            end_to_end()
    _, peak_python = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "training_seed": seed, "variant": variant,
        "device": str(policy.device),
        "parameter_count": sum(parameter.numel() for parameter in policy.model.parameters()),
        "serialized_checkpoint_bytes": model_path.stat().st_size,
        "policy_input_bytes": policy.observation_bytes(observation),
        "peak_python_allocation_bytes": peak_python,
        "peak_cuda_allocation_delta_bytes": peak_device,
    }


def _aggregate(summary_rows: list[dict[str, object]]) -> tuple[list[dict], list[dict]]:
    end_rows = [row for row in summary_rows if row["component"] == "end_to_end_policy"]
    aggregate_rows = []
    for device in sorted({str(row["device"]) for row in end_rows}):
        for variant in VARIANTS:
            selected = [row for row in end_rows if row["device"] == device and row["variant"] == variant]
            for metric in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "decisions_per_second"):
                stats = summarize_values([float(row[metric]) for row in selected]).to_dict()
                aggregate_rows.append({
                    "device": device, "variant": variant,
                    "component": "end_to_end_policy", "metric": metric, **stats,
                })
    paired_rows = []
    lookup = {
        (int(row["training_seed"]), str(row["device"]), str(row["variant"])): row
        for row in end_rows
    }
    for device in sorted({str(row["device"]) for row in end_rows}):
        for variant in (VARIANTS[0], VARIANTS[2], VARIANTS[3]):
            reductions = []
            for seed in SEEDS:
                exact = float(lookup[(seed, device, VARIANTS[1])]["p95_ms"])
                candidate = float(lookup[(seed, device, variant)]["p95_ms"])
                reductions.append(100.0 * (exact - candidate) / exact)
            stats = summarize_values(reductions).to_dict()
            paired_rows.append({
                "device": device, "variant": variant,
                "baseline": VARIANTS[1], "metric": "p95_latency_reduction_percent",
                **stats,
            })
    return aggregate_rows, paired_rows


def _plot(aggregate_rows: list[dict], output_dir: Path) -> None:
    selected = {
        (row["device"], row["variant"]): row for row in aggregate_rows
        if row["metric"] == "p95_ms"
    }
    for device in sorted({key[0] for key in selected}):
        means = [selected[(device, variant)]["mean"] for variant in VARIANTS]
        errors = [
            (selected[(device, variant)]["ci95_high"] - selected[(device, variant)]["ci95_low"]) / 2
            for variant in VARIANTS
        ]
        fig, axis = plt.subplots(figsize=(10, 5))
        axis.bar(range(len(VARIANTS)), means, yerr=errors, capsize=4)
        axis.set_xticks(range(len(VARIANTS)), VARIANTS, rotation=20, ha="right")
        axis.set_ylabel("Raw-decision p95 latency (ms)")
        axis.set_title(f"Same-session randomized-block latency — {device}")
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(output_dir / f"latency_p95_{device}.{suffix}", dpi=220)
        plt.close(fig)


def main() -> int:
    args = parse_args()
    seeds = tuple(args.seed) if args.seed else SEEDS
    if seeds != SEEDS:
        raise ValueError(f"final paper latency requires seeds {SEEDS}, got {seeds}")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    devices = [torch.device("cpu")]
    if torch.cuda.is_available() and not args.cpu_only:
        devices.append(torch.device("cuda"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    cold_rows: list[dict[str, object]] = []
    cost_rows: list[dict[str, object]] = []
    for seed in seeds:
        scenario = phase9_evaluation_scenarios(seed)[0]
        observation, _ = FanetRoutingEnv(scenario.config).reset(
            seed=1_099_999, options=scenario.reset_options
        )
        for device in devices:
            exact = _load(
                args.checkpoint_dir, seed=seed, max_nodes=scenario.config.max_nodes,
                device=device, buffered=False,
            )
            fast = _load_fast(
                args.fast_checkpoint_dir, seed=seed,
                max_nodes=scenario.config.max_nodes, device=device, enable_top2=False,
            )
            top2 = _load_fast(
                args.fast_checkpoint_dir, seed=seed,
                max_nodes=scenario.config.max_nodes, device=device, enable_top2=True,
            )
            specs = [InterleavedSpec(
                VARIANTS[0], "end_to_end_policy", device,
                lambda policy=exact: legacy_repeated_switchglobe_action(policy, observation),
            )]
            specs.extend(_component_specs(VARIANTS[1], exact, observation))
            specs.extend(_component_specs(VARIANTS[2], fast, observation))
            specs.extend(_component_specs(VARIANTS[3], top2, observation))
            cold, seed_cold_rows = _cold_samples(
                seed=seed, device=device, checkpoint_dir=args.checkpoint_dir,
                fast_checkpoint_dir=args.fast_checkpoint_dir,
                repeats=args.cold_repeats,
            )
            summaries, timings = benchmark_interleaved(
                specs, warmup=args.warmup, repeats=args.repeats,
                order_seed=seed + (0 if device.type == "cpu" else 500_000),
                cold_samples=cold,
            )
            summary_rows.extend({"training_seed": seed, **row} for row in summaries)
            raw_rows.extend({"training_seed": seed, **row} for row in timings)
            cold_rows.extend(seed_cold_rows)
            model_paths = {
                VARIANTS[0]: _switchglobe_path(args.checkpoint_dir, seed),
                VARIANTS[1]: _switchglobe_path(args.checkpoint_dir, seed),
                VARIANTS[2]: fast_checkpoint_path(args.fast_checkpoint_dir, seed),
                VARIANTS[3]: fast_checkpoint_path(args.fast_checkpoint_dir, seed),
            }
            policy_map = {
                VARIANTS[0]: exact, VARIANTS[1]: exact,
                VARIANTS[2]: fast, VARIANTS[3]: top2,
            }
            for variant, policy in policy_map.items():
                function = (
                    (lambda policy=policy: legacy_repeated_switchglobe_action(policy, observation))
                    if variant == VARIANTS[0]
                    else (lambda policy=policy: policy.act_with_metadata(observation))
                )
                cost_rows.append(_cost_row(
                    seed=seed, variant=variant, policy=policy,
                    observation=observation, model_path=model_paths[variant],
                    end_to_end=function,
                ))
    aggregate_rows, paired_rows = _aggregate(summary_rows)
    write_csv(args.output_dir / "raw_timings.csv", raw_rows)
    write_csv(args.output_dir / "raw_cold_start.csv", cold_rows)
    write_csv(args.output_dir / "runtime_benchmarks.csv", summary_rows)
    write_csv(args.output_dir / "deployment_costs.csv", cost_rows)
    write_csv(args.output_dir / "aggregate_statistics.csv", aggregate_rows)
    write_csv(args.output_dir / "paired_speedups.csv", paired_rows)
    _plot(aggregate_rows, args.output_dir)
    checkpoint_paths = {}
    for seed in seeds:
        checkpoint_paths[f"switchglobe_exact_seed_{seed}"] = _switchglobe_path(args.checkpoint_dir, seed)
        checkpoint_paths[f"fast_switchglobe_seed_{seed}"] = fast_checkpoint_path(args.fast_checkpoint_dir, seed)
    effective_config = {
        "seeds": list(seeds), "warmup": args.warmup,
        "repeats": args.repeats, "cold_repeats": args.cold_repeats,
        "devices": [str(device) for device in devices],
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "order": "deterministic_randomized_block",
        "batch_size": 1,
    }
    manifest = {
        "schema_version": 1, "complete": True,
        "suite": "switchglobe_final_same_session_latency",
        "primary_device": "cpu", "gpu_role": "secondary",
        "variants": list(VARIANTS), "training_seeds": list(seeds),
        "summary_rows": len(summary_rows), "raw_timing_rows": len(raw_rows),
        "raw_cold_start_rows": len(cold_rows), "deployment_cost_rows": len(cost_rows),
        "config": effective_config, "config_sha256": config_sha256(effective_config),
        "checkpoint_sha256": checkpoint_sha256_map(checkpoint_paths),
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "raw_timing_definition": "one synchronized batch-1 decision per randomized block",
        "cold_start_definition": "first batch-1 decision in a fresh Python process after model load",
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
