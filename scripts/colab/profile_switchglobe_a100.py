"""Collect operator-level A100 traces outside the primary latency timer."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import platform
import shutil
import sys

import torch
from torch.profiler import ProfilerActivity, profile

from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.run_latency_benchmark import (
    _load,
    _load_early_exit,
    _load_fast,
)
from implementations.lite_globe.scenarios import phase9_evaluation_scenarios


ROOT = Path("/content/SwitchGLOBE_globev2")
CHECKPOINT_ROOT = Path("/content/switchglobe_checkpoints")
OUTPUT = Path("/content/switchglobe_globev2_a100_profile_20260905")
ARCHIVE = Path("/content/switchglobe_globev2_a100_profile_20260905.zip")
SEED = 42


def cuda_self_time(event: object) -> float:
    return float(getattr(
        event, "self_device_time_total",
        getattr(event, "self_cuda_time_total", 0.0),
    ))


def cuda_memory(event: object) -> int:
    return int(getattr(
        event, "self_device_memory_usage",
        getattr(event, "self_cuda_memory_usage", 0),
    ))


def main() -> None:
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("This profiler must run on a verified A100")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    scenario = phase9_evaluation_scenarios(SEED)[0]
    observation, _ = FanetRoutingEnv(scenario.config).reset(
        seed=1_099_999, options=scenario.reset_options
    )
    device = torch.device("cuda")
    variants = {
        "exact": _load(
            CHECKPOINT_ROOT / "phase12", seed=SEED,
            max_nodes=scenario.config.max_nodes, device=device, buffered=False,
        ),
        "early_exit": _load_early_exit(
            CHECKPOINT_ROOT / "phase12", seed=SEED,
            max_nodes=scenario.config.max_nodes, device=device,
        ),
        "fast": _load_fast(
            CHECKPOINT_ROOT / "fast", seed=SEED,
            max_nodes=scenario.config.max_nodes, device=device,
            enable_top2=False,
        ),
        "fast_top2": _load_fast(
            CHECKPOINT_ROOT / "fast", seed=SEED,
            max_nodes=scenario.config.max_nodes, device=device,
            enable_top2=True,
        ),
    }
    for name, policy in variants.items():
        with torch.inference_mode():
            for _ in range(50):
                policy.act_with_metadata(observation)
            torch.cuda.synchronize()
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
            ) as prof:
                for _ in range(30):
                    policy.act_with_metadata(observation)
                torch.cuda.synchronize()
        prof.export_chrome_trace(str(OUTPUT / f"{name}_trace.json"))
        events = sorted(
            prof.key_averages(group_by_input_shape=True),
            key=cuda_self_time,
            reverse=True,
        )
        with (OUTPUT / f"{name}_operators.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "operator", "count", "self_cpu_time_total_us",
                "self_cuda_time_total_us", "cpu_memory_usage_bytes",
                "cuda_memory_usage_bytes", "input_shapes",
            ])
            writer.writeheader()
            for event in events:
                writer.writerow({
                    "operator": event.key,
                    "count": int(event.count),
                    "self_cpu_time_total_us": float(event.self_cpu_time_total),
                    "self_cuda_time_total_us": cuda_self_time(event),
                    "cpu_memory_usage_bytes": int(event.self_cpu_memory_usage),
                    "cuda_memory_usage_bytes": cuda_memory(event),
                    "input_shapes": repr(event.input_shapes),
                })
        (OUTPUT / f"{name}_table.txt").write_text(
            prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=40
            ),
            encoding="utf-8",
        )

    manifest = {
        "complete": True,
        "profiling_is_separate_from_primary_timing": True,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "python": sys.version,
        "platform": platform.platform(),
        "seed": SEED,
        "scenario": scenario.name,
        "warmup": 50,
        "profiled_calls": 30,
        "variants": list(variants),
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    shutil.make_archive(str(ARCHIVE.with_suffix("")), "zip", root_dir=OUTPUT)
    print(json.dumps({**manifest, "archive": str(ARCHIVE)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
