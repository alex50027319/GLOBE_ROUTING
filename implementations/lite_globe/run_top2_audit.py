"""Run the Top-2 synthetic stale-primary failover audit (see master prompt §7).

This is NOT a real wireless link-failure test -- see
``evaluation/top2_audit.py`` for why the simulator needs a synthetic
procedure instead.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

import torch

from .evaluation.top2_audit import (
    Top2AuditConfig,
    run_top2_synthetic_failover_audit,
    write_top2_audit_report,
)
from .experiments.latency_optimization_campaign import checkpoint_path as fast_checkpoint_path
from .provenance import checkpoint_sha256_map, config_sha256, git_provenance


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast-checkpoint-dir", type=Path,
        default=Path("artifacts/switchglobe_latency_optimization/fast_switchglobe/checkpoints"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/final_paper_simulation/supplementary/top2_failover"),
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--episodes-per-scenario", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = tuple(args.seed) if args.seed else (42, 77, 123, 314, 2718)
    if args.smoke:
        seeds = tuple(args.seed) if args.seed else (42,)
    config = Top2AuditConfig(
        training_seeds=seeds,
        episodes_per_scenario=3 if args.smoke else args.episodes_per_scenario,
        resolver_warmup=5 if args.smoke else 50,
        resolver_repeats=20 if args.smoke else 500,
    )
    device = _device(args.device)
    result = run_top2_synthetic_failover_audit(
        config, fast_checkpoint_dir=args.fast_checkpoint_dir, device=device,
    )
    checkpoint_paths = {
        f"fast_switchglobe_seed_{seed}": fast_checkpoint_path(args.fast_checkpoint_dir, seed)
        for seed in config.training_seeds
    }
    manifest = write_top2_audit_report(
        args.output_dir,
        metrics=result["metrics"], resolver_rows=result["resolver_latency"],
        metadata={
            "mode": "smoke" if args.smoke else "full",
            "device": str(device), "config": asdict(config),
            "config_sha256": config_sha256(config),
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "fast_checkpoint_dir": str(args.fast_checkpoint_dir),
            "checkpoint_sha256": checkpoint_sha256_map(checkpoint_paths),
            **git_provenance(),
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
