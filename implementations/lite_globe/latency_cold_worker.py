"""Fresh-process first-decision worker for the final latency campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import torch

from .env.fanet_env import FanetRoutingEnv
from .evaluation.latency import legacy_repeated_switchglobe_action, synchronize
from .run_latency_benchmark import _load, _load_fast
from .scenarios import phase9_evaluation_scenarios


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--fast-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--variant",
        choices=("legacy_repeated", "switchglobe_exact", "fast", "fast_top2"),
        required=True,
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(args.device)
    scenario = phase9_evaluation_scenarios(args.seed)[0]
    observation, _ = FanetRoutingEnv(scenario.config).reset(
        seed=1_099_999, options=scenario.reset_options
    )
    if args.variant in {"legacy_repeated", "switchglobe_exact"}:
        policy = _load(
            args.checkpoint_dir, seed=args.seed,
            max_nodes=scenario.config.max_nodes, device=device, buffered=False,
        )
    else:
        policy = _load_fast(
            args.fast_checkpoint_dir, seed=args.seed,
            max_nodes=scenario.config.max_nodes, device=device,
            enable_top2=args.variant == "fast_top2",
        )
    if args.variant == "legacy_repeated":
        function = lambda: legacy_repeated_switchglobe_action(
            policy, observation
        )
    else:
        function = lambda: policy.act_with_metadata(observation)
    with torch.inference_mode():
        synchronize(device)
        started = perf_counter_ns()
        function()
        synchronize(device)
    print(json.dumps({
        "seed": args.seed,
        "variant": args.variant,
        "component": "end_to_end_policy",
        "device": str(device),
        "latency_ms": (perf_counter_ns() - started) / 1_000_000.0,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
