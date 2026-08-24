"""Run deterministic CPU baseline evaluations for the Phase 1 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baselines import GpsrPolicy, RandomPolicy
from .env.config import FanetConfig
from .env.fanet_env import FanetRoutingEnv
from .evaluation import evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config") / "default.yaml",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = FanetConfig.from_yaml(args.config)
    first_seed = config.seed if args.seed is None else args.seed
    seeds = list(range(first_seed, first_seed + args.episodes))
    env = FanetRoutingEnv(config)
    report = {
        "config": str(args.config),
        "seeds": seeds,
        "random": evaluate_policy(
            env, RandomPolicy(env.drop_action), seeds
        ).to_dict(),
        "gpsr": evaluate_policy(
            env, GpsrPolicy(env.drop_action), seeds
        ).to_dict(),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
