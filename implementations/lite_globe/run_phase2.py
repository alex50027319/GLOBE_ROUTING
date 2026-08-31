"""Run a CPU smoke evaluation of the randomly initialized Local Student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from .env.config import FanetConfig
from .env.fanet_env import FanetRoutingEnv
from .evaluation import evaluate_policy
from .models.policy_adapter import StudentPolicyAdapter
from .models.student_policy import LocalStudentPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    package = Path(__file__).parent
    parser.add_argument(
        "--env-config", type=Path, default=package / "config" / "default.yaml"
    )
    parser.add_argument(
        "--student-config", type=Path, default=package / "config" / "student.yaml"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_config = FanetConfig.from_yaml(args.env_config)
    student_config = yaml.safe_load(args.student_config.read_text(encoding="utf-8"))
    seed = env_config.seed if args.seed is None else args.seed
    torch.manual_seed(seed)
    model = LocalStudentPolicy(
        max_nodes=env_config.max_nodes,
        hidden_dim=int(student_config["model"]["hidden_dim"]),
    )
    env = FanetRoutingEnv(env_config)
    policy = StudentPolicyAdapter(model, deterministic=True)
    seeds = list(range(seed, seed + args.episodes))
    report = {
        "phase": 2,
        "status": "architecture_smoke_only",
        "seed": seed,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": evaluate_policy(env, policy, seeds).to_dict(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
