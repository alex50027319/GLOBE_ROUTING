"""Train and gate the privileged Teacher on a deterministic routing hole."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from .algorithms.ppo import PpoConfig
from .algorithms.teacher_trainer import train_teacher
from .baselines import GpsrPolicy, RandomPolicy
from .env.fanet_env import FanetRoutingEnv
from .evaluation import evaluate_policy
from .models.policy_adapter import StudentPolicyAdapter
from .models.student_policy import LocalStudentPolicy
from .models.teacher_adapter import TeacherPolicyAdapter
from .models.teacher_gnn import GlobalTeacherActorCritic
from .scenarios import routing_hole_config, routing_hole_options
from .utils import save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(__file__).with_name("config") / "teacher.yaml"
    parser.add_argument("--config", type=Path, default=default)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--episodes-per-update", type=int)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(raw["runtime"]["seed"] if args.seed is None else args.seed)
    device = _device(args.device or raw["runtime"]["device"])
    seed_everything(seed)
    env_config = routing_hole_config(seed)
    env = FanetRoutingEnv(env_config)
    options = routing_hole_options()
    model = GlobalTeacherActorCritic(
        max_nodes=env_config.max_nodes,
        hidden_dim=int(raw["model"]["hidden_dim"]),
    )
    ppo = PpoConfig(**raw["ppo"])
    updates = int(
        raw["training"]["updates"] if args.updates is None else args.updates
    )
    episodes_per_update = int(
        raw["training"]["episodes_per_update"]
        if args.episodes_per_update is None
        else args.episodes_per_update
    )
    training = train_teacher(
        env,
        model,
        ppo_config=ppo,
        updates=updates,
        episodes_per_update=episodes_per_update,
        seed=seed,
        reset_options=options,
        device=device,
    )
    seeds = list(range(seed + 10_000, seed + 10_000 + args.evaluation_episodes))
    random_metrics = evaluate_policy(
        env,
        RandomPolicy(env.drop_action),
        seeds,
        reset_options=options,
    )
    gpsr_metrics = evaluate_policy(
        env,
        GpsrPolicy(env.drop_action),
        seeds,
        reset_options=options,
    )
    seed_everything(seed)
    local_model = LocalStudentPolicy(env_config.max_nodes, hidden_dim=64)
    local_metrics = evaluate_policy(
        env,
        StudentPolicyAdapter(local_model),
        seeds,
        reset_options=options,
    )
    teacher_metrics = evaluate_policy(
        env,
        TeacherPolicyAdapter(env, model, device=device),
        seeds,
        reset_options=options,
    )
    comparison = {
        "random": random_metrics.to_dict(),
        "gpsr": gpsr_metrics.to_dict(),
        "local_untrained": local_metrics.to_dict(),
        "teacher": teacher_metrics.to_dict(),
    }
    strongest_baseline = max(
        random_metrics.packet_delivery_ratio,
        gpsr_metrics.packet_delivery_ratio,
        local_metrics.packet_delivery_ratio,
    )
    gate_passed = teacher_metrics.packet_delivery_ratio > strongest_baseline
    if args.checkpoint:
        save_checkpoint(
            args.checkpoint,
            model,
            metadata={
                "phase": 3,
                "seed": seed,
                "gate_passed": gate_passed,
                "scenario": "routing_hole",
            },
        )
    report = {
        "phase": 3,
        "teacher_label": "privileged_reference_policy",
        "device": str(device),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "training": {
            **asdict(training),
            "final_metrics": asdict(training.final_metrics),
        },
        "evaluation": comparison,
        "gate": {
            "passed": gate_passed,
            "criterion": "teacher PDR > strongest toy baseline PDR",
            "strongest_baseline_pdr": strongest_baseline,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
