"""Fine-tune a gated KD Student with Teacher-free local PPO."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from .algorithms import StudentFineTuneConfig, fine_tune_student
from .algorithms.ppo import PpoConfig
from .data import DistillationDataset
from .env.fanet_env import FanetRoutingEnv
from .evaluation import evaluate_policy
from .models.policy_adapter import StudentPolicyAdapter
from .models.student_actor_critic import LocalStudentActorCritic
from .models.student_policy import LocalStudentPolicy
from .scenarios import routing_hole_config, routing_hole_options
from .utils import load_checkpoint, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    package = Path(__file__).parent
    parser.add_argument(
        "--config",
        type=Path,
        default=package / "config" / "finetune.yaml",
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=Path("artifacts/lite_globe/student_phase4.pt"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/lite_globe/distillation_dataset.npz"),
    )
    parser.add_argument(
        "--output-checkpoint",
        type=Path,
        default=Path("artifacts/lite_globe/student_phase5.pt"),
    )
    parser.add_argument("--updates", type=int)
    parser.add_argument("--episodes-per-update", type=int)
    parser.add_argument("--kd-lambda", type=float)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
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
    env = FanetRoutingEnv(routing_hole_config(seed))
    options = routing_hole_options()
    policy = LocalStudentPolicy(
        env.config.max_nodes,
        hidden_dim=int(raw["model"]["hidden_dim"]),
    )
    metadata = load_checkpoint(
        args.student_checkpoint,
        policy,
        map_location=device,
    )
    if metadata.get("phase") != 4 or not metadata.get("gate_passed", False):
        raise ValueError("Student checkpoint did not pass the Phase 4 gate")
    actor_critic = LocalStudentActorCritic(policy)

    fine_tune_settings = dict(raw["fine_tuning"])
    if args.updates is not None:
        fine_tune_settings["updates"] = args.updates
    if args.episodes_per_update is not None:
        fine_tune_settings["episodes_per_update"] = args.episodes_per_update
    if args.kd_lambda is not None:
        fine_tune_settings["kd_lambda_initial"] = args.kd_lambda
    fine_tune_config = StudentFineTuneConfig(**fine_tune_settings)
    kd_dataset = None
    if fine_tune_config.kd_lambda_initial > 0:
        if not args.dataset.exists():
            raise FileNotFoundError(
                "optional KD was enabled but the Phase 4 dataset is missing"
            )
        kd_dataset = DistillationDataset.load(args.dataset)

    evaluation_seeds = list(
        range(seed + 300_000, seed + 300_000 + args.evaluation_episodes)
    )
    before = evaluate_policy(
        env,
        StudentPolicyAdapter(policy, device=device),
        evaluation_seeds,
        reset_options=options,
    )
    training = fine_tune_student(
        env,
        actor_critic,
        ppo_config=PpoConfig(**raw["ppo"]),
        fine_tune_config=fine_tune_config,
        seed=seed,
        reset_options=options,
        kd_dataset=kd_dataset,
        device=device,
    )
    after = evaluate_policy(
        env,
        StudentPolicyAdapter(policy, device=device),
        evaluation_seeds,
        reset_options=options,
    )
    gate_passed = (
        after.packet_delivery_ratio >= before.packet_delivery_ratio - 0.05
        and after.mean_episode_reward >= before.mean_episode_reward - 0.5
    )
    save_checkpoint(
        args.output_checkpoint,
        policy,
        metadata={
            "phase": 5,
            "seed": seed,
            "gate_passed": gate_passed,
            "source_checkpoint": str(args.student_checkpoint),
            "teacher_used_during_finetune": False,
            "kd_lambda_initial": fine_tune_config.kd_lambda_initial,
        },
    )
    report = {
        "phase": 5,
        "device": str(device),
        "teacher_used_during_finetune": False,
        "mode": (
            "ppo_plus_decayed_offline_kd"
            if fine_tune_config.kd_lambda_initial > 0
            else "pure_ppo"
        ),
        "parameter_count": sum(
            parameter.numel() for parameter in policy.parameters()
        ),
        "training": {
            **asdict(training),
            "final_metrics": asdict(training.final_metrics),
        },
        "evaluation": {
            "kd_only_before": before.to_dict(),
            "kd_plus_ppo_after": after.to_dict(),
        },
        "gate": {
            "passed": gate_passed,
            "criterion": (
                "PDR degradation <= 0.05 and mean return degradation <= 0.5"
            ),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
