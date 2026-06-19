"""Generate offline Teacher data and distill a local Lite-GLOBE Student."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml

from .algorithms.distillation import (
    DistillationConfig,
    evaluate_distillation,
    train_student_distillation,
)
from .algorithms.ppo import PpoConfig
from .algorithms.teacher_trainer import train_teacher
from .baselines import GpsrPolicy, RandomPolicy
from .data import generate_teacher_dataset, split_by_episode_group
from .env.fanet_env import FanetRoutingEnv
from .evaluation import evaluate_policy
from .models.policy_adapter import StudentPolicyAdapter
from .models.student_policy import LocalStudentPolicy
from .models.teacher_adapter import TeacherPolicyAdapter
from .models.teacher_gnn import GlobalTeacherActorCritic
from .scenarios import routing_hole_config, routing_hole_options
from .utils import load_checkpoint, save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    package = Path(__file__).parent
    parser.add_argument(
        "--config",
        type=Path,
        default=package / "config" / "distillation.yaml",
    )
    parser.add_argument(
        "--teacher-config",
        type=Path,
        default=package / "config" / "teacher.yaml",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        default=Path("artifacts/lite_globe/teacher_phase3.pt"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts/lite_globe/distillation_dataset.npz"),
    )
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=Path("artifacts/lite_globe/student_phase4.pt"),
    )
    parser.add_argument("--dataset-episodes", type=int)
    parser.add_argument("--epochs", type=int)
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


def _prepare_teacher(
    env: FanetRoutingEnv,
    options: dict,
    *,
    teacher_config: dict,
    checkpoint: Path,
    seed: int,
    device: torch.device,
) -> tuple[GlobalTeacherActorCritic, str]:
    teacher = GlobalTeacherActorCritic(
        env.config.max_nodes,
        hidden_dim=int(teacher_config["model"]["hidden_dim"]),
    )
    if checkpoint.exists():
        metadata = load_checkpoint(checkpoint, teacher, map_location=device)
        if (
            metadata.get("phase") != 3
            or not metadata.get("gate_passed", False)
            or metadata.get("scenario") != "routing_hole"
        ):
            raise ValueError("Teacher checkpoint did not pass the Phase 3 gate")
        return teacher.to(device), "loaded_gated_checkpoint"

    training = teacher_config["training"]
    train_teacher(
        env,
        teacher,
        ppo_config=PpoConfig(**teacher_config["ppo"]),
        updates=int(training["updates"]),
        episodes_per_update=int(training["episodes_per_update"]),
        seed=seed,
        reset_options=options,
        device=device,
    )
    evaluation_seeds = list(range(seed + 50_000, seed + 50_100))
    teacher_metrics = evaluate_policy(
        env,
        TeacherPolicyAdapter(env, teacher, device=device),
        evaluation_seeds,
        reset_options=options,
    )
    gpsr_metrics = evaluate_policy(
        env,
        GpsrPolicy(env.drop_action),
        evaluation_seeds,
        reset_options=options,
    )
    random_metrics = evaluate_policy(
        env,
        RandomPolicy(env.drop_action),
        evaluation_seeds,
        reset_options=options,
    )
    seed_everything(seed)
    local_model = LocalStudentPolicy(env.config.max_nodes, hidden_dim=64)
    local_metrics = evaluate_policy(
        env,
        StudentPolicyAdapter(local_model),
        evaluation_seeds,
        reset_options=options,
    )
    strongest_baseline = max(
        gpsr_metrics.packet_delivery_ratio,
        random_metrics.packet_delivery_ratio,
        local_metrics.packet_delivery_ratio,
    )
    if teacher_metrics.packet_delivery_ratio <= strongest_baseline:
        raise RuntimeError("bootstrapped Teacher failed the Phase 3 gate")
    save_checkpoint(
        checkpoint,
        teacher,
        metadata={
            "phase": 3,
            "seed": seed,
            "gate_passed": True,
            "scenario": "routing_hole",
        },
    )
    return teacher, "bootstrapped_and_gated"


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    teacher_raw = yaml.safe_load(
        args.teacher_config.read_text(encoding="utf-8")
    )
    seed = int(raw["runtime"]["seed"] if args.seed is None else args.seed)
    device = _device(args.device or raw["runtime"]["device"])
    seed_everything(seed)
    env = FanetRoutingEnv(routing_hole_config(seed))
    options = routing_hole_options()
    teacher, teacher_source = _prepare_teacher(
        env,
        options,
        teacher_config=teacher_raw,
        checkpoint=args.teacher_checkpoint,
        seed=seed,
        device=device,
    )

    episode_count = int(
        raw["dataset"]["episodes"]
        if args.dataset_episodes is None
        else args.dataset_episodes
    )
    dataset_seeds = list(range(seed + 100_000, seed + 100_000 + episode_count))
    dataset = generate_teacher_dataset(
        env,
        teacher,
        episode_seeds=dataset_seeds,
        scenario_id="routing_hole",
        reset_options=options,
        device=device,
    )
    dataset.save(args.dataset)
    split = split_by_episode_group(
        dataset,
        seed=seed,
        train_fraction=float(raw["dataset"]["train_fraction"]),
        validation_fraction=float(raw["dataset"]["validation_fraction"]),
    )

    distillation_settings = dict(raw["distillation"])
    if args.epochs is not None:
        distillation_settings["epochs"] = args.epochs
    distillation_config = DistillationConfig(**distillation_settings)
    seed_everything(seed)
    student = LocalStudentPolicy(
        env.config.max_nodes,
        hidden_dim=int(raw["model"]["hidden_dim"]),
    )
    initial_test = evaluate_distillation(
        student, split.test, config=distillation_config, device=device
    )
    training = train_student_distillation(
        student,
        split.train,
        split.validation,
        config=distillation_config,
        seed=seed,
        device=device,
    )
    final_test = evaluate_distillation(
        student, split.test, config=distillation_config, device=device
    )

    evaluation_seeds = list(
        range(seed + 200_000, seed + 200_000 + args.evaluation_episodes)
    )
    random_metrics = evaluate_policy(
        env,
        RandomPolicy(env.drop_action),
        evaluation_seeds,
        reset_options=options,
    )
    gpsr_metrics = evaluate_policy(
        env,
        GpsrPolicy(env.drop_action),
        evaluation_seeds,
        reset_options=options,
    )
    teacher_metrics = evaluate_policy(
        env,
        TeacherPolicyAdapter(env, teacher, device=device),
        evaluation_seeds,
        reset_options=options,
    )
    student_metrics = evaluate_policy(
        env,
        StudentPolicyAdapter(student, device=device),
        evaluation_seeds,
        reset_options=options,
    )
    gate_passed = (
        torch.isfinite(torch.tensor(final_test.kl)).item()
        and final_test.kl < initial_test.kl
        and final_test.action_agreement >= 0.9
        and student_metrics.packet_delivery_ratio
        >= teacher_metrics.packet_delivery_ratio - 0.1
    )
    save_checkpoint(
        args.student_checkpoint,
        student,
        metadata={
            "phase": 4,
            "seed": seed,
            "gate_passed": gate_passed,
            "temperature": distillation_config.temperature,
            "teacher_checkpoint": str(args.teacher_checkpoint),
        },
    )
    report = {
        "phase": 4,
        "device": str(device),
        "teacher_source": teacher_source,
        "dataset": {
            "path": str(args.dataset),
            "samples": len(dataset),
            "episode_groups": len(set(dataset.group_ids)),
            "train_samples": len(split.train),
            "validation_samples": len(split.validation),
            "test_samples": len(split.test),
            "contains_global_state": False,
        },
        "distillation": {
            "config": asdict(distillation_config),
            "initial_test": initial_test.to_dict(),
            "train": training.train.to_dict(),
            "validation": training.validation.to_dict(),
            "test": final_test.to_dict(),
        },
        "evaluation": {
            "random": random_metrics.to_dict(),
            "gpsr": gpsr_metrics.to_dict(),
            "teacher": teacher_metrics.to_dict(),
            "student_kd": student_metrics.to_dict(),
        },
        "gate": {
            "passed": gate_passed,
            "criterion": (
                "finite lower test KL, agreement >= 0.9, "
                "Student PDR within 0.1 of Teacher"
            ),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not gate_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
