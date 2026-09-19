"""Curriculum-trained, topology-held-out generalization campaign."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch

from ..algorithms import (
    DistillationConfig,
    StudentFineTuneConfig,
    fine_tune_student,
    train_student_distillation,
    train_teacher,
)
from ..algorithms.ppo import PpoConfig
from ..baselines import (
    GpsrPolicy,
    RandomPolicy,
    ShortestPathOraclePolicy,
)
from ..data import (
    concatenate_datasets,
    generate_teacher_dataset,
    split_by_episode_group,
)
from ..env.fanet_env import FanetRoutingEnv
from ..evaluation import (
    episode_row,
    evaluate_policy_results,
    generalization_summary,
)
from ..models import (
    GlobalTeacherActorCritic,
    LocalStudentActorCritic,
    LocalStudentPolicy,
)
from ..models.policy_adapter import StudentPolicyAdapter
from ..models.teacher_adapter import TeacherPolicyAdapter
from ..scenarios import phase7_curriculum, phase7_evaluation_scenarios
from ..utils import load_checkpoint, save_checkpoint, seed_everything


@dataclass(frozen=True)
class Phase7Config:
    training_seeds: tuple[int, ...]
    evaluation_episodes: int
    hidden_dim: int
    teacher_updates_per_stage: int
    teacher_episodes_per_update: int
    dataset_episodes_per_stage: int
    distillation_epochs: int
    student_updates_per_stage: int
    student_episodes_per_update: int
    kd_lambda_initial: float


NEURAL_METHODS = (
    "Untrained Student",
    "PPO-only Student",
    "KD-only Student",
    "KD+PPO Student",
    "Global Teacher",
)


def _train_methods(
    config: Phase7Config,
    *,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module | None], dict[str, Any]]:
    seed_everything(seed)
    stages = phase7_curriculum(seed)
    max_nodes = stages[0].config.max_nodes
    teacher = GlobalTeacherActorCritic(
        max_nodes, hidden_dim=config.hidden_dim
    )
    teacher_ppo = PpoConfig(
        learning_rate=1e-3,
        entropy_coefficient=0.05,
        update_epochs=4,
        minibatch_size=128,
    )
    student_ppo = PpoConfig(
        learning_rate=3e-4,
        entropy_coefficient=0.02,
        update_epochs=4,
        minibatch_size=128,
    )
    teacher_returns: dict[str, float] = {}
    for stage_index, stage in enumerate(stages):
        result = train_teacher(
            FanetRoutingEnv(stage.config),
            teacher,
            ppo_config=teacher_ppo,
            updates=config.teacher_updates_per_stage,
            episodes_per_update=config.teacher_episodes_per_update,
            seed=seed + stage_index * 10_000,
            reset_options=stage.reset_options,
            device=device,
        )
        teacher_returns[stage.name] = result.mean_training_return

    seed_everything(seed + 100_000)
    untrained = LocalStudentPolicy(max_nodes, hidden_dim=config.hidden_dim)
    ppo_only = copy.deepcopy(untrained)
    ppo_returns: dict[str, float] = {}
    for stage_index, stage in enumerate(stages):
        result = fine_tune_student(
            FanetRoutingEnv(stage.config),
            LocalStudentActorCritic(ppo_only),
            ppo_config=student_ppo,
            fine_tune_config=StudentFineTuneConfig(
                updates=config.student_updates_per_stage,
                episodes_per_update=config.student_episodes_per_update,
            ),
            seed=seed + 200_000 + stage_index * 10_000,
            reset_options=stage.reset_options,
            device=device,
        )
        ppo_returns[stage.name] = result.mean_training_return

    datasets = []
    for stage_index, stage in enumerate(stages):
        datasets.append(
            generate_teacher_dataset(
                FanetRoutingEnv(stage.config),
                teacher,
                episode_seeds=list(
                    range(
                        seed + 300_000 + stage_index * 10_000,
                        seed
                        + 300_000
                        + stage_index * 10_000
                        + config.dataset_episodes_per_stage,
                    )
                ),
                scenario_id=stage.name,
                reset_options=stage.reset_options,
                device=device,
            )
        )
    dataset = concatenate_datasets(datasets)
    split = split_by_episode_group(dataset, seed=seed)
    kd_only = copy.deepcopy(untrained)
    distillation = train_student_distillation(
        kd_only,
        split.train,
        split.validation,
        config=DistillationConfig(
            epochs=config.distillation_epochs,
            learning_rate=1e-3,
            batch_size=128,
        ),
        seed=seed,
        device=device,
    )

    kd_ppo = copy.deepcopy(kd_only)
    kd_ppo_returns: dict[str, float] = {}
    for stage_index, stage in enumerate(stages):
        result = fine_tune_student(
            FanetRoutingEnv(stage.config),
            LocalStudentActorCritic(kd_ppo),
            ppo_config=student_ppo,
            fine_tune_config=StudentFineTuneConfig(
                updates=config.student_updates_per_stage,
                episodes_per_update=config.student_episodes_per_update,
                kd_lambda_initial=config.kd_lambda_initial,
                kd_decay_rate=0.05,
                kd_batch_size=128,
            ),
            seed=seed + 400_000 + stage_index * 10_000,
            reset_options=stage.reset_options,
            kd_dataset=split.train,
            device=device,
        )
        kd_ppo_returns[stage.name] = result.mean_training_return

    models: dict[str, torch.nn.Module | None] = {
        "Random": None,
        "GPSR": None,
        "Untrained Student": untrained,
        "PPO-only Student": ppo_only,
        "KD-only Student": kd_only,
        "KD+PPO Student": kd_ppo,
        "Global Teacher": teacher,
        "Shortest-path Oracle": None,
    }
    metrics: dict[str, Any] = {
        "training_seed": seed,
        "distillation_validation_kl": distillation.validation.kl,
        "distillation_validation_agreement": (
            distillation.validation.action_agreement
        ),
    }
    for stage in stages:
        metrics[f"{stage.name}_teacher_return"] = teacher_returns[stage.name]
        metrics[f"{stage.name}_ppo_only_return"] = ppo_returns[stage.name]
        metrics[f"{stage.name}_kd_ppo_return"] = kd_ppo_returns[stage.name]
    return models, metrics


def _paths(directory: Path) -> dict[str, Path]:
    return {
        method: directory
        / (
            method.lower()
            .replace(" ", "_")
            .replace("+", "_plus_")
            .replace("-", "_")
            + ".pt"
        )
        for method in NEURAL_METHODS
    }


def _save(
    directory: Path,
    models: dict[str, torch.nn.Module | None],
    metrics: dict[str, Any],
    *,
    seed: int,
) -> None:
    for method, path in _paths(directory).items():
        model = models[method]
        assert model is not None
        save_checkpoint(
            path,
            model,
            metadata={"phase": 7, "training_seed": seed, "method": method},
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load(
    directory: Path,
    config: Phase7Config,
    *,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module | None], dict[str, Any]] | None:
    paths = _paths(directory)
    metrics_path = directory / "training_metrics.json"
    if not metrics_path.is_file() or not all(path.is_file() for path in paths.values()):
        return None
    max_nodes = phase7_curriculum(0)[0].config.max_nodes
    models: dict[str, torch.nn.Module | None] = {
        "Random": None,
        "GPSR": None,
        "Shortest-path Oracle": None,
        "Global Teacher": GlobalTeacherActorCritic(
            max_nodes, hidden_dim=config.hidden_dim
        ),
        "Untrained Student": LocalStudentPolicy(
            max_nodes, hidden_dim=config.hidden_dim
        ),
        "PPO-only Student": LocalStudentPolicy(
            max_nodes, hidden_dim=config.hidden_dim
        ),
        "KD-only Student": LocalStudentPolicy(
            max_nodes, hidden_dim=config.hidden_dim
        ),
        "KD+PPO Student": LocalStudentPolicy(
            max_nodes, hidden_dim=config.hidden_dim
        ),
    }
    for method, path in paths.items():
        model = models[method]
        assert model is not None
        load_checkpoint(path, model, map_location=device)
    return models, json.loads(metrics_path.read_text(encoding="utf-8"))


def _policy(
    method: str,
    model: torch.nn.Module | None,
    env: FanetRoutingEnv,
    device: torch.device,
):
    if method == "Random":
        return RandomPolicy(env.drop_action)
    if method == "GPSR":
        return GpsrPolicy(env.drop_action)
    if method == "Shortest-path Oracle":
        return ShortestPathOraclePolicy(env)
    if method == "Global Teacher":
        assert isinstance(model, GlobalTeacherActorCritic)
        return TeacherPolicyAdapter(env, model, device=device)
    assert isinstance(model, LocalStudentPolicy)
    return StudentPolicyAdapter(model, device=device)


def run_phase7_campaign(
    config: Phase7Config,
    *,
    device: torch.device | str = "cpu",
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    device = torch.device(device)
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for training_seed in config.training_seeds:
        directory = (
            checkpoint_dir / f"seed_{training_seed}"
            if checkpoint_dir is not None
            else None
        )
        restored = (
            _load(directory, config, device=device)
            if resume and directory is not None
            else None
        )
        if restored is None:
            models, metrics = _train_methods(
                config, seed=training_seed, device=device
            )
            if directory is not None:
                _save(directory, models, metrics, seed=training_seed)
        else:
            models, metrics = restored
        training_rows.append(metrics)
        for scenario_index, scenario in enumerate(
            phase7_evaluation_scenarios(training_seed)
        ):
            evaluation_seeds = list(
                range(
                    900_000 + scenario_index * 10_000,
                    900_000
                    + scenario_index * 10_000
                    + config.evaluation_episodes,
                )
            )
            env = FanetRoutingEnv(scenario.config)
            for method, model in models.items():
                results = evaluate_policy_results(
                    env,
                    _policy(method, model, env, device),
                    evaluation_seeds,
                    reset_options=scenario.reset_options,
                )
                episode_rows.extend(
                    episode_row(
                        result,
                        method=method,
                        scenario=scenario.name,
                        training_seed=training_seed,
                    )
                    for result in results
                )
                summary_rows.append(
                    generalization_summary(
                        results,
                        method=method,
                        scenario=scenario.name,
                        training_seed=training_seed,
                    )
                )
    return {
        "episodes": episode_rows,
        "seed_summaries": summary_rows,
        "training": training_rows,
    }
