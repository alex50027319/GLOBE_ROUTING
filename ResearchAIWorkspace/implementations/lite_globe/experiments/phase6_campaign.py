"""Multi-seed method comparison and OOD evaluation for Lite-GLOBE."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
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
from ..baselines import GpsrPolicy, RandomPolicy
from ..data import generate_teacher_dataset, split_by_episode_group
from ..env.fanet_env import FanetRoutingEnv
from ..evaluation import (
    episode_row,
    evaluate_policy_results,
    measure_policy_cost,
    summary_row,
    summarize_episode_results,
)
from ..models import (
    GlobalTeacherActorCritic,
    LocalStudentActorCritic,
    LocalStudentPolicy,
)
from ..models.policy_adapter import StudentPolicyAdapter
from ..models.teacher_adapter import TeacherPolicyAdapter
from ..scenarios import phase6_scenarios, routing_hole_config
from ..utils import load_checkpoint, save_checkpoint, seed_everything


@dataclass(frozen=True)
class Phase6Config:
    training_seeds: tuple[int, ...]
    evaluation_episodes: int
    hidden_dim: int
    teacher_updates: int
    teacher_episodes_per_update: int
    dataset_episodes: int
    distillation_epochs: int
    student_updates: int
    student_episodes_per_update: int
    kd_lambda_initial: float
    cost_warmup: int
    cost_repeats: int


def _training_row(
    *,
    seed: int,
    teacher_result: Any,
    distillation_result: Any,
    ppo_result: Any,
    kd_ppo_result: Any,
) -> dict[str, Any]:
    return {
        "training_seed": seed,
        "teacher_mean_return": teacher_result.mean_training_return,
        "teacher_policy_loss": teacher_result.final_metrics.policy_loss,
        "teacher_value_loss": teacher_result.final_metrics.value_loss,
        "teacher_entropy": teacher_result.final_metrics.entropy,
        "distillation_train_kl": distillation_result.train.kl,
        "distillation_validation_kl": distillation_result.validation.kl,
        "distillation_validation_agreement": (
            distillation_result.validation.action_agreement
        ),
        "ppo_only_mean_return": ppo_result.mean_training_return,
        "kd_ppo_mean_return": kd_ppo_result.mean_training_return,
        "kd_ppo_final_kd_lambda": kd_ppo_result.final_metrics.kd_lambda,
    }


def _train_methods(
    config: Phase6Config,
    *,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module | None], dict[str, Any]]:
    seed_everything(seed)
    env = FanetRoutingEnv(routing_hole_config(seed))
    scenario = phase6_scenarios(seed)[0]
    ppo = PpoConfig(
        learning_rate=1e-3,
        entropy_coefficient=0.05,
        update_epochs=4,
        minibatch_size=128,
    )
    student_ppo = PpoConfig(
        learning_rate=3e-4,
        entropy_coefficient=0.01,
        update_epochs=4,
        minibatch_size=128,
    )

    teacher = GlobalTeacherActorCritic(
        env.config.max_nodes, hidden_dim=config.hidden_dim
    )
    teacher_result = train_teacher(
        env,
        teacher,
        ppo_config=ppo,
        updates=config.teacher_updates,
        episodes_per_update=config.teacher_episodes_per_update,
        seed=seed,
        reset_options=scenario.reset_options,
        device=device,
    )

    seed_everything(seed + 10_000)
    untrained = LocalStudentPolicy(
        env.config.max_nodes, hidden_dim=config.hidden_dim
    )
    ppo_only = copy.deepcopy(untrained)
    ppo_result = fine_tune_student(
        env,
        LocalStudentActorCritic(ppo_only),
        ppo_config=student_ppo,
        fine_tune_config=StudentFineTuneConfig(
            updates=config.student_updates,
            episodes_per_update=config.student_episodes_per_update,
        ),
        seed=seed + 20_000,
        reset_options=scenario.reset_options,
        device=device,
    )

    dataset = generate_teacher_dataset(
        env,
        teacher,
        episode_seeds=list(
            range(seed + 100_000, seed + 100_000 + config.dataset_episodes)
        ),
        scenario_id=scenario.name,
        reset_options=scenario.reset_options,
        device=device,
    )
    split = split_by_episode_group(dataset, seed=seed)
    kd_only = copy.deepcopy(untrained)
    distillation_result = train_student_distillation(
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
    kd_ppo_result = fine_tune_student(
        env,
        LocalStudentActorCritic(kd_ppo),
        ppo_config=student_ppo,
        fine_tune_config=StudentFineTuneConfig(
            updates=config.student_updates,
            episodes_per_update=config.student_episodes_per_update,
            kd_lambda_initial=config.kd_lambda_initial,
            kd_decay_rate=0.05,
            kd_batch_size=128,
        ),
        seed=seed + 30_000,
        reset_options=scenario.reset_options,
        kd_dataset=split.train if config.kd_lambda_initial > 0 else None,
        device=device,
    )
    models: dict[str, torch.nn.Module | None] = {
        "Random": None,
        "GPSR": None,
        "Untrained Student": untrained,
        "PPO-only Student": ppo_only,
        "KD-only Student": kd_only,
        "KD+PPO Student": kd_ppo,
        "Global Teacher": teacher,
    }
    metrics = _training_row(
        seed=seed,
        teacher_result=teacher_result,
        distillation_result=distillation_result,
        ppo_result=ppo_result,
        kd_ppo_result=kd_ppo_result,
    )
    return models, metrics


def _checkpoint_paths(directory: Path) -> dict[str, Path]:
    return {
        "Global Teacher": directory / "teacher.pt",
        "Untrained Student": directory / "untrained_student.pt",
        "PPO-only Student": directory / "ppo_only_student.pt",
        "KD-only Student": directory / "kd_only_student.pt",
        "KD+PPO Student": directory / "kd_ppo_student.pt",
    }


def _save_seed_checkpoint(
    directory: Path,
    models: dict[str, torch.nn.Module | None],
    metrics: dict[str, Any],
    *,
    seed: int,
) -> None:
    for method, path in _checkpoint_paths(directory).items():
        model = models[method]
        assert model is not None
        save_checkpoint(
            path,
            model,
            metadata={"phase": 6, "training_seed": seed, "method": method},
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_seed_checkpoint(
    directory: Path,
    config: Phase6Config,
    *,
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module | None], dict[str, Any]] | None:
    paths = _checkpoint_paths(directory)
    metrics_path = directory / "training_metrics.json"
    if not metrics_path.is_file() or not all(path.is_file() for path in paths.values()):
        return None
    max_nodes = routing_hole_config().max_nodes
    models: dict[str, torch.nn.Module | None] = {
        "Random": None,
        "GPSR": None,
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
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return models, metrics


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
    if method == "Global Teacher":
        assert isinstance(model, GlobalTeacherActorCritic)
        return TeacherPolicyAdapter(env, model, device=device)
    assert isinstance(model, LocalStudentPolicy)
    return StudentPolicyAdapter(model, device=device)


def run_phase6_campaign(
    config: Phase6Config,
    *,
    device: torch.device | str = "cpu",
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Train all ablations and retain episode-, seed-, and cost-level rows."""

    if not config.training_seeds or config.evaluation_episodes <= 0:
        raise ValueError("training seeds and evaluation episodes are required")
    device = torch.device(device)
    episode_rows: list[dict[str, Any]] = []
    seed_summary_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []

    for training_seed in config.training_seeds:
        seed_directory = (
            checkpoint_dir / f"seed_{training_seed}"
            if checkpoint_dir is not None
            else None
        )
        restored = (
            _load_seed_checkpoint(seed_directory, config, device=device)
            if resume and seed_directory is not None
            else None
        )
        if restored is None:
            models, training_metrics = _train_methods(
                config, seed=training_seed, device=device
            )
            if seed_directory is not None:
                _save_seed_checkpoint(
                    seed_directory,
                    models,
                    training_metrics,
                    seed=training_seed,
                )
        else:
            models, training_metrics = restored
        training_rows.append(training_metrics)
        for scenario_index, scenario in enumerate(
            phase6_scenarios(training_seed)
        ):
            env = FanetRoutingEnv(scenario.config)
            evaluation_seeds = list(
                range(
                    500_000 + scenario_index * 10_000,
                    500_000
                    + scenario_index * 10_000
                    + config.evaluation_episodes,
                )
            )
            for method, model in models.items():
                policy = _policy(method, model, env, device)
                results = evaluate_policy_results(
                    env,
                    policy,
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
                seed_summary_rows.append(
                    summary_row(
                        summarize_episode_results(results),
                        method=method,
                        scenario=scenario.name,
                        training_seed=training_seed,
                    )
                )

        cost_env = FanetRoutingEnv(routing_hole_config(training_seed))
        local_observation, _ = cost_env.reset(
            seed=training_seed,
            options=phase6_scenarios(training_seed)[0].reset_options,
        )
        global_observation = cost_env.global_observation()
        for method, model in models.items():
            policy = _policy(method, model, cost_env, device)
            cost = measure_policy_cost(
                policy,
                local_observation,
                model=model,
                input_observation=(
                    global_observation
                    if method == "Global Teacher"
                    else local_observation
                ),
                device=device,
                warmup=config.cost_warmup,
                repeats=config.cost_repeats,
            )
            cost_rows.append(
                {
                    "training_seed": training_seed,
                    "method": method,
                    **cost.to_dict(),
                }
            )
    return {
        "episodes": episode_rows,
        "seed_summaries": seed_summary_rows,
        "training": training_rows,
        "costs": cost_rows,
    }
