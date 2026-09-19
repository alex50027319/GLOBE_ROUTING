"""Literature-aligned optimization using existing Phase 7 Teachers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..algorithms import DistillationConfig, train_student_distillation
from ..baselines import GpsrPolicy, ShortestPathOraclePolicy
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
    GeographicResidualStudentPolicy,
    GlobalTeacherActorCritic,
    LocalStudentPolicy,
)
from ..models.policy_adapter import StudentPolicyAdapter
from ..models.teacher_adapter import TeacherPolicyAdapter
from ..scenarios import phase8_curriculum, phase8_evaluation_scenarios
from ..scenarios import (
    phase8_hole_calibration_scenarios,
    phase8_hole_training_scenarios,
)
from ..utils import load_checkpoint, save_checkpoint, seed_everything


@dataclass(frozen=True)
class Phase8Config:
    training_seeds: tuple[int, ...]
    evaluation_episodes: int
    hidden_dim: int
    dataset_episodes_per_stage: int
    distillation_epochs: int
    oracle_coefficient: float
    teacher_action_coefficient: float
    early_stopping_patience: int
    initial_prior_strength: float
    calibration_episodes_per_stage: int
    structural_hole_episodes_per_variant: int
    calibration_pdr_tolerance: float
    prior_pretraining_epochs: int


PHASE8_METHODS = (
    "GPSR",
    "Phase 7 KD-only",
    "Untrained Geo-Residual",
    "Geo-Residual KD",
    "Global Teacher",
    "Shortest-path Oracle",
)


def _load_phase7_models(
    directory: Path,
    *,
    max_nodes: int,
    hidden_dim: int,
    device: torch.device,
) -> tuple[GlobalTeacherActorCritic, LocalStudentPolicy]:
    teacher = GlobalTeacherActorCritic(max_nodes, hidden_dim=hidden_dim)
    kd_student = LocalStudentPolicy(max_nodes, hidden_dim=hidden_dim)
    load_checkpoint(
        directory / "global_teacher.pt",
        teacher,
        map_location=device,
    )
    load_checkpoint(
        directory / "kd_only_student.pt",
        kd_student,
        map_location=device,
    )
    return teacher, kd_student


def _train_optimized_student(
    config: Phase8Config,
    *,
    seed: int,
    teacher: GlobalTeacherActorCritic,
    device: torch.device,
) -> tuple[GeographicResidualStudentPolicy, dict[str, Any]]:
    seed_everything(seed + 800_000)
    stages = phase8_curriculum(seed)
    datasets = []
    for stage_index, stage in enumerate(stages):
        start = seed + 810_000 + stage_index * 10_000
        teacher_count = max(3, config.dataset_episodes_per_stage // 2)
        oracle_count = max(
            3, config.dataset_episodes_per_stage - teacher_count
        )
        datasets.extend(
            [
                generate_teacher_dataset(
                    FanetRoutingEnv(stage.config),
                    teacher,
                    episode_seeds=list(
                        range(start, start + teacher_count)
                    ),
                    scenario_id=f"{stage.name}_teacher_rollout",
                    reset_options=stage.reset_options,
                    rollout_policy="teacher",
                    device=device,
                ),
                generate_teacher_dataset(
                    FanetRoutingEnv(stage.config),
                    teacher,
                    episode_seeds=list(
                        range(
                            start + teacher_count,
                            start + teacher_count + oracle_count,
                        )
                    ),
                    scenario_id=f"{stage.name}_oracle_rollout",
                    reset_options=stage.reset_options,
                    rollout_policy="oracle",
                    device=device,
                ),
            ]
        )
    for variant_index, scenario in enumerate(
        phase8_hole_training_scenarios(seed)
    ):
        start = seed + 840_000 + variant_index * 10_000
        count = config.structural_hole_episodes_per_variant
        datasets.extend(
            [
                generate_teacher_dataset(
                    FanetRoutingEnv(scenario.config),
                    teacher,
                    episode_seeds=list(range(start, start + count)),
                    scenario_id=f"{scenario.name}_teacher_rollout",
                    reset_options=scenario.reset_options,
                    rollout_policy="teacher",
                    device=device,
                ),
                generate_teacher_dataset(
                    FanetRoutingEnv(scenario.config),
                    teacher,
                    episode_seeds=list(
                        range(start + count, start + 2 * count)
                    ),
                    scenario_id=f"{scenario.name}_oracle_rollout",
                    reset_options=scenario.reset_options,
                    rollout_policy="oracle",
                    device=device,
                ),
            ]
        )
    dataset = concatenate_datasets(datasets)
    split = split_by_episode_group(dataset, seed=seed + 820_000)
    model = GeographicResidualStudentPolicy(
        stages[0].config.max_nodes,
        hidden_dim=config.hidden_dim,
        initial_prior_strength=config.initial_prior_strength,
    )
    _pretrain_forwardability_prior(
        model,
        split.train,
        epochs=config.prior_pretraining_epochs,
        seed=seed + 825_000,
        device=device,
    )
    result = train_student_distillation(
        model,
        split.train,
        split.validation,
        config=DistillationConfig(
            epochs=config.distillation_epochs,
            learning_rate=5e-4,
            batch_size=128,
            weight_decay=1e-5,
            oracle_coefficient=config.oracle_coefficient,
            teacher_action_coefficient=config.teacher_action_coefficient,
            early_stopping_patience=config.early_stopping_patience,
        ),
        seed=seed + 830_000,
        device=device,
    )
    calibration = _calibrate_residual_weight(
        model,
        stages,
        seed=seed,
        episodes_per_stage=config.calibration_episodes_per_stage,
        pdr_tolerance=config.calibration_pdr_tolerance,
        device=device,
    )
    test_metrics = result.validation
    return model, {
        "training_seed": seed,
        "dataset_samples": len(dataset),
        "completed_epochs": result.epochs,
        "validation_kl": test_metrics.kl,
        "validation_teacher_agreement": test_metrics.action_agreement,
        "validation_oracle_agreement": (
            test_metrics.oracle_action_agreement
        ),
        "learned_prior_strength": float(
            torch.nn.functional.softplus(
                model.log_prior_strength.detach().cpu()
            ).item()
        ),
        "learned_forwardability_presence_strength": float(
            torch.nn.functional.softplus(
                model.log_forwardability_strength.detach().cpu()
            )[0].item()
        ),
        "learned_forwardability_degree_strength": float(
            torch.nn.functional.softplus(
                model.log_forwardability_strength.detach().cpu()
            )[1].item()
        ),
        "calibrated_residual_weight": calibration["weight"],
        "calibration_score": calibration["score"],
        "calibration_pdr": calibration["pdr"],
        "calibration_hole_pdr": calibration["hole_pdr"],
        "calibration_gpsr_fallback_pdr": (
            calibration["gpsr_fallback_pdr"]
        ),
    }


def _pretrain_forwardability_prior(
    model: GeographicResidualStudentPolicy,
    dataset,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> None:
    """Learn only the deployable forwardability prior before full KD."""

    if epochs <= 0:
        raise ValueError("prior_pretraining_epochs must be positive")
    if "candidate_forwardability" not in dataset.arrays:
        raise ValueError("prior pretraining requires forwardability features")
    model.to(device)
    original_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.log_forwardability_strength.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [model.log_forwardability_strength],
        lr=5e-2,
        weight_decay=1e-4,
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        generator=generator,
    )
    for _ in range(epochs):
        model.train()
        for batch in loader:
            observation = {
                key: value.to(device)
                for key, value in batch.items()
                if key
                in {
                    "self_features",
                    "neighbor_features",
                    "edge_features",
                    "packet_features",
                    "action_mask",
                    "candidate_forwardability",
                }
            }
            output = model(observation)
            loss = torch.nn.functional.cross_entropy(
                output.masked_logits,
                batch["oracle_actions"].to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_requires_grad[name])


def _calibrate_residual_weight(
    model: GeographicResidualStudentPolicy,
    stages,
    *,
    seed: int,
    episodes_per_stage: int,
    pdr_tolerance: float,
    device: torch.device,
) -> dict[str, float]:
    if episodes_per_stage <= 0:
        raise ValueError("calibration_episodes_per_stage must be positive")
    if not 0.0 <= pdr_tolerance <= 1.0:
        raise ValueError("calibration_pdr_tolerance must be in [0, 1]")
    candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
    measurements: list[dict[str, float]] = []
    for weight in candidates:
        model.set_residual_weight(weight)
        generic_delivered = 0
        generic_episodes = 0
        successful_steps: list[int] = []
        energy: list[float] = []
        for stage_index, stage in enumerate(stages):
            start = seed + 850_000 + stage_index * 10_000
            results = evaluate_policy_results(
                FanetRoutingEnv(stage.config),
                StudentPolicyAdapter(model, device=device),
                list(range(start, start + episodes_per_stage)),
                reset_options=stage.reset_options,
            )
            generic_delivered += sum(result.delivered for result in results)
            generic_episodes += len(results)
            successful_steps.extend(
                result.steps for result in results if result.delivered
            )
            energy.extend(
                result.transmission_energy_proxy for result in results
            )
        hole_delivered = 0
        hole_episodes = 0
        for scenario_index, scenario in enumerate(
            phase8_hole_calibration_scenarios(seed)
        ):
            start = seed + 890_000 + scenario_index * 10_000
            results = evaluate_policy_results(
                FanetRoutingEnv(scenario.config),
                StudentPolicyAdapter(model, device=device),
                list(range(start, start + episodes_per_stage)),
                reset_options=scenario.reset_options,
            )
            hole_delivered += sum(result.delivered for result in results)
            hole_episodes += len(results)
        generic_pdr = generic_delivered / generic_episodes
        hole_pdr = hole_delivered / hole_episodes
        mean_delay = (
            sum(successful_steps) / len(successful_steps)
            if successful_steps
            else 100.0
        )
        mean_energy = sum(energy) / len(energy)
        measurements.append(
            {
                "weight": weight,
                "generic_pdr": generic_pdr,
                "hole_pdr": hole_pdr,
                "mean_delay": mean_delay,
                "mean_energy": mean_energy,
            }
        )
    baseline_pdr = measurements[0]["generic_pdr"]
    feasible = [
        item
        for item in measurements
        if item["generic_pdr"] >= baseline_pdr - pdr_tolerance
    ]
    best = max(
        feasible,
        key=lambda item: (
            item["hole_pdr"],
            item["generic_pdr"],
            -item["mean_delay"],
            -item["mean_energy"],
            -item["weight"],
        ),
    )
    model.set_residual_weight(best["weight"])
    score = (
        100.0 * best["generic_pdr"]
        + 100.0 * best["hole_pdr"]
        - best["mean_delay"]
        - 0.1 * best["mean_energy"]
    )
    return {
        "weight": best["weight"],
        "score": score,
        "pdr": best["generic_pdr"],
        "hole_pdr": best["hole_pdr"],
        "gpsr_fallback_pdr": baseline_pdr,
    }


def _policy(
    method: str,
    model: torch.nn.Module | None,
    env: FanetRoutingEnv,
    device: torch.device,
):
    if method == "GPSR":
        return GpsrPolicy(env.drop_action)
    if method == "Shortest-path Oracle":
        return ShortestPathOraclePolicy(env)
    if method == "Global Teacher":
        assert isinstance(model, GlobalTeacherActorCritic)
        return TeacherPolicyAdapter(env, model, device=device)
    assert isinstance(model, LocalStudentPolicy)
    return StudentPolicyAdapter(model, device=device)


def run_phase8_campaign(
    config: Phase8Config,
    *,
    phase7_checkpoint_dir: Path,
    output_checkpoint_dir: Path | None = None,
    device: torch.device | str = "cpu",
    resume: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Train only optimized Students and compare on identical held-out seeds."""

    device = torch.device(device)
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for training_seed in config.training_seeds:
        stages = phase8_curriculum(training_seed)
        max_nodes = stages[0].config.max_nodes
        teacher, phase7_student = _load_phase7_models(
            phase7_checkpoint_dir / f"seed_{training_seed}",
            max_nodes=max_nodes,
            hidden_dim=config.hidden_dim,
            device=device,
        )
        seed_everything(training_seed + 795_000)
        untrained_geo = GeographicResidualStudentPolicy(
            max_nodes,
            hidden_dim=config.hidden_dim,
            initial_prior_strength=config.initial_prior_strength,
        )
        optimized_path = (
            output_checkpoint_dir
            / f"seed_{training_seed}"
            / "geo_residual_kd.pt"
            if output_checkpoint_dir is not None
            else None
        )
        metrics_path = (
            optimized_path.parent / "training_metrics.json"
            if optimized_path is not None
            else None
        )
        if (
            resume
            and optimized_path is not None
            and optimized_path.is_file()
            and metrics_path is not None
            and metrics_path.is_file()
        ):
            optimized = GeographicResidualStudentPolicy(
                max_nodes,
                hidden_dim=config.hidden_dim,
                initial_prior_strength=config.initial_prior_strength,
            )
            load_checkpoint(optimized_path, optimized, map_location=device)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            optimized, metrics = _train_optimized_student(
                config,
                seed=training_seed,
                teacher=teacher,
                device=device,
            )
            if optimized_path is not None and metrics_path is not None:
                save_checkpoint(
                    optimized_path,
                    optimized,
                    metadata={
                        "phase": 8,
                        "training_seed": training_seed,
                        "method": "Geo-Residual KD",
                    },
                )
                metrics_path.write_text(
                    json.dumps(metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        training_rows.append(metrics)
        models: dict[str, torch.nn.Module | None] = {
            "GPSR": None,
            "Phase 7 KD-only": phase7_student,
            "Untrained Geo-Residual": untrained_geo,
            "Geo-Residual KD": optimized,
            "Global Teacher": teacher,
            "Shortest-path Oracle": None,
        }
        for scenario_index, scenario in enumerate(
            phase8_evaluation_scenarios(training_seed)
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
