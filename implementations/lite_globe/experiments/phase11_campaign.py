"""Phase 11: Lite-GLOBE-P predictive residual optimization."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..algorithms import DistillationConfig, train_student_distillation
from ..baselines import (
    GpsrPolicy,
    PredictiveGeographicPolicy,
    RiskAwareOraclePolicy,
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
    GeographicResidualStudentPolicy,
    GlobalTeacherActorCritic,
    LiteGlobePStudentPolicy,
)
from ..models.policy_adapter import StudentPolicyAdapter
from ..models.teacher_adapter import TeacherPolicyAdapter
from ..scenarios import (
    phase9_curriculum,
    phase9_evaluation_scenarios,
    phase9_hole_calibration_scenarios,
    phase9_hole_training_scenarios,
    phase9_predictive_calibration_scenarios,
    phase9_predictive_training_scenarios,
)
from ..utils import load_checkpoint, save_checkpoint, seed_everything


@dataclass(frozen=True)
class Phase11Config:
    training_seeds: tuple[int, ...]
    evaluation_episodes: int
    hidden_dim: int
    dataset_episodes_per_stage: int
    distillation_epochs: int
    oracle_coefficient: float
    risk_oracle_coefficient: float
    teacher_action_coefficient: float
    early_stopping_patience: int
    initial_prior_strength: float
    initial_predictive_strength: tuple[float, float, float, float]
    initial_break_penalty: float
    initial_residual_bound: float
    calibration_episodes_per_stage: int
    structural_hole_episodes_per_variant: int
    calibration_pdr_tolerance: float
    predictive_replay_multiplier: int
    predictive_pretraining_epochs: int


PHASE11_METHODS = (
    "GPSR",
    "Predictive Geographic",
    "Phase 8 Geo-Residual KD",
    "Lite-GLOBE-P no-predictive ablation",
    "Lite-GLOBE-P predictive-prior only",
    "Lite-GLOBE-P",
    "Global Teacher",
    "Shortest-path Oracle",
    "Risk-aware Oracle",
)


def _load_source_models(
    *,
    training_seed: int,
    phase7_checkpoint_dir: Path,
    phase8_checkpoint_dir: Path,
    max_nodes: int,
    hidden_dim: int,
    device: torch.device,
) -> tuple[GlobalTeacherActorCritic, GeographicResidualStudentPolicy]:
    teacher = GlobalTeacherActorCritic(max_nodes, hidden_dim=hidden_dim)
    load_checkpoint(
        phase7_checkpoint_dir
        / f"seed_{training_seed}"
        / "global_teacher.pt",
        teacher,
        map_location=device,
    )
    phase8 = GeographicResidualStudentPolicy(
        max_nodes,
        hidden_dim=hidden_dim,
    )
    load_checkpoint(
        phase8_checkpoint_dir
        / f"seed_{training_seed}"
        / "geo_residual_kd.pt",
        phase8,
        map_location=device,
    )
    return teacher, phase8


def _initialize_from_phase8(
    phase8: GeographicResidualStudentPolicy,
    *,
    max_nodes: int,
    hidden_dim: int,
    config: Phase11Config,
) -> LiteGlobePStudentPolicy:
    model = LiteGlobePStudentPolicy(
        max_nodes,
        hidden_dim=hidden_dim,
        initial_prior_strength=config.initial_prior_strength,
        initial_predictive_strength=config.initial_predictive_strength,
        initial_break_penalty=config.initial_break_penalty,
        initial_residual_bound=config.initial_residual_bound,
    )
    incompatible = model.load_state_dict(phase8.state_dict(), strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"unexpected Phase 8 keys: {incompatible.unexpected_keys}"
        )
    allowed_missing = {
        "log_predictive_strength",
        "log_break_penalty",
        "log_residual_bound",
        "predictive_weight",
        "lifetime_gate",
        "onward_gate",
        "margin_gate",
    }
    if set(incompatible.missing_keys) != allowed_missing:
        raise ValueError(
            f"unexpected Lite-GLOBE-P missing keys: "
            f"{incompatible.missing_keys}"
        )
    return model


def _collect_phase11_dataset(
    config: Phase11Config,
    *,
    seed: int,
    teacher: GlobalTeacherActorCritic,
    device: torch.device,
):
    datasets = []
    base_scenarios = [
        *phase9_curriculum(seed),
        *phase9_hole_training_scenarios(seed),
    ]
    predictive_scenarios = phase9_predictive_training_scenarios(seed)
    scenarios = [
        *base_scenarios,
        *(
            predictive_scenarios
            * max(1, config.predictive_replay_multiplier)
        ),
    ]
    for index, scenario in enumerate(scenarios):
        count = (
            config.structural_hole_episodes_per_variant
            if "structural_hole" in scenario.name
            else config.dataset_episodes_per_stage
        )
        per_policy = max(3, count // 3)
        start = seed + 1_210_000 + index * 30_000
        for rollout_index, rollout_policy in enumerate(
            ("teacher", "oracle", "risk_oracle")
        ):
            rollout_start = start + rollout_index * per_policy
            datasets.append(
                generate_teacher_dataset(
                    FanetRoutingEnv(scenario.config),
                    teacher,
                    episode_seeds=list(
                        range(rollout_start, rollout_start + per_policy)
                    ),
                    scenario_id=(
                        f"{scenario.name}_{rollout_policy}_rollout"
                    ),
                    reset_options=scenario.reset_options,
                    rollout_policy=rollout_policy,
                    device=device,
                )
            )
    return concatenate_datasets(datasets)


def _pretrain_predictive_prior(
    model: LiteGlobePStudentPolicy,
    dataset,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> None:
    if epochs <= 0:
        raise ValueError("predictive_pretraining_epochs must be positive")
    required = {"candidate_risk_features", "risk_oracle_actions"}
    if not required.issubset(dataset.arrays):
        raise ValueError("predictive pretraining requires risk targets")
    model.to(device)
    original_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in (
        model.log_predictive_strength,
        model.log_break_penalty,
        model.log_residual_bound,
    ):
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [
            model.log_predictive_strength,
            model.log_break_penalty,
            model.log_residual_bound,
        ],
        lr=2e-2,
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
                    "candidate_risk_features",
                }
            }
            output = model(observation)
            loss = torch.nn.functional.cross_entropy(
                output.masked_logits,
                batch["risk_oracle_actions"].to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_requires_grad[name])


def _measure_candidate(
    model: LiteGlobePStudentPolicy,
    scenarios,
    *,
    seed: int,
    episodes_per_stage: int,
    device: torch.device,
) -> dict[str, float]:
    delivered = 0
    deadline_met = 0
    episodes = 0
    delay = 0.0
    energy = 0.0
    queue_delay = 0.0
    for index, scenario in enumerate(scenarios):
        start = seed + 1_270_000 + index * 10_000
        results = evaluate_policy_results(
            FanetRoutingEnv(scenario.config),
            StudentPolicyAdapter(
                model,
                device=device,
                force_forward_if_available=True,
            ),
            list(range(start, start + episodes_per_stage)),
            reset_options=scenario.reset_options,
        )
        delivered += sum(result.delivered for result in results)
        deadline_met += sum(result.deadline_met for result in results)
        episodes += len(results)
        delay += sum(
            result.steps for result in results if result.delivered
        )
        energy += sum(
            result.transmission_energy_proxy for result in results
        )
        queue_delay += sum(
            result.cumulative_queue_delay_proxy for result in results
        )
    return {
        "pdr": delivered / episodes,
        "deadline_delivery_ratio": deadline_met / episodes,
        "mean_success_delay": delay / max(delivered, 1),
        "mean_energy": energy / episodes,
        "mean_queue_delay": queue_delay / episodes,
    }


def _calibrate_phase11_weights(
    model: LiteGlobePStudentPolicy,
    *,
    seed: int,
    episodes_per_stage: int,
    pdr_tolerance: float,
    device: torch.device,
) -> dict[str, float]:
    generic = phase9_curriculum(seed)
    holes = phase9_hole_calibration_scenarios(seed)
    predictive = phase9_predictive_calibration_scenarios(seed)
    original_residual = float(model.residual_weight.item())
    original_predictive = float(model.predictive_weight.item())
    residual_candidates = sorted({0.0, 0.25, 0.5, 0.75, original_residual})
    predictive_candidates = sorted({0.5, 0.75, 1.0, original_predictive})
    calibration_seed = seed + 1_000
    model.set_predictive_weight(0.0)
    baseline = _measure_candidate(
        model,
        generic,
        seed=calibration_seed,
        episodes_per_stage=episodes_per_stage,
        device=device,
    )
    measurements = []
    for residual_weight in residual_candidates:
        for predictive_weight in predictive_candidates:
            model.set_residual_weight(residual_weight)
            model.set_predictive_weight(predictive_weight)
            generic_result = _measure_candidate(
                model,
                generic,
                seed=calibration_seed,
                episodes_per_stage=episodes_per_stage,
                device=device,
            )
            hole_result = _measure_candidate(
                model,
                holes,
                seed=calibration_seed + 1_000,
                episodes_per_stage=episodes_per_stage,
                device=device,
            )
            predictive_result = _measure_candidate(
                model,
                predictive,
                seed=calibration_seed + 2_000,
                episodes_per_stage=episodes_per_stage,
                device=device,
            )
            measurements.append(
                {
                    "residual_weight": residual_weight,
                    "predictive_weight": predictive_weight,
                    "generic_pdr": generic_result["pdr"],
                    "hole_pdr": hole_result["pdr"],
                    "predictive_pdr": predictive_result["pdr"],
                    "deadline_delivery_ratio": generic_result[
                        "deadline_delivery_ratio"
                    ],
                    "mean_success_delay": generic_result[
                        "mean_success_delay"
                    ],
                    "mean_energy": generic_result["mean_energy"],
                    "mean_queue_delay": generic_result[
                        "mean_queue_delay"
                    ],
                }
            )
    feasible = [
        item
        for item in measurements
        if item["generic_pdr"] >= baseline["pdr"] - pdr_tolerance
    ]
    best = max(
        feasible,
        key=lambda item: (
            item["predictive_pdr"],
            item["hole_pdr"],
            item["generic_pdr"],
            item["deadline_delivery_ratio"],
            -item["mean_success_delay"],
            -item["mean_energy"],
            -item["mean_queue_delay"],
            -item["residual_weight"],
        ),
    )
    model.set_residual_weight(best["residual_weight"])
    model.set_predictive_weight(best["predictive_weight"])
    return {
        **best,
        "phase8_like_generic_pdr": baseline["pdr"],
    }


def _train_phase11_student(
    config: Phase11Config,
    *,
    seed: int,
    teacher: GlobalTeacherActorCritic,
    phase8: GeographicResidualStudentPolicy,
    device: torch.device,
) -> tuple[LiteGlobePStudentPolicy, dict[str, Any]]:
    seed_everything(seed + 1_200_000)
    max_nodes = phase9_curriculum(seed)[0].config.max_nodes
    model = _initialize_from_phase8(
        phase8,
        max_nodes=max_nodes,
        hidden_dim=config.hidden_dim,
        config=config,
    )
    dataset = _collect_phase11_dataset(
        config,
        seed=seed,
        teacher=teacher,
        device=device,
    )
    split = split_by_episode_group(dataset, seed=seed + 1_230_000)
    _pretrain_predictive_prior(
        model,
        split.train,
        epochs=config.predictive_pretraining_epochs,
        seed=seed + 1_240_000,
        device=device,
    )
    result = train_student_distillation(
        model,
        split.train,
        split.validation,
        config=DistillationConfig(
            epochs=config.distillation_epochs,
            learning_rate=3e-4,
            batch_size=128,
            weight_decay=1e-5,
            oracle_coefficient=config.oracle_coefficient,
            risk_oracle_coefficient=config.risk_oracle_coefficient,
            teacher_action_coefficient=config.teacher_action_coefficient,
            early_stopping_patience=config.early_stopping_patience,
        ),
        seed=seed + 1_250_000,
        device=device,
    )
    calibration = _calibrate_phase11_weights(
        model,
        seed=seed,
        episodes_per_stage=config.calibration_episodes_per_stage,
        pdr_tolerance=config.calibration_pdr_tolerance,
        device=device,
    )
    strengths = torch.nn.functional.softplus(
        model.log_predictive_strength.detach().cpu()
    )
    return model, {
        "training_seed": seed,
        "dataset_samples": len(dataset),
        "completed_epochs": result.epochs,
        "validation_kl": result.validation.kl,
        "validation_teacher_agreement": (
            result.validation.action_agreement
        ),
        "validation_shortest_oracle_agreement": (
            result.validation.oracle_action_agreement
        ),
        "validation_risk_oracle_agreement": (
            result.validation.risk_oracle_action_agreement
        ),
        "predictive_margin_strength": float(strengths[0].item()),
        "predictive_lifetime_strength": float(strengths[1].item()),
        "predictive_queue_headroom_strength": float(strengths[2].item()),
        "predictive_onward_lifetime_strength": float(strengths[3].item()),
        "break_penalty": float(
            torch.nn.functional.softplus(
                model.log_break_penalty.detach().cpu()
            ).item()
        ),
        "residual_bound": float(
            torch.nn.functional.softplus(
                model.log_residual_bound.detach().cpu()
            ).item()
        ),
        **calibration,
    }


def _phase11_ablations(
    model: LiteGlobePStudentPolicy,
) -> dict[str, LiteGlobePStudentPolicy]:
    no_predictive = copy.deepcopy(model)
    no_predictive.set_predictive_weight(0.0)
    predictive_only = copy.deepcopy(model)
    predictive_only.set_residual_weight(0.0)
    return {
        "Lite-GLOBE-P no-predictive ablation": no_predictive,
        "Lite-GLOBE-P predictive-prior only": predictive_only,
    }


def _policy(
    method: str,
    model: torch.nn.Module | None,
    env: FanetRoutingEnv,
    device: torch.device,
):
    if method == "GPSR":
        return GpsrPolicy(env.drop_action)
    if method == "Predictive Geographic":
        return PredictiveGeographicPolicy(env.drop_action)
    if method == "Shortest-path Oracle":
        return ShortestPathOraclePolicy(env)
    if method == "Risk-aware Oracle":
        return RiskAwareOraclePolicy(env)
    if method == "Global Teacher":
        assert isinstance(model, GlobalTeacherActorCritic)
        return TeacherPolicyAdapter(env, model, device=device)
    assert isinstance(
        model,
        (GeographicResidualStudentPolicy, LiteGlobePStudentPolicy),
    )
    return StudentPolicyAdapter(
        model,
        device=device,
        force_forward_if_available=True,
    )


def run_phase11_campaign(
    config: Phase11Config,
    *,
    phase7_checkpoint_dir: Path,
    phase8_checkpoint_dir: Path,
    output_checkpoint_dir: Path | None = None,
    device: torch.device | str = "cpu",
    resume: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Train Lite-GLOBE-P and evaluate it on external baseline scenario axes."""

    device = torch.device(device)
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for training_seed in config.training_seeds:
        max_nodes = phase9_curriculum(training_seed)[0].config.max_nodes
        teacher, phase8 = _load_source_models(
            training_seed=training_seed,
            phase7_checkpoint_dir=phase7_checkpoint_dir,
            phase8_checkpoint_dir=phase8_checkpoint_dir,
            max_nodes=max_nodes,
            hidden_dim=config.hidden_dim,
            device=device,
        )
        checkpoint = (
            output_checkpoint_dir
            / f"seed_{training_seed}"
            / "lite_globe_p.pt"
            if output_checkpoint_dir is not None
            else None
        )
        metrics_path = (
            checkpoint.parent / "training_metrics.json"
            if checkpoint is not None
            else None
        )
        if (
            resume
            and checkpoint is not None
            and checkpoint.is_file()
            and metrics_path is not None
            and metrics_path.is_file()
        ):
            optimized = LiteGlobePStudentPolicy(
                max_nodes,
                hidden_dim=config.hidden_dim,
                initial_prior_strength=config.initial_prior_strength,
                initial_predictive_strength=(
                    config.initial_predictive_strength
                ),
                initial_break_penalty=config.initial_break_penalty,
                initial_residual_bound=config.initial_residual_bound,
            )
            load_checkpoint(checkpoint, optimized, map_location=device)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            optimized, metrics = _train_phase11_student(
                config,
                seed=training_seed,
                teacher=teacher,
                phase8=phase8,
                device=device,
            )
            if checkpoint is not None and metrics_path is not None:
                save_checkpoint(
                    checkpoint,
                    optimized,
                    metadata={
                        "phase": 11,
                        "training_seed": training_seed,
                        "method": "Lite-GLOBE-P",
                    },
                )
                metrics_path.write_text(
                    json.dumps(metrics, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        training_rows.append(metrics)
        models: dict[str, torch.nn.Module | None] = {
            "GPSR": None,
            "Predictive Geographic": None,
            "Phase 8 Geo-Residual KD": phase8,
            **_phase11_ablations(optimized),
            "Lite-GLOBE-P": optimized,
            "Global Teacher": teacher,
            "Shortest-path Oracle": None,
            "Risk-aware Oracle": None,
        }
        for scenario_index, scenario in enumerate(
            phase9_evaluation_scenarios(training_seed)
        ):
            evaluation_seeds = list(
                range(
                    1_100_000 + scenario_index * 10_000,
                    1_100_000
                    + scenario_index * 10_000
                    + config.evaluation_episodes,
                )
            )
            env = FanetRoutingEnv(scenario.config)
            for method in PHASE11_METHODS:
                results = evaluate_policy_results(
                    env,
                    _policy(method, models[method], env, device),
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
