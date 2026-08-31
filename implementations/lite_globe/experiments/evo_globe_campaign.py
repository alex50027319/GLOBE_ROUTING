"""Gated Evo-inspired, deployment-neutral SwitchGLOBE experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from ..algorithms import DistillationConfig, train_student_distillation
from ..data import (
    concatenate_datasets,
    generate_return_guided_dataset,
    split_by_episode_group,
)
from ..env.fanet_env import FanetRoutingEnv
from ..evaluation import evaluate_policy_results
from ..models import SwitchGlobePolicy
from ..models.policy_adapter import StudentPolicyAdapter
from ..scenarios import (
    phase9_compositional_predictive_training_scenarios,
    phase9_curriculum,
    phase9_hole_training_scenarios,
    phase9_predictive_training_scenarios,
)


@dataclass(frozen=True)
class CostToGoDistillationConfig:
    dataset_episodes_per_scenario: int = 9
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    return_discount: float = 0.85
    return_action_coefficient: float = 0.10
    return_weight_temperature: float = 5.0
    early_stopping_patience: int = 3

    def validate(self) -> None:
        if self.dataset_episodes_per_scenario < 9:
            raise ValueError(
                "dataset_episodes_per_scenario must be at least 9"
            )
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid")
        if not 0.0 <= self.return_discount <= 1.0:
            raise ValueError("return_discount must be in [0, 1]")
        if self.return_action_coefficient < 0:
            raise ValueError("return_action_coefficient must be non-negative")
        if self.return_weight_temperature <= 0:
            raise ValueError("return_weight_temperature must be positive")


@dataclass(frozen=True)
class CompositionalCurriculumConfig:
    """Fine-tune only predictive-prior scalars on compound disruptions."""

    dataset_episodes_per_scenario: int = 30
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-2
    weight_decay: float = 1e-5
    risk_oracle_coefficient: float = 1.0
    link_loss_scale: float = 1.0
    early_stopping_patience: int = 5

    def validate(self) -> None:
        if self.dataset_episodes_per_scenario < 9:
            raise ValueError(
                "dataset_episodes_per_scenario must be at least 9"
            )
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer settings are invalid")
        if self.risk_oracle_coefficient <= 0:
            raise ValueError("risk_oracle_coefficient must be positive")
        if not 0.0 < self.link_loss_scale <= 1.5:
            raise ValueError("link_loss_scale must be in (0, 1.5]")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")


def cost_to_go_training_scenarios(seed: int):
    """Use the existing training distributions without evaluation leakage."""

    return [
        *phase9_curriculum(seed),
        *phase9_hole_training_scenarios(seed),
        *phase9_predictive_training_scenarios(seed),
    ]


def collect_cost_to_go_dataset(
    reference: SwitchGlobePolicy,
    config: CostToGoDistillationConfig,
    *,
    seed: int,
    device: torch.device | str = "cpu",
):
    """Collect reference/oracle/risk-oracle rollouts with return targets."""

    config.validate()
    datasets = []
    per_policy = max(3, config.dataset_episodes_per_scenario // 3)
    for scenario_index, scenario in enumerate(
        cost_to_go_training_scenarios(seed)
    ):
        start = seed + 1_510_000 + scenario_index * 30_000
        for rollout_index, rollout_policy in enumerate(
            ("reference", "oracle", "risk_oracle")
        ):
            rollout_start = start + rollout_index * per_policy
            datasets.append(
                generate_return_guided_dataset(
                    FanetRoutingEnv(scenario.config),
                    reference,
                    episode_seeds=list(
                        range(rollout_start, rollout_start + per_policy)
                    ),
                    scenario_id=(
                        f"{scenario.name}_{rollout_policy}_rollout"
                    ),
                    reset_options=scenario.reset_options,
                    rollout_policy=rollout_policy,
                    return_discount=config.return_discount,
                    device=device,
                )
            )
    return concatenate_datasets(datasets)


def train_cost_to_go_switchglobe(
    reference: SwitchGlobePolicy,
    config: CostToGoDistillationConfig,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[SwitchGlobePolicy, dict[str, Any]]:
    """Fine-tune an exact-architecture clone with multi-step return targets."""

    config.validate()
    device = torch.device(device)
    torch.manual_seed(seed + 1_500_000)
    np.random.seed(seed + 1_500_000)
    reference = reference.to(device).eval()
    dataset = collect_cost_to_go_dataset(
        reference,
        config,
        seed=seed,
        device=device,
    )
    split = split_by_episode_group(dataset, seed=seed + 1_520_000)
    candidate = deepcopy(reference).to(device)
    state_keys_before = tuple(candidate.state_dict())
    parameter_count_before = sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    result = train_student_distillation(
        candidate,
        split.train,
        split.validation,
        config=DistillationConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            return_action_coefficient=config.return_action_coefficient,
            return_weight_temperature=config.return_weight_temperature,
            early_stopping_patience=config.early_stopping_patience,
        ),
        seed=seed + 1_530_000,
        device=device,
    )
    state_keys_after = tuple(candidate.state_dict())
    parameter_count_after = sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    if state_keys_after != state_keys_before:
        raise RuntimeError("cost-to-go training changed deployment state keys")
    if parameter_count_after != parameter_count_before:
        raise RuntimeError("cost-to-go training changed deployment parameters")
    metrics = {
        "training_seed": seed,
        "dataset_samples": len(dataset),
        "train_samples": len(split.train),
        "validation_samples": len(split.validation),
        "test_samples": len(split.test),
        "completed_epochs": result.epochs,
        "validation_kl": result.validation.kl,
        "validation_reference_agreement": (
            result.validation.action_agreement
        ),
        "validation_rollout_agreement": (
            result.validation.rollout_action_agreement
        ),
        "validation_mean_discounted_return": (
            result.validation.mean_discounted_return
        ),
        "parameter_count": parameter_count_after,
        "config": asdict(config),
    }
    return candidate.eval(), metrics


def collect_compositional_curriculum_dataset(
    reference: SwitchGlobePolicy,
    config: CompositionalCurriculumConfig,
    *,
    seed: int,
    device: torch.device | str = "cpu",
):
    """Collect disjoint-angle compound-disruption trajectories."""

    config.validate()
    datasets = []
    per_policy = max(3, config.dataset_episodes_per_scenario // 3)
    scenarios = phase9_compositional_predictive_training_scenarios(
        seed,
        link_loss_scale=config.link_loss_scale,
    )
    for scenario_index, scenario in enumerate(scenarios):
        start = seed + 1_610_000 + scenario_index * 30_000
        for rollout_index, rollout_policy in enumerate(
            ("reference", "oracle", "risk_oracle")
        ):
            rollout_start = start + rollout_index * per_policy
            datasets.append(
                generate_return_guided_dataset(
                    FanetRoutingEnv(scenario.config),
                    reference.predictive_policy,
                    episode_seeds=list(
                        range(rollout_start, rollout_start + per_policy)
                    ),
                    scenario_id=f"{scenario.name}_{rollout_policy}_rollout",
                    reset_options=scenario.reset_options,
                    rollout_policy=rollout_policy,
                    device=device,
                )
            )
    return concatenate_datasets(datasets)


def _predictive_prior_values(model: SwitchGlobePolicy) -> dict[str, Any]:
    predictive = model.predictive_policy
    return {
        "predictive_strength": torch.nn.functional.softplus(
            predictive.log_predictive_strength.detach().cpu()
        ).tolist(),
        "break_penalty": float(
            torch.nn.functional.softplus(
                predictive.log_break_penalty.detach().cpu()
            ).item()
        ),
    }


def train_compositional_switchglobe(
    reference: SwitchGlobePolicy,
    config: CompositionalCurriculumConfig,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> tuple[SwitchGlobePolicy, dict[str, Any]]:
    """Tune five predictive-prior scalars without changing inference shape."""

    config.validate()
    device = torch.device(device)
    torch.manual_seed(seed + 1_600_000)
    np.random.seed(seed + 1_600_000)
    reference = reference.to(device).eval()
    dataset = collect_compositional_curriculum_dataset(
        reference,
        config,
        seed=seed,
        device=device,
    )
    split = split_by_episode_group(dataset, seed=seed + 1_620_000)
    candidate = deepcopy(reference).to(device)
    state_before = {
        key: value.detach().cpu().clone()
        for key, value in candidate.state_dict().items()
    }
    parameters_before = sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    prior_before = _predictive_prior_values(candidate)
    trainable_names = {
        "log_predictive_strength",
        "log_break_penalty",
    }
    original_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in candidate.predictive_policy.named_parameters()
    }
    for name, parameter in candidate.predictive_policy.named_parameters():
        parameter.requires_grad_(name in trainable_names)
    result = train_student_distillation(
        candidate.predictive_policy,
        split.train,
        split.validation,
        config=DistillationConfig(
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            risk_oracle_coefficient=config.risk_oracle_coefficient,
            early_stopping_patience=config.early_stopping_patience,
        ),
        seed=seed + 1_630_000,
        device=device,
    )
    for name, parameter in candidate.predictive_policy.named_parameters():
        parameter.requires_grad_(original_requires_grad[name])
    candidate.predictive_policy.set_residual_weight(0.0)
    state_after = candidate.state_dict()
    allowed_changes = {
        "predictive_policy.log_predictive_strength",
        "predictive_policy.log_break_penalty",
    }
    changed = {
        key
        for key, before in state_before.items()
        if not torch.equal(before, state_after[key].detach().cpu())
    }
    if not changed.issubset(allowed_changes):
        raise RuntimeError(
            f"compositional training changed protected state: "
            f"{sorted(changed - allowed_changes)}"
        )
    parameters_after = sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    if tuple(state_after) != tuple(state_before):
        raise RuntimeError("compositional training changed deployment state keys")
    if parameters_after != parameters_before:
        raise RuntimeError("compositional training changed deployment parameters")
    metrics = {
        "training_seed": seed,
        "dataset_samples": len(dataset),
        "train_samples": len(split.train),
        "validation_samples": len(split.validation),
        "test_samples": len(split.test),
        "completed_epochs": result.epochs,
        "validation_kl": result.validation.kl,
        "validation_reference_agreement": result.validation.action_agreement,
        "validation_risk_oracle_agreement": (
            result.validation.risk_oracle_action_agreement
        ),
        "changed_state_keys": sorted(changed),
        "parameter_count": parameters_after,
        "prior_before": prior_before,
        "prior_after": _predictive_prior_values(candidate),
        "config": asdict(config),
    }
    return candidate.eval(), metrics


def evaluate_cost_to_go_candidate(
    model: SwitchGlobePolicy,
    scenarios,
    *,
    episode_seed_base: int,
    episodes_per_scenario: int,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Aggregate a frozen candidate on paired scenario episode seeds."""

    delivered = connected = deadline = episodes = 0
    total_delivered = 0
    energy = 0.0
    scenario_rows = []
    adapter = StudentPolicyAdapter(
        model,
        device=device,
        force_forward_if_available=True,
    )
    for scenario_index, scenario in enumerate(scenarios):
        seeds = list(
            range(
                episode_seed_base + scenario_index * 10_000,
                episode_seed_base
                + scenario_index * 10_000
                + episodes_per_scenario,
            )
        )
        results = evaluate_policy_results(
            FanetRoutingEnv(scenario.config),
            adapter,
            seeds,
            reset_options=scenario.reset_options,
        )
        connected_results = [
            result for result in results if result.initially_connected
        ]
        scenario_delivered = sum(
            result.delivered for result in connected_results
        )
        scenario_deadline = sum(result.deadline_met for result in results)
        scenario_energy = sum(
            result.transmission_energy_proxy for result in results
        )
        scenario_total_delivered = sum(
            result.delivered for result in results
        )
        delivered += scenario_delivered
        total_delivered += scenario_total_delivered
        connected += len(connected_results)
        deadline += scenario_deadline
        episodes += len(results)
        energy += scenario_energy
        scenario_rows.append(
            {
                "scenario": scenario.name,
                "connected_pair_pdr": (
                    scenario_delivered / max(len(connected_results), 1)
                ),
                "deadline_delivery_ratio": (
                    scenario_deadline / len(results)
                ),
                "energy_per_delivered_packet": (
                    scenario_energy / max(scenario_total_delivered, 1)
                ),
            }
        )
    return {
        "connected_pair_pdr": delivered / connected,
        "deadline_delivery_ratio": deadline / episodes,
        "energy_per_delivered_packet": energy / max(total_delivered, 1),
        "scenario_rows": scenario_rows,
    }
