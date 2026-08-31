"""Leave-one-component-out ablation for SwitchGLOBE novelty claims."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..env.fanet_env import FanetRoutingEnv
from ..evaluation import episode_row, evaluate_policy_results, generalization_summary
from ..models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    LocalStudentPolicy,
    SwitchGlobePolicy,
)
from ..models.policy_adapter import StudentPolicyAdapter
from ..scenarios import phase9_evaluation_scenarios
from ..utils import load_checkpoint, seed_everything
from .ablation_campaign import _load_phase8
from .external_comparison_campaign import load_switchglobe


FULL = "Full SwitchGLOBE"
NO_RISK_SWITCH = "w/o Risk-Switch"
NO_GEO_RESIDUAL = "w/o Geo-Residual"
NO_DISTILLATION = "w/o Distillation"

NOVELTY_ABLATION_METHODS = (
    FULL,
    NO_RISK_SWITCH,
    NO_GEO_RESIDUAL,
    NO_DISTILLATION,
)


@dataclass(frozen=True)
class NoveltyAblationConfig:
    training_seeds: tuple[int, ...] = (42, 77, 123, 314, 2718)
    evaluation_episodes: int = 200
    hidden_dim: int = 64


def _load_phase7_student(
    *,
    seed: int,
    checkpoint_dir: Path,
    filename: str,
    max_nodes: int,
    hidden_dim: int,
    device: torch.device,
) -> LocalStudentPolicy:
    model = LocalStudentPolicy(max_nodes, hidden_dim=hidden_dim)
    load_checkpoint(
        checkpoint_dir / f"seed_{seed}" / filename,
        model,
        map_location=device,
    )
    return model.to(device).eval()


def _switch_parameters(model: SwitchGlobePolicy) -> dict[str, float]:
    return {
        "switch_threshold": float(model.switch_threshold.item()),
        "margin_gate": float(model.margin_gate.item()),
        "lifetime_gate": float(model.lifetime_gate.item()),
        "onward_gate": float(model.onward_gate.item()),
    }


def _assert_phase8_matches_full(
    phase8: GeographicResidualStudentPolicy,
    full: SwitchGlobePolicy,
) -> None:
    phase8_state = phase8.state_dict()
    full_state = full.normal_policy.state_dict()
    if tuple(phase8_state) != tuple(full_state) or any(
        not torch.equal(phase8_state[key], full_state[key])
        for key in phase8_state
    ):
        raise ValueError(
            "Phase 8 checkpoint does not match the full policy normal branch"
        )


def build_novelty_ablation_policies(
    config: NoveltyAblationConfig,
    *,
    seed: int,
    max_nodes: int,
    phase7_checkpoint_dir: Path,
    phase8_checkpoint_dir: Path,
    switchglobe_checkpoint_dir: Path,
    device: torch.device,
) -> dict[str, StudentPolicyAdapter]:
    """Build causal leave-one-out variants from seed-matched checkpoints."""

    full_adapter = load_switchglobe(
        switchglobe_checkpoint_dir,
        seed=seed,
        max_nodes=max_nodes,
        hidden_dim=config.hidden_dim,
        device=device,
    )
    full = full_adapter.model
    if not isinstance(full, SwitchGlobePolicy):
        raise TypeError("expected a SwitchGlobePolicy checkpoint")
    phase8 = _load_phase8(
        seed=seed,
        checkpoint_dir=phase8_checkpoint_dir,
        max_nodes=max_nodes,
        hidden_dim=config.hidden_dim,
        device=device,
    )
    _assert_phase8_matches_full(phase8, full)

    kd_only = _load_phase7_student(
        seed=seed,
        checkpoint_dir=phase7_checkpoint_dir,
        filename="kd_only_student.pt",
        max_nodes=max_nodes,
        hidden_dim=config.hidden_dim,
        device=device,
    )
    no_geo = SwitchGlobePolicy(
        kd_only,
        deepcopy(full.predictive_policy),
        **_switch_parameters(full),
    ).to(device).eval()

    seed_everything(seed + 1_700_000)
    analytic_normal = GeographicResidualStudentPolicy(
        max_nodes,
        hidden_dim=config.hidden_dim,
    )
    analytic_predictive = LiteGlobePStudentPolicy(
        max_nodes,
        hidden_dim=config.hidden_dim,
    )
    no_distillation = SwitchGlobePolicy(
        analytic_normal,
        analytic_predictive,
        **_switch_parameters(full),
    ).to(device).eval()

    return {
        FULL: full_adapter,
        NO_RISK_SWITCH: StudentPolicyAdapter(
            phase8,
            device=device,
            force_forward_if_available=True,
        ),
        NO_GEO_RESIDUAL: StudentPolicyAdapter(
            no_geo,
            device=device,
            force_forward_if_available=True,
        ),
        NO_DISTILLATION: StudentPolicyAdapter(
            no_distillation,
            device=device,
            force_forward_if_available=True,
        ),
    }


def run_novelty_ablation_campaign(
    config: NoveltyAblationConfig,
    *,
    phase7_checkpoint_dir: Path,
    phase8_checkpoint_dir: Path,
    switchglobe_checkpoint_dir: Path,
    device: torch.device | str = "cpu",
) -> dict[str, list[dict[str, Any]]]:
    """Evaluate every variant on identical scenarios and episode seeds."""

    if not config.training_seeds:
        raise ValueError("at least one training seed is required")
    if len(config.training_seeds) != len(set(config.training_seeds)):
        raise ValueError("training seeds must be unique")
    if config.evaluation_episodes <= 0:
        raise ValueError("evaluation_episodes must be positive")
    device = torch.device(device)
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for seed in config.training_seeds:
        scenarios = phase9_evaluation_scenarios(seed)
        max_nodes = scenarios[0].config.max_nodes
        policies = build_novelty_ablation_policies(
            config,
            seed=seed,
            max_nodes=max_nodes,
            phase7_checkpoint_dir=phase7_checkpoint_dir,
            phase8_checkpoint_dir=phase8_checkpoint_dir,
            switchglobe_checkpoint_dir=switchglobe_checkpoint_dir,
            device=device,
        )
        for scenario_index, scenario in enumerate(scenarios):
            evaluation_seeds = list(
                range(
                    1_100_000 + scenario_index * 10_000,
                    1_100_000
                    + scenario_index * 10_000
                    + config.evaluation_episodes,
                )
            )
            env = FanetRoutingEnv(scenario.config)
            for method in NOVELTY_ABLATION_METHODS:
                results = evaluate_policy_results(
                    env,
                    policies[method],
                    evaluation_seeds,
                    reset_options=scenario.reset_options,
                )
                episode_rows.extend(
                    episode_row(
                        result,
                        method=method,
                        scenario=scenario.name,
                        training_seed=seed,
                    )
                    for result in results
                )
                summary_rows.append(
                    generalization_summary(
                        results,
                        method=method,
                        scenario=scenario.name,
                        training_seed=seed,
                    )
                )
    return {"episodes": episode_rows, "seed_summaries": summary_rows}
