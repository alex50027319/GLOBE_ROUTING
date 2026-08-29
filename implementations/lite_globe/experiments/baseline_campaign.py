"""Train and evaluate external routing baselines against SwitchGLOBE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch

from ..baselines import (
    DramaPolicy,
    EvoQGeoPolicy,
    GpsrPolicy,
    IqmrPolicy,
    PredictiveGeographicPolicy,
)
from ..baselines.external_rl import (
    train_drama_baseline,
    train_value_baseline,
)
from ..env.config import FanetConfig
from ..env.fanet_env import FanetRoutingEnv
from ..evaluation import (
    episode_row,
    evaluate_policy_results,
    generalization_summary,
)
from ..models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from ..models.policy_adapter import StudentPolicyAdapter
from ..scenarios import (
    phase9_curriculum,
    phase9_evaluation_scenarios,
    phase9_hole_training_scenarios,
    phase9_predictive_training_scenarios,
)
from ..utils import load_checkpoint


@dataclass(frozen=True)
class BaselineConfig:
    training_seeds: tuple[int, ...]
    evaluation_episodes: int
    hidden_dim: int
    tabular_episodes_per_stage: int
    drama_episodes_per_stage: int
    drama_batch_size: int
    drama_replay_capacity: int
    drama_learning_rate: float
    drama_auxiliary_coefficient: float


BASELINE_METHODS = (
    "GPSR",
    "Predictive Geographic",
    "Evo-QGeo",
    "IQMR Q(lambda)",
    "DRAMA",
    "SwitchGLOBE",
)


def _training_scenarios(seed: int):
    return [
        *phase9_curriculum(seed),
        *phase9_hole_training_scenarios(seed),
        *phase9_predictive_training_scenarios(seed),
    ]


def _env_factory(config: FanetConfig) -> FanetRoutingEnv:
    return FanetRoutingEnv(config)


def _checkpoint_path(
    checkpoint_dir: Path,
    *,
    training_seed: int,
    method: str,
) -> Path:
    slug = (
        method.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("lambda", "lambda")
    )
    return checkpoint_dir / f"seed_{training_seed}" / f"{slug}.pt"


def _save_policy(path: Path, policy, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state": policy.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def _load_policy(path: Path, policy) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    policy.load_state_dict(checkpoint["policy_state"])
    return dict(checkpoint.get("metadata", {}))


def _load_switchglobe_policy(
    *,
    training_seed: int,
    switchglobe_checkpoint_dir: Path,
    max_nodes: int,
    hidden_dim: int,
    device: torch.device,
) -> SwitchGlobePolicy:
    normal = GeographicResidualStudentPolicy(
        max_nodes=max_nodes,
        hidden_dim=hidden_dim,
    )
    predictive = LiteGlobePStudentPolicy(
        max_nodes=max_nodes,
        hidden_dim=hidden_dim,
    )
    model = SwitchGlobePolicy(normal, predictive)
    load_checkpoint(
        switchglobe_checkpoint_dir
        / f"seed_{training_seed}"
        / "switchglobe.pt",
        model,
        map_location=device,
    )
    return model


def _train_or_load_external(
    config: BaselineConfig,
    *,
    training_seed: int,
    checkpoint_dir: Path | None,
    device: torch.device,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, object]]:
    max_nodes = phase9_curriculum(training_seed)[0].config.max_nodes
    scenarios = _training_scenarios(training_seed)
    policies: dict[str, object] = {
        "Evo-QGeo": EvoQGeoPolicy(max_nodes),
        "IQMR Q(lambda)": IqmrPolicy(max_nodes),
        "DRAMA": DramaPolicy(max_nodes, hidden_dim=config.hidden_dim, device=device),
    }
    rows: dict[str, Any] = {"training_seed": training_seed}
    for method, policy in policies.items():
        checkpoint = (
            _checkpoint_path(
                checkpoint_dir,
                training_seed=training_seed,
                method=method,
            )
            if checkpoint_dir is not None
            else None
        )
        if resume and checkpoint is not None and checkpoint.is_file():
            metadata = _load_policy(checkpoint, policy)
            result = metadata.get("training_result")
        else:
            if isinstance(policy, DramaPolicy):
                result_obj = train_drama_baseline(
                    _env_factory,
                    policy,
                    scenarios,
                    training_seed=training_seed,
                    episodes_per_stage=config.drama_episodes_per_stage,
                    batch_size=config.drama_batch_size,
                    replay_capacity=config.drama_replay_capacity,
                    learning_rate=config.drama_learning_rate,
                    auxiliary_coefficient=(
                        config.drama_auxiliary_coefficient
                    ),
                )
            else:
                result_obj = train_value_baseline(
                    _env_factory,
                    policy,
                    scenarios,
                    training_seed=training_seed,
                    episodes_per_stage=config.tabular_episodes_per_stage,
                )
            result = asdict(result_obj)
            if checkpoint is not None:
                _save_policy(
                    checkpoint,
                    policy,
                    metadata={
                        "suite": "external_baselines",
                        "training_seed": training_seed,
                        "method": method,
                        "training_result": result,
                    },
                )
        assert isinstance(result, dict)
        for key, value in result.items():
            if key in {"method", "training_seed"}:
                continue
            rows[f"{method}:{key}"] = value
    return rows, policies


def _policy(
    method: str,
    policy_or_model,
    env: FanetRoutingEnv,
    device: torch.device,
):
    if method == "GPSR":
        return GpsrPolicy(env.drop_action)
    if method == "Predictive Geographic":
        return PredictiveGeographicPolicy(env.drop_action)
    if method == "SwitchGLOBE":
        assert isinstance(policy_or_model, SwitchGlobePolicy)
        return StudentPolicyAdapter(
            policy_or_model,
            device=device,
            force_forward_if_available=True,
        )
    return policy_or_model


def run_baseline_campaign(
    config: BaselineConfig,
    *,
    switchglobe_checkpoint_dir: Path,
    output_checkpoint_dir: Path | None = None,
    device: torch.device | str = "cpu",
    resume: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Train external baselines and evaluate them with paired seeds."""

    device = torch.device(device)
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    for training_seed in config.training_seeds:
        training_row, external_policies = _train_or_load_external(
            config,
            training_seed=training_seed,
            checkpoint_dir=output_checkpoint_dir,
            device=device,
            resume=resume,
        )
        training_rows.append(training_row)
        max_nodes = phase9_curriculum(training_seed)[0].config.max_nodes
        switchglobe = _load_switchglobe_policy(
            training_seed=training_seed,
            switchglobe_checkpoint_dir=switchglobe_checkpoint_dir,
            max_nodes=max_nodes,
            hidden_dim=config.hidden_dim,
            device=device,
        )
        models: dict[str, object | None] = {
            "GPSR": None,
            "Predictive Geographic": None,
            **external_policies,
            "SwitchGLOBE": switchglobe,
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
            for method in BASELINE_METHODS:
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
