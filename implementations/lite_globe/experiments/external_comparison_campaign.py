"""Resumable fair external comparison campaign for SwitchGLOBE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..baselines.common import atomic_torch_save, snapshot_from_env
from ..baselines.evo_qgeo import EvoQGeoAdaptedPolicy
from ..baselines.gat_gru_ddqn import GatGruDdqnPolicy
from ..baselines.rdqn_herp import RdqnHerpAdaptedPolicy
from ..baselines.registry import COMPARISON_METHODS, EXTERNAL_METHODS, METHOD_REGISTRY, build_method
from ..baselines.external_rl import train_value_baseline
from ..env.fanet_env import FanetRoutingEnv
from ..evaluation import episode_row, evaluate_policy_results, generalization_summary, measure_policy_cost
from ..models import GeographicResidualStudentPolicy, LiteGlobePStudentPolicy, SwitchGlobePolicy
from ..models.policy_adapter import StudentPolicyAdapter
from ..scenarios import (
    phase9_curriculum,
    phase9_evaluation_scenarios,
    phase9_hole_training_scenarios,
    phase9_predictive_training_scenarios,
)
from ..utils import load_checkpoint


@dataclass(frozen=True)
class ExternalComparisonConfig:
    training_seeds: tuple[int, ...] = (42, 77, 123, 314, 2718)
    evaluation_episodes: int = 200
    hidden_dim: int = 64
    tabular_episodes_per_stage: int = 350
    neural_episodes_per_stage: int = 450
    neural_batch_size: int = 64


def training_scenarios(seed: int):
    return [
        *phase9_curriculum(seed),
        *phase9_hole_training_scenarios(seed),
        *phase9_predictive_training_scenarios(seed),
    ]


def checkpoint_path(root: Path, seed: int, method: str) -> Path:
    return root / f"seed_{seed}" / f"{METHOD_REGISTRY[method].slug}.pt"


def _load_complete(path: Path, policy: object, *, method: str, seed: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not payload.get("complete") or payload.get("method") != method or int(payload.get("training_seed", -1)) != seed:
        return None
    if isinstance(policy, (RdqnHerpAdaptedPolicy, GatGruDdqnPolicy)):
        policy.load_checkpoint_state(payload["policy_state"])
    elif hasattr(policy, "load_state_dict"):
        policy.load_state_dict(payload["policy_state"])
    return dict(payload["training"])


def _save_complete(path: Path, policy: object, *, method: str, seed: int, training: dict[str, Any]) -> None:
    if isinstance(policy, (RdqnHerpAdaptedPolicy, GatGruDdqnPolicy)):
        state = policy.checkpoint_state()
    elif hasattr(policy, "state_dict"):
        state = policy.state_dict()
    else:
        state = {}
    atomic_torch_save(
        {
            "schema_version": 1,
            "complete": True,
            "method": method,
            "training_seed": seed,
            "policy_state": state,
            "training": training,
        },
        path,
    )


def _train_neural(policy: RdqnHerpAdaptedPolicy, scenarios, *, seed: int,
                  episodes_per_stage: int, batch_size: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    losses: list[float] = []
    delivered = dropped = episodes = 0
    for scenario_index, scenario in enumerate(scenarios):
        env = FanetRoutingEnv(scenario.config)
        for episode_index in range(episodes_per_stage):
            index = scenario_index * episodes_per_stage + episode_index
            policy.reset(seed + 3_000_000 + index)
            observation, _ = env.reset(
                seed=int(rng.integers(0, 2**31 - 1)),
                options=scenario.reset_options,
            )
            done = False
            while not done:
                candidates = np.flatnonzero(observation["action_mask"][:policy.drop_action])
                epsilon = max(0.03, 0.35 * (1.0 - index / max(len(scenarios) * episodes_per_stage, 1)))
                action = int(rng.choice(candidates)) if candidates.size and rng.random() < epsilon else policy.act(observation)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                unstable = 1.0
                if action < policy.drop_action and "candidate_risk_features" in observation:
                    unstable = 1.0 - float(observation["candidate_risk_features"][action, 1])
                policy.observe(observation, action, float(reward), next_observation, done, unstable)
                loss = policy.learn(batch_size=batch_size)
                if loss is not None:
                    losses.append(loss)
                observation = next_observation
            delivered += int(info["delivered"])
            dropped += int(info["dropped"])
            episodes += 1
    return {
        "episodes": episodes,
        "updates": policy.updates,
        "environment_steps": policy.environment_steps,
        "mean_loss": float(np.mean(losses)) if losses else 0.0,
        "delivered": delivered,
        "dropped": dropped,
    }


def train_or_load_methods(config: ExternalComparisonConfig, *, seed: int,
                          checkpoint_dir: Path, device: torch.device,
                          resume: bool) -> tuple[dict[str, object], list[dict[str, Any]]]:
    max_nodes = phase9_curriculum(seed)[0].config.max_nodes
    scenarios = training_scenarios(seed)
    policies: dict[str, object] = {}
    rows: list[dict[str, Any]] = []
    for method in EXTERNAL_METHODS:
        spec = METHOD_REGISTRY[method]
        policy = build_method(method, max_nodes=max_nodes, hidden_dim=config.hidden_dim, device=device)
        path = checkpoint_path(checkpoint_dir, seed, method)
        training = _load_complete(path, policy, method=method, seed=seed) if resume else None
        resumed = training is not None
        if training is None:
            if isinstance(policy, EvoQGeoAdaptedPolicy):
                training = asdict(train_value_baseline(
                    FanetRoutingEnv, policy, scenarios, training_seed=seed,
                    episodes_per_stage=config.tabular_episodes_per_stage,
                ))
            elif isinstance(policy, (RdqnHerpAdaptedPolicy, GatGruDdqnPolicy)):
                training = _train_neural(
                    policy, scenarios, seed=seed,
                    episodes_per_stage=config.neural_episodes_per_stage,
                    batch_size=config.neural_batch_size,
                )
            else:
                training = {"episodes": 0, "updates": 0, "environment_steps": 0}
            _save_complete(path, policy, method=method, seed=seed, training=training)
        policies[method] = policy
        rows.append({**training, "method": method, "training_seed": seed, "resumed": int(resumed)})
    return policies, rows


def _switchglobe_path(root: Path, seed: int) -> Path:
    candidates = (
        root / f"seed_{seed}" / "switchglobe.pt",
        root / f"seed_{seed}" / "risk_switch_lite_globe_p.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"SwitchGLOBE checkpoint not found for seed {seed} under {root}")


def load_switchglobe(root: Path, *, seed: int, max_nodes: int, hidden_dim: int,
                     device: torch.device) -> StudentPolicyAdapter:
    model = SwitchGlobePolicy(
        GeographicResidualStudentPolicy(max_nodes, hidden_dim=hidden_dim),
        LiteGlobePStudentPolicy(max_nodes, hidden_dim=hidden_dim),
    )
    load_checkpoint(_switchglobe_path(root, seed), model, map_location=device)
    model.eval()
    return StudentPolicyAdapter(model, device=device, force_forward_if_available=True)


def run_external_comparison(config: ExternalComparisonConfig, *, switchglobe_checkpoint_dir: Path,
                            checkpoint_dir: Path, device: torch.device | str = "cpu",
                            resume: bool = False) -> dict[str, Any]:
    device = torch.device(device)
    episode_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    deployment_rows: list[dict[str, Any]] = []
    for seed in config.training_seeds:
        policies, seed_training = train_or_load_methods(
            config, seed=seed, checkpoint_dir=checkpoint_dir, device=device, resume=resume,
        )
        training_rows.extend(seed_training)
        max_nodes = phase9_curriculum(seed)[0].config.max_nodes
        policies["SwitchGLOBE"] = load_switchglobe(
            switchglobe_checkpoint_dir, seed=seed, max_nodes=max_nodes,
            hidden_dim=config.hidden_dim, device=device,
        )
        cost_scenario = phase9_evaluation_scenarios(seed)[0]
        for method in COMPARISON_METHODS:
            cost_env = FanetRoutingEnv(cost_scenario.config)
            cost_observation, _ = cost_env.reset(
                seed=1_099_999, options=cost_scenario.reset_options,
            )
            policy = policies[method]
            tick = getattr(policy, "protocol_tick", None)
            prepare = (lambda tick=tick, env=cost_env: tick(snapshot_from_env(env))) if tick is not None else None
            checkpoint = (
                _switchglobe_path(switchglobe_checkpoint_dir, seed)
                if method == "SwitchGLOBE"
                else checkpoint_path(checkpoint_dir, seed, method)
            )
            model = getattr(policy, "model", None)
            if model is None:
                model = getattr(policy, "online", None)
            cost = measure_policy_cost(
                policy, cost_observation, model=model,
                input_observation={key: cost_observation[key] for key in (
                    METHOD_REGISTRY[method].observation_fields
                    if method in METHOD_REGISTRY
                    else tuple(cost_observation)
                ) if key in cost_observation},
                device=device, warmup=2, repeats=10,
                serialized_model_path=checkpoint, prepare=prepare,
            )
            deployment_rows.append({"method": method, "training_seed": seed, **cost.to_dict()})
        for scenario_index, scenario in enumerate(phase9_evaluation_scenarios(seed)):
            evaluation_seeds = list(range(
                1_100_000 + scenario_index * 10_000,
                1_100_000 + scenario_index * 10_000 + config.evaluation_episodes,
            ))
            for method in COMPARISON_METHODS:
                results = evaluate_policy_results(
                    FanetRoutingEnv(scenario.config), policies[method], evaluation_seeds,
                    reset_options=scenario.reset_options,
                )
                episode_rows.extend(episode_row(result, method=method, scenario=scenario.name, training_seed=seed) for result in results)
                summaries.append(generalization_summary(results, method=method, scenario=scenario.name, training_seed=seed))
    return {
        "episodes": episode_rows,
        "seed_summaries": summaries,
        "training": training_rows,
        "deployment_costs": deployment_rows,
        "method_contracts": [METHOD_REGISTRY[name].manifest_dict() for name in EXTERNAL_METHODS],
    }
