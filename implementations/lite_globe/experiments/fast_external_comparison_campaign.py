"""Train and evaluate FastSwitchGLOBE on the external-comparison contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ..env.fanet_env import FanetRoutingEnv
from ..evaluation import (
    episode_row,
    evaluate_policy_results,
    generalization_summary,
    measure_policy_cost,
)
from ..models.policy_adapter import StudentPolicyAdapter
from ..scenarios import phase9_evaluation_scenarios
from .external_comparison_campaign import load_switchglobe
from .latency_optimization_campaign import (
    LatencyOptimizationConfig,
    checkpoint_path,
    train_or_load_fast,
)


FAST_METHOD = "FastSwitchGLOBE"


@dataclass(frozen=True)
class FastExternalComparisonConfig:
    training_seeds: tuple[int, ...] = (42, 77, 123, 314, 2718)
    evaluation_episodes: int = 200
    exact_hidden_dim: int = 64
    fast_hidden_dim: int = 32
    dataset_episodes_per_scenario: int = 100
    epochs: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    temperature: float = 1.0
    action_coefficient: float = 1.0
    switch_coefficient: float = 0.2

    def training_config(self) -> LatencyOptimizationConfig:
        return LatencyOptimizationConfig(
            training_seeds=self.training_seeds,
            dataset_episodes_per_scenario=self.dataset_episodes_per_scenario,
            evaluation_episodes=self.evaluation_episodes,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            temperature=self.temperature,
            action_coefficient=self.action_coefficient,
            switch_coefficient=self.switch_coefficient,
            hidden_dim=self.fast_hidden_dim,
        )


def run_fast_external_comparison(
    config: FastExternalComparisonConfig,
    *,
    switchglobe_checkpoint_dir: Path,
    fast_checkpoint_dir: Path,
    device: torch.device | str = "cpu",
    resume: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Run the Fast model only, using the exact baseline campaign's episode seeds.

    FastSwitchGLOBE here is the single-pass distilled model without Top-2 failover
    or freshness caching. Tensor-buffer reuse is an implementation optimization and
    does not change the selected action.
    """

    device = torch.device(device)
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    deployment_rows: list[dict[str, Any]] = []
    training_config = config.training_config()

    for seed in config.training_seeds:
        scenarios = phase9_evaluation_scenarios(seed)
        max_nodes = scenarios[0].config.max_nodes
        teacher = load_switchglobe(
            switchglobe_checkpoint_dir,
            seed=seed,
            max_nodes=max_nodes,
            hidden_dim=config.exact_hidden_dim,
            device=device,
        )
        trained, training = train_or_load_fast(
            teacher,
            config=training_config,
            seed=seed,
            checkpoint_dir=fast_checkpoint_dir,
            device=device,
            resume=resume,
        )
        policy = StudentPolicyAdapter(
            trained.model,
            device=device,
            force_forward_if_available=True,
            reuse_tensor_buffer=True,
            enable_fast_failover=False,
            enable_freshness_cache=False,
        )
        training_rows.append({"method": FAST_METHOD, "training_seed": seed, **training})

        cost_scenario = scenarios[0]
        cost_env = FanetRoutingEnv(cost_scenario.config)
        cost_observation, _ = cost_env.reset(
            seed=1_099_999, options=cost_scenario.reset_options
        )
        cost = measure_policy_cost(
            policy,
            cost_observation,
            model=policy.model,
            input_observation={
                key: cost_observation[key]
                for key in policy.model.observation_fields
                if key in cost_observation
            },
            device=device,
            warmup=2,
            repeats=10,
            serialized_model_path=checkpoint_path(fast_checkpoint_dir, seed),
        )
        deployment_rows.append(
            {"method": FAST_METHOD, "training_seed": seed, **cost.to_dict()}
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
            results = evaluate_policy_results(
                FanetRoutingEnv(scenario.config),
                policy,
                evaluation_seeds,
                reset_options=scenario.reset_options,
            )
            episode_rows.extend(
                episode_row(
                    result,
                    method=FAST_METHOD,
                    scenario=scenario.name,
                    training_seed=seed,
                )
                for result in results
            )
            summary_rows.append(
                generalization_summary(
                    results,
                    method=FAST_METHOD,
                    scenario=scenario.name,
                    training_seed=seed,
                )
            )

    return {
        "episodes": episode_rows,
        "seed_summaries": summary_rows,
        "training": training_rows,
        "deployment_costs": deployment_rows,
    }
