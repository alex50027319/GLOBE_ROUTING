"""On-policy PPO trainer for the privileged global Teacher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from ..env.fanet_env import FanetRoutingEnv
from ..models.teacher_gnn import GlobalTeacherActorCritic
from ..models.tensor_observation import observation_to_tensors
from .ppo import PpoBatch, PpoConfig, PpoMetrics, update_ppo


@dataclass(frozen=True)
class TeacherTrainingResult:
    updates: int
    episodes: int
    transitions: int
    final_metrics: PpoMetrics
    mean_training_return: float


def _stack_observations(
    observations: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    return {
        key: torch.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


def train_teacher(
    env: FanetRoutingEnv,
    model: GlobalTeacherActorCritic,
    *,
    ppo_config: PpoConfig,
    updates: int,
    episodes_per_update: int,
    seed: int,
    reset_options: dict[str, Any] | None = None,
    device: torch.device | str = "cpu",
) -> TeacherTrainingResult:
    """Train the Teacher with complete-episode Monte Carlo PPO rollouts."""

    if updates <= 0 or episodes_per_update <= 0:
        raise ValueError("updates and episodes_per_update must be positive")
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=ppo_config.learning_rate
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    all_returns: list[float] = []
    transition_count = 0
    final_metrics = PpoMetrics(0.0, 0.0, 0.0)

    for update_index in range(updates):
        observations: list[dict[str, Tensor]] = []
        actions: list[int] = []
        old_log_probabilities: list[float] = []
        returns: list[float] = []
        values: list[float] = []
        for episode_index in range(episodes_per_update):
            episode_seed = (
                seed + update_index * episodes_per_update + episode_index
            )
            env.reset(seed=episode_seed, options=reset_options)
            rewards: list[float] = []
            episode_values: list[float] = []
            terminated = False
            truncated = False
            while not (terminated or truncated):
                tensors = observation_to_tensors(
                    env.global_observation(), device=device
                )
                with torch.no_grad():
                    output = model(tensors)
                    action_tensor = torch.multinomial(
                        output.probabilities,
                        num_samples=1,
                        generator=generator,
                    )
                    action = int(action_tensor.item())
                    log_probability = torch.log(
                        output.probabilities[action].clamp_min(1e-8)
                    )
                _, reward, terminated, truncated, _ = env.step(action)
                observations.append(
                    {key: value.detach() for key, value in tensors.items()}
                )
                actions.append(action)
                old_log_probabilities.append(float(log_probability.item()))
                rewards.append(float(reward))
                episode_values.append(float(output.value.item()))

            discounted = 0.0
            episode_returns = [0.0] * len(rewards)
            for index in range(len(rewards) - 1, -1, -1):
                discounted = rewards[index] + ppo_config.gamma * discounted
                episode_returns[index] = discounted
            returns.extend(episode_returns)
            values.extend(episode_values)
            all_returns.append(sum(rewards))

        returns_tensor = torch.tensor(
            returns, dtype=torch.float32, device=device
        )
        values_tensor = torch.tensor(
            values, dtype=torch.float32, device=device
        )
        batch = PpoBatch(
            observations=_stack_observations(observations),
            actions=torch.tensor(actions, dtype=torch.long, device=device),
            old_log_probabilities=torch.tensor(
                old_log_probabilities, dtype=torch.float32, device=device
            ),
            returns=returns_tensor,
            advantages=returns_tensor - values_tensor,
        )
        final_metrics = update_ppo(
            model,
            optimizer,
            batch,
            ppo_config,
            generator=generator,
        )
        transition_count += len(actions)

    return TeacherTrainingResult(
        updates=updates,
        episodes=updates * episodes_per_update,
        transitions=transition_count,
        final_metrics=final_metrics,
        mean_training_return=float(np.mean(all_returns)),
    )
