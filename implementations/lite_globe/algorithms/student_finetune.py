"""Teacher-free local PPO fine-tuning with optional decayed offline KD."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..data.distillation_dataset import DistillationDataset
from ..env.fanet_env import FanetRoutingEnv
from ..models.student_actor_critic import LocalStudentActorCritic
from ..models.tensor_observation import observation_to_tensors
from .distillation import forward_kl_loss
from .ppo import PpoBatch, PpoConfig


@dataclass(frozen=True)
class StudentFineTuneConfig:
    updates: int = 40
    episodes_per_update: int = 32
    kd_lambda_initial: float = 0.0
    kd_decay_rate: float = 0.05
    kd_temperature: float = 1.0
    kd_batch_size: int = 128


@dataclass(frozen=True)
class StudentFineTuneMetrics:
    policy_loss: float
    value_loss: float
    entropy: float
    kd_loss: float
    kd_lambda: float


@dataclass(frozen=True)
class StudentFineTuneResult:
    updates: int
    episodes: int
    transitions: int
    mean_training_return: float
    final_metrics: StudentFineTuneMetrics


def kd_lambda_at_update(
    initial: float,
    decay_rate: float,
    update_index: int,
) -> float:
    if initial < 0 or decay_rate < 0 or update_index < 0:
        raise ValueError("KD schedule values must be non-negative")
    return float(initial * np.exp(-decay_rate * update_index))


def _stack_observations(
    observations: list[dict[str, Tensor]],
) -> dict[str, Tensor]:
    return {
        key: torch.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


def _kd_loader(
    dataset: DistillationDataset | None,
    *,
    batch_size: int,
    seed: int,
) -> Iterator[dict[str, Tensor]] | None:
    if dataset is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    def repeat() -> Iterator[dict[str, Tensor]]:
        while True:
            yield from loader

    return repeat()


def _update_student(
    model: LocalStudentActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: PpoBatch,
    *,
    ppo_config: PpoConfig,
    kd_iterator: Iterator[dict[str, Tensor]] | None,
    kd_lambda: float,
    kd_temperature: float,
    generator: torch.Generator,
    device: torch.device,
) -> StudentFineTuneMetrics:
    sample_count = batch.actions.shape[0]
    advantages = batch.advantages
    if sample_count > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
    totals = torch.zeros(4, dtype=torch.float64)
    optimization_steps = 0
    for _ in range(ppo_config.update_epochs):
        permutation = torch.randperm(
            sample_count,
            generator=generator,
            device=device,
        )
        for start in range(0, sample_count, ppo_config.minibatch_size):
            indices = permutation[start : start + ppo_config.minibatch_size]
            observations = {
                key: value[indices] for key, value in batch.observations.items()
            }
            output = model(observations)
            distribution = torch.distributions.Categorical(
                logits=output.masked_logits
            )
            log_probabilities = distribution.log_prob(batch.actions[indices])
            ratio = torch.exp(
                log_probabilities - batch.old_log_probabilities[indices]
            )
            unclipped = ratio * advantages[indices]
            clipped = torch.clamp(
                ratio,
                1.0 - ppo_config.clip_ratio,
                1.0 + ppo_config.clip_ratio,
            ) * advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.mean(
                (output.value - batch.returns[indices]) ** 2
            )
            entropy = distribution.entropy().mean()
            kd_loss = torch.zeros((), device=device)
            if kd_lambda > 0:
                if kd_iterator is None:
                    raise ValueError(
                        "positive kd_lambda requires a distillation dataset"
                    )
                kd_batch = next(kd_iterator)
                kd_observation = {
                    key: kd_batch[key].to(device)
                    for key in (
                        "self_features",
                        "neighbor_features",
                        "edge_features",
                        "packet_features",
                        "action_mask",
                    )
                }
                kd_output = model.policy(kd_observation)
                kd_loss, _, _ = forward_kl_loss(
                    kd_batch["teacher_logits"].to(device),
                    kd_output.logits,
                    kd_observation["action_mask"],
                    temperature=kd_temperature,
                )
            loss = (
                policy_loss
                + ppo_config.value_coefficient * value_loss
                - ppo_config.entropy_coefficient * entropy
                + kd_lambda * kd_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), ppo_config.max_grad_norm
            )
            optimizer.step()
            totals += torch.tensor(
                [
                    policy_loss.detach().item(),
                    value_loss.detach().item(),
                    entropy.detach().item(),
                    kd_loss.detach().item(),
                ],
                dtype=torch.float64,
            )
            optimization_steps += 1
    averages = totals / optimization_steps
    return StudentFineTuneMetrics(
        policy_loss=float(averages[0]),
        value_loss=float(averages[1]),
        entropy=float(averages[2]),
        kd_loss=float(averages[3]),
        kd_lambda=kd_lambda,
    )


def fine_tune_student(
    env: FanetRoutingEnv,
    model: LocalStudentActorCritic,
    *,
    ppo_config: PpoConfig,
    fine_tune_config: StudentFineTuneConfig,
    seed: int,
    reset_options: dict[str, Any] | None = None,
    kd_dataset: DistillationDataset | None = None,
    device: torch.device | str = "cpu",
) -> StudentFineTuneResult:
    """Fine-tune from local observations without querying a Teacher."""

    if fine_tune_config.updates <= 0:
        raise ValueError("updates must be positive")
    if fine_tune_config.episodes_per_update <= 0:
        raise ValueError("episodes_per_update must be positive")
    if fine_tune_config.kd_lambda_initial > 0 and kd_dataset is None:
        raise ValueError("optional KD requires an offline dataset")
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=ppo_config.learning_rate
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    kd_iterator = _kd_loader(
        kd_dataset,
        batch_size=fine_tune_config.kd_batch_size,
        seed=seed,
    )
    all_returns: list[float] = []
    transitions = 0
    final_metrics = StudentFineTuneMetrics(0.0, 0.0, 0.0, 0.0, 0.0)
    for update_index in range(fine_tune_config.updates):
        observations: list[dict[str, Tensor]] = []
        actions: list[int] = []
        old_log_probabilities: list[float] = []
        returns: list[float] = []
        values: list[float] = []
        for episode_index in range(fine_tune_config.episodes_per_update):
            episode_seed = (
                seed
                + update_index * fine_tune_config.episodes_per_update
                + episode_index
            )
            observation, _ = env.reset(
                seed=episode_seed,
                options=reset_options,
            )
            rewards: list[float] = []
            episode_values: list[float] = []
            terminated = False
            truncated = False
            while not (terminated or truncated):
                tensors = observation_to_tensors(observation, device=device)
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
                observation, reward, terminated, truncated, _ = env.step(
                    action
                )
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
        batch = PpoBatch(
            observations=_stack_observations(observations),
            actions=torch.tensor(actions, dtype=torch.long, device=device),
            old_log_probabilities=torch.tensor(
                old_log_probabilities, dtype=torch.float32, device=device
            ),
            returns=returns_tensor,
            advantages=returns_tensor
            - torch.tensor(values, dtype=torch.float32, device=device),
        )
        current_kd_lambda = kd_lambda_at_update(
            fine_tune_config.kd_lambda_initial,
            fine_tune_config.kd_decay_rate,
            update_index,
        )
        final_metrics = _update_student(
            model,
            optimizer,
            batch,
            ppo_config=ppo_config,
            kd_iterator=kd_iterator,
            kd_lambda=current_kd_lambda,
            kd_temperature=fine_tune_config.kd_temperature,
            generator=generator,
            device=device,
        )
        transitions += len(actions)
    return StudentFineTuneResult(
        updates=fine_tune_config.updates,
        episodes=(
            fine_tune_config.updates
            * fine_tune_config.episodes_per_update
        ),
        transitions=transitions,
        mean_training_return=float(np.mean(all_returns)),
        final_metrics=final_metrics,
    )
