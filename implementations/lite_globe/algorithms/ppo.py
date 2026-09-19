"""Small, dependency-free clipped PPO update for discrete masked policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class ActorCriticOutput(Protocol):
    masked_logits: Tensor
    value: Tensor


class ActorCriticModel(Protocol):
    def forward(self, observation: dict[str, Tensor]) -> ActorCriticOutput: ...

    def __call__(self, observation: dict[str, Tensor]) -> ActorCriticOutput: ...

    def parameters(self): ...


@dataclass(frozen=True)
class PpoConfig:
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    update_epochs: int = 4
    minibatch_size: int = 128
    max_grad_norm: float = 0.5


@dataclass(frozen=True)
class PpoBatch:
    observations: dict[str, Tensor]
    actions: Tensor
    old_log_probabilities: Tensor
    returns: Tensor
    advantages: Tensor


@dataclass(frozen=True)
class PpoMetrics:
    policy_loss: float
    value_loss: float
    entropy: float


def update_ppo(
    model: ActorCriticModel,
    optimizer: torch.optim.Optimizer,
    batch: PpoBatch,
    config: PpoConfig,
    *,
    generator: torch.Generator,
) -> PpoMetrics:
    """Perform clipped PPO epochs over one on-policy rollout batch."""

    sample_count = batch.actions.shape[0]
    if sample_count == 0:
        raise ValueError("PPO batch must contain at least one transition")
    advantages = batch.advantages
    if sample_count > 1:
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )
    totals = torch.zeros(3, dtype=torch.float64)
    updates = 0
    for _ in range(config.update_epochs):
        permutation = torch.randperm(
            sample_count,
            generator=generator,
            device=batch.actions.device,
        )
        for start in range(0, sample_count, config.minibatch_size):
            indices = permutation[start : start + config.minibatch_size]
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
                1.0 - config.clip_ratio,
                1.0 + config.clip_ratio,
            ) * advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = torch.mean(
                (output.value - batch.returns[indices]) ** 2
            )
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + config.value_coefficient * value_loss
                - config.entropy_coefficient * entropy
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.max_grad_norm
            )
            optimizer.step()
            totals += torch.tensor(
                [
                    policy_loss.detach().item(),
                    value_loss.detach().item(),
                    entropy.detach().item(),
                ],
                dtype=torch.float64,
            )
            updates += 1
    averages = totals / updates
    return PpoMetrics(
        policy_loss=float(averages[0]),
        value_loss=float(averages[1]),
        entropy=float(averages[2]),
    )
