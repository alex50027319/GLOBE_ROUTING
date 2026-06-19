"""Local-only actor-critic used for Student PPO fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from ..env.observation import (
    EDGE_FEATURES,
    NEIGHBOR_FEATURES,
    PACKET_FEATURES,
    SELF_FEATURES,
)
from .student_policy import LocalStudentPolicy


@dataclass(frozen=True)
class StudentActorCriticOutput:
    """Local Student policy tensors and local value estimate."""

    logits: Tensor
    masked_logits: Tensor
    probabilities: Tensor
    value: Tensor


class LocalStudentActorCritic(nn.Module):
    """Attach a local value network without changing the Student actor."""

    def __init__(self, policy: LocalStudentPolicy) -> None:
        super().__init__()
        self.policy = policy
        self.max_nodes = policy.max_nodes
        self.hidden_dim = policy.hidden_dim
        critic_input = (
            SELF_FEATURES
            + PACKET_FEATURES
            + NEIGHBOR_FEATURES
            + EDGE_FEATURES
        )
        self.value_network = nn.Sequential(
            nn.Linear(critic_input, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self, observation: Mapping[str, Tensor]
    ) -> StudentActorCriticOutput:
        policy_output = self.policy(observation)
        unbatched = observation["self_features"].ndim == 1
        self_features = observation["self_features"]
        neighbors = observation["neighbor_features"]
        edges = observation["edge_features"]
        packet = observation["packet_features"]
        action_mask = observation["action_mask"]
        if unbatched:
            self_features = self_features.unsqueeze(0)
            neighbors = neighbors.unsqueeze(0)
            edges = edges.unsqueeze(0)
            packet = packet.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)
        valid = action_mask[:, : self.max_nodes].unsqueeze(-1).to(
            neighbors.dtype
        )
        count = valid.sum(dim=1).clamp_min(1.0)
        mean_neighbor = (neighbors * valid).sum(dim=1) / count
        mean_edge = (edges * valid).sum(dim=1) / count
        value = self.value_network(
            torch.cat(
                (self_features, packet, mean_neighbor, mean_edge),
                dim=-1,
            )
        ).squeeze(-1)
        if unbatched:
            value = value.squeeze(0)
        return StudentActorCriticOutput(
            logits=policy_output.logits,
            masked_logits=policy_output.masked_logits,
            probabilities=policy_output.probabilities,
            value=value,
        )
