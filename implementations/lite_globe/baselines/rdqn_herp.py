"""Common-contract RDQN-HERP adaptation for one-hop FANET routing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .common import Observation, PrioritizedReplay, ReplayTransition, filtered_observation_bytes
from .external_rl import candidate_feature_matrix


class NoisyLinear(nn.Module):
    """Factorised Gaussian NoisyNet layer."""

    def __init__(self, in_features: int, out_features: int, sigma0: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.zeros(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.zeros(out_features))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, sigma0 / math.sqrt(in_features))
        nn.init.constant_(self.bias_sigma, sigma0 / math.sqrt(out_features))
        self.reset_noise()

    @staticmethod
    def _noise(size: int, device: torch.device) -> Tensor:
        value = torch.randn(size, device=device)
        return value.sign() * value.abs().sqrt()

    def reset_noise(self) -> None:
        input_noise = self._noise(self.in_features, self.weight_mu.device)
        output_noise = self._noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(output_noise.outer(input_noise))
        self.bias_epsilon.copy_(output_noise)

    def forward(self, value: Tensor) -> Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight, bias = self.weight_mu, self.bias_mu
        return torch.nn.functional.linear(value, weight, bias)


class DuelingCandidateQNetwork(nn.Module):
    def __init__(self, max_nodes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.max_nodes = max_nodes
        self.encoder = nn.Sequential(nn.Linear(13, hidden_dim), nn.ReLU())
        self.value = nn.Sequential(NoisyLinear(hidden_dim, hidden_dim), nn.ReLU(), NoisyLinear(hidden_dim, 1))
        self.advantage = nn.Sequential(NoisyLinear(hidden_dim, hidden_dim), nn.ReLU(), NoisyLinear(hidden_dim, 1))
        self.drop = nn.Parameter(torch.tensor(-4.0))

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, features: Tensor, mask: Tensor) -> Tensor:
        encoded = self.encoder(features)
        valid = mask.to(encoded.dtype).unsqueeze(-1)
        context = (encoded * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        value = self.value(context)
        advantage = self.advantage(encoded).squeeze(-1)
        masked_advantage = (advantage * mask).sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp_min(1)
        candidate_q = value + advantage - masked_advantage
        drop_q = self.drop.expand(features.shape[0], 1)
        return torch.cat((candidate_q, drop_q), dim=-1)


def _clone_observation(observation: Observation) -> Observation:
    return {key: np.asarray(value).copy() for key, value in observation.items()}


def _batch(observations: list[Observation], drop_action: int, device: torch.device) -> tuple[Tensor, Tensor]:
    features, masks = [], []
    for observation in observations:
        matrix, valid, _ = candidate_feature_matrix(observation, drop_action)
        features.append(matrix)
        masks.append(valid)
    return (
        torch.as_tensor(np.stack(features), dtype=torch.float32, device=device),
        torch.as_tensor(np.stack(masks), dtype=torch.bool, device=device),
    )


@dataclass(frozen=True)
class NeuralTrainingProgress:
    episodes: int
    environment_steps: int
    updates: int
    mean_loss: float


class RdqnHerpAdaptedPolicy:
    """DDQN + dueling NoisyNet + PER/n-step with three HERP proxy tiers."""

    source = "10.1109/TVT.2026.3668740"
    fidelity = "common-contract adaptation; unavailable details are explicit assumptions"
    observation_fields = ("self_features", "neighbor_features", "edge_features", "packet_features", "action_mask")

    def __init__(self, max_nodes: int, *, hidden_dim: int = 64, learning_rate: float = 1e-3,
                 replay_capacity: int = 20_000, gamma: float = 0.99, n_step: int = 3,
                 target_interval: int = 100, device: torch.device | str = "cpu") -> None:
        self.drop_action = max_nodes
        self.device = torch.device(device)
        self.online = DuelingCandidateQNetwork(max_nodes, hidden_dim).to(self.device)
        self.target = DuelingCandidateQNetwork(max_nodes, hidden_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.replay = PrioritizedReplay(replay_capacity)
        self.gamma = gamma
        self.n_step = n_step
        self.target_interval = target_interval
        self.nstep_buffer: deque[tuple[Observation, int, float, Observation, bool, int]] = deque()
        self.environment_steps = 0
        self.updates = 0
        self.completed_episodes = 0
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)
        self.nstep_buffer.clear()
        self.online.eval()

    def observation_bytes(self, observation: Observation) -> int:
        return filtered_observation_bytes(observation, self.observation_fields)

    def q_values(self, observation: Observation) -> Tensor:
        features, mask = _batch([observation], self.drop_action, self.device)
        self.online.eval()
        with torch.no_grad():
            values = self.online(features, mask)[0]
            full_mask = torch.cat((mask[0], torch.ones(1, dtype=torch.bool, device=self.device)))
            return values.masked_fill(~full_mask, -torch.inf)

    def act(self, observation: Observation) -> int:
        return int(torch.argmax(self.q_values(observation)).item())

    @staticmethod
    def herp_tier(reward: float, done: bool, unstable_factor: float) -> int:
        if done and reward > 0:
            return 2
        if done or unstable_factor >= 0.5:
            return 1
        return 0

    def observe(self, state: Observation, action: int, reward: float, next_state: Observation,
                done: bool, unstable_factor: float) -> None:
        tier = self.herp_tier(reward, done, unstable_factor)
        self.nstep_buffer.append((_clone_observation(state), action, reward, _clone_observation(next_state), done, tier))
        self.environment_steps += 1
        if len(self.nstep_buffer) >= self.n_step or done:
            total = 0.0
            discount = 1.0
            last_next = next_state
            last_done = False
            max_tier = 0
            for _, _, item_reward, item_next, item_done, item_tier in list(self.nstep_buffer)[: self.n_step]:
                total += discount * item_reward
                discount *= self.gamma
                last_next, last_done = item_next, item_done
                max_tier = max(max_tier, item_tier)
                if item_done:
                    break
            first_state, first_action, *_ = self.nstep_buffer[0]
            self.replay.add(ReplayTransition(first_state, first_action, total, _clone_observation(last_next), last_done, discount, max_tier))
            self.nstep_buffer.popleft()
        if done:
            self.completed_episodes += 1
            self.nstep_buffer.clear()

    def learn(self, batch_size: int = 32) -> float | None:
        if len(self.replay) < max(2, batch_size):
            return None
        transitions, indices = self.replay.sample(batch_size, self.rng)
        states, masks = _batch([item.state for item in transitions], self.drop_action, self.device)
        next_states, next_masks = _batch([item.next_state for item in transitions], self.drop_action, self.device)
        actions = torch.tensor([item.action for item in transitions], device=self.device)
        rewards = torch.tensor([item.reward for item in transitions], dtype=torch.float32, device=self.device)
        discounts = torch.tensor([item.discount for item in transitions], dtype=torch.float32, device=self.device)
        dones = torch.tensor([item.done for item in transitions], dtype=torch.float32, device=self.device)
        self.online.train()
        self.online.reset_noise()
        predicted = self.online(states, masks).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            online_next = self.online(next_states, next_masks)
            target_next = self.target(next_states, next_masks)
            full_mask = torch.cat((next_masks, torch.ones((len(transitions), 1), dtype=torch.bool, device=self.device)), dim=1)
            next_actions = online_next.masked_fill(~full_mask, -torch.inf).argmax(1)
            bootstrap = target_next.gather(1, next_actions[:, None]).squeeze(1)
            targets = rewards + (1.0 - dones) * discounts * bootstrap
        td_error = targets - predicted
        loss = torch.nn.functional.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step()
        self.replay.update_priorities(indices, td_error.detach().abs().cpu().numpy())
        self.updates += 1
        if self.updates % self.target_interval == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "online": self.online.state_dict(), "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(), "replay": self.replay.state_dict(),
            "environment_steps": self.environment_steps, "updates": self.updates,
            "completed_episodes": self.completed_episodes,
            "rng_state": self.rng.bit_generator.state,
            "nstep_buffer": list(self.nstep_buffer),
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.replay.load_state_dict(state["replay"])
        self.environment_steps = int(state["environment_steps"])
        self.updates = int(state["updates"])
        self.completed_episodes = int(state["completed_episodes"])
        self.rng.bit_generator.state = state["rng_state"]
        self.nstep_buffer = deque(state.get("nstep_buffer", []))
