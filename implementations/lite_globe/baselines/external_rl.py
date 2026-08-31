"""Recent RL routing baselines adapted to the Lite-GLOBE FANET setting.

The original papers use different simulators and state variables.  These
implementations preserve the deployable information pattern and learning
update of each method, while mapping unavailable physical quantities to the
local proxies already exposed by :mod:`implementations.lite_globe.env`.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import copy
import random
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch
from torch import nn
from torch.nn import functional as F


Observation = dict[str, NDArray[np.generic]]


def _candidate_nodes(
    observation: Observation,
    drop_action: int,
) -> NDArray[np.int64]:
    return np.flatnonzero(observation["action_mask"][:drop_action]).astype(
        np.int64
    )


def _progress_values(
    observation: Observation,
    candidates: NDArray[np.int64],
) -> tuple[float, NDArray[np.float32], NDArray[np.float32]]:
    destination_delta = observation["packet_features"][:2].astype(np.float32)
    neighbor_delta = observation["neighbor_features"][candidates, :2].astype(
        np.float32
    )
    current_distance = float(np.linalg.norm(destination_delta))
    next_distance = np.linalg.norm(
        destination_delta[None, :] - neighbor_delta,
        axis=1,
    ).astype(np.float32)
    progress = (current_distance - next_distance).astype(np.float32)
    return current_distance, next_distance, progress


def candidate_feature_matrix(
    observation: Observation,
    drop_action: int,
) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]]:
    """Return fixed candidate features, valid mask, and geographic prior.

    Feature order:
    bias, signed progress, normalized progress, destination closeness, link
    margin, predicted lifetime, queue headroom, onward flag, onward count,
    destination flag, collision-safety proxy, TTL ratio, hop ratio.
    """

    max_nodes = drop_action
    features = np.zeros((max_nodes, 13), dtype=np.float32)
    valid_mask = observation["action_mask"][:max_nodes].astype(bool)
    candidates = np.flatnonzero(valid_mask).astype(np.int64)
    if candidates.size == 0:
        return features, valid_mask, np.full(max_nodes, -1e6, dtype=np.float32)

    current_distance, next_distance, progress = _progress_values(
        observation,
        candidates,
    )
    edge = observation["edge_features"][candidates]
    neighbor = observation["neighbor_features"][candidates]
    risk = observation.get("candidate_risk_features")
    forward = observation.get("candidate_forwardability")

    margin = (
        risk[candidates, 0].astype(np.float32)
        if risk is not None
        else np.clip(1.0 - edge[:, 0], 0.0, 1.0).astype(np.float32)
    )
    lifetime = (
        risk[candidates, 1].astype(np.float32)
        if risk is not None
        else margin.copy()
    )
    queue_headroom = (
        risk[candidates, 2].astype(np.float32)
        if risk is not None
        else np.clip(1.0 - neighbor[:, 4], 0.0, 1.0).astype(np.float32)
    )
    onward_lifetime = (
        risk[candidates, 3].astype(np.float32)
        if risk is not None
        else lifetime.copy()
    )
    onward_flag = (
        forward[candidates, 0].astype(np.float32)
        if forward is not None
        else np.ones(candidates.size, dtype=np.float32)
    )
    onward_count = (
        forward[candidates, 1].astype(np.float32)
        if forward is not None
        else np.zeros(candidates.size, dtype=np.float32)
    )
    destination_flag = neighbor[:, 5].astype(np.float32)
    collision_safety = np.clip(edge[:, 0], 0.0, 1.0).astype(np.float32)
    signed_progress = np.clip(progress, -1.0, 1.0)
    normalizer = max(current_distance, 1e-6)
    normalized_progress = np.clip(progress / normalizer, -1.0, 1.0)
    destination_closeness = np.clip(
        1.0 - next_distance / max(current_distance, 1e-6),
        -1.0,
        1.0,
    ).astype(np.float32)
    ttl_ratio = float(observation["packet_features"][2])
    hop_ratio = float(observation["packet_features"][3])

    features[candidates] = np.stack(
        [
            np.ones(candidates.size, dtype=np.float32),
            signed_progress,
            normalized_progress.astype(np.float32),
            destination_closeness,
            margin,
            lifetime,
            queue_headroom,
            onward_flag,
            onward_count,
            destination_flag,
            collision_safety,
            np.full(candidates.size, ttl_ratio, dtype=np.float32),
            np.full(candidates.size, hop_ratio, dtype=np.float32),
        ],
        axis=1,
    )
    prior = np.full(max_nodes, -1e6, dtype=np.float32)
    prior[candidates] = (
        4.0 * normalized_progress
        + 2.0 * destination_flag
        + 1.5 * margin
        + 1.5 * lifetime
        + 0.7 * onward_flag
        + 0.5 * onward_lifetime
        + 0.2 * onward_count
        + 0.3 * queue_headroom
    ).astype(np.float32)
    return features, valid_mask, prior


def _discretize(value: float, bins: tuple[float, ...]) -> int:
    return int(np.digitize([value], bins)[0])


class EvoQGeoPolicy:
    """Evo-QGeo style online Q-learning with future link-state scoring."""

    method_name = "Evo-QGeo"

    def __init__(
        self,
        drop_action: int,
        *,
        learning_rate: float = 0.25,
        gamma: float = 0.85,
        epsilon: float = 0.0,
        link_state_weight: float = 1.0,
        hole_bypass_bonus: float = 2.0,
    ) -> None:
        self.drop_action = drop_action
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.link_state_weight = link_state_weight
        self.hole_bypass_bonus = hole_bypass_bonus
        self.q_table: defaultdict[tuple[tuple[int, ...], tuple[int, ...]], float]
        self.q_table = defaultdict(float)
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def observation_bytes(self, observation: Observation) -> int:
        keys = (
            "neighbor_features",
            "edge_features",
            "packet_features",
            "action_mask",
            "candidate_forwardability",
            "candidate_risk_features",
        )
        return sum(
            int(observation[key].nbytes)
            for key in keys
            if key in observation
        )

    def _state_key(self, observation: Observation) -> tuple[int, ...]:
        candidates = _candidate_nodes(observation, self.drop_action)
        current_distance = (
            _progress_values(observation, candidates)[0]
            if candidates.size
            else float(np.linalg.norm(observation["packet_features"][:2]))
        )
        self_queue = float(observation["self_features"][4])
        return (
            _discretize(current_distance, (0.2, 0.4, 0.7, 1.0, 1.5)),
            _discretize(float(observation["packet_features"][2]), (0.25, 0.5, 0.75)),
            _discretize(float(observation["packet_features"][3]), (0.25, 0.5, 0.75)),
            _discretize(len(candidates), (1, 2, 4, 8, 16)),
            _discretize(self_queue, (0.25, 0.5, 0.75)),
        )

    def _action_key(self, features: NDArray[np.float32], action: int) -> tuple[int, ...]:
        row = features[action]
        return (
            _discretize(float(row[2]), (-0.2, 0.0, 0.2, 0.5)),
            _discretize(float(row[4]), (0.2, 0.4, 0.6, 0.8)),
            _discretize(float(row[5]), (0.2, 0.4, 0.6, 0.8)),
            _discretize(float(row[7]), (0.5,)),
            _discretize(float(row[8]), (0.05, 0.15, 0.3)),
            _discretize(float(row[9]), (0.5,)),
        )

    def _link_state_scores(
        self,
        observation: Observation,
    ) -> tuple[NDArray[np.float32], NDArray[np.bool_], NDArray[np.float32]]:
        features, valid_mask, prior = candidate_feature_matrix(
            observation,
            self.drop_action,
        )
        scores = prior.copy()
        candidates = np.flatnonzero(valid_mask)
        if candidates.size and not np.any(features[candidates, 2] > 0.0):
            scores[candidates] += self.hole_bypass_bonus * (
                features[candidates, 7]
                + features[candidates, 8]
                + features[candidates, 5]
            )
        return features, valid_mask, scores

    def q_values(self, observation: Observation) -> NDArray[np.float32]:
        features, valid_mask, scores = self._link_state_scores(observation)
        state_key = self._state_key(observation)
        values = np.full(self.drop_action, -1e6, dtype=np.float32)
        for action in np.flatnonzero(valid_mask):
            q_key = (state_key, self._action_key(features, int(action)))
            learned = self.q_table[q_key]
            values[action] = learned + self.link_state_weight * scores[action]
        return values

    def select_action(self, observation: Observation, epsilon: float | None = None) -> int:
        candidates = _candidate_nodes(observation, self.drop_action)
        if candidates.size == 0:
            return self.drop_action
        eps = self.epsilon if epsilon is None else epsilon
        if self.rng.random() < eps:
            return int(self.rng.choice(candidates))
        values = self.q_values(observation)
        return int(candidates[int(np.argmax(values[candidates]))])

    def act(self, observation: Observation) -> int:
        return self.select_action(observation, epsilon=0.0)

    def update(
        self,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Observation,
        done: bool,
        *,
        delivered: bool,
        dropped: bool,
    ) -> float:
        if action == self.drop_action:
            return 0.0
        features, _, link_scores = self._link_state_scores(observation)
        state_key = self._state_key(observation)
        q_key = (state_key, self._action_key(features, action))
        current = self.q_table[q_key]
        if delivered:
            paper_reward = 10.0
        elif dropped:
            paper_reward = -10.0
        else:
            paper_reward = float(link_scores[action])
        next_max = 0.0 if done else float(np.max(self.q_values(next_observation)))
        target = paper_reward + self.gamma * next_max
        updated = current + self.learning_rate * (target - current)
        self.q_table[q_key] = updated
        return float(abs(target - current))

    def state_dict(self) -> dict[str, Any]:
        return {
            "q_table": dict(self.q_table),
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "link_state_weight": self.link_state_weight,
            "hole_bypass_bonus": self.hole_bypass_bonus,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.learning_rate = float(state["learning_rate"])
        self.gamma = float(state["gamma"])
        self.link_state_weight = float(state["link_state_weight"])
        self.hole_bypass_bonus = float(state["hole_bypass_bonus"])
        self.q_table = defaultdict(float, state["q_table"])


class IqmrPolicy:
    """IQMR-style multi-objective Q(lambda) routing baseline."""

    method_name = "IQMR Q(lambda)"

    def __init__(
        self,
        drop_action: int,
        *,
        lambda_trace: float = 0.65,
        beta_min: float = 0.01,
        beta_max: float = 1.0,
        gamma_min: float = 0.1,
        gamma_max: float = 0.9,
        epsilon: float = 0.0,
    ) -> None:
        self.drop_action = drop_action
        self.lambda_trace = lambda_trace
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.epsilon = epsilon
        self.weights = np.zeros(13, dtype=np.float32)
        self.weights[:10] = np.array(
            [0.0, 0.1, 0.6, 0.2, 0.8, 0.6, 0.35, 0.35, 0.2, 1.0],
            dtype=np.float32,
        )
        self.eligibility = np.zeros_like(self.weights)
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        self.eligibility.fill(0.0)
        self.rng = np.random.default_rng(seed)

    def observation_bytes(self, observation: Observation) -> int:
        keys = (
            "neighbor_features",
            "edge_features",
            "packet_features",
            "action_mask",
            "candidate_forwardability",
            "candidate_risk_features",
        )
        return sum(
            int(observation[key].nbytes)
            for key in keys
            if key in observation
        )

    def q_values(self, observation: Observation) -> NDArray[np.float32]:
        features, valid_mask, _ = candidate_feature_matrix(
            observation,
            self.drop_action,
        )
        values = features @ self.weights
        values[~valid_mask] = -1e6
        return values.astype(np.float32)

    def select_action(self, observation: Observation, epsilon: float | None = None) -> int:
        candidates = _candidate_nodes(observation, self.drop_action)
        if candidates.size == 0:
            return self.drop_action
        eps = self.epsilon if epsilon is None else epsilon
        if self.rng.random() < eps:
            return int(self.rng.choice(candidates))
        values = self.q_values(observation)
        return int(candidates[int(np.argmax(values[candidates]))])

    def act(self, observation: Observation) -> int:
        return self.select_action(observation, epsilon=0.0)

    def _adaptive_rates(
        self,
        observation: Observation,
        action: int,
    ) -> tuple[float, float]:
        features, valid_mask, _ = candidate_feature_matrix(
            observation,
            self.drop_action,
        )
        coverage = float(features[action, 8] + 0.5 * features[action, 7])
        coverage = float(np.clip(coverage, 0.0, 1.0))
        candidate_count = int(np.count_nonzero(valid_mask))
        beta = self.beta_min + (self.beta_max - self.beta_min) * (1.0 - coverage)
        gamma = self.gamma_min + (
            self.gamma_max - self.gamma_min
        ) * candidate_count / max(self.drop_action, 1)
        return float(np.clip(beta, self.beta_min, self.beta_max)), float(
            np.clip(gamma, self.gamma_min, self.gamma_max)
        )

    def _paper_reward(
        self,
        observation: Observation,
        action: int,
        *,
        delivered: bool,
        dropped: bool,
    ) -> float:
        features, valid_mask, _ = candidate_feature_matrix(
            observation,
            self.drop_action,
        )
        if np.count_nonzero(valid_mask) == 0 or action == self.drop_action:
            return 0.0
        if dropped:
            return 0.0
        row = features[action]
        collision_safety = row[10]
        l3_reception = 1.0 if delivered else 0.5 * (row[4] + row[5])
        l2_reception = row[4]
        coverage = max(row[7], row[8])
        residual_energy_proxy = row[6]
        return float(
            0.40 * collision_safety
            + 0.25 * l3_reception
            + 0.15 * l2_reception
            + 0.12 * coverage
            + 0.08 * residual_energy_proxy
        )

    def update(
        self,
        observation: Observation,
        action: int,
        reward: float,
        next_observation: Observation,
        done: bool,
        *,
        delivered: bool,
        dropped: bool,
    ) -> float:
        del reward
        if action == self.drop_action:
            self.eligibility.fill(0.0)
            return 0.0
        features, _, _ = candidate_feature_matrix(observation, self.drop_action)
        phi = features[action]
        current = float(phi @ self.weights)
        beta, gamma = self._adaptive_rates(observation, action)
        next_max = 0.0 if done else float(np.max(self.q_values(next_observation)))
        target = self._paper_reward(
            observation,
            action,
            delivered=delivered,
            dropped=dropped,
        ) + gamma * next_max
        delta = target - current
        self.eligibility = gamma * self.lambda_trace * self.eligibility + phi
        self.weights += beta * delta * self.eligibility
        self.weights = np.clip(self.weights, -25.0, 25.0)
        if done:
            self.eligibility.fill(0.0)
        return float(abs(delta))

    def state_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "lambda_trace": self.lambda_trace,
            "beta_min": self.beta_min,
            "beta_max": self.beta_max,
            "gamma_min": self.gamma_min,
            "gamma_max": self.gamma_max,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.weights = np.asarray(state["weights"], dtype=np.float32)
        self.eligibility = np.zeros_like(self.weights)
        self.lambda_trace = float(state["lambda_trace"])
        self.beta_min = float(state["beta_min"])
        self.beta_max = float(state["beta_max"])
        self.gamma_min = float(state["gamma_min"])
        self.gamma_max = float(state["gamma_max"])


class DramaGraphQNetwork(nn.Module):
    """Graph-DQN used by the DRAMA common-environment adaptation."""

    def __init__(
        self,
        feature_dim: int = 13,
        hidden_dim: int = 64,
        message_rounds: int = 2,
        attention_tau: float = 0.25,
    ) -> None:
        super().__init__()
        self.message_rounds = message_rounds
        self.attention_tau = attention_tau
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.message = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * (message_rounds + 1), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(features)
        history = [hidden]
        for _ in range(self.message_rounds):
            query = self.query(hidden)
            key = self.key(hidden)
            value = self.value(hidden)
            logits = (
                torch.matmul(query, key.transpose(-1, -2))
                * self.attention_tau
                / max(hidden.shape[-1] ** 0.5, 1.0)
            )
            logits = logits.masked_fill(~mask[:, None, :], -1e9)
            attention = torch.softmax(logits, dim=-1)
            attention = attention.masked_fill(~mask[:, :, None], 0.0)
            aggregated = torch.matmul(attention, value)
            hidden = self.message(aggregated)
            history.append(hidden)
        q_values = self.q_head(torch.cat(history, dim=-1)).squeeze(-1)
        return q_values.masked_fill(~mask, -1e9)


@dataclass(frozen=True)
class DramaTransition:
    features: NDArray[np.float32]
    mask: NDArray[np.bool_]
    action: int
    reward: float
    next_features: NDArray[np.float32]
    next_mask: NDArray[np.bool_]
    done: bool
    aux_targets: NDArray[np.float32]


class DramaPolicy:
    """DRAMA-style DQN policy with emergent local candidate communication."""

    method_name = "DRAMA"

    def __init__(
        self,
        drop_action: int,
        *,
        hidden_dim: int = 64,
        message_rounds: int = 2,
        device: torch.device | str = "cpu",
    ) -> None:
        self.drop_action = drop_action
        self.device = torch.device(device)
        self.model = DramaGraphQNetwork(
            hidden_dim=hidden_dim,
            message_rounds=message_rounds,
        ).to(self.device)
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def observation_bytes(self, observation: Observation) -> int:
        keys = (
            "neighbor_features",
            "edge_features",
            "packet_features",
            "action_mask",
            "candidate_forwardability",
            "candidate_risk_features",
        )
        return sum(
            int(observation[key].nbytes)
            for key in keys
            if key in observation
        )

    def q_values(self, observation: Observation) -> NDArray[np.float32]:
        features, mask, _ = candidate_feature_matrix(observation, self.drop_action)
        with torch.no_grad():
            values = self.model(
                torch.as_tensor(features[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(mask[None], dtype=torch.bool, device=self.device),
            )[0]
        return values.detach().cpu().numpy().astype(np.float32)

    def select_action(self, observation: Observation, epsilon: float = 0.0) -> int:
        candidates = _candidate_nodes(observation, self.drop_action)
        if candidates.size == 0:
            return self.drop_action
        if self.rng.random() < epsilon:
            return int(self.rng.choice(candidates))
        values = self.q_values(observation)
        return int(candidates[int(np.argmax(values[candidates]))])

    def act(self, observation: Observation) -> int:
        return self.select_action(observation, epsilon=0.0)

    def state_dict(self) -> dict[str, Any]:
        return {
            "model_state": self.model.state_dict(),
            "hidden_dim": self.model.encoder[0].out_features,
            "message_rounds": self.model.message_rounds,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        hidden_dim = int(state.get("hidden_dim", 64))
        rounds = int(state.get("message_rounds", 2))
        self.model = DramaGraphQNetwork(
            hidden_dim=hidden_dim,
            message_rounds=rounds,
        ).to(self.device)
        self.model.load_state_dict(state["model_state"])


@dataclass(frozen=True)
class ExternalTrainingResult:
    method: str
    training_seed: int
    episodes: int
    updates: int
    mean_training_td_error: float
    delivered: int
    dropped: int


def _auxiliary_targets(
    observation: Observation,
    drop_action: int,
) -> NDArray[np.float32]:
    features, valid_mask, _ = candidate_feature_matrix(observation, drop_action)
    targets = np.full(drop_action, 0.0, dtype=np.float32)
    targets[valid_mask] = (
        2.5 * features[valid_mask, 2]
        + 1.0 * features[valid_mask, 4]
        + 1.0 * features[valid_mask, 5]
        + 0.8 * features[valid_mask, 7]
        + 0.4 * features[valid_mask, 8]
        - 0.5 * (1.0 - features[valid_mask, 6])
    ).astype(np.float32)
    return targets


def train_value_baseline(
    env_factory,
    policy: EvoQGeoPolicy | IqmrPolicy,
    scenarios,
    *,
    training_seed: int,
    episodes_per_stage: int,
    epsilon_start: float = 0.35,
    epsilon_end: float = 0.03,
) -> ExternalTrainingResult:
    rng = np.random.default_rng(training_seed)
    errors: list[float] = []
    delivered = 0
    dropped = 0
    episodes = 0
    for scenario_index, scenario in enumerate(scenarios):
        env = env_factory(scenario.config)
        for episode_index in range(episodes_per_stage):
            global_index = scenario_index * episodes_per_stage + episode_index
            progress = global_index / max(
                len(scenarios) * episodes_per_stage - 1,
                1,
            )
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
            seed = int(training_seed + 2_000_000 + global_index)
            policy.reset(seed)
            observation, _ = env.reset(
                seed=int(rng.integers(0, 2**31 - 1)),
                options=scenario.reset_options,
            )
            done = False
            while not done:
                action = policy.select_action(observation, epsilon=epsilon)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                errors.append(
                    policy.update(
                        observation,
                        action,
                        float(reward),
                        next_observation,
                        done,
                        delivered=bool(info["delivered"]),
                        dropped=bool(info["dropped"]),
                    )
                )
                observation = next_observation
            delivered += int(info["delivered"])
            dropped += int(info["dropped"])
            episodes += 1
    return ExternalTrainingResult(
        method=policy.method_name,
        training_seed=training_seed,
        episodes=episodes,
        updates=len(errors),
        mean_training_td_error=float(np.mean(errors)) if errors else 0.0,
        delivered=delivered,
        dropped=dropped,
    )


def train_drama_baseline(
    env_factory,
    policy: DramaPolicy,
    scenarios,
    *,
    training_seed: int,
    episodes_per_stage: int,
    batch_size: int = 64,
    replay_capacity: int = 30_000,
    learning_rate: float = 1e-3,
    gamma: float = 0.99,
    target_tau: float = 0.01,
    auxiliary_coefficient: float = 0.35,
    epsilon_start: float = 0.50,
    epsilon_end: float = 0.05,
) -> ExternalTrainingResult:
    torch.manual_seed(training_seed)
    random.seed(training_seed)
    rng = np.random.default_rng(training_seed)
    model = policy.model
    target = copy.deepcopy(model).to(policy.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    replay: deque[DramaTransition] = deque(maxlen=replay_capacity)
    losses: list[float] = []
    delivered = 0
    dropped = 0
    episodes = 0
    updates = 0

    for scenario_index, scenario in enumerate(scenarios):
        env = env_factory(scenario.config)
        for episode_index in range(episodes_per_stage):
            global_index = scenario_index * episodes_per_stage + episode_index
            progress = global_index / max(
                len(scenarios) * episodes_per_stage - 1,
                1,
            )
            epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress
            policy.reset(int(training_seed + 3_000_000 + global_index))
            observation, _ = env.reset(
                seed=int(rng.integers(0, 2**31 - 1)),
                options=scenario.reset_options,
            )
            done = False
            while not done:
                features, mask, _ = candidate_feature_matrix(
                    observation,
                    policy.drop_action,
                )
                action = policy.select_action(observation, epsilon=epsilon)
                next_observation, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                next_features, next_mask, _ = candidate_feature_matrix(
                    next_observation,
                    policy.drop_action,
                )
                shaped_reward = float(reward) / 10.0
                if info["delivered"]:
                    shaped_reward += 1.0
                elif info["dropped"]:
                    shaped_reward -= 1.0
                if action != policy.drop_action:
                    replay.append(
                        DramaTransition(
                            features=features,
                            mask=mask,
                            action=int(action),
                            reward=shaped_reward,
                            next_features=next_features,
                            next_mask=next_mask,
                            done=done,
                            aux_targets=_auxiliary_targets(
                                observation,
                                policy.drop_action,
                            ),
                        )
                    )
                observation = next_observation
                if len(replay) < batch_size:
                    continue
                batch = random.sample(replay, batch_size)
                feature_batch = torch.as_tensor(
                    np.stack([item.features for item in batch]),
                    dtype=torch.float32,
                    device=policy.device,
                )
                mask_batch = torch.as_tensor(
                    np.stack([item.mask for item in batch]),
                    dtype=torch.bool,
                    device=policy.device,
                )
                actions = torch.as_tensor(
                    [item.action for item in batch],
                    dtype=torch.int64,
                    device=policy.device,
                )
                rewards = torch.as_tensor(
                    [item.reward for item in batch],
                    dtype=torch.float32,
                    device=policy.device,
                )
                next_features = torch.as_tensor(
                    np.stack([item.next_features for item in batch]),
                    dtype=torch.float32,
                    device=policy.device,
                )
                next_mask = torch.as_tensor(
                    np.stack([item.next_mask for item in batch]),
                    dtype=torch.bool,
                    device=policy.device,
                )
                done_batch = torch.as_tensor(
                    [item.done for item in batch],
                    dtype=torch.float32,
                    device=policy.device,
                )
                aux_targets = torch.as_tensor(
                    np.stack([item.aux_targets for item in batch]),
                    dtype=torch.float32,
                    device=policy.device,
                )
                q_values = model(feature_batch, mask_batch)
                selected_q = q_values.gather(1, actions[:, None]).squeeze(1)
                with torch.no_grad():
                    next_all = target(next_features, next_mask)
                    has_next = next_mask.any(dim=1)
                    next_q = torch.where(
                        has_next,
                        next_all.max(dim=1).values,
                        torch.zeros_like(rewards),
                    )
                    td_target = rewards + (1.0 - done_batch) * gamma * next_q
                td_loss = F.smooth_l1_loss(selected_q, td_target)
                valid_aux = mask_batch
                aux_loss = F.mse_loss(q_values[valid_aux], aux_targets[valid_aux])
                loss = td_loss + auxiliary_coefficient * aux_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                with torch.no_grad():
                    for target_param, source_param in zip(
                        target.parameters(),
                        model.parameters(),
                    ):
                        target_param.mul_(1.0 - target_tau).add_(
                            source_param,
                            alpha=target_tau,
                        )
                losses.append(float(td_loss.detach().cpu()))
                updates += 1
            delivered += int(info["delivered"])
            dropped += int(info["dropped"])
            episodes += 1
    return ExternalTrainingResult(
        method=policy.method_name,
        training_seed=training_seed,
        episodes=episodes,
        updates=updates,
        mean_training_td_error=float(np.mean(losses)) if losses else 0.0,
        delivered=delivered,
        dropped=dropped,
    )
