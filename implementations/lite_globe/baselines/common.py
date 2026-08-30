"""Shared contracts and deterministic utilities for external baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray


Observation = dict[str, NDArray[np.generic]]


@dataclass(frozen=True)
class ProtocolSnapshot:
    """Physical connectivity used only to deliver simulated control messages."""

    adjacency: NDArray[np.bool_]
    positions: NDArray[np.float32]
    step: int
    current_node: int
    destination: int


@dataclass(frozen=True)
class ObservationContract:
    fields: tuple[str, ...]
    hop_radius: str
    privileged_information: bool
    control_plane: bool


class ExternalPolicy(Protocol):
    def reset(self, seed: int | None = None) -> None: ...
    def act(self, observation: Observation) -> int: ...


class ControlPlanePolicy(Protocol):
    def protocol_tick(self, snapshot: ProtocolSnapshot) -> None: ...


def snapshot_from_env(env: Any) -> ProtocolSnapshot:
    """Copy topology for message delivery without exposing the env object."""

    return ProtocolSnapshot(
        adjacency=np.asarray(env.adjacency, dtype=np.bool_).copy(),
        positions=np.asarray(env.mobility.positions, dtype=np.float32).copy(),
        step=int(env.episode_step),
        current_node=int(env.packet.current),
        destination=int(env.packet.destination),
    )


def valid_candidates(observation: Observation, drop_action: int) -> NDArray[np.int64]:
    return np.flatnonzero(observation["action_mask"][:drop_action]).astype(np.int64)


def filtered_observation_bytes(observation: Observation, fields: tuple[str, ...]) -> int:
    return int(sum(observation[name].nbytes for name in fields if name in observation))


def atomic_torch_save(payload: Any, path: Path) -> None:
    """Write a checkpoint atomically so interrupted files are never accepted."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_json_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly copy used in checkpoint manifests."""

    return {
        key: asdict(value) if hasattr(value, "__dataclass_fields__") else value
        for key, value in metadata.items()
    }


@dataclass
class ReplayTransition:
    state: dict[str, NDArray[np.generic]]
    action: int
    reward: float
    next_state: dict[str, NDArray[np.generic]]
    done: bool
    discount: float
    tier: int = 1


class PrioritizedReplay:
    """Small deterministic proportional replay with explicit HERP tiers."""

    def __init__(self, capacity: int, *, alpha: float = 0.6) -> None:
        if capacity <= 0 or not 0.0 <= alpha <= 1.0:
            raise ValueError("invalid replay configuration")
        self.capacity = capacity
        self.alpha = alpha
        self.transitions: list[ReplayTransition] = []
        self.priorities: list[float] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, transition: ReplayTransition, priority: float | None = None) -> None:
        base = max(self.priorities, default=1.0) if priority is None else float(priority)
        tier_weight = (1.0, 2.0, 4.0)[max(0, min(2, transition.tier))]
        value = max(abs(base) * tier_weight, 1e-6)
        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
            self.priorities.append(value)
        else:
            self.transitions[self.position] = transition
            self.priorities[self.position] = value
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, rng: np.random.Generator) -> tuple[list[ReplayTransition], NDArray[np.int64]]:
        if not self.transitions:
            raise ValueError("cannot sample empty replay")
        probabilities = np.asarray(self.priorities, dtype=np.float64) ** self.alpha
        probabilities /= probabilities.sum()
        size = min(batch_size, len(self.transitions))
        indices = rng.choice(len(self.transitions), size=size, replace=False, p=probabilities)
        return [self.transitions[int(i)] for i in indices], indices.astype(np.int64)

    def update_priorities(self, indices: NDArray[np.int64], priorities: NDArray[np.float32]) -> None:
        for index, priority in zip(indices, priorities, strict=True):
            self.priorities[int(index)] = max(float(abs(priority)), 1e-6)

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "transitions": self.transitions,
            "priorities": self.priorities,
            "position": self.position,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.transitions = list(state["transitions"])
        self.priorities = [float(value) for value in state["priorities"]]
        self.position = int(state["position"])
