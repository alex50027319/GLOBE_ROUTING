"""Privileged multi-objective shortest-path reference for Phase 9."""

from __future__ import annotations

from dataclasses import dataclass
import heapq

import numpy as np
from numpy.typing import NDArray

from ..env.fanet_env import FanetRoutingEnv
from ..env.link_model import predicted_link_lifetime_steps


@dataclass(frozen=True)
class RiskCostWeights:
    """Pre-registered edge-cost weights with hop count as the anchor."""

    hop: float = 1.0
    instability: float = 4.0
    energy: float = 0.35
    queue: float = 0.35

    def __post_init__(self) -> None:
        if min(
            self.hop,
            self.instability,
            self.energy,
            self.queue,
        ) < 0:
            raise ValueError("risk cost weights must be non-negative")
        if self.hop <= 0:
            raise ValueError("hop cost must be positive")


def risk_aware_shortest_path(
    env: FanetRoutingEnv,
    *,
    weights: RiskCostWeights | None = None,
) -> list[int] | None:
    """Run Dijkstra on the current graph using stability-aware edge costs."""

    weights = weights or RiskCostWeights()
    source = env.packet.current
    destination = env.packet.destination
    if source == destination:
        return [source]
    adjacency = env.adjacency.copy()
    for visited in env.packet.path[:-1]:
        if visited != destination:
            adjacency[visited, :] = False
            adjacency[:, visited] = False

    node_count = adjacency.shape[0]
    costs = np.full(node_count, np.inf, dtype=np.float64)
    parents = np.full(node_count, -1, dtype=np.int64)
    costs[source] = 0.0
    queue: list[tuple[float, int]] = [(0.0, source)]
    while queue:
        cost, current = heapq.heappop(queue)
        if cost > costs[current]:
            continue
        if current == destination:
            break
        for neighbor_value in np.flatnonzero(adjacency[current]):
            neighbor = int(neighbor_value)
            distance_ratio = float(
                env.distances[current, neighbor]
                / env.config.communication_radius
            )
            lifetime = predicted_link_lifetime_steps(
                env.mobility.positions[neighbor]
                - env.mobility.positions[current],
                env.mobility.velocities[neighbor]
                - env.mobility.velocities[current],
                communication_radius=env.config.communication_radius,
                time_step=env.config.time_step,
                horizon_steps=env.config.max_episode_steps,
            )
            instability = 1.0 - lifetime / max(
                env.config.max_episode_steps, 1
            )
            queue_occupancy = float(
                env.queues[neighbor]
                / max(env.config.max_queue_size, 1)
            )
            edge_cost = (
                weights.hop
                + weights.instability * instability
                + weights.energy * distance_ratio**2
                + weights.queue * queue_occupancy
            )
            candidate = cost + edge_cost
            if candidate + 1e-12 < costs[neighbor]:
                costs[neighbor] = candidate
                parents[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))
    if not np.isfinite(costs[destination]):
        return None
    path = [destination]
    while path[-1] != source:
        parent = int(parents[path[-1]])
        if parent < 0:
            return None
        path.append(parent)
    return list(reversed(path))


class RiskAwareOraclePolicy:
    """Choose the next hop of a privileged stability-aware Dijkstra path."""

    def __init__(
        self,
        env: FanetRoutingEnv,
        *,
        weights: RiskCostWeights | None = None,
    ) -> None:
        self.env = env
        self.weights = weights or RiskCostWeights()

    def reset(self, seed: int | None = None) -> None:
        del seed

    def observation_bytes(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> int:
        del observation
        return int(
            self.env.adjacency.nbytes
            + self.env.distances.nbytes
            + self.env.mobility.positions.nbytes
            + self.env.mobility.velocities.nbytes
            + self.env.queues.nbytes
        )

    def act(self, observation: dict[str, NDArray[np.generic]]) -> int:
        del observation
        path = risk_aware_shortest_path(self.env, weights=self.weights)
        if path is None or len(path) < 2:
            return self.env.drop_action
        return path[1]
