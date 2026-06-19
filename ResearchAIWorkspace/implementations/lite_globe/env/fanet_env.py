"""Gymnasium-compatible Phase 1 FANET packet-routing environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .config import FanetConfig
from .global_observation import build_global_observation
from .graph_utils import connected_pairs, shortest_path
from .link_model import RadiusLinkModel
from .mobility import MobilityState, RandomWaypointMobility
from .observation import (
    EDGE_FEATURES,
    FORWARDABILITY_FEATURES,
    NEIGHBOR_FEATURES,
    PACKET_FEATURES,
    RISK_FEATURES,
    SELF_FEATURES,
    build_observation,
)
from .packet import PacketState
from .reward import RoutingReward


class FanetRoutingEnv(gym.Env[dict[str, NDArray[np.generic]], int]):
    """Route one packet through a mobile 2D FANET using one-hop actions."""

    metadata = {"render_modes": []}

    def __init__(self, config: FanetConfig | None = None) -> None:
        super().__init__()
        self.config = config or FanetConfig()
        self.drop_action = self.config.max_nodes
        self.action_space = spaces.Discrete(self.config.max_nodes + 1)
        self.observation_space = spaces.Dict(
            {
                "self_features": spaces.Box(
                    -1.0, 1.0, shape=(SELF_FEATURES,), dtype=np.float32
                ),
                "neighbor_features": spaces.Box(
                    -1.0,
                    1.0,
                    shape=(self.config.max_nodes, NEIGHBOR_FEATURES),
                    dtype=np.float32,
                ),
                "edge_features": spaces.Box(
                    0.0,
                    np.finfo(np.float32).max,
                    shape=(self.config.max_nodes, EDGE_FEATURES),
                    dtype=np.float32,
                ),
                "packet_features": spaces.Box(
                    -1.0, 1.0, shape=(PACKET_FEATURES,), dtype=np.float32
                ),
                "action_mask": spaces.MultiBinary(self.config.max_nodes + 1),
            }
        )
        if self.config.include_forwardability:
            self.observation_space.spaces["candidate_forwardability"] = (
                spaces.Box(
                    0.0,
                    1.0,
                    shape=(
                        self.config.max_nodes,
                        FORWARDABILITY_FEATURES,
                    ),
                    dtype=np.float32,
                )
            )
        if self.config.include_risk_features:
            self.observation_space.spaces["candidate_risk_features"] = (
                spaces.Box(
                    0.0,
                    1.0,
                    shape=(self.config.max_nodes, RISK_FEATURES),
                    dtype=np.float32,
                )
            )
        self.mobility_model = RandomWaypointMobility(
            self.config.num_nodes,
            self.config.area_size,
            self.config.min_speed,
            self.config.max_speed,
            self.config.time_step,
            self.config.waypoint_tolerance,
        )
        self.link_model = RadiusLinkModel(
            self.config.communication_radius,
            self.config.stochastic_link_loss,
        )
        self.reward_model = RoutingReward(
            self.config.reward_delivery,
            self.config.reward_delay,
            self.config.reward_failure,
            self.config.reward_progress,
        )
        self.mobility: MobilityState
        self.adjacency: NDArray[np.bool_]
        self.distances: NDArray[np.float32]
        self.queues: NDArray[np.float32]
        self.packet: PacketState
        self.episode_step = 0
        self.initially_connected = False
        self.initial_shortest_hops: int | None = None
        self.transmission_attempts = 0
        self.cumulative_link_distance = 0.0
        self.transmission_energy_proxy = 0.0
        self.minimum_link_lifetime_steps: float | None = None
        self.cumulative_queue_delay_proxy = 0.0
        self.minimum_link_margin: float | None = None
        self._fixed_velocity_mode = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any]]:
        """Reset with optional fixed positions and source/destination for tests."""

        super().reset(seed=self.config.seed if seed is None else seed)
        require_connected = bool(options and options.get("require_connected"))
        min_hops = int(options.get("min_shortest_hops", 1)) if options else 1
        max_topology_attempts = (
            int(options.get("max_topology_attempts", 100))
            if options
            else 100
        )
        fixed_positions = bool(options and "positions" in options)
        self._fixed_velocity_mode = bool(
            options and "velocities" in options
        )
        endpoint_candidates: list[tuple[int, int, int]] = []
        for _ in range(max_topology_attempts):
            self.mobility = self.mobility_model.reset(self.np_random)
            if fixed_positions:
                positions = np.asarray(options["positions"], dtype=np.float32)
                if positions.shape != (self.config.num_nodes, 2):
                    raise ValueError(
                        "fixed positions must have shape (num_nodes, 2)"
                    )
                self.mobility.positions = positions.copy()
                if self._fixed_velocity_mode:
                    velocities = np.asarray(
                        options["velocities"], dtype=np.float32
                    )
                    if velocities.shape != positions.shape:
                        raise ValueError(
                            "fixed velocities must match positions"
                        )
                    self.mobility.velocities = velocities.copy()
                else:
                    self.mobility.velocities = np.zeros_like(positions)
            self._refresh_links()
            if not require_connected:
                break
            endpoint_candidates = connected_pairs(
                self.adjacency, min_hops=min_hops
            )
            if endpoint_candidates:
                break
            if fixed_positions:
                break
        if require_connected and not endpoint_candidates:
            raise RuntimeError(
                "could not sample a connected endpoint pair for this topology"
            )
        self.queues = self.np_random.integers(
            0,
            self.config.max_queue_size + 1,
            size=self.config.num_nodes,
        ).astype(np.float32)
        source, destination = self._sample_endpoints(
            options,
            endpoint_candidates=endpoint_candidates,
        )
        self.packet = PacketState.create(source, destination, self.config.packet_ttl)
        self.episode_step = 0
        initial_path = shortest_path(self.adjacency, source, destination)
        self.initially_connected = initial_path is not None
        self.initial_shortest_hops = (
            len(initial_path) - 1 if initial_path is not None else None
        )
        self.transmission_attempts = 0
        self.cumulative_link_distance = 0.0
        self.transmission_energy_proxy = 0.0
        self.minimum_link_lifetime_steps = None
        self.cumulative_queue_delay_proxy = 0.0
        self.minimum_link_margin = None
        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[
        dict[str, NDArray[np.generic]], float, bool, bool, dict[str, Any]
    ]:
        """Apply a next-hop node id or the explicit DROP action."""

        if not self.action_space.contains(action):
            raise ValueError(f"action {action} is outside the action space")
        if self.packet.delivered or self.packet.dropped:
            raise RuntimeError("step called after episode termination")
        self.episode_step += 1
        previous_distance = float(
            np.linalg.norm(
                self.mobility.positions[self.packet.destination]
                - self.mobility.positions[self.packet.current]
            )
        )
        delivered = False
        failed = False
        truncated = False
        mask = self._observation()["action_mask"]
        if action == self.drop_action:
            self._drop("agent_drop")
            failed = True
        elif mask[action] == 0:
            self._drop("invalid_action")
            failed = True
        else:
            self._record_transmission(int(action))
            if action in self.packet.path:
                self.packet.path.append(action)
                self.packet.ttl_remaining -= 1
                self._drop("routing_loop")
                failed = True
            else:
                self.packet.current = int(action)
                self.packet.path.append(int(action))
                self.packet.ttl_remaining -= 1
            if failed:
                pass
            elif self.packet.current == self.packet.destination:
                self.packet.delivered = True
                delivered = True
            elif self.packet.ttl_remaining <= 0:
                self._drop("ttl_expired")
                failed = True
        terminated = self.packet.delivered or self.packet.dropped
        if not terminated and self.episode_step >= self.config.max_episode_steps:
            self._drop("time_limit")
            failed = True
            truncated = True
        next_distance = float(
            np.linalg.norm(
                self.mobility.positions[self.packet.destination]
                - self.mobility.positions[self.packet.current]
            )
        )
        reward = self.reward_model.calculate(
            delivered=delivered,
            failed=failed,
            normalized_progress=(
                (previous_distance - next_distance) / self.config.area_size
            ),
        )
        if not terminated and not truncated:
            if self._fixed_velocity_mode:
                self.mobility.positions = np.clip(
                    self.mobility.positions
                    + self.mobility.velocities * self.config.time_step,
                    0.0,
                    self.config.area_size,
                ).astype(np.float32)
            else:
                self.mobility = self.mobility_model.step(
                    self.mobility, self.np_random
                )
            self._refresh_links()
        return self._observation(), reward, terminated, truncated, self._info()

    def _sample_endpoints(
        self,
        options: dict[str, Any] | None,
        *,
        endpoint_candidates: list[tuple[int, int, int]],
    ) -> tuple[int, int]:
        if endpoint_candidates:
            index = int(self.np_random.integers(len(endpoint_candidates)))
            source, destination, _ = endpoint_candidates[index]
            return source, destination
        source = int(options["source"]) if options and "source" in options else -1
        destination = (
            int(options["destination"])
            if options and "destination" in options
            else -1
        )
        if source < 0:
            source = int(self.np_random.integers(self.config.num_nodes))
        if destination < 0:
            choices = [node for node in range(self.config.num_nodes) if node != source]
            destination = int(self.np_random.choice(choices))
        if not 0 <= source < self.config.num_nodes:
            raise ValueError("source is outside the active node range")
        if not 0 <= destination < self.config.num_nodes or destination == source:
            raise ValueError("destination must be a different active node")
        return source, destination

    def _refresh_links(self) -> None:
        self.adjacency, self.distances = self.link_model.sample(
            self.mobility.positions, self.np_random
        )

    def _observation(self) -> dict[str, NDArray[np.generic]]:
        return build_observation(
            config=self.config,
            positions=self.mobility.positions,
            velocities=self.mobility.velocities,
            queues=self.queues,
            adjacency=self.adjacency,
            distances=self.distances,
            packet=self.packet,
        )

    def global_observation(self) -> dict[str, NDArray[np.generic]]:
        """Return privileged full-graph state for training the Teacher only."""

        return build_global_observation(
            config=self.config,
            positions=self.mobility.positions,
            velocities=self.mobility.velocities,
            queues=self.queues,
            adjacency=self.adjacency,
            distances=self.distances,
            packet=self.packet,
        )

    def _drop(self, reason: str) -> None:
        self.packet.dropped = True
        self.packet.drop_reason = reason

    def _record_transmission(self, next_hop: int) -> None:
        current = self.packet.current
        distance = float(self.distances[current, next_hop])
        normalized_distance = distance / self.config.communication_radius
        self.transmission_attempts += 1
        self.cumulative_link_distance += distance
        self.transmission_energy_proxy += normalized_distance**2
        self.cumulative_queue_delay_proxy += (
            1.0
            + float(self.queues[next_hop])
            / max(self.config.max_queue_size, 1)
        )
        margin = max(0.0, 1.0 - normalized_distance)
        if self.minimum_link_margin is None:
            self.minimum_link_margin = margin
        else:
            self.minimum_link_margin = min(
                self.minimum_link_margin, margin
            )
        lifetime = self._predicted_link_lifetime_steps(current, next_hop)
        if self.minimum_link_lifetime_steps is None:
            self.minimum_link_lifetime_steps = lifetime
        else:
            self.minimum_link_lifetime_steps = min(
                self.minimum_link_lifetime_steps, lifetime
            )

    def _predicted_link_lifetime_steps(
        self, current: int, next_hop: int
    ) -> float:
        from .link_model import predicted_link_lifetime_steps

        return predicted_link_lifetime_steps(
            self.mobility.positions[next_hop]
            - self.mobility.positions[current],
            self.mobility.velocities[next_hop]
            - self.mobility.velocities[current],
            communication_radius=self.config.communication_radius,
            time_step=self.config.time_step,
            horizon_steps=self.config.max_episode_steps,
        )

    def _info(self) -> dict[str, Any]:
        return {
            "source": self.packet.source,
            "destination": self.packet.destination,
            "current_node": self.packet.current,
            "path": tuple(self.packet.path),
            "hop_count": self.packet.hop_count,
            "ttl_remaining": self.packet.ttl_remaining,
            "delivered": self.packet.delivered,
            "dropped": self.packet.dropped,
            "drop_reason": self.packet.drop_reason,
            "episode_step": self.episode_step,
            "initially_connected": self.initially_connected,
            "initial_shortest_hops": self.initial_shortest_hops,
            "transmission_attempts": self.transmission_attempts,
            "cumulative_link_distance": self.cumulative_link_distance,
            "transmission_energy_proxy": self.transmission_energy_proxy,
            "minimum_link_lifetime_steps": self.minimum_link_lifetime_steps,
            "cumulative_queue_delay_proxy": (
                self.cumulative_queue_delay_proxy
            ),
            "minimum_link_margin": self.minimum_link_margin,
        }
