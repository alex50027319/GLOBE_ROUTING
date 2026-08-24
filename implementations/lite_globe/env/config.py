"""Typed configuration for the Lite-GLOBE FANET environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FanetConfig:
    """Environment settings with conservative Phase 1 defaults."""

    num_nodes: int = 12
    max_nodes: int = 32
    area_size: float = 1000.0
    communication_radius: float = 350.0
    max_episode_steps: int = 64
    packet_ttl: int = 20
    max_queue_size: int = 16
    stochastic_link_loss: float = 0.0
    min_speed: float = 2.0
    max_speed: float = 12.0
    time_step: float = 1.0
    waypoint_tolerance: float = 1.0
    reward_delivery: float = 10.0
    reward_delay: float = 0.1
    reward_failure: float = 5.0
    reward_progress: float = 0.0
    include_node_ids: bool = True
    mask_visited_actions: bool = False
    include_forwardability: bool = False
    include_risk_features: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        if not 2 <= self.num_nodes <= self.max_nodes:
            raise ValueError("num_nodes must be between 2 and max_nodes")
        if self.area_size <= 0 or self.communication_radius <= 0:
            raise ValueError("area_size and communication_radius must be positive")
        if not 0.0 <= self.stochastic_link_loss < 1.0:
            raise ValueError("stochastic_link_loss must be in [0, 1)")
        if self.min_speed < 0 or self.max_speed < self.min_speed:
            raise ValueError("mobility speed range is invalid")
        if self.packet_ttl <= 0 or self.max_episode_steps <= 0:
            raise ValueError("packet_ttl and max_episode_steps must be positive")
        if self.reward_progress < 0:
            raise ValueError("reward_progress must be non-negative")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FanetConfig":
        """Load the nested project YAML format into a flat typed config."""

        raw: dict[str, Any] = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        env = raw.get("environment", {})
        mobility = raw.get("mobility", {})
        reward = raw.get("reward", {})
        runtime = raw.get("runtime", {})
        return cls(
            num_nodes=int(env.get("num_nodes", cls.num_nodes)),
            max_nodes=int(env.get("max_nodes", cls.max_nodes)),
            area_size=float(env.get("area_size", cls.area_size)),
            communication_radius=float(
                env.get("communication_radius", cls.communication_radius)
            ),
            max_episode_steps=int(
                env.get("max_episode_steps", cls.max_episode_steps)
            ),
            packet_ttl=int(env.get("packet_ttl", cls.packet_ttl)),
            max_queue_size=int(env.get("max_queue_size", cls.max_queue_size)),
            stochastic_link_loss=float(
                env.get("stochastic_link_loss", cls.stochastic_link_loss)
            ),
            min_speed=float(mobility.get("min_speed", cls.min_speed)),
            max_speed=float(mobility.get("max_speed", cls.max_speed)),
            time_step=float(mobility.get("time_step", cls.time_step)),
            waypoint_tolerance=float(
                mobility.get("waypoint_tolerance", cls.waypoint_tolerance)
            ),
            reward_delivery=float(reward.get("delivery", cls.reward_delivery)),
            reward_delay=float(reward.get("delay", cls.reward_delay)),
            reward_failure=float(reward.get("failure", cls.reward_failure)),
            reward_progress=float(reward.get("progress", cls.reward_progress)),
            include_node_ids=bool(
                env.get("include_node_ids", cls.include_node_ids)
            ),
            mask_visited_actions=bool(
                env.get("mask_visited_actions", cls.mask_visited_actions)
            ),
            include_forwardability=bool(
                env.get("include_forwardability", cls.include_forwardability)
            ),
            include_risk_features=bool(
                env.get("include_risk_features", cls.include_risk_features)
            ),
            seed=int(runtime.get("seed", cls.seed)),
        )
