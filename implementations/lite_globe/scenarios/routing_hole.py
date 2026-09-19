"""A fixed topology where greedy geographic forwarding reaches a dead end."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..env.config import FanetConfig


def routing_hole_config(seed: int = 42) -> FanetConfig:
    return FanetConfig(
        num_nodes=6,
        max_nodes=6,
        area_size=10.0,
        communication_radius=1.85,
        max_episode_steps=8,
        packet_ttl=8,
        max_queue_size=4,
        min_speed=0.0,
        max_speed=0.0,
        seed=seed,
    )


def routing_hole_options() -> dict[str, Any]:
    """Return source 0, destination 5, one greedy trap, and one valid detour."""

    positions = np.array(
        [
            [0.0, 0.5],
            [1.4, 0.0],
            [0.0, 1.9],
            [1.4, 1.9],
            [2.8, 1.9],
            [4.0, 0.5],
        ],
        dtype=np.float32,
    )
    return {"positions": positions, "source": 0, "destination": 5}
