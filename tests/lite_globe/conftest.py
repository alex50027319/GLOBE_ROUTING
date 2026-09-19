"""Shared deterministic Lite-GLOBE test fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from implementations.lite_globe.env.config import FanetConfig
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv


@pytest.fixture
def line_config() -> FanetConfig:
    return FanetConfig(
        num_nodes=3,
        max_nodes=4,
        area_size=10.0,
        communication_radius=1.1,
        max_episode_steps=8,
        packet_ttl=4,
        max_queue_size=4,
        min_speed=0.0,
        max_speed=0.0,
        seed=7,
    )


@pytest.fixture
def line_positions() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=np.float32)


@pytest.fixture
def line_env(line_config: FanetConfig) -> FanetRoutingEnv:
    return FanetRoutingEnv(line_config)
