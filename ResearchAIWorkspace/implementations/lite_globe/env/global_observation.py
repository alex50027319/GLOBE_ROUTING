"""Privileged global graph observation used only by the Teacher."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import FanetConfig
from .packet import PacketState


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]

GLOBAL_NODE_FEATURES = 9
GLOBAL_EDGE_FEATURES = 2
GLOBAL_PACKET_FEATURES = 2


def build_global_observation(
    *,
    config: FanetConfig,
    positions: FloatArray,
    velocities: FloatArray,
    queues: FloatArray,
    adjacency: BoolArray,
    distances: FloatArray,
    packet: PacketState,
) -> dict[str, NDArray[np.generic]]:
    """Build a padded full-graph observation and local next-hop support."""

    nodes = np.zeros(
        (config.max_nodes, GLOBAL_NODE_FEATURES), dtype=np.float32
    )
    max_speed = max(config.max_speed, 1e-6)
    active = np.zeros(config.max_nodes, dtype=np.int8)
    active[: config.num_nodes] = 1
    visited = np.zeros(config.max_nodes, dtype=np.float32)
    visited[packet.path] = 1.0
    nodes[: config.num_nodes] = np.column_stack(
        (
            positions[:, 0] / config.area_size,
            positions[:, 1] / config.area_size,
            velocities[:, 0] / max_speed,
            velocities[:, 1] / max_speed,
            queues / config.max_queue_size,
            np.arange(config.num_nodes) == packet.current,
            np.arange(config.num_nodes) == packet.destination,
            np.arange(config.num_nodes) == packet.source,
            visited[: config.num_nodes],
        )
    ).astype(np.float32)

    graph = np.zeros((config.max_nodes, config.max_nodes), dtype=np.int8)
    graph[: config.num_nodes, : config.num_nodes] = adjacency.astype(np.int8)
    edges = np.zeros(
        (config.max_nodes, config.max_nodes, GLOBAL_EDGE_FEATURES),
        dtype=np.float32,
    )
    radius = max(config.communication_radius, 1e-6)
    edges[: config.num_nodes, : config.num_nodes, 0] = distances / radius
    edges[: config.num_nodes, : config.num_nodes, 1] = adjacency.astype(
        np.float32
    )

    action_mask = np.zeros(config.max_nodes + 1, dtype=np.int8)
    action_mask[: config.num_nodes] = adjacency[packet.current].astype(np.int8)
    if config.mask_visited_actions:
        for node in packet.path[:-1]:
            action_mask[node] = 0
    action_mask[config.max_nodes] = 1
    packet_features = np.array(
        [
            packet.ttl_remaining / config.packet_ttl,
            packet.hop_count / max(config.packet_ttl, 1),
        ],
        dtype=np.float32,
    )
    observation: dict[str, NDArray[np.generic]] = {
        "node_features": nodes,
        "adjacency": graph,
        "edge_features": edges,
        "node_mask": active,
        "packet_features": packet_features,
        "action_mask": action_mask,
    }
    validate_global_observation(observation, config)
    return observation


def validate_global_observation(
    observation: dict[str, NDArray[np.generic]],
    config: FanetConfig,
) -> None:
    expected = {
        "node_features": (config.max_nodes, GLOBAL_NODE_FEATURES),
        "adjacency": (config.max_nodes, config.max_nodes),
        "edge_features": (
            config.max_nodes,
            config.max_nodes,
            GLOBAL_EDGE_FEATURES,
        ),
        "node_mask": (config.max_nodes,),
        "packet_features": (GLOBAL_PACKET_FEATURES,),
        "action_mask": (config.max_nodes + 1,),
    }
    for key, shape in expected.items():
        if observation[key].shape != shape:
            raise ValueError(f"{key} has shape {observation[key].shape}, expected {shape}")
        if not np.all(np.isfinite(observation[key])):
            raise ValueError(f"{key} contains non-finite values")
    if not np.array_equal(
        observation["adjacency"], observation["adjacency"].T
    ):
        raise ValueError("global adjacency must be symmetric")
    if observation["action_mask"][config.max_nodes] != 1:
        raise ValueError("global action support must include DROP")
