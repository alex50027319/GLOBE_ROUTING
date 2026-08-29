"""Construction and validation of local 1-hop routing observations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import FanetConfig
from .link_model import predicted_link_lifetime_steps
from .packet import PacketState


FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]

SELF_FEATURES = 6
NEIGHBOR_FEATURES = 7
EDGE_FEATURES = 2
PACKET_FEATURES = 6
FORWARDABILITY_FEATURES = 2
RISK_FEATURES = 4


def build_observation(
    *,
    config: FanetConfig,
    positions: FloatArray,
    velocities: FloatArray,
    queues: FloatArray,
    adjacency: BoolArray,
    distances: FloatArray,
    packet: PacketState,
) -> dict[str, NDArray[np.generic]]:
    """Build a padded local observation indexed by global node id."""

    current = packet.current
    destination = packet.destination
    max_speed = max(config.max_speed, 1e-6)
    self_features = np.array(
        [
            positions[current, 0] / config.area_size,
            positions[current, 1] / config.area_size,
            velocities[current, 0] / max_speed,
            velocities[current, 1] / max_speed,
            queues[current] / config.max_queue_size,
            float(current == destination),
        ],
        dtype=np.float32,
    )
    neighbor_features = np.zeros(
        (config.max_nodes, NEIGHBOR_FEATURES), dtype=np.float32
    )
    edge_features = np.zeros(
        (config.max_nodes, EDGE_FEATURES), dtype=np.float32
    )
    candidate_forwardability = np.zeros(
        (config.max_nodes, FORWARDABILITY_FEATURES), dtype=np.float32
    )
    candidate_risk_features = np.zeros(
        (config.max_nodes, RISK_FEATURES), dtype=np.float32
    )
    action_mask = np.zeros(config.max_nodes + 1, dtype=np.int8)
    valid = np.flatnonzero(adjacency[current])
    for node in valid:
        relative = (positions[node] - positions[current]) / config.area_size
        neighbor_features[node] = np.array(
            [
                relative[0],
                relative[1],
                velocities[node, 0] / max_speed,
                velocities[node, 1] / max_speed,
                queues[node] / config.max_queue_size,
                float(node == destination),
                1.0,
            ],
            dtype=np.float32,
        )
        edge_features[node] = np.array(
            [
                distances[current, node] / config.communication_radius,
                1.0,
            ],
            dtype=np.float32,
        )
        onward = adjacency[node].copy()
        onward[current] = False
        for visited in packet.path:
            onward[visited] = False
        onward_count = int(np.count_nonzero(onward))
        candidate_forwardability[node] = np.array(
            [
                float(node == destination or onward_count > 0),
                onward_count / max(config.num_nodes - 1, 1),
            ],
            dtype=np.float32,
        )
        lifetime = predicted_link_lifetime_steps(
            positions[node] - positions[current],
            velocities[node] - velocities[current],
            communication_radius=config.communication_radius,
            time_step=config.time_step,
            horizon_steps=config.max_episode_steps,
        )
        onward_lifetimes = []
        for onward_node in np.flatnonzero(onward):
            onward_lifetimes.append(
                predicted_link_lifetime_steps(
                    positions[int(onward_node)] - positions[node],
                    velocities[int(onward_node)] - velocities[node],
                    communication_radius=config.communication_radius,
                    time_step=config.time_step,
                    horizon_steps=config.max_episode_steps,
                )
            )
        best_onward_lifetime = (
            max(onward_lifetimes)
            if onward_lifetimes
            else (
                float(config.max_episode_steps)
                if node == destination
                else 0.0
            )
        )
        candidate_risk_features[node] = np.array(
            [
                np.clip(
                    1.0
                    - distances[current, node]
                    / config.communication_radius,
                    0.0,
                    1.0,
                ),
                np.clip(
                    lifetime / max(config.max_episode_steps, 1),
                    0.0,
                    1.0,
                ),
                np.clip(
                    1.0 - queues[node] / max(config.max_queue_size, 1),
                    0.0,
                    1.0,
                ),
                np.clip(
                    best_onward_lifetime
                    / max(config.max_episode_steps, 1),
                    0.0,
                    1.0,
                ),
            ],
            dtype=np.float32,
        )
        action_mask[node] = 1
    if config.mask_visited_actions:
        for node in packet.path[:-1]:
            action_mask[node] = 0
    action_mask[config.max_nodes] = 1  # Explicit DROP action.
    destination_delta = (
        positions[destination] - positions[current]
    ) / config.area_size
    packet_features = np.array(
        [
            destination_delta[0],
            destination_delta[1],
            packet.ttl_remaining / config.packet_ttl,
            packet.hop_count / max(config.packet_ttl, 1),
            (
                packet.source / max(config.num_nodes - 1, 1)
                if config.include_node_ids
                else 0.0
            ),
            (
                destination / max(config.num_nodes - 1, 1)
                if config.include_node_ids
                else 0.0
            ),
        ],
        dtype=np.float32,
    )
    observation: dict[str, NDArray[np.generic]] = {
        "self_features": self_features,
        "neighbor_features": neighbor_features,
        "edge_features": edge_features,
        "packet_features": packet_features,
        "action_mask": action_mask,
    }
    if config.include_forwardability:
        observation["candidate_forwardability"] = candidate_forwardability
    if config.include_risk_features:
        observation["candidate_risk_features"] = candidate_risk_features
    validate_observation(observation, config)
    return observation


def validate_observation(
    observation: dict[str, NDArray[np.generic]], config: FanetConfig
) -> None:
    """Fail loudly when feature shapes, dtypes, or masks are inconsistent."""

    expected = {
        "self_features": (SELF_FEATURES,),
        "neighbor_features": (config.max_nodes, NEIGHBOR_FEATURES),
        "edge_features": (config.max_nodes, EDGE_FEATURES),
        "packet_features": (PACKET_FEATURES,),
        "action_mask": (config.max_nodes + 1,),
    }
    if config.include_forwardability:
        expected["candidate_forwardability"] = (
            config.max_nodes,
            FORWARDABILITY_FEATURES,
        )
    if config.include_risk_features:
        expected["candidate_risk_features"] = (
            config.max_nodes,
            RISK_FEATURES,
        )
    for key, shape in expected.items():
        if observation[key].shape != shape:
            raise ValueError(f"{key} has shape {observation[key].shape}, expected {shape}")
        if not np.all(np.isfinite(observation[key])):
            raise ValueError(f"{key} contains non-finite values")
    mask = observation["action_mask"]
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("action_mask must be binary")
    if mask[config.max_nodes] != 1:
        raise ValueError("DROP action must always be valid")
