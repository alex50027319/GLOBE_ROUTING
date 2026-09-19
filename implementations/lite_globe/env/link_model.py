"""Communication-radius FANET link model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float32]


def predicted_link_lifetime_steps(
    relative_position: NDArray[np.floating],
    relative_velocity: NDArray[np.floating],
    *,
    communication_radius: float,
    time_step: float,
    horizon_steps: float,
) -> float:
    """Predict when a constant-velocity link exits the radio radius."""

    position = np.asarray(relative_position, dtype=np.float64)
    velocity = np.asarray(relative_velocity, dtype=np.float64)
    a = float(np.dot(velocity, velocity))
    if a <= 1e-12:
        return float(horizon_steps)
    b = 2.0 * float(np.dot(position, velocity))
    c = float(np.dot(position, position)) - communication_radius**2
    discriminant = max(b * b - 4.0 * a * c, 0.0)
    exit_time = (-b + float(np.sqrt(discriminant))) / (2.0 * a)
    if exit_time <= 0:
        return 0.0
    return min(
        exit_time / max(time_step, 1e-12),
        float(horizon_steps),
    )


class RadiusLinkModel:
    """Construct undirected links from distance and optional random loss."""

    def __init__(self, communication_radius: float, loss_probability: float) -> None:
        self.communication_radius = communication_radius
        self.loss_probability = loss_probability

    def sample(
        self, positions: FloatArray, rng: np.random.Generator
    ) -> tuple[BoolArray, FloatArray]:
        """Return a symmetric adjacency matrix and pairwise distances."""

        delta = positions[:, None, :] - positions[None, :, :]
        distances = np.linalg.norm(delta, axis=-1).astype(np.float32)
        adjacency = distances <= self.communication_radius
        np.fill_diagonal(adjacency, False)
        if self.loss_probability > 0:
            random_keep = rng.random(adjacency.shape) >= self.loss_probability
            random_keep = np.triu(random_keep, 1)
            random_keep = random_keep | random_keep.T
            adjacency &= random_keep
        return adjacency.astype(np.bool_), distances
