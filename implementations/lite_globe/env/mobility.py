"""Deterministic, seed-controlled Random Waypoint mobility."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]


@dataclass
class MobilityState:
    """Positions, velocities, targets, and scalar speeds for all UAVs."""

    positions: FloatArray
    velocities: FloatArray
    targets: FloatArray
    speeds: FloatArray


class RandomWaypointMobility:
    """A minimal 2D Random Waypoint model with no pause interval."""

    def __init__(
        self,
        num_nodes: int,
        area_size: float,
        min_speed: float,
        max_speed: float,
        time_step: float,
        waypoint_tolerance: float,
    ) -> None:
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.time_step = time_step
        self.waypoint_tolerance = waypoint_tolerance

    def reset(self, rng: np.random.Generator) -> MobilityState:
        """Sample initial positions, destinations, and speeds."""

        positions = rng.uniform(
            0.0, self.area_size, size=(self.num_nodes, 2)
        ).astype(np.float32)
        targets = rng.uniform(
            0.0, self.area_size, size=(self.num_nodes, 2)
        ).astype(np.float32)
        speeds = rng.uniform(
            self.min_speed, self.max_speed, size=self.num_nodes
        ).astype(np.float32)
        velocities = self._velocities(positions, targets, speeds)
        return MobilityState(positions, velocities, targets, speeds)

    def step(
        self, state: MobilityState, rng: np.random.Generator
    ) -> MobilityState:
        """Advance every UAV by one simulation interval."""

        positions = state.positions.copy()
        targets = state.targets.copy()
        speeds = state.speeds.copy()
        displacement = targets - positions
        distance = np.linalg.norm(displacement, axis=1)
        arrived = distance <= np.maximum(
            self.waypoint_tolerance, speeds * self.time_step
        )
        if np.any(arrived):
            count = int(np.sum(arrived))
            targets[arrived] = rng.uniform(
                0.0, self.area_size, size=(count, 2)
            ).astype(np.float32)
            speeds[arrived] = rng.uniform(
                self.min_speed, self.max_speed, size=count
            ).astype(np.float32)
        velocities = self._velocities(positions, targets, speeds)
        positions += velocities * self.time_step
        positions = np.clip(positions, 0.0, self.area_size).astype(np.float32)
        return MobilityState(positions, velocities, targets, speeds)

    @staticmethod
    def _velocities(
        positions: FloatArray, targets: FloatArray, speeds: FloatArray
    ) -> FloatArray:
        direction = targets - positions
        norms = np.linalg.norm(direction, axis=1, keepdims=True)
        unit = np.divide(
            direction,
            norms,
            out=np.zeros_like(direction),
            where=norms > 1e-8,
        )
        return (unit * speeds[:, None]).astype(np.float32)
