"""Greedy geographic forwarding baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class GpsrPolicy:
    """Forward to the observed neighbor closest to the destination."""

    def __init__(self, drop_action: int) -> None:
        self.drop_action = drop_action

    def reset(self, seed: int | None = None) -> None:
        del seed

    def observation_bytes(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> int:
        return sum(
            int(observation[key].nbytes)
            for key in (
                "neighbor_features",
                "packet_features",
                "action_mask",
            )
        )

    def act(self, observation: dict[str, NDArray[np.generic]]) -> int:
        mask = observation["action_mask"]
        candidates = np.flatnonzero(mask[: self.drop_action])
        if candidates.size == 0:
            return self.drop_action

        destination_delta = observation["packet_features"][:2]
        neighbor_deltas = observation["neighbor_features"][candidates, :2]
        remaining = neighbor_deltas - destination_delta
        distances = np.linalg.norm(remaining, axis=1)
        return int(candidates[int(np.argmin(distances))])
