"""Uniform random forwarding over structurally valid neighbors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class RandomPolicy:
    """Choose a valid next hop uniformly, falling back to explicit DROP."""

    def __init__(self, drop_action: int) -> None:
        self.drop_action = drop_action
        self._rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def observation_bytes(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> int:
        return int(observation["action_mask"].nbytes)

    def act(self, observation: dict[str, NDArray[np.generic]]) -> int:
        mask = observation["action_mask"]
        candidates = np.flatnonzero(mask[: self.drop_action])
        if candidates.size == 0:
            return self.drop_action
        return int(self._rng.choice(candidates))
