"""Dynamic shortest-path reference policy with privileged topology access."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..env.fanet_env import FanetRoutingEnv
from ..env.graph_utils import shortest_path


class ShortestPathOraclePolicy:
    """Choose the next hop on the current graph's unweighted shortest path."""

    def __init__(self, env: FanetRoutingEnv) -> None:
        self.env = env

    def reset(self, seed: int | None = None) -> None:
        del seed

    def observation_bytes(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> int:
        del observation
        return int(self.env.adjacency.nbytes + 2 * np.dtype(np.int64).itemsize)

    def act(self, observation: dict[str, NDArray[np.generic]]) -> int:
        del observation
        adjacency = self.env.adjacency.copy()
        for visited in self.env.packet.path[:-1]:
            if visited != self.env.packet.destination:
                adjacency[visited, :] = False
                adjacency[:, visited] = False
        path = shortest_path(
            adjacency,
            self.env.packet.current,
            self.env.packet.destination,
        )
        if path is None or len(path) < 2:
            return self.env.drop_action
        return path[1]
