"""Small graph utilities for connectivity-aware routing experiments."""

from __future__ import annotations

from collections import deque

import numpy as np
from numpy.typing import NDArray


BoolArray = NDArray[np.bool_]


def shortest_path(
    adjacency: BoolArray,
    source: int,
    destination: int,
) -> list[int] | None:
    """Return an unweighted shortest path, or ``None`` when disconnected."""

    if source == destination:
        return [source]
    parents = np.full(adjacency.shape[0], -1, dtype=np.int64)
    parents[source] = source
    queue: deque[int] = deque([source])
    while queue:
        current = queue.popleft()
        for neighbor in np.flatnonzero(adjacency[current]):
            neighbor = int(neighbor)
            if parents[neighbor] != -1:
                continue
            parents[neighbor] = current
            if neighbor == destination:
                path = [destination]
                while path[-1] != source:
                    path.append(int(parents[path[-1]]))
                return list(reversed(path))
            queue.append(neighbor)
    return None


def connected_pairs(
    adjacency: BoolArray,
    *,
    min_hops: int = 1,
) -> list[tuple[int, int, int]]:
    """Enumerate ordered connected endpoint pairs and shortest hop counts."""

    if min_hops < 1:
        raise ValueError("min_hops must be positive")
    pairs: list[tuple[int, int, int]] = []
    for source in range(adjacency.shape[0]):
        for destination in range(adjacency.shape[0]):
            if source == destination:
                continue
            path = shortest_path(adjacency, source, destination)
            if path is not None and len(path) - 1 >= min_hops:
                pairs.append((source, destination, len(path) - 1))
    return pairs
