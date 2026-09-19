"""Explicit HELLO/TC OLSR control-plane adaptation of RFC 3626."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .common import Observation, ProtocolSnapshot, filtered_observation_bytes, valid_candidates


HELLO_BASE_BYTES = 16
TC_BASE_BYTES = 16


@dataclass
class TopologyTuple:
    destination: int
    last_hop: int
    expires_at: int


class OlsrPolicy:
    """Proactive routing derived only from periodically disseminated tuples."""

    source = "RFC 3626"
    fidelity = "common-contract adaptation"
    observation_fields = ("action_mask", "packet_features")

    def __init__(self, drop_action: int, *, hello_interval: int = 1, tc_interval: int = 2, hold_time: int = 6) -> None:
        self.drop_action = drop_action
        self.hello_interval = hello_interval
        self.tc_interval = tc_interval
        self.hold_time = hold_time
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.snapshot: ProtocolSnapshot | None = None
        self.one_hop: dict[int, set[int]] = {}
        self.two_hop: dict[int, set[int]] = {}
        self.mprs: dict[int, set[int]] = {}
        self.topology: dict[tuple[int, int], TopologyTuple] = {}
        self.control_messages = 0
        self.control_bytes = 0
        self._last_hello = -10**9
        self._last_tc = -10**9
        self._hello_expires = -1

    def observation_bytes(self, observation: Observation) -> int:
        return filtered_observation_bytes(observation, self.observation_fields)

    @staticmethod
    def _select_mprs(neighbors: set[int], neighbor_sets: dict[int, set[int]], origin: int) -> set[int]:
        uncovered = set().union(*(neighbor_sets.get(n, set()) for n in neighbors)) - neighbors - {origin}
        selected: set[int] = set()
        while uncovered:
            best = max(neighbors - selected, key=lambda n: (len(neighbor_sets.get(n, set()) & uncovered), -n), default=None)
            if best is None or not (neighbor_sets.get(best, set()) & uncovered):
                break
            selected.add(best)
            uncovered -= neighbor_sets.get(best, set())
        return selected

    def protocol_tick(self, snapshot: ProtocolSnapshot) -> None:
        self.snapshot = snapshot
        step = snapshot.step
        if self._hello_expires >= 0 and step >= self._hello_expires:
            self.one_hop.clear()
            self.two_hop.clear()
            self.mprs.clear()
        for key, value in list(self.topology.items()):
            if value.expires_at <= step:
                del self.topology[key]
        if step - self._last_hello >= self.hello_interval:
            current_sets = {
                node: set(int(v) for v in np.flatnonzero(snapshot.adjacency[node]))
                for node in range(snapshot.adjacency.shape[0])
            }
            self.one_hop = current_sets
            self.two_hop = {
                node: set().union(*(current_sets[n] for n in neighbors)) - neighbors - {node}
                if neighbors else set()
                for node, neighbors in current_sets.items()
            }
            self.mprs = {
                node: self._select_mprs(neighbors, current_sets, node)
                for node, neighbors in current_sets.items()
            }
            for neighbors in current_sets.values():
                self.control_messages += 1
                self.control_bytes += HELLO_BASE_BYTES + 4 * len(neighbors)
            self._last_hello = step
            self._hello_expires = step + self.hold_time
        if step - self._last_tc >= self.tc_interval:
            for origin, selected in self.mprs.items():
                advertised = {node for node, mprs in self.mprs.items() if origin in mprs}
                if not advertised:
                    advertised = self.one_hop.get(origin, set())
                transmissions = 1 + len(selected)
                self.control_messages += transmissions
                self.control_bytes += transmissions * (TC_BASE_BYTES + 4 * len(advertised))
                for destination in advertised:
                    self.topology[(origin, destination)] = TopologyTuple(
                        destination=destination,
                        last_hop=origin,
                        expires_at=step + self.hold_time,
                    )
            self._last_tc = step

    def _route(self, source: int, destination: int) -> int | None:
        graph: dict[int, set[int]] = {node: set(self.one_hop.get(node, set())) for node in self.one_hop}
        for value in self.topology.values():
            graph[value.last_hop].add(value.destination)
            graph[value.destination].add(value.last_hop)
        queue: deque[tuple[int, int | None]] = deque([(source, None)])
        seen = {source}
        while queue:
            node, first = queue.popleft()
            for neighbor in sorted(graph.get(node, set())):
                if neighbor in seen:
                    continue
                next_hop = neighbor if first is None else first
                if neighbor == destination:
                    return next_hop
                seen.add(neighbor)
                queue.append((neighbor, next_hop))
        return None

    def act(self, observation: Observation) -> int:
        if self.snapshot is None:
            return self.drop_action
        next_hop = self._route(self.snapshot.current_node, self.snapshot.destination)
        candidates = set(valid_candidates(observation, self.drop_action).tolist())
        return int(next_hop) if next_hop in candidates else self.drop_action

    def episode_diagnostics(self) -> dict[str, float]:
        return {
            "control_messages": float(self.control_messages),
            "control_bytes": float(self.control_bytes),
            "mpr_count": float(sum(len(value) for value in self.mprs.values())),
            "topology_tuple_count": float(len(self.topology)),
        }
