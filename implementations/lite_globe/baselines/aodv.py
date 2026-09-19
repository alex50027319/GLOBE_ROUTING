"""Message-driven AODV adaptation following RFC 3561 core behavior."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .common import Observation, ProtocolSnapshot, filtered_observation_bytes, valid_candidates


RREQ_BYTES = 24
RREP_BYTES = 20
RERR_BASE_BYTES = 12


@dataclass
class RouteEntry:
    destination: int
    next_hop: int
    hop_count: int
    sequence_number: int
    expires_at: int
    valid: bool = True


class AodvPolicy:
    """RFC-3561-inspired control plane with explicit RREQ/RREP/RERR accounting.

    The simulator snapshot is used solely as the delivery medium for control
    packets. Forwarding decisions read the resulting route table, never a
    shortest path over the current graph.
    """

    source = "RFC 3561"
    fidelity = "common-contract adaptation"
    observation_fields = ("action_mask", "packet_features")

    def __init__(self, drop_action: int, *, discovery_ttl: int = 2, route_lifetime: int = 5) -> None:
        self.drop_action = drop_action
        self.discovery_ttl = discovery_ttl
        self.route_lifetime = route_lifetime
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.routes: dict[tuple[int, int], RouteEntry] = {}
        self.sequence_numbers = np.zeros(self.drop_action, dtype=np.int64)
        self.seen_rreq: set[tuple[int, int]] = set()
        self.rreq_id = 0
        self.snapshot: ProtocolSnapshot | None = None
        self.control_messages = 0
        self.control_bytes = 0
        self.duplicate_rreq_suppressed = 0
        self.route_discoveries = 0
        self.route_errors = 0

    def observation_bytes(self, observation: Observation) -> int:
        return filtered_observation_bytes(observation, self.observation_fields)

    def protocol_tick(self, snapshot: ProtocolSnapshot) -> None:
        previous = self.snapshot
        self.snapshot = snapshot
        for key, route in list(self.routes.items()):
            if route.expires_at <= snapshot.step:
                route.valid = False
            origin, _ = key
            if previous is not None and route.valid and not snapshot.adjacency[origin, route.next_hop]:
                route.valid = False
                self.control_messages += 1
                self.control_bytes += RERR_BASE_BYTES + 4
                self.route_errors += 1

    def _install_path(self, path: list[int], destination: int, destination_seq: int) -> None:
        assert self.snapshot is not None
        for index, node in enumerate(path[:-1]):
            self.routes[(node, destination)] = RouteEntry(
                destination=destination,
                next_hop=path[index + 1],
                hop_count=len(path) - index - 1,
                sequence_number=destination_seq,
                expires_at=self.snapshot.step + self.route_lifetime,
            )
        for index in range(1, len(path)):
            node = path[index]
            self.routes[(node, path[0])] = RouteEntry(
                destination=path[0],
                next_hop=path[index - 1],
                hop_count=index,
                sequence_number=int(self.sequence_numbers[path[0]]),
                expires_at=self.snapshot.step + self.route_lifetime,
            )

    def discover(self, origin: int, destination: int) -> bool:
        if self.snapshot is None:
            return False
        self.route_discoveries += 1
        self.rreq_id += 1
        self.sequence_numbers[origin] += 1
        request = (origin, self.rreq_id)
        queue: deque[tuple[int, list[int], int]] = deque([(origin, [origin], 0)])
        local_seen: set[int] = set()
        max_ttl = max(self.discovery_ttl, self.drop_action)
        ttl = self.discovery_ttl
        while ttl <= max_ttl:
            while queue:
                node, path, hops = queue.popleft()
                if node in local_seen:
                    self.duplicate_rreq_suppressed += 1
                    continue
                local_seen.add(node)
                self.seen_rreq.add(request)
                if node == destination:
                    self.sequence_numbers[destination] += 1
                    self.control_messages += max(len(path) - 1, 1)
                    self.control_bytes += RREP_BYTES * max(len(path) - 1, 1)
                    self._install_path(path, destination, int(self.sequence_numbers[destination]))
                    return True
                if hops >= ttl:
                    continue
                for neighbor in np.flatnonzero(self.snapshot.adjacency[node]):
                    self.control_messages += 1
                    self.control_bytes += RREQ_BYTES
                    queue.append((int(neighbor), [*path, int(neighbor)], hops + 1))
            if ttl == max_ttl:
                break
            ttl = min(max_ttl, ttl + 2)
            queue = deque([(origin, [origin], 0)])
            local_seen.clear()
        return False

    def act(self, observation: Observation) -> int:
        if self.snapshot is None:
            return self.drop_action
        current = self.snapshot.current_node
        destination = self.snapshot.destination
        route = self.routes.get((current, destination))
        if route is None or not route.valid or route.expires_at <= self.snapshot.step:
            self.discover(current, destination)
            route = self.routes.get((current, destination))
        candidates = set(valid_candidates(observation, self.drop_action).tolist())
        if route is None or not route.valid or route.next_hop not in candidates:
            if route is not None:
                route.valid = False
            return self.drop_action
        route.expires_at = self.snapshot.step + self.route_lifetime
        return route.next_hop

    def episode_diagnostics(self) -> dict[str, float]:
        return {
            "control_messages": float(self.control_messages),
            "control_bytes": float(self.control_bytes),
            "route_discoveries": float(self.route_discoveries),
            "route_errors": float(self.route_errors),
            "duplicate_rreq_suppressed": float(self.duplicate_rreq_suppressed),
        }
