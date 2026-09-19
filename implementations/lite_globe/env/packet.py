"""Packet state for sequential hop-by-hop routing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PacketState:
    """Mutable state for one packet during an episode."""

    source: int
    destination: int
    current: int
    ttl_remaining: int
    path: list[int] = field(default_factory=list)
    delivered: bool = False
    dropped: bool = False
    drop_reason: str = ""

    @classmethod
    def create(cls, source: int, destination: int, ttl: int) -> "PacketState":
        return cls(
            source=source,
            destination=destination,
            current=source,
            ttl_remaining=ttl,
            path=[source],
        )

    @property
    def hop_count(self) -> int:
        return max(0, len(self.path) - 1)
