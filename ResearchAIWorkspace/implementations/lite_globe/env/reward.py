"""Minimal Lite-GLOBE routing reward."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingReward:
    """Delivery, per-step delay, and terminal failure terms only."""

    delivery: float
    delay: float
    failure: float
    progress: float = 0.0

    def calculate(
        self,
        *,
        delivered: bool,
        failed: bool,
        normalized_progress: float = 0.0,
    ) -> float:
        reward = -self.delay
        reward += self.progress * normalized_progress
        if delivered:
            reward += self.delivery
        if failed:
            reward -= self.failure
        return float(reward)
