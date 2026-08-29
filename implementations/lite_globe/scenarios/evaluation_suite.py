"""In-distribution and OOD scenarios for the Phase 6 campaign."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..env.config import FanetConfig
from .routing_hole import routing_hole_config, routing_hole_options


@dataclass(frozen=True)
class EvaluationScenario:
    name: str
    config: FanetConfig
    reset_options: dict[str, Any] | None
    distribution: str


def phase6_scenarios(seed: int) -> list[EvaluationScenario]:
    """Return one training distribution and three controlled shifts."""

    base = routing_hole_config(seed)
    return [
        EvaluationScenario(
            "routing_hole",
            base,
            routing_hole_options(),
            "in_distribution",
        ),
        EvaluationScenario(
            "routing_hole_link_loss",
            replace(base, stochastic_link_loss=0.15),
            routing_hole_options(),
            "ood_link_loss",
        ),
        EvaluationScenario(
            "mobile_dense",
            replace(
                base,
                area_size=6.0,
                communication_radius=3.0,
                min_speed=0.05,
                max_speed=0.20,
            ),
            None,
            "ood_mobility",
        ),
        EvaluationScenario(
            "mobile_sparse",
            replace(
                base,
                area_size=10.0,
                communication_radius=2.5,
                min_speed=0.10,
                max_speed=0.35,
            ),
            None,
            "ood_density",
        ),
    ]
