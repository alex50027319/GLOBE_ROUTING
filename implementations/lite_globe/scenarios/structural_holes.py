"""Rotated routing-hole families for learning non-greedy detours."""

from __future__ import annotations

from dataclasses import replace
from math import cos, radians, sin

import numpy as np

from ..env.config import FanetConfig
from .evaluation_suite import EvaluationScenario


def structural_hole_config(
    seed: int,
    *,
    stochastic_link_loss: float = 0.0,
) -> FanetConfig:
    return FanetConfig(
        num_nodes=8,
        max_nodes=12,
        area_size=10.0,
        communication_radius=1.85,
        max_episode_steps=12,
        packet_ttl=12,
        max_queue_size=8,
        stochastic_link_loss=stochastic_link_loss,
        min_speed=0.0,
        max_speed=0.0,
        reward_delivery=10.0,
        reward_delay=0.1,
        reward_failure=5.0,
        reward_progress=2.0,
        include_node_ids=False,
        mask_visited_actions=True,
        include_forwardability=True,
        seed=seed,
    )


def structural_hole_options(angle_degrees: float) -> dict[str, object]:
    """Rotate a greedy trap while preserving graph distances and endpoints."""

    positions = np.array(
        [
            [3.0, 4.5],
            [4.4, 4.0],
            [3.0, 5.9],
            [4.4, 5.9],
            [5.8, 5.9],
            [7.0, 4.5],
            [1.0, 1.0],
            [9.0, 9.0],
        ],
        dtype=np.float32,
    )
    center = np.array([5.0, 5.0], dtype=np.float32)
    theta = radians(angle_degrees)
    rotation = np.array(
        [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]],
        dtype=np.float32,
    )
    rotated = (positions - center) @ rotation.T + center
    return {
        "positions": rotated.astype(np.float32),
        "source": 0,
        "destination": 5,
    }


def phase8_hole_training_scenarios(seed: int) -> list[EvaluationScenario]:
    config = structural_hole_config(seed)
    return [
        EvaluationScenario(
            f"train_structural_hole_{angle}",
            config,
            structural_hole_options(float(angle)),
            "training_structural_hole",
        )
        for angle in (0, 90, 180)
    ]


def phase8_hole_calibration_scenarios(seed: int) -> list[EvaluationScenario]:
    return [
        EvaluationScenario(
            "calibration_structural_hole_270",
            structural_hole_config(seed),
            structural_hole_options(270.0),
            "calibration_structural_hole",
        )
    ]


def phase8_hole_evaluation_scenarios(seed: int) -> list[EvaluationScenario]:
    base = structural_hole_config(seed)
    return [
        EvaluationScenario(
            "structural_hole_45",
            base,
            structural_hole_options(45.0),
            "ood_structural_hole",
        ),
        EvaluationScenario(
            "structural_hole_225_link_loss",
            replace(base, stochastic_link_loss=0.10),
            structural_hole_options(225.0),
            "ood_structural_hole_link_loss",
        ),
    ]


def phase9_hole_training_scenarios(seed: int) -> list[EvaluationScenario]:
    """Reuse training holes with Phase 9 observation capacity."""

    return [
        EvaluationScenario(
            scenario.name,
            replace(
                scenario.config,
                max_nodes=32,
                include_risk_features=True,
            ),
            scenario.reset_options,
            scenario.distribution,
        )
        for scenario in phase8_hole_training_scenarios(seed)
    ]


def phase9_hole_calibration_scenarios(seed: int) -> list[EvaluationScenario]:
    """Use a held-out rotation and a lossy rotation for calibration."""

    base = replace(
        structural_hole_config(seed),
        max_nodes=32,
        include_risk_features=True,
    )
    return [
        EvaluationScenario(
            "calibration_structural_hole_270",
            base,
            structural_hole_options(270.0),
            "calibration_structural_hole",
        ),
        EvaluationScenario(
            "calibration_structural_hole_315_link_loss",
            replace(base, stochastic_link_loss=0.15),
            structural_hole_options(315.0),
            "calibration_structural_hole_link_loss",
        ),
    ]
