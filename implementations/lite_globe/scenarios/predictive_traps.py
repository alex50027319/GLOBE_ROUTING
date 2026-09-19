"""Rotated mobility traps where a currently valid onward link soon breaks."""

from __future__ import annotations

from dataclasses import replace
from math import cos, radians, sin

import numpy as np

from ..env.config import FanetConfig
from .evaluation_suite import EvaluationScenario


def predictive_break_config(
    seed: int,
    *,
    stochastic_link_loss: float = 0.0,
) -> FanetConfig:
    return FanetConfig(
        num_nodes=9,
        max_nodes=32,
        area_size=10.0,
        communication_radius=2.1,
        max_episode_steps=16,
        packet_ttl=16,
        max_queue_size=8,
        stochastic_link_loss=stochastic_link_loss,
        min_speed=0.0,
        max_speed=1.5,
        time_step=1.0,
        waypoint_tolerance=0.25,
        reward_delivery=10.0,
        reward_delay=0.1,
        reward_failure=5.0,
        reward_progress=2.0,
        include_node_ids=False,
        mask_visited_actions=True,
        include_forwardability=True,
        include_risk_features=True,
        seed=seed,
    )


def predictive_break_options(
    angle_degrees: float,
) -> dict[str, object]:
    """Rotate positions and velocities without changing graph geometry."""

    positions = np.array(
        [
            [1.0, 1.0],
            [3.0, 1.0],
            [5.0, 1.0],
            [1.0, 3.0],
            [2.5, 4.2],
            [4.5, 4.2],
            [6.0, 3.5],
            [7.0, 2.0],
            [7.0, 1.0],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)
    velocities[2] = np.array([0.0, 1.5], dtype=np.float32)
    geometry_center = np.array([4.0, 2.5], dtype=np.float32)
    area_center = np.array([5.0, 5.0], dtype=np.float32)
    theta = radians(angle_degrees)
    rotation = np.array(
        [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]],
        dtype=np.float32,
    )
    rotated_positions = (
        (positions - geometry_center) @ rotation.T + area_center
    )
    rotated_velocities = velocities @ rotation.T
    return {
        "positions": rotated_positions.astype(np.float32),
        "velocities": rotated_velocities.astype(np.float32),
        "source": 0,
        "destination": 8,
    }


def phase9_predictive_training_scenarios(
    seed: int,
) -> list[EvaluationScenario]:
    config = predictive_break_config(seed)
    return [
        EvaluationScenario(
            f"train_predictive_break_{angle}",
            config,
            predictive_break_options(float(angle)),
            "training_predictive_break",
        )
        for angle in (0, 90, 180)
    ]


def phase9_predictive_link_loss_training_scenarios(
    seed: int,
) -> list[EvaluationScenario]:
    """Expose Predictive Student training to transient, recoverable link loss.

    ``phase9_predictive_training_scenarios`` uses angles 0/90/180, all with
    ``stochastic_link_loss=0.0``: the model only ever sees the deterministic
    node-2 break, never a transient failure a retry could recover from. This
    reuses the same three angles (not the held-out eval angles 45/225) so the
    model learns to distinguish "wait/retry" from "reroute/give up" at
    topologies it already trains on, rather than learning a new geometry.
    """

    config = replace(predictive_break_config(seed), stochastic_link_loss=0.10)
    return [
        EvaluationScenario(
            f"train_predictive_break_{angle}_link_loss",
            config,
            predictive_break_options(float(angle)),
            "training_predictive_break_link_loss",
        )
        for angle in (0, 90, 180)
    ]


def phase9_predictive_calibration_scenarios(
    seed: int,
) -> list[EvaluationScenario]:
    return [
        EvaluationScenario(
            "calibration_predictive_break_270",
            predictive_break_config(seed),
            predictive_break_options(270.0),
            "calibration_predictive_break",
        )
    ]


def phase9_predictive_link_loss_calibration_scenarios(
    seed: int,
) -> list[EvaluationScenario]:
    """Expose risk-switch calibration to transient, recoverable link loss.

    ``phase9_predictive_calibration_scenarios`` (angle 270) and the training
    scenarios (angles 0/90/180) all use ``stochastic_link_loss=0.0``, so the
    Phase 12 switch-threshold search has only ever seen the deterministic
    node-2 break, never a transient failure a retry could recover from. This
    uses angle 135 (untouched by every other predictive-break scenario) so
    ``predictive_break_45``/``predictive_break_225_link_loss`` stay held-out.
    """

    return [
        EvaluationScenario(
            "calibration_predictive_break_135_link_loss",
            replace(predictive_break_config(seed), stochastic_link_loss=0.10),
            predictive_break_options(135.0),
            "calibration_predictive_break_link_loss",
        )
    ]


def phase9_predictive_evaluation_scenarios(
    seed: int,
) -> list[EvaluationScenario]:
    base = predictive_break_config(seed)
    return [
        EvaluationScenario(
            "predictive_break_45",
            base,
            predictive_break_options(45.0),
            "ood_predictive_break",
        ),
        EvaluationScenario(
            "predictive_break_225_link_loss",
            replace(base, stochastic_link_loss=0.10),
            predictive_break_options(225.0),
            "ood_predictive_break_link_loss",
        ),
    ]
