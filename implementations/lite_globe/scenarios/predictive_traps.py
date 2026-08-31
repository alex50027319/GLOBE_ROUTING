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
    *,
    break_speed_scale: float = 1.0,
    velocity_offset_degrees: float = 0.0,
) -> dict[str, object]:
    """Rotate positions and velocities without changing graph geometry."""

    if not 0.0 < break_speed_scale <= 1.0:
        raise ValueError("break_speed_scale must be in (0, 1]")

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
    velocity_theta = radians(velocity_offset_degrees)
    velocities[2] = (
        1.5
        * break_speed_scale
        * np.array(
            [-sin(velocity_theta), cos(velocity_theta)],
            dtype=np.float32,
        )
    )
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


def phase9_compositional_predictive_training_scenarios(
    seed: int,
    *,
    link_loss_scale: float = 1.0,
) -> list[EvaluationScenario]:
    """Combine break prediction, mobility variation, and transient loss.

    Training retains the historical 0/90/180-degree geometries.  Speed and
    direction offsets alter the local lifetime evidence, while stochastic
    loss prevents the prior from treating every failed transmission as a
    deterministic break.  Calibration and evaluation rotations remain
    disjoint from these scenarios.
    """

    if not 0.0 < link_loss_scale <= 1.5:
        raise ValueError("link_loss_scale must be in (0, 1.5]")
    variants = (
        (0, 0.05, 0.65, -12.0),
        (90, 0.10, 0.85, 15.0),
        (180, 0.15, 1.00, -8.0),
    )
    scenarios = []
    for angle, link_loss, speed_scale, offset in variants:
        effective_loss = min(0.25, link_loss * link_loss_scale)
        scenarios.append(
            EvaluationScenario(
                f"train_composite_break_{angle}_loss_{effective_loss:.3f}",
                replace(
                    predictive_break_config(seed),
                    stochastic_link_loss=effective_loss,
                ),
                predictive_break_options(
                    float(angle),
                    break_speed_scale=speed_scale,
                    velocity_offset_degrees=offset,
                ),
                "training_composite_predictive_break",
            )
        )
    return scenarios


def phase9_compositional_predictive_calibration_scenarios(
    seed: int,
) -> list[EvaluationScenario]:
    """Held-out rotations for selecting one compositional curriculum."""

    variants = (
        (135, 0.12, 0.75, 10.0),
        (270, 0.18, 0.95, -10.0),
    )
    return [
        EvaluationScenario(
            f"calibration_composite_break_{angle}_loss_{link_loss:.2f}",
            replace(
                predictive_break_config(seed),
                stochastic_link_loss=link_loss,
            ),
            predictive_break_options(
                float(angle),
                break_speed_scale=speed_scale,
                velocity_offset_degrees=offset,
            ),
            "calibration_composite_predictive_break",
        )
        for angle, link_loss, speed_scale, offset in variants
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
