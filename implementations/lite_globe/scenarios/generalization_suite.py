"""Dynamic curriculum and held-out topology families for Phase 7."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..env.config import FanetConfig
from .evaluation_suite import EvaluationScenario


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    config: FanetConfig
    reset_options: dict[str, Any]


def _base(seed: int) -> FanetConfig:
    return FanetConfig(
        num_nodes=8,
        max_nodes=12,
        area_size=10.0,
        communication_radius=4.5,
        max_episode_steps=12,
        packet_ttl=12,
        max_queue_size=8,
        stochastic_link_loss=0.0,
        min_speed=0.0,
        max_speed=0.05,
        time_step=1.0,
        waypoint_tolerance=0.25,
        reward_delivery=10.0,
        reward_delay=0.1,
        reward_failure=5.0,
        reward_progress=2.0,
        include_node_ids=False,
        mask_visited_actions=False,
        seed=seed,
    )


def _connected_options() -> dict[str, Any]:
    return {
        "require_connected": True,
        "min_shortest_hops": 2,
        "max_topology_attempts": 500,
    }


def phase7_curriculum(seed: int) -> list[CurriculumStage]:
    base = _base(seed)
    options = _connected_options()
    return [
        CurriculumStage("train_easy", base, options),
        CurriculumStage(
            "train_medium",
            replace(
                base,
                communication_radius=4.0,
                min_speed=0.05,
                max_speed=0.20,
            ),
            options,
        ),
        CurriculumStage(
            "train_hard",
            replace(
                base,
                communication_radius=3.5,
                stochastic_link_loss=0.05,
                min_speed=0.10,
                max_speed=0.35,
            ),
            options,
        ),
    ]


def phase7_evaluation_scenarios(seed: int) -> list[EvaluationScenario]:
    medium = phase7_curriculum(seed)[1].config
    connected = _connected_options()
    return [
        EvaluationScenario(
            "heldout_medium",
            medium,
            connected,
            "in_distribution_heldout",
        ),
        EvaluationScenario(
            "ood_link_loss",
            replace(medium, stochastic_link_loss=0.15),
            connected,
            "ood_link_loss",
        ),
        EvaluationScenario(
            "ood_fast_mobility",
            replace(medium, min_speed=0.30, max_speed=0.70),
            connected,
            "ood_mobility",
        ),
        EvaluationScenario(
            "ood_sparse",
            replace(medium, communication_radius=3.0),
            connected,
            "ood_density",
        ),
        EvaluationScenario(
            "ood_nodes_10",
            replace(
                medium,
                num_nodes=10,
                area_size=11.0,
                communication_radius=3.8,
            ),
            connected,
            "ood_node_count",
        ),
        EvaluationScenario(
            "unconditional_sparse",
            replace(medium, communication_radius=3.0),
            None,
            "unconditional",
        ),
    ]


def phase8_curriculum(seed: int) -> list[CurriculumStage]:
    """Reuse Phase 7 topology families with loop-safe action support."""

    return [
        CurriculumStage(
            stage.name,
            replace(
                stage.config,
                mask_visited_actions=True,
                include_forwardability=True,
            ),
            stage.reset_options,
        )
        for stage in phase7_curriculum(seed)
    ]


def phase8_evaluation_scenarios(seed: int) -> list[EvaluationScenario]:
    """Evaluate optimized policies on the same held-out families."""

    scenarios = [
        EvaluationScenario(
            scenario.name,
            replace(
                scenario.config,
                mask_visited_actions=True,
                include_forwardability=True,
            ),
            scenario.reset_options,
            scenario.distribution,
        )
        for scenario in phase7_evaluation_scenarios(seed)
    ]
    from .structural_holes import phase8_hole_evaluation_scenarios

    return scenarios + phase8_hole_evaluation_scenarios(seed)


def phase9_curriculum(seed: int) -> list[CurriculumStage]:
    """Add predictive local features and stress stages for robust KD."""

    base_stages = [
        CurriculumStage(
            stage.name,
            replace(
                stage.config,
                max_nodes=32,
                include_risk_features=True,
            ),
            stage.reset_options,
        )
        for stage in phase8_curriculum(seed)
    ]
    medium = base_stages[1].config
    options = _connected_options()
    return base_stages + [
        CurriculumStage(
            "train_mobility_stress",
            replace(
                medium,
                min_speed=0.35,
                max_speed=0.85,
            ),
            options,
        ),
        CurriculumStage(
            "train_link_loss_stress",
            replace(
                medium,
                stochastic_link_loss=0.20,
                communication_radius=3.8,
            ),
            options,
        ),
    ]


def phase9_evaluation_scenarios(seed: int) -> list[EvaluationScenario]:
    """Evaluate robustness, scalability, and structural-hole recovery."""

    base = [
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
        for scenario in phase8_evaluation_scenarios(seed)
    ]
    medium = phase9_curriculum(seed)[1].config
    connected = _connected_options()
    from .predictive_traps import (
        phase9_predictive_evaluation_scenarios,
    )

    return base + phase9_predictive_evaluation_scenarios(seed) + [
        EvaluationScenario(
            "ood_link_loss_30",
            replace(medium, stochastic_link_loss=0.30),
            connected,
            "ood_severe_link_loss",
        ),
        EvaluationScenario(
            "ood_extreme_mobility",
            replace(medium, min_speed=0.60, max_speed=1.20),
            connected,
            "ood_extreme_mobility",
        ),
        EvaluationScenario(
            "ood_nodes_16",
            replace(
                medium,
                num_nodes=16,
                area_size=14.0,
                communication_radius=4.2,
            ),
            connected,
            "ood_node_count",
        ),
        EvaluationScenario(
            "ood_nodes_24",
            replace(
                medium,
                num_nodes=24,
                area_size=17.0,
                communication_radius=4.4,
            ),
            connected,
            "ood_node_count",
        ),
    ]


def phase9_density_training_scenarios(seed: int) -> list[EvaluationScenario]:
    """Expose KD training to higher node counts than the base curriculum.

    The base curriculum (``phase9_curriculum``) trains entirely at
    ``num_nodes=8``, so ``ood_nodes_10``/``ood_nodes_16``/``ood_nodes_24``
    evaluate a density regime the student never saw. These two stages use
    different node counts (12, 20) than every ``ood_nodes_*`` evaluation
    scenario so those remain genuinely held-out, while still giving KD
    training exposure to a wider density range.
    """

    medium = phase9_curriculum(seed)[1].config
    connected = _connected_options()
    return [
        EvaluationScenario(
            "train_nodes_12",
            replace(
                medium,
                num_nodes=12,
                area_size=12.0,
                communication_radius=4.1,
            ),
            connected,
            "training_node_count",
        ),
        EvaluationScenario(
            "train_nodes_20",
            replace(
                medium,
                num_nodes=20,
                area_size=15.5,
                communication_radius=4.3,
            ),
            connected,
            "training_node_count",
        ),
    ]
