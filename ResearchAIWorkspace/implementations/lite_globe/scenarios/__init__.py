"""Deterministic scenarios used for architecture and learning gates."""

from .evaluation_suite import EvaluationScenario, phase6_scenarios
from .generalization_suite import (
    CurriculumStage,
    phase7_curriculum,
    phase7_evaluation_scenarios,
    phase8_curriculum,
    phase8_evaluation_scenarios,
    phase9_curriculum,
    phase9_evaluation_scenarios,
)
from .routing_hole import routing_hole_config, routing_hole_options
from .predictive_traps import (
    phase9_predictive_calibration_scenarios,
    phase9_predictive_evaluation_scenarios,
    phase9_predictive_training_scenarios,
    predictive_break_config,
    predictive_break_options,
)
from .structural_holes import (
    phase8_hole_calibration_scenarios,
    phase8_hole_evaluation_scenarios,
    phase8_hole_training_scenarios,
    phase9_hole_calibration_scenarios,
    phase9_hole_training_scenarios,
    structural_hole_config,
    structural_hole_options,
)

__all__ = [
    "EvaluationScenario",
    "CurriculumStage",
    "phase6_scenarios",
    "phase7_curriculum",
    "phase7_evaluation_scenarios",
    "phase8_curriculum",
    "phase8_evaluation_scenarios",
    "phase9_curriculum",
    "phase9_evaluation_scenarios",
    "routing_hole_config",
    "routing_hole_options",
    "predictive_break_config",
    "predictive_break_options",
    "phase9_predictive_training_scenarios",
    "phase9_predictive_calibration_scenarios",
    "phase9_predictive_evaluation_scenarios",
    "phase8_hole_calibration_scenarios",
    "phase8_hole_evaluation_scenarios",
    "phase8_hole_training_scenarios",
    "phase9_hole_calibration_scenarios",
    "phase9_hole_training_scenarios",
    "structural_hole_config",
    "structural_hole_options",
]
