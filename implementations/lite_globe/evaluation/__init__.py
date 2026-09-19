"""Evaluation helpers and routing metrics."""

from .evaluator import (
    EpisodeResult,
    EvaluationSummary,
    evaluate_policy,
    evaluate_policy_results,
    run_episode,
    summarize_episode_results,
)
from .costs import PolicyCost, measure_policy_cost
from .latency import (
    LatencyBenchmark, benchmark_callable, benchmark_resolver,
    legacy_repeated_switchglobe_action, profile_student_policy,
)
from .records import episode_row, summary_row
from .external_comparison_reporting import write_external_comparison_artifacts
from .ablation_reporting import write_ablation_artifacts
from .reporting import aggregate_seed_summaries
from .statistics import Statistic, summarize_values
from .generalization import GENERALIZATION_METRICS, generalization_summary
from .phase7_reporting import write_phase7_artifacts
from .phase8_reporting import write_phase8_artifacts
from .phase11_reporting import write_phase11_artifacts
from .phase12_reporting import write_phase12_artifacts

write_switchglobe_artifacts = write_phase12_artifacts

__all__ = [
    "EpisodeResult",
    "EvaluationSummary",
    "evaluate_policy",
    "evaluate_policy_results",
    "run_episode",
    "summarize_episode_results",
    "PolicyCost",
    "LatencyBenchmark",
    "benchmark_callable",
    "benchmark_resolver",
    "legacy_repeated_switchglobe_action",
    "profile_student_policy",
    "Statistic",
    "aggregate_seed_summaries",
    "episode_row",
    "measure_policy_cost",
    "summary_row",
    "write_external_comparison_artifacts",
    "write_ablation_artifacts",
    "summarize_values",
    "GENERALIZATION_METRICS",
    "generalization_summary",
    "write_phase7_artifacts",
    "write_phase8_artifacts",
    "write_phase11_artifacts",
    "write_phase12_artifacts",
    "write_switchglobe_artifacts",
]
