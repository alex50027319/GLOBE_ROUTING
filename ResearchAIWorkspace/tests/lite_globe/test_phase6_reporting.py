"""Phase 6 statistics, scenarios, costs, and artifact guarantees."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from implementations.lite_globe.baselines import GpsrPolicy
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.evaluation.costs import measure_policy_cost
from implementations.lite_globe.evaluation.reporting import (
    METHOD_ORDER,
    aggregate_seed_summaries,
    write_phase6_artifacts,
)
from implementations.lite_globe.evaluation.statistics import summarize_values
from implementations.lite_globe.scenarios import (
    phase6_scenarios,
    routing_hole_config,
    routing_hole_options,
)


def _summary_rows() -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for scenario in (
        "routing_hole",
        "routing_hole_link_loss",
        "mobile_dense",
        "mobile_sparse",
    ):
        for method_index, method in enumerate(METHOD_ORDER):
            for seed in (1, 2):
                value = 0.1 * method_index + 0.01 * seed
                rows.append(
                    {
                        "method": method,
                        "scenario": scenario,
                        "training_seed": seed,
                        "episodes": 10,
                        "delivered": 5,
                        "dropped": 5,
                        "packet_delivery_ratio": value,
                        "mean_delay_steps": value + 1,
                        "mean_hop_count": value + 2,
                        "throughput_packets_per_step": value / 2,
                        "loop_drop_rate": value / 3,
                        "mean_episode_reward": value - 1,
                    }
                )
    return rows


def test_student_t_confidence_interval_and_single_seed() -> None:
    statistic = summarize_values([1.0, 2.0, 3.0])
    assert statistic.mean == 2.0
    assert statistic.ci95_low < statistic.mean < statistic.ci95_high
    single = summarize_values([4.0])
    assert single.standard_deviation == 0.0
    assert single.ci95_low == single.ci95_high == 4.0


def test_phase6_scenarios_share_shape_but_shift_distribution() -> None:
    scenarios = phase6_scenarios(42)
    assert len(scenarios) == 4
    assert len({scenario.name for scenario in scenarios}) == 4
    assert all(scenario.config.max_nodes == 6 for scenario in scenarios)
    assert scenarios[0].distribution == "in_distribution"
    assert all(
        scenario.distribution.startswith("ood") for scenario in scenarios[1:]
    )


def test_policy_cost_is_finite() -> None:
    env = FanetRoutingEnv(routing_hole_config())
    observation, _ = env.reset(seed=1, options=routing_hole_options())
    cost = measure_policy_cost(
        GpsrPolicy(env.drop_action),
        observation,
        warmup=1,
        repeats=3,
    )
    assert cost.parameter_count == 0
    assert cost.input_bytes == sum(value.nbytes for value in observation.values())
    assert np.isfinite(cost.mean_latency_ms)
    assert cost.peak_python_memory_bytes >= 0


def test_phase6_artifacts_include_tables_and_vector_figures(
    tmp_path: Path,
) -> None:
    summary_rows = _summary_rows()
    aggregates = aggregate_seed_summaries(summary_rows)
    assert len(aggregates) == 4 * len(METHOD_ORDER) * 6
    manifest = write_phase6_artifacts(
        tmp_path,
        episode_rows=[
            {
                "method": "GPSR",
                "scenario": "routing_hole",
                "training_seed": 1,
                "evaluation_seed": 2,
                "delivered": 1,
            }
        ],
        seed_summary_rows=summary_rows,
        training_rows=[{"training_seed": 1}],
        cost_rows=[{"training_seed": 1, "method": "GPSR"}],
        metadata={"phase": 6, "mode": "test"},
    )
    assert manifest["statistics_rows"] == len(aggregates)
    for relative in (
        "raw/episodes.csv",
        "summaries/statistics.json",
        "tables/main_results.md",
        "tables/main_results.tex",
        "figures/pdr_comparison.png",
        "figures/pdr_comparison.pdf",
        "figures/pdr_comparison.svg",
        "figures/ood_pdr.png",
        "manifest.json",
    ):
        assert (tmp_path / relative).is_file()


def test_delay_excludes_seeds_without_deliveries() -> None:
    rows = _summary_rows()
    target = [
        row
        for row in rows
        if row["scenario"] == "routing_hole" and row["method"] == "GPSR"
    ]
    target[0]["delivered"] = 0
    aggregates = aggregate_seed_summaries(rows)
    delay = next(
        row
        for row in aggregates
        if row["scenario"] == "routing_hole"
        and row["method"] == "GPSR"
        and row["metric"] == "mean_delay_steps"
    )
    assert delay["count"] == 1
