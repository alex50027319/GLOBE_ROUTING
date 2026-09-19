"""Unit tests for the combined FastSwitchGLOBE/SwitchGLOBE Exact/baseline reporting module."""

from __future__ import annotations

from pathlib import Path

import pytest

from implementations.lite_globe.evaluation.combined_comparison_reporting import (
    COMBINED_METHODS,
    DEFAULT_PAIRS,
    FAST_SWITCHGLOBE,
    PRIMARY_METRICS,
    SCENARIOS,
    SWITCHGLOBE_EXACT,
    aggregate,
    macro_aggregate,
    macro_paired_effects,
    paired_effects,
    validate_rows,
    write_combined_comparison_artifacts,
)

SEEDS = (42, 77, 123, 314, 2718)


def _metric_value(method: str, metric: str, seed: int) -> float:
    # Deterministic per (method, metric, seed) so paired contrasts are non-trivial
    # and never accidentally zero (which would break relative-percent asserts).
    base = {SWITCHGLOBE_EXACT: 0.90, FAST_SWITCHGLOBE: 0.88}.get(method, 0.70)
    return base + 0.001 * seed / 1000 + (0.01 if metric == "overall_pdr" else 0.0)


def _summary_rows() -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        for method in COMBINED_METHODS:
            for seed in SEEDS:
                row = {"method": method, "scenario": scenario, "training_seed": str(seed)}
                for metric in PRIMARY_METRICS:
                    row[metric] = _metric_value(method, metric, seed)
                rows.append(row)
    return rows


def _episode_rows() -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        for method in COMBINED_METHODS:
            for seed in SEEDS:
                rows.append({
                    "method": method, "scenario": scenario, "training_seed": str(seed),
                    "evaluation_seed": str(1_100_000 + seed),
                })
    return rows


def test_validate_rows_accepts_complete_grid():
    validate_rows(_summary_rows(), training_seeds=SEEDS)


def test_validate_rows_rejects_missing_cell():
    rows = [row for row in _summary_rows() if not (row["method"] == FAST_SWITCHGLOBE and row["scenario"] == SCENARIOS[0] and row["training_seed"] == "42")]
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_rows(rows, training_seeds=SEEDS)


def test_validate_rows_rejects_duplicate():
    rows = _summary_rows()
    rows.append(dict(rows[0]))
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows(rows, training_seeds=SEEDS)


def test_validate_rows_rejects_unexpected_method():
    rows = _summary_rows()
    bad = dict(rows[0])
    bad["method"] = "Not A Real Method"
    rows.append(bad)
    with pytest.raises(ValueError, match="unexpected method"):
        validate_rows(rows, training_seeds=SEEDS)


def test_aggregate_covers_every_scenario_method_metric():
    stats = aggregate(_summary_rows())
    assert len(stats) == len(SCENARIOS) * len(COMBINED_METHODS) * len(PRIMARY_METRICS)
    for row in stats:
        assert row["count"] == len(SEEDS)


def test_macro_aggregate_one_row_per_method_metric():
    stats = macro_aggregate(_summary_rows())
    assert len(stats) == len(COMBINED_METHODS) * len(PRIMARY_METRICS)
    assert all(row["count"] == len(SEEDS) for row in stats)


def test_default_pairs_include_fast_vs_everything_and_exact_vs_baselines():
    assert (FAST_SWITCHGLOBE, SWITCHGLOBE_EXACT) in DEFAULT_PAIRS
    fast_pairs = [pair for pair in DEFAULT_PAIRS if pair[0] == FAST_SWITCHGLOBE]
    assert len(fast_pairs) == len(COMBINED_METHODS) - 1
    exact_pairs = [pair for pair in DEFAULT_PAIRS if pair[0] == SWITCHGLOBE_EXACT]
    assert len(exact_pairs) == len(COMBINED_METHODS) - 2


def test_paired_effects_direction_is_seed_paired():
    rows = _summary_rows()
    effects = paired_effects(rows, pairs=((FAST_SWITCHGLOBE, SWITCHGLOBE_EXACT),), training_seeds=SEEDS)
    assert len(effects) == len(SCENARIOS) * len(PRIMARY_METRICS)
    for row in effects:
        # FastSwitchGLOBE's fixture value (0.88ish) is below SwitchGLOBE Exact's (0.90ish)
        # on every higher-is-better metric, so the paired mean must be negative.
        if row["direction"] == "higher_is_better":
            assert row["mean"] < 0


def test_macro_paired_effects_matches_scenario_macro_convention():
    rows = _summary_rows()
    effects = macro_paired_effects(rows, pairs=((FAST_SWITCHGLOBE, SWITCHGLOBE_EXACT),), training_seeds=SEEDS)
    assert len(effects) == len(PRIMARY_METRICS)
    assert all(row["evidence"] == "five-seed scenario-macro paired mean" for row in effects)


def test_write_combined_comparison_artifacts_end_to_end(tmp_path: Path):
    metadata = {"training_seeds": list(SEEDS)}
    manifest = write_combined_comparison_artifacts(
        tmp_path, seed_summary_rows=_summary_rows(), episode_rows=_episode_rows(), metadata=metadata,
    )
    assert manifest["complete"] is True
    assert manifest["seed_summary_rows"] == manifest["expected_seed_summary_rows"]
    assert (tmp_path / "raw" / "seed_summaries.csv").exists()
    assert (tmp_path / "tables" / "combined_comparison.md").exists()
    assert (tmp_path / "tables" / "combined_comparison.tex").exists()
    for figure in ("fig_combined_connected_pdr_by_method", "fig_combined_deadline_ratio_by_method", "fig_combined_pdr_deadline_energy_by_method"):
        for suffix in ("png", "pdf", "svg"):
            assert (tmp_path / "figures" / f"{figure}.{suffix}").exists()
