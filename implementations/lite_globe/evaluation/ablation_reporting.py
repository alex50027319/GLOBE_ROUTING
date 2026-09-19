"""Validated statistics, tables, and raw-data checks for the SwitchGLOBE ablation."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ..scenarios import phase9_evaluation_scenarios
from .reporting import write_csv
from .statistics import summarize_values


SCENARIOS = tuple(item.name for item in phase9_evaluation_scenarios(0))

GEO_RESIDUAL = "Geo-Residual Student"
PREDICTIVE_PRIOR_ONLY = "Predictive Prior Only"
PREDICTIVE_NO_SWITCH = "Predictive Student (No Switch)"
SWITCHGLOBE_EXACT = "SwitchGLOBE Exact"
FAST_SWITCHGLOBE = "FastSwitchGLOBE"
FAST_SWITCHGLOBE_TOP2 = "FastSwitchGLOBE + Top-2"

ABLATION_METHODS = (
    GEO_RESIDUAL,
    PREDICTIVE_PRIOR_ONLY,
    PREDICTIVE_NO_SWITCH,
    SWITCHGLOBE_EXACT,
    FAST_SWITCHGLOBE,
    FAST_SWITCHGLOBE_TOP2,
)

PRIMARY_METRICS = (
    "connected_pair_pdr",
    "deadline_delivery_ratio",
    "p95_success_delay",
    "energy_per_delivered_packet",
    "decision_latency_p95_ms",
    "mean_policy_input_bytes",
    "switch_activation_rate",
    "backup_availability_rate",
    "fast_failover_success_rate",
)
LOWER_IS_BETTER = {
    "p95_success_delay", "energy_per_delivered_packet",
    "decision_latency_p95_ms", "mean_policy_input_bytes",
}
DELIVERY_DENOMINATOR_METRICS = {
    "p95_success_delay", "energy_per_delivered_packet",
}
# These are structurally zero for variants without the corresponding
# mechanism (e.g. Geo-Residual has no switch, FastSwitchGLOBE without Top-2
# has no backup) rather than NaN, so they never fail the finiteness check.


def validate_summary_rows(rows: list[dict[str, Any]], *, training_seeds: tuple[int, ...]) -> None:
    expected = {
        (method, scenario, seed)
        for method in ABLATION_METHODS
        for scenario in SCENARIOS
        for seed in training_seeds
    }
    actual: set[tuple[str, str, int]] = set()
    for row in rows:
        key = (str(row["method"]), str(row["scenario"]), int(row["training_seed"]))
        if key in actual:
            raise ValueError(f"duplicate method/scenario/seed summary: {key}")
        actual.add(key)
        for metric in PRIMARY_METRICS:
            raw = row[metric]
            if raw is None or raw == "":
                if metric in DELIVERY_DENOMINATOR_METRICS and int(row["delivered"]) == 0:
                    continue
                raise ValueError(f"unexpected undefined {metric} for {key}")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"non-finite {metric} for {key}")
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(f"ablation summary contract mismatch: missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}")


def validate_fast_top2_outcome_equality(episode_rows: list[dict[str, Any]]) -> int:
    """Standard-routing outcomes must match exactly between Fast and Fast+Top-2.

    Returns the number of mismatched (scenario, training_seed, evaluation_seed)
    triples; raises if any exist, since Top-2 backup availability must never
    change normal-routing behaviour.
    """

    fast: dict[tuple[str, int, int], dict[str, Any]] = {}
    top2: dict[tuple[str, int, int], dict[str, Any]] = {}
    fields = ("delivered", "dropped", "drop_reason", "steps", "hop_count", "transmission_attempts")
    for row in episode_rows:
        key = (str(row["scenario"]), int(row["training_seed"]), int(row["evaluation_seed"]))
        if row["method"] == FAST_SWITCHGLOBE:
            fast[key] = row
        elif row["method"] == FAST_SWITCHGLOBE_TOP2:
            top2[key] = row
    mismatches = [
        key for key in fast
        if key in top2 and any(fast[key][field] != top2[key][field] for field in fields)
    ]
    if mismatches:
        raise ValueError(
            f"FastSwitchGLOBE and FastSwitchGLOBE + Top-2 standard-routing outcomes "
            f"diverged on {len(mismatches)} episodes, e.g. {mismatches[:3]}"
        )
    return 0


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric in PRIMARY_METRICS:
            raw = row[metric]
            if raw is None or raw == "":
                continue
            grouped[(str(row["scenario"]), str(row["method"]), metric)].append(float(raw))
    output = []
    for scenario in SCENARIOS:
        for method in ABLATION_METHODS:
            for metric in PRIMARY_METRICS:
                values = grouped[(scenario, method, metric)]
                stats = (
                    summarize_values(values).to_dict()
                    if values
                    else {
                        "count": 0, "mean": None, "standard_deviation": None,
                        "ci95_low": None, "ci95_high": None,
                    }
                )
                output.append({"scenario": scenario, "method": method, "metric": metric, **stats})
    return output


def paired_effects_vs_exact(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paired (by training seed) contrasts of every variant against SwitchGLOBE Exact."""

    lookup = {(str(row["scenario"]), str(row["method"]), int(row["training_seed"])): row for row in rows}
    contrasts = tuple(method for method in ABLATION_METHODS if method != SWITCHGLOBE_EXACT)
    output = []
    for scenario in SCENARIOS:
        for method in contrasts:
            for metric in PRIMARY_METRICS:
                differences, relative = [], []
                for row in rows:
                    if row["scenario"] != scenario or row["method"] != method:
                        continue
                    seed = int(row["training_seed"])
                    exact_row = lookup.get((scenario, SWITCHGLOBE_EXACT, seed))
                    if exact_row is None:
                        continue
                    variant_raw, exact_raw = row[metric], exact_row[metric]
                    if variant_raw is None or variant_raw == "" or exact_raw is None or exact_raw == "":
                        continue
                    variant_value, exact_value = float(variant_raw), float(exact_raw)
                    difference = (
                        exact_value - variant_value if metric in LOWER_IS_BETTER
                        else variant_value - exact_value
                    )
                    differences.append(difference)
                    if abs(exact_value) > 1e-12:
                        relative.append(100.0 * difference / abs(exact_value))
                stats = (
                    summarize_values(differences).to_dict()
                    if differences
                    else {
                        "count": 0, "mean": None, "standard_deviation": None,
                        "ci95_low": None, "ci95_high": None,
                    }
                )
                relative_stats = summarize_values(relative).to_dict() if relative else None
                output.append({
                    "scenario": scenario, "variant": method, "baseline": SWITCHGLOBE_EXACT, "metric": metric,
                    "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                    **stats,
                    "relative_mean_percent": relative_stats["mean"] if relative_stats else None,
                    "relative_ci95_low_percent": relative_stats["ci95_low"] if relative_stats else None,
                    "relative_ci95_high_percent": relative_stats["ci95_high"] if relative_stats else None,
                })
    return output


def _table(rows: list[dict[str, Any]]) -> str:
    lookup = {(row["scenario"], row["method"], row["metric"]): row for row in rows}
    lines = [
        "# SwitchGLOBE ablation (verified aggregates)", "",
        "| Variant | Scenario | Connected PDR | Deadline ratio | Delay p95 | Energy/delivered | Switch rate | Backup availability |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in SCENARIOS:
        for method in ABLATION_METHODS:
            values = []
            for metric in (
                "connected_pair_pdr", "deadline_delivery_ratio", "p95_success_delay",
                "energy_per_delivered_packet", "switch_activation_rate", "backup_availability_rate",
            ):
                row = lookup[(scenario, method, metric)]
                if row["mean"] is None:
                    values.append("undefined (no deliveries)")
                else:
                    half = (float(row["ci95_high"]) - float(row["ci95_low"])) / 2
                    values.append(f"{float(row['mean']):.4f} ± {half:.4f}")
            lines.append(f"| {method} | {scenario} | " + " | ".join(values) + " |")
    lines.extend(["", "Energy는 simulator transmission proxy이며 Joule이 아니다. Input bytes는 control overhead가 아니다."])
    return "\n".join(lines) + "\n"


def _plot(rows: list[dict[str, Any]], metric: str, path: Path) -> None:
    lookup = {(row["scenario"], row["method"], row["metric"]): row for row in rows}
    fig, axis = plt.subplots(figsize=(14, 6))
    for method in ABLATION_METHODS:
        axis.plot(
            SCENARIOS,
            [
                float(lookup[(scenario, method, metric)]["mean"])
                if lookup[(scenario, method, metric)]["mean"] is not None
                else math.nan
                for scenario in SCENARIOS
            ],
            marker="o", label=method,
        )
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=220)
    plt.close(fig)


def write_ablation_artifacts(
    output_dir: Path, *, episode_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]], metadata: dict[str, Any],
) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in metadata["config"]["training_seeds"])
    validate_summary_rows(summary_rows, training_seeds=seeds)
    fast_top2_mismatches = validate_fast_top2_outcome_equality(episode_rows)
    expected_episodes = len(ABLATION_METHODS) * len(SCENARIOS) * len(seeds) * int(metadata["config"]["evaluation_episodes"])
    if len(episode_rows) != expected_episodes:
        raise ValueError(f"episode row count {len(episode_rows)} != {expected_episodes}")
    statistics = aggregate(summary_rows)
    effects = paired_effects_vs_exact(summary_rows)
    write_csv(output_dir / "raw" / "episodes.csv", episode_rows)
    write_csv(output_dir / "raw" / "seed_summaries.csv", summary_rows)
    write_csv(output_dir / "summaries" / "statistics.csv", statistics)
    write_csv(output_dir / "summaries" / "paired_effects.csv", effects)
    (output_dir / "summaries").mkdir(parents=True, exist_ok=True)
    (output_dir / "summaries" / "statistics.json").write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summaries" / "paired_effects.json").write_text(json.dumps(effects, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables" / "ablation.md").write_text(_table(statistics), encoding="utf-8")
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    for metric in ("connected_pair_pdr", "p95_success_delay", "energy_per_delivered_packet", "switch_activation_rate"):
        _plot(statistics, metric, output_dir / "figures" / metric)
    manifest = {
        "schema_version": 1, "complete": True, "suite": "switchglobe_ablation",
        "mode": metadata["mode"], "methods": list(ABLATION_METHODS), "scenarios": list(SCENARIOS),
        "episode_rows": len(episode_rows), "seed_summary_rows": len(summary_rows),
        "expected_episode_rows": expected_episodes, "statistics_rows": len(statistics),
        "paired_effect_rows": len(effects), "fast_top2_outcome_mismatches": fast_top2_mismatches,
        "metadata": metadata,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
