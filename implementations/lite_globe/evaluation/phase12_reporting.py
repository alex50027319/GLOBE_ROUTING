"""Paper-ready reports for Phase 12 Risk-Switch Lite-GLOBE-P."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ..experiments.phase12_campaign import PHASE12_METHODS
from ..scenarios import phase9_evaluation_scenarios
from .generalization import GENERALIZATION_METRICS
from .reporting import write_csv
from .statistics import summarize_values


PHASE12_SCENARIOS = tuple(
    scenario.name for scenario in phase9_evaluation_scenarios(0)
)
PROPOSED_METHOD = "Risk-Switch Lite-GLOBE-P"
CORE_BASELINES = (
    "GPSR",
    "Predictive Geographic",
    "Phase 8 Geo-Residual KD",
    "Lite-GLOBE-P no-switch",
)


def aggregate_phase12(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric in GENERALIZATION_METRICS:
            if (
                metric
                in {
                    "mean_success_delay",
                    "p95_success_delay",
                    "mean_path_stretch",
                }
                and int(row["delivered"]) == 0
            ):
                continue
            grouped[(row["scenario"], row["method"], metric)].append(
                float(row[metric])
            )
    output: list[dict[str, Any]] = []
    for scenario in PHASE12_SCENARIOS:
        for method in PHASE12_METHODS:
            for metric in GENERALIZATION_METRICS:
                values = grouped.get((scenario, method, metric), [])
                stats = (
                    summarize_values(values).to_dict()
                    if values
                    else {
                        "count": 0,
                        "mean": None,
                        "standard_deviation": None,
                        "ci95_low": None,
                        "ci95_high": None,
                    }
                )
                output.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "metric": metric,
                        **stats,
                    }
                )
    return output


def paired_phase12_effects(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_key = {
        (
            str(row["scenario"]),
            str(row["method"]),
            int(row["training_seed"]),
        ): row
        for row in rows
    }
    metrics = (
        "connected_pair_pdr",
        "deadline_delivery_ratio",
        "p95_success_delay",
        "mean_transmission_energy_proxy",
        "mean_policy_input_bytes",
    )
    lower_is_better = {
        "p95_success_delay",
        "mean_transmission_energy_proxy",
        "mean_policy_input_bytes",
    }
    output: list[dict[str, Any]] = []
    for scenario in PHASE12_SCENARIOS:
        for baseline in CORE_BASELINES:
            for metric in metrics:
                differences = []
                relative = []
                for key, proposed_row in by_key.items():
                    row_scenario, method, seed = key
                    if row_scenario != scenario or method != PROPOSED_METHOD:
                        continue
                    baseline_row = by_key.get((scenario, baseline, seed))
                    if baseline_row is None:
                        continue
                    proposed = float(proposed_row[metric])
                    base = float(baseline_row[metric])
                    diff = (
                        base - proposed
                        if metric in lower_is_better
                        else proposed - base
                    )
                    differences.append(diff)
                    if abs(base) > 1e-8:
                        relative.append(100.0 * diff / abs(base))
                if differences:
                    stats = summarize_values(differences).to_dict()
                    rel_stats = (
                        summarize_values(relative).to_dict()
                        if relative
                        else None
                    )
                    output.append(
                        {
                            "scenario": scenario,
                            "baseline": baseline,
                            "metric": metric,
                            "direction": (
                                "lower_is_better"
                                if metric in lower_is_better
                                else "higher_is_better"
                            ),
                            **stats,
                            "relative_mean_percent": (
                                rel_stats["mean"] if rel_stats else None
                            ),
                            "relative_ci95_low_percent": (
                                rel_stats["ci95_low"] if rel_stats else None
                            ),
                            "relative_ci95_high_percent": (
                                rel_stats["ci95_high"] if rel_stats else None
                            ),
                        }
                    )
    return output


def _lookup(
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (row["scenario"], row["method"], row["metric"]): row
        for row in rows
    }


def _estimate(row: dict[str, Any]) -> str:
    if row["mean"] is None:
        return "N/A"
    half = (float(row["ci95_high"]) - float(row["ci95_low"])) / 2
    return f'{float(row["mean"]):.3f} ± {half:.3f}'


def _main_table(rows: list[dict[str, Any]]) -> str:
    lookup = _lookup(rows)
    metrics = (
        "connected_pair_pdr",
        "deadline_delivery_ratio",
        "p95_success_delay",
        "mean_transmission_energy_proxy",
        "mean_policy_input_bytes",
    )
    lines = [
        "# Phase 12 Risk-Switch Lite-GLOBE-P Results",
        "",
        "| Method | Scenario | Connected PDR | Deadline delivery | "
        "Delay p95 | Energy proxy | Input bytes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in PHASE12_SCENARIOS:
        for method in PHASE12_METHODS:
            values = [
                _estimate(lookup[(scenario, method, metric)])
                for metric in metrics
            ]
            lines.append(
                f"| {method} | {scenario} | "
                + " | ".join(values)
                + " |"
            )
    return "\n".join(lines) + "\n"


def _effect_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 12 Risk-Switch Paired Effects",
        "",
        "| Scenario | Baseline | Metric | Difference | Relative | 95% CI |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        relative = row["relative_mean_percent"]
        relative_text = (
            "N/A" if relative is None else f"{float(relative):+.2f}%"
        )
        lines.append(
            f"| {row['scenario']} | {row['baseline']} | "
            f"{row['metric']} | {float(row['mean']):+.4f} | "
            f"{relative_text} | "
            f"[{float(row['ci95_low']):+.4f}, "
            f"{float(row['ci95_high']):+.4f}] |"
        )
    return "\n".join(lines) + "\n"


def _plot_metric(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    ylabel: str,
    path: Path,
) -> None:
    lookup = _lookup(rows)
    figure, axis = plt.subplots(figsize=(14, 6))
    for method in PHASE12_METHODS:
        means = []
        for scenario in PHASE12_SCENARIOS:
            mean = lookup[(scenario, method, metric)]["mean"]
            means.append(float("nan") if mean is None else float(mean))
        axis.plot(PHASE12_SCENARIOS, means, marker="o", label=method)
    if "pdr" in metric or "ratio" in metric:
        axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel(ylabel)
    axis.set_title(f"Phase 12 {ylabel}")
    axis.tick_params(axis="x", rotation=28)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    for extension in ("png", "pdf", "svg"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=220)
    plt.close(figure)


def write_phase12_artifacts(
    output_dir: Path,
    *,
    episode_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    raw = output_dir / "raw"
    summaries = output_dir / "summaries"
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    for directory in (raw, summaries, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)
    aggregates = aggregate_phase12(summary_rows)
    effects = paired_phase12_effects(summary_rows)
    write_csv(raw / "episodes.csv", episode_rows)
    write_csv(raw / "seed_summaries.csv", summary_rows)
    write_csv(raw / "training_metrics.csv", training_rows)
    write_csv(summaries / "statistics.csv", aggregates)
    write_csv(summaries / "paired_effects.csv", effects)
    (summaries / "statistics.json").write_text(
        json.dumps(aggregates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (summaries / "paired_effects.json").write_text(
        json.dumps(effects, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (tables / "risk_switch_results.md").write_text(
        _main_table(aggregates), encoding="utf-8"
    )
    (tables / "risk_switch_paired_effects.md").write_text(
        _effect_table(effects), encoding="utf-8"
    )
    _plot_metric(
        aggregates,
        metric="connected_pair_pdr",
        ylabel="Connected-pair PDR",
        path=figures / "risk_switch_pdr",
    )
    _plot_metric(
        aggregates,
        metric="p95_success_delay",
        ylabel="P95 Delay",
        path=figures / "risk_switch_delay_p95",
    )
    _plot_metric(
        aggregates,
        metric="mean_policy_input_bytes",
        ylabel="Policy Input Bytes",
        path=figures / "risk_switch_input_bytes",
    )
    manifest = {
        **metadata,
        "episode_rows": len(episode_rows),
        "seed_summary_rows": len(summary_rows),
        "statistics_rows": len(aggregates),
        "paired_effect_rows": len(effects),
        "methods": list(PHASE12_METHODS),
        "scenarios": list(PHASE12_SCENARIOS),
        "primary_metric": "connected_pair_pdr",
        "proposed_method": PROPOSED_METHOD,
        "metric_policy": (
            "PDR is primary; input bytes are counted according to switch use."
        ),
        "artifacts": sorted(
            str(path.relative_to(output_dir))
            for path in output_dir.rglob("*")
            if path.is_file()
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
