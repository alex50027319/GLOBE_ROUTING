"""Statistics and figures for the SwitchGLOBE novelty ablation."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..experiments.novelty_ablation_campaign import (
    FULL,
    NO_DISTILLATION,
    NO_GEO_RESIDUAL,
    NO_RISK_SWITCH,
    NOVELTY_ABLATION_METHODS,
)
from ..scenarios import phase9_evaluation_scenarios
from .reporting import write_csv
from .statistics import summarize_values


SCENARIOS = tuple(item.name for item in phase9_evaluation_scenarios(0))
COMPONENTS = {
    "Risk-Switch": NO_RISK_SWITCH,
    "Geo-Residual": NO_GEO_RESIDUAL,
    "Distillation": NO_DISTILLATION,
}
PRIMARY_METRICS = (
    "connected_pair_pdr",
    "deadline_delivery_ratio",
    "p95_success_delay",
    "energy_per_delivered_packet",
)
LOWER_IS_BETTER = {
    "p95_success_delay",
    "energy_per_delivered_packet",
}
METRIC_LABELS = {
    "connected_pair_pdr": "Connected-pair PDR",
    "deadline_delivery_ratio": "Deadline delivery ratio",
    "p95_success_delay": "P95 success delay (steps)",
    "energy_per_delivered_packet": "Energy proxy / delivered packet",
}


def _as_float(row: dict[str, Any], key: str) -> float:
    value = row[key]
    if value is None or value == "":
        return math.nan
    return float(value)


def validate_rows(
    episode_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    training_seeds: tuple[int, ...],
    evaluation_episodes: int,
) -> None:
    expected_episode_rows = (
        len(NOVELTY_ABLATION_METHODS)
        * len(SCENARIOS)
        * len(training_seeds)
        * evaluation_episodes
    )
    if len(episode_rows) != expected_episode_rows:
        raise ValueError(
            f"episode row count {len(episode_rows)} != {expected_episode_rows}"
        )
    episode_keys = {
        (
            str(row["method"]),
            str(row["scenario"]),
            int(row["training_seed"]),
            int(row["evaluation_seed"]),
        )
        for row in episode_rows
    }
    if len(episode_keys) != expected_episode_rows:
        raise ValueError("episode rows contain duplicate paired keys")
    expected_summary = {
        (method, scenario, seed)
        for method in NOVELTY_ABLATION_METHODS
        for scenario in SCENARIOS
        for seed in training_seeds
    }
    summary_keys = {
        (
            str(row["method"]),
            str(row["scenario"]),
            int(row["training_seed"]),
        )
        for row in summary_rows
    }
    if summary_keys != expected_summary:
        raise ValueError("summary rows do not match the ablation contract")


def seed_overall_metrics(
    episode_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in episode_rows:
        grouped[(str(row["method"]), int(row["training_seed"]))].append(row)
    output = []
    for method in NOVELTY_ABLATION_METHODS:
        seeds = sorted(seed for name, seed in grouped if name == method)
        for seed in seeds:
            rows = grouped[(method, seed)]
            connected = [row for row in rows if int(row["initially_connected"])]
            delivered = [row for row in rows if int(row["delivered"])]
            connected_delivered = [
                row for row in connected if int(row["delivered"])
            ]
            success_delays = [float(row["steps"]) for row in delivered]
            energy = sum(float(row["transmission_energy_proxy"]) for row in rows)
            output.append(
                {
                    "method": method,
                    "training_seed": seed,
                    "episodes": len(rows),
                    "connected_episodes": len(connected),
                    "delivered": len(delivered),
                    "connected_pair_pdr": (
                        len(connected_delivered) / max(len(connected), 1)
                    ),
                    "deadline_delivery_ratio": (
                        sum(int(row["deadline_met"]) for row in rows)
                        / len(rows)
                    ),
                    "p95_success_delay": (
                        float(np.percentile(success_delays, 95))
                        if success_delays
                        else None
                    ),
                    "energy_per_delivered_packet": (
                        energy / len(delivered) if delivered else None
                    ),
                    "switch_activation_rate": (
                        sum(float(row["switch_steps"]) for row in rows)
                        / max(sum(float(row["steps"]) for row in rows), 1.0)
                    ),
                }
            )
    return output


def aggregate_seed_metrics(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for method in NOVELTY_ABLATION_METHODS:
        selected = [row for row in seed_rows if row["method"] == method]
        for metric in PRIMARY_METRICS:
            values = [_as_float(row, metric) for row in selected]
            stats = summarize_values(values).to_dict()
            output.append({"method": method, "metric": metric, **stats})
    return output


def _sign_flip_pvalue(differences: list[float]) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0 or np.allclose(values, 0.0):
        return 1.0
    observed = abs(float(np.mean(values)))
    extreme = 0
    total = 0
    for signs in product((-1.0, 1.0), repeat=values.size):
        permuted = abs(float(np.mean(values * np.asarray(signs))))
        extreme += int(permuted >= observed - 1e-15)
        total += 1
    return extreme / total


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    for metric in PRIMARY_METRICS:
        selected = [row for row in rows if row["metric"] == metric]
        ordered = sorted(selected, key=lambda row: float(row["p_value"]))
        running = 0.0
        count = len(ordered)
        for rank, row in enumerate(ordered):
            adjusted = min(1.0, (count - rank) * float(row["p_value"]))
            running = max(running, adjusted)
            row["holm_adjusted_p_value"] = running


def paired_component_effects(
    seed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (str(row["method"]), int(row["training_seed"])): row
        for row in seed_rows
    }
    seeds = sorted({int(row["training_seed"]) for row in seed_rows})
    output = []
    for component, ablation in COMPONENTS.items():
        for metric in PRIMARY_METRICS:
            differences = []
            relative = []
            for seed in seeds:
                full = _as_float(lookup[(FULL, seed)], metric)
                removed = _as_float(lookup[(ablation, seed)], metric)
                if not math.isfinite(full) or not math.isfinite(removed):
                    continue
                difference = (
                    removed - full
                    if metric in LOWER_IS_BETTER
                    else full - removed
                )
                differences.append(difference)
                if abs(full) > 1e-12:
                    relative.append(100.0 * difference / abs(full))
            stats = summarize_values(differences).to_dict()
            relative_stats = summarize_values(relative).to_dict()
            output.append(
                {
                    "component": component,
                    "ablation": ablation,
                    "baseline": FULL,
                    "metric": metric,
                    "direction": "positive_means_full_is_better",
                    **stats,
                    "relative_mean_percent": relative_stats["mean"],
                    "relative_ci95_low_percent": relative_stats["ci95_low"],
                    "relative_ci95_high_percent": relative_stats["ci95_high"],
                    "p_value": _sign_flip_pvalue(differences),
                }
            )
    _holm_adjust(output)
    return output


def scenario_component_effects(
    summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["method"]),
            str(row["scenario"]),
            int(row["training_seed"]),
        ): row
        for row in summary_rows
    }
    seeds = sorted({int(row["training_seed"]) for row in summary_rows})
    output = []
    for scenario in SCENARIOS:
        for component, ablation in COMPONENTS.items():
            for metric in PRIMARY_METRICS:
                differences = []
                for seed in seeds:
                    full = _as_float(lookup[(FULL, scenario, seed)], metric)
                    removed = _as_float(
                        lookup[(ablation, scenario, seed)], metric
                    )
                    if not math.isfinite(full) or not math.isfinite(removed):
                        continue
                    differences.append(
                        removed - full
                        if metric in LOWER_IS_BETTER
                        else full - removed
                    )
                if not differences:
                    continue
                stats = summarize_values(differences).to_dict()
                output.append(
                    {
                        "scenario": scenario,
                        "component": component,
                        "ablation": ablation,
                        "metric": metric,
                        **stats,
                    }
                )
    return output


def _error(row: dict[str, Any]) -> tuple[float, float]:
    mean = float(row["mean"])
    return mean - float(row["ci95_low"]), float(row["ci95_high"]) - mean


def _plot_absolute(rows: list[dict[str, Any]], output_dir: Path) -> None:
    lookup = {(row["method"], row["metric"]): row for row in rows}
    labels = ("Full", "No Risk-Switch", "No Geo-Residual", "No Distillation")
    colors = ("#173F5F", "#D95F02", "#7570B3", "#1B9E77")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric in zip(axes.flat, PRIMARY_METRICS, strict=True):
        selected = [lookup[(method, metric)] for method in NOVELTY_ABLATION_METHODS]
        means = [float(row["mean"]) for row in selected]
        errors = np.asarray([_error(row) for row in selected]).T
        axis.bar(labels, means, yerr=errors, color=colors, capsize=4)
        axis.set_title(METRIC_LABELS[metric])
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("SwitchGLOBE leave-one-component-out ablation (5 seeds)")
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output_dir / f"novelty_ablation_four_panel.{suffix}", dpi=240)
    plt.close(figure)


def _plot_contributions(rows: list[dict[str, Any]], output_dir: Path) -> None:
    lookup = {(row["component"], row["metric"]): row for row in rows}
    components = tuple(COMPONENTS)
    colors = ("#D95F02", "#7570B3", "#1B9E77")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric in zip(axes.flat, PRIMARY_METRICS, strict=True):
        selected = [lookup[(component, metric)] for component in components]
        if metric in {"connected_pair_pdr", "deadline_delivery_ratio"}:
            means = [100.0 * float(row["mean"]) for row in selected]
            lows = [100.0 * float(row["ci95_low"]) for row in selected]
            highs = [100.0 * float(row["ci95_high"]) for row in selected]
            ylabel = "Contribution (percentage points)"
        else:
            means = [float(row["relative_mean_percent"]) for row in selected]
            lows = [float(row["relative_ci95_low_percent"]) for row in selected]
            highs = [float(row["relative_ci95_high_percent"]) for row in selected]
            ylabel = "Contribution relative to Full (%)"
        errors = np.asarray([
            (mean - low, high - mean)
            for mean, low, high in zip(means, lows, highs, strict=True)
        ]).T
        axis.bar(components, means, yerr=errors, color=colors, capsize=4)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(METRIC_LABELS[metric])
        axis.set_ylabel(ylabel)
        axis.tick_params(axis="x", rotation=15)
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("Marginal contribution: positive means Full is better")
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output_dir / f"component_contribution_four_panel.{suffix}", dpi=240)
    plt.close(figure)


def _plot_pdr_heatmap(rows: list[dict[str, Any]], output_dir: Path) -> None:
    lookup = {
        (row["component"], row["scenario"]): 100.0 * float(row["mean"])
        for row in rows
        if row["metric"] == "connected_pair_pdr"
    }
    components = tuple(COMPONENTS)
    values = np.asarray([
        [lookup[(component, scenario)] for scenario in SCENARIOS]
        for component in components
    ])
    limit = max(float(np.max(np.abs(values))), 1.0)
    figure, axis = plt.subplots(figsize=(15, 4.5))
    image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(len(SCENARIOS)))
    axis.set_xticklabels(SCENARIOS, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(components)))
    axis.set_yticklabels(components)
    axis.set_title("Connected-PDR contribution by scenario (percentage points)")
    figure.colorbar(image, ax=axis, label="Full minus ablation (pp)")
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output_dir / f"scenario_pdr_contribution_heatmap.{suffix}", dpi=240)
    plt.close(figure)


def _markdown_table(
    aggregate_rows: list[dict[str, Any]],
    effect_rows: list[dict[str, Any]],
) -> str:
    aggregate = {(row["method"], row["metric"]): row for row in aggregate_rows}
    effects = {(row["component"], row["metric"]): row for row in effect_rows}
    lines = [
        "# SwitchGLOBE novelty ablation",
        "",
        "## Five-seed overall metrics",
        "",
        "| Variant | Connected PDR | Deadline ratio | P95 delay | Energy/delivered |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in NOVELTY_ABLATION_METHODS:
        values = []
        for metric in PRIMARY_METRICS:
            row = aggregate[(method, metric)]
            half = (float(row["ci95_high"]) - float(row["ci95_low"])) / 2.0
            values.append(f"{float(row['mean']):.4f} ± {half:.4f}")
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Paired marginal contribution of each component",
            "",
            "Positive values mean that the Full model is better after removing the component.",
            "",
            "| Component | PDR (pp) | Deadline (pp) | Delay (%) | Energy (%) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for component in COMPONENTS:
        pdr = 100.0 * float(effects[(component, "connected_pair_pdr")]["mean"])
        deadline = 100.0 * float(effects[(component, "deadline_delivery_ratio")]["mean"])
        delay = float(effects[(component, "p95_success_delay")]["relative_mean_percent"])
        energy = float(effects[(component, "energy_per_delivered_packet")]["relative_mean_percent"])
        lines.append(
            f"| {component} | {pdr:.3f} | {deadline:.3f} | {delay:.2f} | {energy:.2f} |"
        )
    lines.extend(
        [
            "",
            "Inference uses training seed as the paired statistical unit (n=5). "
            "P-values are exact two-sided sign-flip tests with Holm correction. "
            "Energy is a simulator proxy, not Joules.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_novelty_ablation_artifacts(
    output_dir: Path,
    *,
    episode_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in metadata["config"]["training_seeds"])
    episodes = int(metadata["config"]["evaluation_episodes"])
    validate_rows(
        episode_rows,
        summary_rows,
        training_seeds=seeds,
        evaluation_episodes=episodes,
    )
    seed_rows = seed_overall_metrics(episode_rows)
    aggregates = aggregate_seed_metrics(seed_rows)
    effects = paired_component_effects(seed_rows)
    scenario_effects = scenario_component_effects(summary_rows)
    write_csv(output_dir / "raw" / "episodes.csv", episode_rows)
    write_csv(output_dir / "raw" / "seed_summaries.csv", summary_rows)
    write_csv(output_dir / "summaries" / "seed_overall.csv", seed_rows)
    write_csv(output_dir / "summaries" / "overall_statistics.csv", aggregates)
    write_csv(output_dir / "summaries" / "paired_component_effects.csv", effects)
    write_csv(output_dir / "summaries" / "scenario_component_effects.csv", scenario_effects)
    (output_dir / "summaries").mkdir(parents=True, exist_ok=True)
    (output_dir / "summaries" / "paired_component_effects.json").write_text(
        json.dumps(effects, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables" / "novelty_ablation.md").write_text(
        _markdown_table(aggregates, effects),
        encoding="utf-8",
    )
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _plot_absolute(aggregates, figure_dir)
    _plot_contributions(effects, figure_dir)
    _plot_pdr_heatmap(scenario_effects, figure_dir)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "suite": "switchglobe_novelty_ablation",
        "mode": metadata["mode"],
        "methods": list(NOVELTY_ABLATION_METHODS),
        "components": list(COMPONENTS),
        "scenarios": list(SCENARIOS),
        "episode_rows": len(episode_rows),
        "expected_episode_rows": (
            len(NOVELTY_ABLATION_METHODS) * len(SCENARIOS) * len(seeds) * episodes
        ),
        "seed_summary_rows": len(summary_rows),
        "seed_overall_rows": len(seed_rows),
        "paired_effect_rows": len(effects),
        "metadata": metadata,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
