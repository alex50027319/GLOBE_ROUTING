"""Generate paper-ready Phase 9 figures from the full Colab results."""

from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE9 = ROOT / "artifacts" / "lite_globe" / "phase9"
SUMMARY_CSV = PHASE9 / "raw" / "seed_summaries.csv"
EPISODE_CSV = PHASE9 / "raw" / "episodes.csv"
TRAINING_CSV = PHASE9 / "raw" / "training_metrics.csv"
EFFECT_CSV = PHASE9 / "summaries" / "paired_effects.csv"
OUTPUT = PHASE9 / "figures" / "paper"

METHODS = (
    "GPSR",
    "Predictive Geographic",
    "Phase 8 Geo-Residual KD",
    "Phase 9 Risk-Aware KD",
    "Global Teacher",
)
SHORT = {
    "GPSR": "GPSR",
    "Predictive Geographic": "Predictive Geo",
    "Phase 8 Geo-Residual KD": "Phase 8",
    "Phase 9 Risk-Aware KD": "Phase 9",
    "Global Teacher": "Teacher",
    "Phase 9 no-risk ablation": "No risk",
    "Phase 9 no-forwardability ablation": "No forwardability",
    "Phase 9 geographic+risk only": "Geo+risk only",
}
COLORS = {
    "GPSR": "#8ec5f7",
    "Predictive Geographic": "#4f98df",
    "Phase 8 Geo-Residual KD": "#35a853",
    "Phase 9 Risk-Aware KD": "#d62728",
    "Global Teacher": "#7e57c2",
    "Phase 9 no-risk ablation": "#ffb74d",
    "Phase 9 no-forwardability ablation": "#26a69a",
    "Phase 9 geographic+risk only": "#ec407a",
}
MARKERS = {
    "GPSR": "o",
    "Predictive Geographic": "s",
    "Phase 8 Geo-Residual KD": "^",
    "Phase 9 Risk-Aware KD": "D",
    "Global Teacher": "X",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else float("nan")


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(figure: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(OUTPUT / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(figure)


def aggregate(
    rows: Iterable[dict[str, str]],
    scenario: str,
    method: str,
    metric: str,
) -> tuple[float, float]:
    values = [
        number(row, metric)
        for row in rows
        if row["scenario"] == scenario and row["method"] == method
    ]
    values = [value for value in values if np.isfinite(value)]
    if not values:
        return float("nan"), float("nan")
    return float(np.mean(values)), float(np.std(values, ddof=1))


def error_line(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    scenarios: list[str],
    labels: list[str],
    metric: str,
    methods: Iterable[str] = METHODS,
) -> None:
    x = np.arange(len(scenarios))
    for method in methods:
        estimates = [aggregate(rows, scenario, method, metric) for scenario in scenarios]
        axis.errorbar(
            x,
            [item[0] for item in estimates],
            yerr=[item[1] for item in estimates],
            color=COLORS[method],
            marker=MARKERS.get(method, "o"),
            linewidth=1.8,
            capsize=3,
            label=SHORT[method],
        )
    axis.set_xticks(x, labels)


def grouped_bars(
    axis: plt.Axes,
    rows: list[dict[str, str]],
    scenarios: list[str],
    labels: list[str],
    metric: str,
    methods: list[str],
) -> None:
    x = np.arange(len(scenarios))
    width = 0.8 / len(methods)
    for index, method in enumerate(methods):
        estimates = [aggregate(rows, scenario, method, metric) for scenario in scenarios]
        axis.bar(
            x + (index - (len(methods) - 1) / 2) * width,
            [item[0] for item in estimates],
            width,
            yerr=[item[1] for item in estimates],
            capsize=2,
            color=COLORS[method],
            edgecolor="white",
            linewidth=0.5,
            label=SHORT[method],
        )
    axis.set_xticks(x, labels)


def scalability_figure(rows: list[dict[str, str]]) -> None:
    scenarios = ["ood_nodes_10", "ood_nodes_16", "ood_nodes_24"]
    labels = ["10", "16", "24"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    for axis, metric, title, ylabel in (
        (axes[0], "connected_pair_pdr", "(a) PDR vs. number of UAVs", "Connected-pair PDR"),
        (
            axes[1],
            "deadline_delivery_ratio",
            "(b) Deadline delivery vs. number of UAVs",
            "Deadline delivery ratio",
        ),
        (
            axes[2],
            "mean_policy_input_bytes",
            "(c) Input cost vs. number of UAVs",
            "Policy input bytes",
        ),
    ):
        error_line(axis, rows, scenarios, labels, metric)
        axis.set_xlabel("Number of UAVs")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
    axes[0].set_ylim(0.80, 1.01)
    axes[1].set_ylim(0.78, 1.01)
    axes[0].legend(loc="lower left", frameon=True)
    save(figure, "01_scalability_uav_count")


def robustness_figure(rows: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.1))
    error_line(
        axes[0],
        rows,
        ["ood_link_loss", "ood_link_loss_30"],
        ["15%", "30%"],
        "connected_pair_pdr",
    )
    axes[0].set_title("(a) PDR vs. link-loss probability")
    axes[0].set_xlabel("Link-loss probability")
    axes[0].set_ylabel("Connected-pair PDR")
    axes[0].set_ylim(0.65, 0.96)

    error_line(
        axes[1],
        rows,
        ["heldout_medium", "ood_fast_mobility", "ood_extreme_mobility"],
        ["Nominal", "Fast", "Extreme"],
        "connected_pair_pdr",
    )
    axes[1].set_title("(b) PDR under mobility shift")
    axes[1].set_xlabel("Mobility regime")
    axes[1].set_ylabel("Connected-pair PDR")
    axes[1].set_ylim(0.88, 1.0)

    error_line(
        axes[2],
        rows,
        ["heldout_medium", "ood_sparse", "unconditional_sparse"],
        ["Medium", "Sparse", "Unconditional"],
        "connected_pair_pdr",
    )
    axes[2].set_title("(c) PDR under topology sparsity")
    axes[2].set_xlabel("Topology regime")
    axes[2].set_ylabel("Connected-pair PDR")
    axes[2].set_ylim(0.90, 1.0)
    axes[0].legend(loc="lower left", frameon=True)
    save(figure, "02_robustness_sweeps")


def challenge_figure(rows: list[dict[str, str]]) -> None:
    scenarios = [
        "structural_hole_45",
        "structural_hole_225_link_loss",
        "predictive_break_45",
        "predictive_break_225_link_loss",
    ]
    labels = ["Hole 45°", "Hole 225°+loss", "Break 45°", "Break 225°+loss"]
    methods = list(METHODS[:4])
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    grouped_bars(axes[0], rows, scenarios, labels, "connected_pair_pdr", methods)
    grouped_bars(axes[1], rows, scenarios, labels, "deadline_delivery_ratio", methods)
    axes[0].set_title("(a) Routing challenge PDR")
    axes[1].set_title("(b) Routing challenge deadline delivery")
    axes[0].set_ylabel("Connected-pair PDR")
    axes[1].set_ylabel("Deadline delivery ratio")
    for axis in axes:
        axis.set_ylim(0, 1.08)
        axis.tick_params(axis="x", rotation=16)
    axes[0].legend(ncol=2, loc="upper right", frameon=True)
    save(figure, "03_routing_challenges")


def ablation_figure(rows: list[dict[str, str]]) -> None:
    scenarios = [
        "structural_hole_45",
        "predictive_break_45",
        "predictive_break_225_link_loss",
        "ood_nodes_24",
    ]
    labels = ["Structural hole", "Predictive break", "Break + loss", "24 UAVs"]
    methods = [
        "Phase 8 Geo-Residual KD",
        "Phase 9 no-risk ablation",
        "Phase 9 no-forwardability ablation",
        "Phase 9 geographic+risk only",
        "Phase 9 Risk-Aware KD",
    ]
    figure, axis = plt.subplots(figsize=(11, 4.8))
    grouped_bars(axis, rows, scenarios, labels, "connected_pair_pdr", methods)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Connected-pair PDR")
    axis.set_title("Phase 9 component ablation")
    axis.legend(ncol=3, loc="upper center", frameon=True)
    save(figure, "04_component_ablation")


def seed_stability_figure(rows: list[dict[str, str]]) -> None:
    seeds = [42, 77, 123, 314, 2718]
    methods = [
        "Phase 8 Geo-Residual KD",
        "Phase 9 no-risk ablation",
        "Phase 9 geographic+risk only",
        "Phase 9 Risk-Aware KD",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.3), sharey=True)
    for axis, scenario, title in (
        (axes[0], "predictive_break_45", "(a) Clean predictive break"),
        (
            axes[1],
            "predictive_break_225_link_loss",
            "(b) Predictive break with link loss",
        ),
    ):
        for method in methods:
            values = []
            for seed in seeds:
                match = next(
                    row
                    for row in rows
                    if row["scenario"] == scenario
                    and row["method"] == method
                    and int(row["training_seed"]) == seed
                )
                values.append(number(match, "connected_pair_pdr"))
            axis.plot(
                range(len(seeds)),
                values,
                marker=MARKERS.get(method, "o"),
                color=COLORS[method],
                linewidth=1.8,
                label=SHORT[method],
            )
        axis.set_xticks(range(len(seeds)), [str(seed) for seed in seeds])
        axis.set_xlabel("Training seed")
        axis.set_title(title)
        axis.set_ylim(-0.03, 1.05)
    axes[0].set_ylabel("Connected-pair PDR")
    axes[1].legend(loc="lower right", frameon=True)
    save(figure, "05_seed_stability")


def tradeoff_scatter(rows: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for axis, scenario, title in (
        (
            axes[0],
            "predictive_break_225_link_loss",
            "(a) Predictive-break reliability/cost",
        ),
        (axes[1], "ood_nodes_24", "(b) 24-UAV reliability/cost"),
    ):
        for method in METHODS:
            pdr, pdr_std = aggregate(rows, scenario, method, "connected_pair_pdr")
            cost, cost_std = aggregate(
                rows, scenario, method, "mean_policy_input_bytes"
            )
            axis.errorbar(
                cost,
                pdr,
                xerr=cost_std,
                yerr=pdr_std,
                marker=MARKERS[method],
                color=COLORS[method],
                markersize=8,
                capsize=3,
                linestyle="none",
                label=SHORT[method],
            )
            axis.annotate(
                SHORT[method],
                (cost, pdr),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_xlabel("Policy input bytes")
        axis.set_ylabel("Connected-pair PDR")
        axis.set_title(title)
    save(figure, "06_reliability_input_tradeoff")


def improvement_figure(rows: list[dict[str, str]]) -> None:
    scenario = "predictive_break_225_link_loss"
    metrics = [
        ("connected_pair_pdr", "PDR", True),
        ("deadline_delivery_ratio", "Deadline", True),
        ("delivery_energy_efficiency_proxy", "Delivery/energy", True),
        ("delivery_transmission_efficiency_proxy", "Delivery/ETX", True),
        ("mean_path_stretch", "Path stretch", False),
        ("mean_queue_delay_proxy", "Queue delay", False),
        ("mean_policy_input_bytes", "Input bytes", False),
    ]
    values = []
    for metric, _, higher_is_better in metrics:
        baseline = aggregate(
            rows, scenario, "Phase 8 Geo-Residual KD", metric
        )[0]
        proposed = aggregate(
            rows, scenario, "Phase 9 Risk-Aware KD", metric
        )[0]
        change = (
            (proposed - baseline) / baseline
            if higher_is_better
            else (baseline - proposed) / baseline
        )
        values.append(change * 100)
    figure, axis = plt.subplots(figsize=(10, 4.6))
    colors = ["#2e9d57" if value >= 0 else "#d9534f" for value in values]
    bars = axis.bar(
        [item[1] for item in metrics],
        values,
        color=colors,
        edgecolor="white",
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Improvement over Phase 8 (%)")
    axis.set_title("Metric-wise Phase 9 improvement in predictive-break + loss")
    axis.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + (8 if value >= 0 else -8),
            f"{value:+.1f}%",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=8,
        )
    save(figure, "07_metric_improvement")


def heatmap_figure(rows: list[dict[str, str]]) -> None:
    scenarios = [
        "heldout_medium",
        "ood_link_loss",
        "ood_fast_mobility",
        "ood_sparse",
        "ood_nodes_10",
        "structural_hole_45",
        "structural_hole_225_link_loss",
        "predictive_break_45",
        "predictive_break_225_link_loss",
        "ood_link_loss_30",
        "ood_extreme_mobility",
        "ood_nodes_16",
        "ood_nodes_24",
    ]
    labels = [
        "Heldout",
        "Loss 15%",
        "Fast",
        "Sparse",
        "10 UAV",
        "Hole",
        "Hole+loss",
        "Break",
        "Break+loss",
        "Loss 30%",
        "Extreme",
        "16 UAV",
        "24 UAV",
    ]
    matrix = np.array(
        [
            [
                aggregate(rows, scenario, method, "connected_pair_pdr")[0]
                for scenario in scenarios
            ]
            for method in METHODS
        ]
    )
    figure, axis = plt.subplots(figsize=(13, 4.4))
    image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    axis.set_xticks(range(len(scenarios)), labels, rotation=35, ha="right")
    axis.set_yticks(range(len(METHODS)), [SHORT[method] for method in METHODS])
    axis.set_title("Connected-pair PDR heatmap")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            axis.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > 0.72 else "black",
            )
    figure.colorbar(image, ax=axis, label="Connected-pair PDR")
    save(figure, "08_pdr_heatmap")


def delay_stretch_figure(rows: list[dict[str, str]]) -> None:
    scenarios = [
        "heldout_medium",
        "ood_fast_mobility",
        "predictive_break_225_link_loss",
        "ood_nodes_24",
    ]
    labels = ["Heldout", "Fast", "Break+loss", "24 UAV"]
    methods = list(METHODS[:4])
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    grouped_bars(axes[0], rows, scenarios, labels, "p95_success_delay", methods)
    grouped_bars(axes[1], rows, scenarios, labels, "mean_path_stretch", methods)
    axes[0].set_title("(a) Tail delay")
    axes[1].set_title("(b) Path stretch")
    axes[0].set_ylabel("p95 successful delay (steps)")
    axes[1].set_ylabel("Mean path stretch")
    axes[0].legend(ncol=2, frameon=True)
    save(figure, "09_delay_and_path_stretch")


def efficiency_figure(rows: list[dict[str, str]]) -> None:
    scenarios = [
        "heldout_medium",
        "ood_fast_mobility",
        "predictive_break_225_link_loss",
        "ood_nodes_24",
    ]
    labels = ["Heldout", "Fast", "Break+loss", "24 UAV"]
    methods = list(METHODS[:4])
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    grouped_bars(
        axes[0],
        rows,
        scenarios,
        labels,
        "delivery_energy_efficiency_proxy",
        methods,
    )
    grouped_bars(
        axes[1],
        rows,
        scenarios,
        labels,
        "delivery_transmission_efficiency_proxy",
        methods,
    )
    axes[0].set_title("(a) Delivery/energy proxy")
    axes[1].set_title("(b) Delivery/ETX proxy")
    axes[0].set_ylabel("Delivered packets per energy proxy")
    axes[1].set_ylabel("Delivered packets per ETX proxy")
    axes[0].legend(ncol=2, frameon=True)
    save(figure, "10_efficiency_proxies")


def forest_figure(effect_rows: list[dict[str, str]]) -> None:
    labels = {
        "heldout_medium": "Heldout",
        "ood_link_loss": "Loss 15%",
        "ood_fast_mobility": "Fast mobility",
        "ood_sparse": "Sparse",
        "ood_nodes_10": "10 UAV",
        "structural_hole_45": "Structural hole",
        "structural_hole_225_link_loss": "Hole + loss",
        "predictive_break_45": "Predictive break",
        "predictive_break_225_link_loss": "Break + loss",
        "ood_link_loss_30": "Loss 30%",
        "ood_extreme_mobility": "Extreme mobility",
        "ood_nodes_16": "16 UAV",
        "ood_nodes_24": "24 UAV",
    }
    rows = [
        row
        for row in effect_rows
        if row["comparison"]
        == "Phase 9 Risk-Aware KD - Phase 8 Geo-Residual KD"
        and row["scenario"] in labels
    ]
    rows.sort(key=lambda row: number(row, "mean"))
    y = np.arange(len(rows))
    means = np.array([number(row, "mean") for row in rows])
    lows = np.array([number(row, "ci95_low") for row in rows])
    highs = np.array([number(row, "ci95_high") for row in rows])
    figure, axis = plt.subplots(figsize=(8.5, 6))
    colors = [
        "#2e9d57" if low > 0 else "#d9534f" if high < 0 else "#7f8c8d"
        for low, high in zip(lows, highs)
    ]
    for index in range(len(rows)):
        axis.errorbar(
            means[index],
            y[index],
            xerr=np.array(
                [
                    [means[index] - lows[index]],
                    [highs[index] - means[index]],
                ]
            ),
            fmt="o",
            color=colors[index],
            capsize=3,
        )
    axis.axvline(0, color="black", linewidth=0.9)
    axis.set_yticks(y, [labels[row["scenario"]] for row in rows])
    axis.set_xlabel("Paired PDR difference: Phase 9 - Phase 8")
    axis.set_title("Phase 9 paired effects with 95% confidence intervals")
    save(figure, "11_paired_effect_forest")


def ecdf(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def delay_ecdf_figure(episode_rows: list[dict[str, str]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    for axis, scenario, title in (
        (axes[0], "heldout_medium", "(a) Held-out delay ECDF"),
        (
            axes[1],
            "predictive_break_225_link_loss",
            "(b) Predictive-break + loss delay ECDF",
        ),
    ):
        for method in METHODS[:4]:
            values = [
                number(row, "delay_steps")
                for row in episode_rows
                if row["scenario"] == scenario
                and row["method"] == method
                and row["delivered"] in {"1", "True", "true"}
            ]
            if not values:
                continue
            x, y = ecdf(values)
            axis.step(x, y, where="post", color=COLORS[method], label=SHORT[method])
        axis.set_xlabel("Successful delivery delay (steps)")
        axis.set_ylabel("Empirical CDF")
        axis.set_ylim(0, 1.02)
        axis.set_title(title)
    axes[0].legend(frameon=True)
    save(figure, "12_success_delay_ecdf")


def hop_distribution_figure(episode_rows: list[dict[str, str]]) -> None:
    scenarios = ["heldout_medium", "predictive_break_225_link_loss"]
    titles = ["(a) Held-out hop count", "(b) Predictive-break + loss hop count"]
    methods = list(METHODS[:4])
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    for axis, scenario, title in zip(axes, scenarios, titles):
        data = []
        labels = []
        for method in methods:
            values = [
                number(row, "hop_count")
                for row in episode_rows
                if row["scenario"] == scenario
                and row["method"] == method
                and row["delivered"] in {"1", "True", "true"}
            ]
            data.append(values)
            labels.append(SHORT[method])
        boxes = axis.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        for box, method in zip(boxes["boxes"], methods):
            box.set_facecolor(COLORS[method])
            box.set_alpha(0.8)
        axis.set_ylabel("Hop count of delivered packets")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=15)
    save(figure, "13_hop_count_distribution")


def drop_breakdown_figure(rows: list[dict[str, str]]) -> None:
    scenario = "predictive_break_225_link_loss"
    methods = list(METHODS[:4])
    fields = [
        ("connected_pair_pdr", "Delivered", "#2e9d57"),
        ("agent_drop_rate", "Agent drop", "#f0ad4e"),
        ("ttl_drop_rate", "TTL drop", "#d9534f"),
        ("invalid_action_drop_rate", "Invalid action", "#8e44ad"),
        ("loop_drop_rate", "Loop", "#34495e"),
    ]
    figure, axis = plt.subplots(figsize=(9, 4.7))
    bottom = np.zeros(len(methods))
    for field, label, color in fields:
        values = [aggregate(rows, scenario, method, field)[0] for method in methods]
        axis.bar(
            range(len(methods)),
            values,
            bottom=bottom,
            color=color,
            label=label,
            edgecolor="white",
        )
        bottom += np.asarray(values)
    axis.set_xticks(range(len(methods)), [SHORT[method] for method in methods])
    axis.set_ylabel("Episode fraction")
    axis.set_title("Outcome breakdown in predictive-break + loss")
    axis.set_ylim(0, max(1.0, float(np.max(bottom)) * 1.05))
    axis.legend(ncol=3, frameon=True)
    save(figure, "14_outcome_breakdown")


def training_parameter_figure(training_rows: list[dict[str, str]]) -> None:
    rows = sorted(training_rows, key=lambda row: int(row["training_seed"]))
    seeds = [row["training_seed"] for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
    for field, label, color in (
        ("risk_lifetime_strength", "Current lifetime", "#4f98df"),
        ("risk_onward_lifetime_strength", "Onward lifetime", "#d62728"),
        ("risk_margin_strength", "Link margin", "#35a853"),
        ("risk_queue_headroom_strength", "Queue headroom", "#ff9800"),
    ):
        axes[0].plot(
            range(len(rows)),
            [number(row, field) for row in rows],
            marker="o",
            label=label,
            color=color,
        )
    axes[0].set_xticks(range(len(rows)), seeds)
    axes[0].set_xlabel("Training seed")
    axes[0].set_ylabel("Learned risk coefficient")
    axes[0].set_title("(a) Risk coefficient variability")
    axes[0].legend(frameon=True)

    axes[1].bar(
        np.arange(len(rows)) - 0.18,
        [number(row, "residual_weight") for row in rows],
        width=0.36,
        label="Residual weight",
        color="#7e57c2",
    )
    axes[1].bar(
        np.arange(len(rows)) + 0.18,
        [number(row, "risk_weight") for row in rows],
        width=0.36,
        label="Risk weight",
        color="#ec407a",
    )
    axes[1].set_xticks(range(len(rows)), seeds)
    axes[1].set_xlabel("Training seed")
    axes[1].set_ylabel("Calibrated mixture weight")
    axes[1].set_title("(b) Residual/risk calibration")
    axes[1].legend(frameon=True)
    save(figure, "15_training_parameter_variability")


def write_catalog() -> None:
    entries = [
        ("01_scalability_uav_count", "UAV 수 증가에 따른 PDR, deadline delivery, 입력비용"),
        ("02_robustness_sweeps", "링크 손실, 이동성, 희소 topology 강건성"),
        ("03_routing_challenges", "Structural hole과 predictive-break 성능"),
        ("04_component_ablation", "Risk, forwardability, learned residual 구성요소 ablation"),
        ("05_seed_stability", "Predictive-break에서 학습 seed 안정성"),
        ("06_reliability_input_tradeoff", "PDR과 실행 입력비용의 trade-off"),
        ("07_metric_improvement", "Phase 8 대비 지표별 상대 개선율"),
        ("08_pdr_heatmap", "주요 방법과 시나리오 전체 PDR heatmap"),
        ("09_delay_and_path_stretch", "p95 delay와 path stretch"),
        ("10_efficiency_proxies", "Delivery/energy와 Delivery/ETX proxy"),
        ("11_paired_effect_forest", "Phase 8 대비 paired PDR 차이와 95% CI"),
        ("12_success_delay_ecdf", "성공 패킷 지연의 경험적 누적분포"),
        ("13_hop_count_distribution", "전달 성공 패킷의 hop 수 분포"),
        ("14_outcome_breakdown", "Predictive-break+loss 결과 및 실패 원인"),
        ("15_training_parameter_variability", "Seed별 risk 계수와 calibration 변동"),
    ]
    lines = [
        "# Phase 9 Paper Figure Catalog",
        "",
        "모든 그림은 5개 training seed와 full Colab 결과에서 생성했다.",
        "각 그림은 PNG, PDF, SVG 형식으로 제공된다.",
        "",
        "| ID | 그림 | 논문 활용 위치 |",
        "| --- | --- | --- |",
    ]
    purposes = [
        "Scalability",
        "Robustness",
        "Reliability",
        "Ablation",
        "Stability",
        "Efficiency",
        "Trade-off",
        "Overview",
        "Latency",
        "Efficiency",
        "Statistical significance",
        "Latency distribution",
        "Route analysis",
        "Failure analysis",
        "Training stability",
    ]
    for index, ((name, description), purpose) in enumerate(
        zip(entries, purposes), start=1
    ):
        lines.append(f"| {index:02d} | `{name}` — {description} | {purpose} |")
    lines.extend(
        [
            "",
            "## 권장 본문 그림",
            "",
            "1. `01_scalability_uav_count`",
            "2. `02_robustness_sweeps`",
            "3. `03_routing_challenges`",
            "4. `04_component_ablation`",
            "5. `11_paired_effect_forest`",
            "",
            "나머지는 지면이 부족하면 supplementary material에 배치한다.",
            "",
            "## 주의",
            "",
            "- Phase 9가 낮은 시나리오도 제거하지 않았다.",
            "- energy, ETX, queue delay는 simulator proxy다.",
            "- clean predictive-break 평균은 seed 분산이 크므로 "
            "`05_seed_stability`와 함께 해석한다.",
        ]
    )
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_manifest() -> None:
    manifest_path = PHASE9 / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["paper_figure_count"] = 15
    manifest["artifacts"] = sorted(
        str(path.relative_to(PHASE9))
        for path in PHASE9.rglob("*")
        if path.is_file() and path != manifest_path
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    configure_style()
    summary_rows = read_csv(SUMMARY_CSV)
    episode_rows = read_csv(EPISODE_CSV)
    training_rows = read_csv(TRAINING_CSV)
    effect_rows = read_csv(EFFECT_CSV)
    scalability_figure(summary_rows)
    robustness_figure(summary_rows)
    challenge_figure(summary_rows)
    ablation_figure(summary_rows)
    seed_stability_figure(summary_rows)
    tradeoff_scatter(summary_rows)
    improvement_figure(summary_rows)
    heatmap_figure(summary_rows)
    delay_stretch_figure(summary_rows)
    efficiency_figure(summary_rows)
    forest_figure(effect_rows)
    delay_ecdf_figure(episode_rows)
    hop_distribution_figure(episode_rows)
    drop_breakdown_figure(summary_rows)
    training_parameter_figure(training_rows)
    write_catalog()
    update_manifest()
    print(f"Generated 15 paper figures in {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
