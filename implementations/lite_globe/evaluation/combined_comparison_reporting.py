"""Statistics, tables and figures for the combined FastSwitchGLOBE vs SwitchGLOBE
Exact vs external-baseline comparison.

This merges Phase B (external baseline full comparison) and Phase C (ablation)
seed-summary rows into one 8-method view. Only routing-reliability/QoS metrics
that both source campaigns recorded under directly comparable protocol are
included; wall-clock latency is intentionally excluded (see
``EXCLUDED_METRICS_NOTE``) because the two campaigns ran on separate Colab
sessions/devices.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "switchglobe_matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..baselines.registry import EXTERNAL_METHODS
from ..scenarios import phase9_evaluation_scenarios
from .reporting import write_csv
from .statistics import summarize_values


SCENARIOS = tuple(item.name for item in phase9_evaluation_scenarios(0))

SWITCHGLOBE_EXACT = "SwitchGLOBE Exact"
FAST_SWITCHGLOBE = "FastSwitchGLOBE"
COMBINED_METHODS = (*EXTERNAL_METHODS, SWITCHGLOBE_EXACT, FAST_SWITCHGLOBE)
PROPOSED_FAMILY = (SWITCHGLOBE_EXACT, FAST_SWITCHGLOBE)

PRIMARY_METRICS = (
    "connected_pair_pdr",
    "overall_pdr",
    "deadline_delivery_ratio",
    "p95_success_delay",
    "energy_per_delivered_packet",
    "mean_policy_input_bytes",
)
LOWER_IS_BETTER = {"p95_success_delay", "energy_per_delivered_packet", "mean_policy_input_bytes"}

EXCLUDED_METRICS_NOTE = (
    "decision_latency_p95_ms is excluded from this combined comparison: Phase B and "
    "Phase C episodes were captured in separate Colab sessions on different devices, "
    "so their wall-clock timings are not a fair same-session comparison per the "
    "master prompt's latency-fairness rule. Use the dedicated Phase E same-session "
    "latency benchmark for CPU/GPU decision-latency claims instead."
)

DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    *((FAST_SWITCHGLOBE, other) for other in (SWITCHGLOBE_EXACT, *EXTERNAL_METHODS)),
    *((SWITCHGLOBE_EXACT, other) for other in EXTERNAL_METHODS),
)


def validate_rows(rows: list[dict[str, Any]], *, training_seeds: tuple[int, ...],
                   methods: tuple[str, ...] = COMBINED_METHODS) -> None:
    expected = {
        (method, scenario, seed)
        for method in methods
        for scenario in SCENARIOS
        for seed in training_seeds
    }
    actual: set[tuple[str, str, int]] = set()
    for row in rows:
        method = str(row["method"])
        if method not in methods:
            raise ValueError(f"unexpected method in combined rows: {method!r}")
        key = (method, str(row["scenario"]), int(row["training_seed"]))
        if key in actual:
            raise ValueError(f"duplicate method/scenario/seed summary: {key}")
        actual.add(key)
        for metric in PRIMARY_METRICS:
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {metric} for {key}")
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(f"summary contract mismatch: missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}")


def aggregate(rows: list[dict[str, Any]], *, methods: tuple[str, ...] = COMBINED_METHODS) -> list[dict[str, Any]]:
    """Per (scenario, method, metric) 5-seed aggregate."""

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        method = str(row["method"])
        if method not in methods:
            continue
        for metric in PRIMARY_METRICS:
            grouped[(str(row["scenario"]), method, metric)].append(float(row[metric]))
    output = []
    for scenario in SCENARIOS:
        for method in methods:
            for metric in PRIMARY_METRICS:
                values = grouped[(scenario, method, metric)]
                if not values:
                    raise ValueError(f"empty confidence-interval input: {(scenario, method, metric)}")
                output.append({"scenario": scenario, "method": method, "metric": metric, **summarize_values(values).to_dict()})
    return output


def macro_aggregate(rows: list[dict[str, Any]], *, methods: tuple[str, ...] = COMBINED_METHODS) -> list[dict[str, Any]]:
    """Per (method, metric) overall aggregate: mean each seed across all 14
    scenarios first (scenario-macro average), then 95% CI across the 5 seeds."""

    per_seed: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        method = str(row["method"])
        if method not in methods:
            continue
        seed = int(row["training_seed"])
        for metric in PRIMARY_METRICS:
            per_seed[(method, metric)][seed].append(float(row[metric]))
    output = []
    for method in methods:
        for metric in PRIMARY_METRICS:
            seed_means = [sum(values) / len(values) for values in per_seed[(method, metric)].values()]
            if not seed_means:
                raise ValueError(f"empty macro-aggregate input: {(method, metric)}")
            output.append({"method": method, "metric": metric, **summarize_values(seed_means).to_dict()})
    return output


def _paired_seed_series(rows: list[dict[str, Any]], *, scenario: str | None,
                         method: str, metric: str,
                         training_seeds: tuple[int, ...]) -> dict[int, float]:
    matches = [
        row for row in rows
        if str(row["method"]) == method and (scenario is None or str(row["scenario"]) == scenario)
    ]
    if scenario is None:
        per_seed: dict[int, list[float]] = defaultdict(list)
        for row in matches:
            per_seed[int(row["training_seed"])].append(float(row[metric]))
        return {seed: sum(values) / len(values) for seed, values in per_seed.items() if seed in training_seeds}
    return {
        int(row["training_seed"]): float(row[metric])
        for row in matches
        if int(row["training_seed"]) in training_seeds
    }


def paired_effects(rows: list[dict[str, Any]], *, pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS,
                    training_seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    """Paired (method_a vs method_b) contrasts, paired within training seed.

    A positive ``mean`` means ``method_a`` is better than ``method_b`` on that
    metric's own direction.
    """

    output = []
    for scenario in SCENARIOS:
        for method_a, method_b in pairs:
            if method_a == method_b:
                raise ValueError("paired contrast requires two distinct methods")
            for metric in PRIMARY_METRICS:
                a_by_seed = _paired_seed_series(rows, scenario=scenario, method=method_a, metric=metric, training_seeds=training_seeds)
                b_by_seed = _paired_seed_series(rows, scenario=scenario, method=method_b, metric=metric, training_seeds=training_seeds)
                differences, relative = [], []
                for seed in training_seeds:
                    if seed not in a_by_seed or seed not in b_by_seed:
                        continue
                    avalue, bvalue = a_by_seed[seed], b_by_seed[seed]
                    difference = bvalue - avalue if metric in LOWER_IS_BETTER else avalue - bvalue
                    differences.append(difference)
                    if abs(bvalue) > 1e-12:
                        relative.append(100.0 * difference / abs(bvalue))
                if not differences:
                    raise ValueError(f"empty paired contrast for {method_a}/{scenario}/{method_b}/{metric}")
                stats = summarize_values(differences).to_dict()
                relative_stats = summarize_values(relative).to_dict() if relative else None
                output.append({
                    "method_a": method_a, "method_b": method_b,
                    "scenario": scenario, "metric": metric,
                    "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                    **stats,
                    "relative_mean_percent": relative_stats["mean"] if relative_stats else None,
                    "relative_ci95_low_percent": relative_stats["ci95_low"] if relative_stats else None,
                    "relative_ci95_high_percent": relative_stats["ci95_high"] if relative_stats else None,
                })
    return output


def macro_paired_effects(rows: list[dict[str, Any]], *, pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS,
                          training_seeds: tuple[int, ...]) -> list[dict[str, Any]]:
    """Same as ``paired_effects`` but scenario-macro-averaged first (one row
    per pair per metric, matching the acceptance-gate "scenario-macro paired
    mean" convention already used for the Phase C gates)."""

    output = []
    for method_a, method_b in pairs:
        if method_a == method_b:
            raise ValueError("paired contrast requires two distinct methods")
        for metric in PRIMARY_METRICS:
            a_by_seed = _paired_seed_series(rows, scenario=None, method=method_a, metric=metric, training_seeds=training_seeds)
            b_by_seed = _paired_seed_series(rows, scenario=None, method=method_b, metric=metric, training_seeds=training_seeds)
            differences, relative = [], []
            for seed in training_seeds:
                if seed not in a_by_seed or seed not in b_by_seed:
                    continue
                avalue, bvalue = a_by_seed[seed], b_by_seed[seed]
                difference = bvalue - avalue if metric in LOWER_IS_BETTER else avalue - bvalue
                differences.append(difference)
                if abs(bvalue) > 1e-12:
                    relative.append(100.0 * difference / abs(bvalue))
            if not differences:
                raise ValueError(f"empty scenario-macro paired contrast for {method_a}/{method_b}/{metric}")
            stats = summarize_values(differences).to_dict()
            relative_stats = summarize_values(relative).to_dict() if relative else None
            output.append({
                "method_a": method_a, "method_b": method_b, "metric": metric,
                "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                "evidence": "five-seed scenario-macro paired mean",
                **stats,
                "relative_mean_percent": relative_stats["mean"] if relative_stats else None,
                "relative_ci95_low_percent": relative_stats["ci95_low"] if relative_stats else None,
                "relative_ci95_high_percent": relative_stats["ci95_high"] if relative_stats else None,
            })
    return output


def _best_second_best(rows: list[dict[str, Any]], metric: str, methods: tuple[str, ...]) -> tuple[str | None, str | None]:
    ranked = sorted(
        (row for row in rows if row["metric"] == metric and row["method"] in methods),
        key=lambda row: row["mean"],
        reverse=(metric not in LOWER_IS_BETTER),
    )
    if not ranked:
        return None, None
    best = ranked[0]["method"]
    second = ranked[1]["method"] if len(ranked) > 1 else None
    return best, second


def _overview_table(macro_rows: list[dict[str, Any]], *, methods: tuple[str, ...] = COMBINED_METHODS) -> str:
    lookup = {(row["method"], row["metric"]): row for row in macro_rows}
    best_second = {metric: _best_second_best(macro_rows, metric, methods) for metric in PRIMARY_METRICS}
    header = ["Method", "Connected-pair PDR", "Overall PDR", "Deadline ratio", "P95 success delay", "Energy/delivered", "Policy input bytes"]
    lines = [
        "# Combined comparison: external baselines + SwitchGLOBE Exact + FastSwitchGLOBE",
        "",
        "5-seed scenario-macro aggregate (mean across all 14 scenarios, then 95% CI across seeds 42/77/123/314/2718).",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for method in methods:
        cells = [method]
        for metric in PRIMARY_METRICS:
            row = lookup[(method, metric)]
            half = (float(row["ci95_high"]) - float(row["ci95_low"])) / 2
            text = f"{float(row['mean']):.4f} ± {half:.4f}"
            best, second = best_second[metric]
            if method == best:
                text = f"**{text}**"
            elif method == second:
                text = f"*{text}*"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend([
        "",
        "Bold = best per metric direction, italic = second-best (computed automatically, not manual).",
        "Energy는 simulator transmission-energy proxy이며 Joule이 아니다. Policy input bytes는 routing-control overhead가 아니다.",
        EXCLUDED_METRICS_NOTE,
    ])
    return "\n".join(lines) + "\n"


def _csv_table(macro_rows: list[dict[str, Any]], *, methods: tuple[str, ...] = COMBINED_METHODS) -> list[dict[str, Any]]:
    lookup = {(row["method"], row["metric"]): row for row in macro_rows}
    out = []
    for method in methods:
        row = {"method": method}
        for metric in PRIMARY_METRICS:
            source = lookup[(method, metric)]
            row[f"{metric}_mean"] = source["mean"]
            row[f"{metric}_ci95_low"] = source["ci95_low"]
            row[f"{metric}_ci95_high"] = source["ci95_high"]
        out.append(row)
    return out


def _latex_table(macro_rows: list[dict[str, Any]], *, methods: tuple[str, ...] = COMBINED_METHODS) -> str:
    lookup = {(row["method"], row["metric"]): row for row in macro_rows}
    best_second = {metric: _best_second_best(macro_rows, metric, methods) for metric in PRIMARY_METRICS}
    lines = [
        r"\begin{tabular}{l" + "r" * len(PRIMARY_METRICS) + "}",
        r"\toprule",
        "Method & " + " & ".join(metric.replace("_", r"\_") for metric in PRIMARY_METRICS) + r" \\",
        r"\midrule",
    ]
    for method in methods:
        cells = []
        for metric in PRIMARY_METRICS:
            row = lookup[(method, metric)]
            half = (float(row["ci95_high"]) - float(row["ci95_low"])) / 2
            text = f"{float(row['mean']):.4f} $\\pm$ {half:.4f}"
            best, second = best_second[metric]
            if method == best:
                text = r"\textbf{" + text + "}"
            elif method == second:
                text = r"\textit{" + text + "}"
            cells.append(text)
        lines.append(method.replace("_", r"\_") + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _bar_figure(macro_rows: list[dict[str, Any]], metric: str, path: Path, *,
                methods: tuple[str, ...] = COMBINED_METHODS) -> None:
    lookup = {(row["method"], row["metric"]): row for row in macro_rows}
    palette = {method: ("#1b9e77" if method in PROPOSED_FAMILY else "#7570b3") for method in methods}
    means = [lookup[(method, metric)]["mean"] for method in methods]
    lows = [lookup[(method, metric)]["mean"] - lookup[(method, metric)]["ci95_low"] for method in methods]
    highs = [lookup[(method, metric)]["ci95_high"] - lookup[(method, metric)]["mean"] for method in methods]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(methods, means, yerr=[lows, highs], capsize=4, color=[palette[m] for m in methods])
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", rotation=30)
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def _panel_figure(macro_rows: list[dict[str, Any]], path: Path, *,
                   methods: tuple[str, ...] = COMBINED_METHODS) -> None:
    metrics = ("connected_pair_pdr", "deadline_delivery_ratio", "energy_per_delivered_packet")
    lookup = {(row["method"], row["metric"]): row for row in macro_rows}
    palette = {method: ("#1b9e77" if method in PROPOSED_FAMILY else "#7570b3") for method in methods}
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    for axis, metric in zip(axes, metrics):
        means = [lookup[(method, metric)]["mean"] for method in methods]
        lows = [lookup[(method, metric)]["mean"] - lookup[(method, metric)]["ci95_low"] for method in methods]
        highs = [lookup[(method, metric)]["ci95_high"] - lookup[(method, metric)]["mean"] for method in methods]
        axis.bar(methods, means, yerr=[lows, highs], capsize=4, color=[palette[m] for m in methods])
        axis.set_title(metric)
        axis.tick_params(axis="x", rotation=45)
        axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=300)
    plt.close(fig)


def write_combined_comparison_artifacts(
    output_dir: Path, *, seed_summary_rows: list[dict[str, Any]], episode_rows: list[dict[str, Any]],
    metadata: dict[str, Any], methods: tuple[str, ...] = COMBINED_METHODS,
    pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS,
) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in metadata["training_seeds"])
    validate_rows(seed_summary_rows, training_seeds=seeds, methods=methods)
    expected_summary_rows = len(methods) * len(SCENARIOS) * len(seeds)
    if len(seed_summary_rows) != expected_summary_rows:
        raise ValueError(f"seed summary row count {len(seed_summary_rows)} != {expected_summary_rows}")

    per_scenario_stats = aggregate(seed_summary_rows, methods=methods)
    macro_stats = macro_aggregate(seed_summary_rows, methods=methods)
    per_scenario_pairs = paired_effects(seed_summary_rows, pairs=pairs, training_seeds=seeds)
    macro_pairs = macro_paired_effects(seed_summary_rows, pairs=pairs, training_seeds=seeds)

    write_csv(output_dir / "raw" / "episodes.csv", episode_rows)
    write_csv(output_dir / "raw" / "seed_summaries.csv", seed_summary_rows)
    write_csv(output_dir / "summaries" / "statistics_by_scenario.csv", per_scenario_stats)
    write_csv(output_dir / "summaries" / "statistics_overall.csv", macro_stats)
    write_csv(output_dir / "summaries" / "paired_effects_by_scenario.csv", per_scenario_pairs)
    write_csv(output_dir / "summaries" / "paired_effects_overall.csv", macro_pairs)
    write_csv(output_dir / "tables" / "combined_comparison.csv", _csv_table(macro_stats, methods=methods))
    (output_dir / "summaries").mkdir(parents=True, exist_ok=True)
    (output_dir / "summaries" / "statistics_by_scenario.json").write_text(json.dumps(per_scenario_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summaries" / "statistics_overall.json").write_text(json.dumps(macro_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summaries" / "paired_effects_by_scenario.json").write_text(json.dumps(per_scenario_pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summaries" / "paired_effects_overall.json").write_text(json.dumps(macro_pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables" / "combined_comparison.md").write_text(_overview_table(macro_stats, methods=methods), encoding="utf-8")
    (output_dir / "tables" / "combined_comparison.tex").write_text(_latex_table(macro_stats, methods=methods), encoding="utf-8")

    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    _bar_figure(macro_stats, "connected_pair_pdr", output_dir / "figures" / "fig_combined_connected_pdr_by_method", methods=methods)
    _bar_figure(macro_stats, "deadline_delivery_ratio", output_dir / "figures" / "fig_combined_deadline_ratio_by_method", methods=methods)
    _panel_figure(macro_stats, output_dir / "figures" / "fig_combined_pdr_deadline_energy_by_method", methods=methods)

    manifest = {
        "schema_version": 1,
        "complete": True,
        "suite": "switchglobe_combined_comparison",
        "methods": list(methods),
        "scenarios": list(SCENARIOS),
        "primary_metrics": list(PRIMARY_METRICS),
        "excluded_metrics_note": EXCLUDED_METRICS_NOTE,
        "seed_summary_rows": len(seed_summary_rows),
        "expected_seed_summary_rows": expected_summary_rows,
        "episode_rows": len(episode_rows),
        "statistics_by_scenario_rows": len(per_scenario_stats),
        "statistics_overall_rows": len(macro_stats),
        "paired_effect_by_scenario_rows": len(per_scenario_pairs),
        "paired_effect_overall_rows": len(macro_pairs),
        "metadata": metadata,
    }
    return manifest
