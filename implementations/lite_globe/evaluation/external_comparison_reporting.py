"""Validated statistics, tables and figures for external comparison only."""

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

from ..baselines.registry import COMPARISON_METHODS, EXTERNAL_METHODS, PROPOSED_METHOD
from ..scenarios import phase9_evaluation_scenarios
from .reporting import write_csv
from .statistics import summarize_values


SCENARIOS = tuple(item.name for item in phase9_evaluation_scenarios(0))
PRIMARY_METRICS = (
    "connected_pair_pdr",
    "deadline_delivery_ratio",
    "p95_success_delay",
    "energy_per_delivered_packet",
    "decision_latency_p95_ms",
    "mean_policy_input_bytes",
)
LOWER_IS_BETTER = {
    "p95_success_delay", "energy_per_delivered_packet",
    "decision_latency_p95_ms", "mean_policy_input_bytes",
}


def validate_rows(rows: list[dict[str, Any]], *, training_seeds: tuple[int, ...],
                  comparison_methods: tuple[str, ...] = COMPARISON_METHODS) -> None:
    expected = {
        (method, scenario, seed)
        for method in comparison_methods
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
            value = float(row[metric])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {metric} for {key}")
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(f"summary contract mismatch: missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}")


def aggregate(rows: list[dict[str, Any]], *,
              comparison_methods: tuple[str, ...] = COMPARISON_METHODS) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        for metric in PRIMARY_METRICS:
            grouped[(str(row["scenario"]), str(row["method"]), metric)].append(float(row[metric]))
    output = []
    for scenario in SCENARIOS:
        for method in comparison_methods:
            for metric in PRIMARY_METRICS:
                values = grouped[(scenario, method, metric)]
                if not values:
                    raise ValueError(f"empty confidence-interval input: {(scenario, method, metric)}")
                output.append({"scenario": scenario, "method": method, "metric": metric, **summarize_values(values).to_dict()})
    return output


def paired_effects(rows: list[dict[str, Any]], *,
                   proposed_methods: tuple[str, ...] = (PROPOSED_METHOD,),
                   external_methods: tuple[str, ...] = EXTERNAL_METHODS) -> list[dict[str, Any]]:
    lookup = {(str(row["scenario"]), str(row["method"]), int(row["training_seed"])): row for row in rows}
    output = []
    for scenario in SCENARIOS:
        for proposed_method in proposed_methods:
            for baseline in external_methods:
                if baseline == proposed_method:
                    raise ValueError("proposed method must not appear as its own paired baseline")
                for metric in PRIMARY_METRICS:
                    differences, relative = [], []
                    for (row_scenario, method, seed), proposed in lookup.items():
                        if row_scenario != scenario or method != proposed_method:
                            continue
                        base = lookup[(scenario, baseline, seed)]
                        pvalue, bvalue = float(proposed[metric]), float(base[metric])
                        difference = bvalue - pvalue if metric in LOWER_IS_BETTER else pvalue - bvalue
                        differences.append(difference)
                        if abs(bvalue) > 1e-12:
                            relative.append(100.0 * difference / abs(bvalue))
                    if not differences:
                        raise ValueError(
                            f"empty paired contrast for {proposed_method}/{scenario}/{baseline}/{metric}"
                        )
                    stats = summarize_values(differences).to_dict()
                    relative_stats = summarize_values(relative).to_dict() if relative else None
                    output.append({
                        "proposed_method": proposed_method,
                        "scenario": scenario, "baseline": baseline, "metric": metric,
                        "direction": "lower_is_better" if metric in LOWER_IS_BETTER else "higher_is_better",
                        **stats,
                        "relative_mean_percent": relative_stats["mean"] if relative_stats else None,
                        "relative_ci95_low_percent": relative_stats["ci95_low"] if relative_stats else None,
                        "relative_ci95_high_percent": relative_stats["ci95_high"] if relative_stats else None,
                    })
    return output


def _table(rows: list[dict[str, Any]], *, comparison_methods: tuple[str, ...]) -> str:
    lookup = {(row["scenario"], row["method"], row["metric"]): row for row in rows}
    lines = [
        "# External comparison (verified aggregates)", "",
        "| Method | Scenario | Connected PDR | Deadline ratio | Delay p95 | Energy/delivered | Decision p95 ms | Input bytes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in SCENARIOS:
        for method in comparison_methods:
            values = []
            for metric in PRIMARY_METRICS:
                row = lookup[(scenario, method, metric)]
                half = (float(row["ci95_high"]) - float(row["ci95_low"])) / 2
                values.append(f"{float(row['mean']):.4f} ± {half:.4f}")
            lines.append(f"| {method} | {scenario} | " + " | ".join(values) + " |")
    lines.extend(["", "Energy는 simulator transmission proxy이며 Joule이 아니다. Input bytes는 control overhead가 아니다."])
    return "\n".join(lines) + "\n"


def _plot(rows: list[dict[str, Any]], metric: str, path: Path, *,
          comparison_methods: tuple[str, ...]) -> None:
    lookup = {(row["scenario"], row["method"], row["metric"]): row for row in rows}
    fig, axis = plt.subplots(figsize=(14, 6))
    for method in comparison_methods:
        axis.plot(SCENARIOS, [lookup[(scenario, method, metric)]["mean"] for scenario in SCENARIOS], marker="o", label=method)
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", rotation=25)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=220)
    plt.close(fig)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_external_comparison_artifacts(output_dir: Path, *, episode_rows: list[dict[str, Any]],
                                        summary_rows: list[dict[str, Any]], training_rows: list[dict[str, Any]],
                                        deployment_rows: list[dict[str, Any]], method_contracts: list[dict[str, Any]],
                                        metadata: dict[str, Any],
                                        comparison_methods: tuple[str, ...] = COMPARISON_METHODS,
                                        proposed_methods: tuple[str, ...] = (PROPOSED_METHOD,),
                                        external_methods: tuple[str, ...] = EXTERNAL_METHODS) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in metadata["config"]["training_seeds"])
    if len(comparison_methods) != len(set(comparison_methods)):
        raise ValueError("comparison methods must be unique")
    if not set(proposed_methods).issubset(comparison_methods):
        raise ValueError("all proposed methods must be present in comparison methods")
    if not set(external_methods).issubset(comparison_methods):
        raise ValueError("all external methods must be present in comparison methods")
    validate_rows(
        summary_rows, training_seeds=seeds,
        comparison_methods=comparison_methods,
    )
    expected_episodes = len(comparison_methods) * len(SCENARIOS) * len(seeds) * int(metadata["config"]["evaluation_episodes"])
    if len(episode_rows) != expected_episodes:
        raise ValueError(f"episode row count {len(episode_rows)} != {expected_episodes}")
    statistics = aggregate(summary_rows, comparison_methods=comparison_methods)
    effects = paired_effects(
        summary_rows, proposed_methods=proposed_methods,
        external_methods=external_methods,
    )
    write_csv(output_dir / "raw" / "episodes.csv", episode_rows)
    write_csv(output_dir / "raw" / "seed_summaries.csv", summary_rows)
    write_csv(output_dir / "raw" / "training.csv", training_rows)
    write_csv(output_dir / "raw" / "deployment_costs.csv", deployment_rows)
    write_csv(output_dir / "summaries" / "statistics.csv", statistics)
    write_csv(output_dir / "summaries" / "paired_effects.csv", effects)
    (output_dir / "summaries").mkdir(parents=True, exist_ok=True)
    (output_dir / "summaries" / "statistics.json").write_text(json.dumps(statistics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summaries" / "paired_effects.json").write_text(json.dumps(effects, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables" / "external_comparison.md").write_text(
        _table(statistics, comparison_methods=comparison_methods), encoding="utf-8"
    )
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    for metric in ("connected_pair_pdr", "p95_success_delay", "energy_per_delivered_packet", "decision_latency_p95_ms"):
        _plot(
            statistics, metric, output_dir / "figures" / metric,
            comparison_methods=comparison_methods,
        )
    manifest = {
        "schema_version": 1, "complete": True, "suite": "external_comparison",
        "mode": metadata["mode"], "methods": list(comparison_methods), "scenarios": list(SCENARIOS),
        "proposed_methods": list(proposed_methods),
        "external_methods": list(external_methods),
        "episode_rows": len(episode_rows), "seed_summary_rows": len(summary_rows),
        "expected_episode_rows": expected_episodes, "statistics_rows": len(statistics),
        "paired_effect_rows": len(effects), "method_contracts": method_contracts, "metadata": metadata,
        "deployment_cost_rows": len(deployment_rows),
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest
