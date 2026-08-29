"""Paper-ready reports for the external routing baseline suite."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ..experiments.baseline_campaign import BASELINE_METHODS
from ..scenarios import phase9_evaluation_scenarios
from .generalization import GENERALIZATION_METRICS
from .reporting import write_csv
from .statistics import summarize_values


BASELINE_SCENARIOS = tuple(
    scenario.name for scenario in phase9_evaluation_scenarios(0)
)
PROPOSED_METHOD = "SwitchGLOBE"
EXTERNAL_BASELINES = (
    "GPSR",
    "Predictive Geographic",
    "Evo-QGeo",
    "IQMR Q(lambda)",
    "DRAMA",
)


def aggregate_baseline(
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
    for scenario in BASELINE_SCENARIOS:
        for method in BASELINE_METHODS:
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


def paired_baseline_effects(
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
    output: list[dict[str, Any]] = []
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
    for scenario in BASELINE_SCENARIOS:
        for baseline in EXTERNAL_BASELINES:
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
                    diff = base - proposed if metric in lower_is_better else proposed - base
                    differences.append(diff)
                    if abs(base) > 1e-8:
                        relative.append(100.0 * diff / abs(base))
                if differences:
                    stats = summarize_values(differences).to_dict()
                    rel_stats = summarize_values(relative).to_dict() if relative else None
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


def _compact_main_table(rows: list[dict[str, Any]]) -> str:
    lookup = _lookup(rows)
    metrics = (
        "connected_pair_pdr",
        "deadline_delivery_ratio",
        "p95_success_delay",
        "mean_transmission_energy_proxy",
        "mean_policy_input_bytes",
    )
    lines = [
        "# SwitchGLOBE and External Baseline Results",
        "",
        "| Method | Scenario | Connected PDR | Deadline delivery | "
        "Delay p95 | Energy proxy | Input bytes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for scenario in BASELINE_SCENARIOS:
        for method in BASELINE_METHODS:
            values = [
                _estimate(lookup[(scenario, method, metric)])
                for metric in metrics
            ]
            lines.append(
                f"| {method} | {scenario} | "
                + " | ".join(values)
                + " |"
            )
    lines.extend(
        [
            "",
            "값은 training seed 평균 ± 95% t 신뢰구간 반폭이다.",
            "Energy와 input bytes는 simulator-level proxy이며 실제 Joule 또는 "
            "MAC control packet overhead로 직접 해석하지 않는다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _improvement_table(rows: list[dict[str, Any]]) -> str:
    pdr_rows = [
        row
        for row in rows
        if row["metric"] == "connected_pair_pdr"
    ]
    lines = [
        "# SwitchGLOBE Improvement over External Baselines",
        "",
        "| Scenario | Baseline | PDR difference | Relative improvement | 95% CI |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in pdr_rows:
        relative = row["relative_mean_percent"]
        relative_text = (
            "N/A" if relative is None else f"{float(relative):+.2f}%"
        )
        lines.append(
            f"| {row['scenario']} | {row['baseline']} | "
            f"{float(row['mean']):+.4f} | "
            f"{relative_text} | "
            f"[{float(row['ci95_low']):+.4f}, "
            f"{float(row['ci95_high']):+.4f}] |"
        )
    lines.extend(
        [
            "",
            "동일 training seed와 동일 evaluation seed를 묶어 계산한 paired "
            "차이다. 신뢰구간이 0을 포함하면 해당 scenario에서는 우월성 "
            "주장을 보수적으로 제한한다.",
        ]
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
    for method in BASELINE_METHODS:
        means = []
        for scenario in BASELINE_SCENARIOS:
            mean = lookup[(scenario, method, metric)]["mean"]
            means.append(float("nan") if mean is None else float(mean))
        axis.plot(
            BASELINE_SCENARIOS,
            means,
            marker="o",
            label=method,
        )
    if "pdr" in metric or "ratio" in metric:
        axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel(ylabel)
    axis.set_title(f"SwitchGLOBE vs External Baselines: {ylabel}")
    axis.tick_params(axis="x", rotation=25)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=3)
    figure.tight_layout()
    for extension in ("png", "pdf", "svg"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=220)
    plt.close(figure)


def write_baseline_artifacts(
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
    statistics = aggregate_baseline(summary_rows)
    effects = paired_baseline_effects(summary_rows)

    write_csv(raw / "episodes.csv", episode_rows)
    write_csv(raw / "seed_summaries.csv", summary_rows)
    write_csv(raw / "training.csv", training_rows)
    write_csv(summaries / "statistics.csv", statistics)
    write_csv(summaries / "paired_effects.csv", effects)
    summaries.mkdir(parents=True, exist_ok=True)
    (summaries / "statistics.json").write_text(
        json.dumps(statistics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (summaries / "paired_effects.json").write_text(
        json.dumps(effects, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "external_baseline_results.md").write_text(
        _compact_main_table(statistics),
        encoding="utf-8",
    )
    (tables / "switchglobe_improvement_over_external_baselines.md").write_text(
        _improvement_table(effects),
        encoding="utf-8",
    )
    figures.mkdir(parents=True, exist_ok=True)
    _plot_metric(
        statistics,
        metric="connected_pair_pdr",
        ylabel="Connected Pair PDR",
        path=figures / "external_baseline_pdr",
    )
    _plot_metric(
        statistics,
        metric="p95_success_delay",
        ylabel="P95 Success Delay",
        path=figures / "external_baseline_delay_p95",
    )
    _plot_metric(
        statistics,
        metric="mean_transmission_energy_proxy",
        ylabel="Transmission Energy Proxy",
        path=figures / "external_baseline_energy",
    )
    _plot_metric(
        statistics,
        metric="mean_policy_input_bytes",
        ylabel="Policy Input Bytes",
        path=figures / "external_baseline_input_bytes",
    )
    manifest = {
        "suite": "external_baselines",
        "proposed_method": PROPOSED_METHOD,
        "methods": list(BASELINE_METHODS),
        "scenarios": list(BASELINE_SCENARIOS),
        "episode_rows": len(episode_rows),
        "seed_summary_rows": len(summary_rows),
        "statistics_rows": len(statistics),
        "paired_effect_rows": len(effects),
        "training_rows": len(training_rows),
        "metadata": metadata,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest
