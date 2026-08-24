"""Paper-ready Phase 9 reports with paired effects and proxy disclosures."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ..scenarios import phase9_evaluation_scenarios
from .generalization import GENERALIZATION_METRICS
from .reporting import write_csv
from .statistics import summarize_values


PHASE9_SCENARIOS = tuple(
    scenario.name for scenario in phase9_evaluation_scenarios(0)
)
PHASE9_METHODS = (
    "GPSR",
    "Predictive Geographic",
    "Phase 8 Geo-Residual KD",
    "Phase 9 no-risk ablation",
    "Phase 9 no-forwardability ablation",
    "Phase 9 geographic+risk only",
    "Phase 9 Risk-Aware KD",
    "Global Teacher",
    "Shortest-path Oracle",
    "Risk-aware Oracle",
)
MAIN_METHODS = (
    "GPSR",
    "Predictive Geographic",
    "Phase 8 Geo-Residual KD",
    "Phase 9 Risk-Aware KD",
    "Global Teacher",
    "Shortest-path Oracle",
    "Risk-aware Oracle",
)


def aggregate_phase9(
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
    for scenario in PHASE9_SCENARIOS:
        for method in PHASE9_METHODS:
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


def paired_phase9_effects(
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
    output = []
    for scenario in PHASE9_SCENARIOS:
        for baseline in (
            "GPSR",
            "Predictive Geographic",
            "Phase 8 Geo-Residual KD",
        ):
            differences = []
            for key in by_key:
                row_scenario, method, seed = key
                if (
                    row_scenario != scenario
                    or method != "Phase 9 Risk-Aware KD"
                ):
                    continue
                baseline_row = by_key.get((scenario, baseline, seed))
                if baseline_row is None:
                    continue
                differences.append(
                    float(by_key[key]["connected_pair_pdr"])
                    - float(baseline_row["connected_pair_pdr"])
                )
            if differences:
                output.append(
                    {
                        "scenario": scenario,
                        "comparison": (
                            f"Phase 9 Risk-Aware KD - {baseline}"
                        ),
                        **summarize_values(differences).to_dict(),
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
        "mean_path_stretch",
    )
    lines = [
        "# Phase 9 Reliability Results",
        "",
        "| Method | Scenario | Connected PDR | Deadline delivery | "
        "Delay p95 | Path stretch |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for method in PHASE9_METHODS:
        for scenario in PHASE9_SCENARIOS:
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
            "`Deadline delivery`는 초기 최단 hop 수의 1.5배에 1 step을 더한 "
            "사전 정의 deadline 안에 전달된 비율이다.",
            "PDR을 최우선 지표로 유지하며 deadline이나 종합점수로 실패 "
            "패킷을 숨기지 않는다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _resource_table(rows: list[dict[str, Any]]) -> str:
    lookup = _lookup(rows)
    metrics = (
        "delivery_energy_efficiency_proxy",
        "delivery_transmission_efficiency_proxy",
        "mean_queue_delay_proxy",
        "mean_minimum_link_lifetime_steps",
        "mean_minimum_link_margin",
        "mean_policy_input_bytes",
    )
    lines = [
        "# Phase 9 Efficiency and Stability Results",
        "",
        "| Method | Scenario | Delivery/energy | Delivery/ETX | "
        "Queue-delay proxy | Link lifetime | Link margin | Input bytes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in PHASE9_METHODS:
        for scenario in PHASE9_SCENARIOS:
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
            "Energy, ETX, queue delay, link lifetime은 simulator-level proxy다.",
            "실제 Joule, MAC 재전송, queueing delay, control packet overhead로 "
            "해석하지 않는다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _effect_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Phase 9 Paired PDR Effects",
        "",
        "| Scenario | Comparison | Mean difference | 95% CI |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['comparison']} | "
            f"{float(row['mean']):+.4f} | "
            f"[{float(row['ci95_low']):+.4f}, "
            f"{float(row['ci95_high']):+.4f}] |"
        )
    lines.extend(
        [
            "",
            "동일 training seed와 동일 evaluation seed 집합을 사용한 "
            "seed-level paired 차이다. 신뢰구간이 0을 포함하면 우월성을 "
            "확정하지 않는다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    lookup = _lookup(rows)
    figure, axis = plt.subplots(figsize=(14, 6))
    for method in MAIN_METHODS:
        axis.plot(
            PHASE9_SCENARIOS,
            [
                float(
                    lookup[
                        (scenario, method, "connected_pair_pdr")
                    ]["mean"]
                )
                for scenario in PHASE9_SCENARIOS
            ],
            marker="o",
            label=method,
        )
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("Connected-pair PDR")
    axis.set_title("Phase 9 Risk-Aware Geo-Residual Evaluation")
    axis.tick_params(axis="x", rotation=28)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    for extension in ("png", "pdf", "svg"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=220)
    plt.close(figure)


def write_phase9_artifacts(
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
    aggregates = aggregate_phase9(summary_rows)
    effects = paired_phase9_effects(summary_rows)
    write_csv(raw / "episodes.csv", episode_rows)
    write_csv(raw / "seed_summaries.csv", summary_rows)
    write_csv(raw / "training_metrics.csv", training_rows)
    write_csv(summaries / "statistics.csv", aggregates)
    write_csv(summaries / "paired_effects.csv", effects)
    (summaries / "statistics.json").write_text(
        json.dumps(aggregates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (tables / "reliability_results.md").write_text(
        _main_table(aggregates), encoding="utf-8"
    )
    (tables / "efficiency_results.md").write_text(
        _resource_table(aggregates), encoding="utf-8"
    )
    (tables / "paired_effects.md").write_text(
        _effect_table(effects), encoding="utf-8"
    )
    _plot(aggregates, figures / "phase9_pdr")
    manifest = {
        **metadata,
        "episode_rows": len(episode_rows),
        "seed_summary_rows": len(summary_rows),
        "statistics_rows": len(aggregates),
        "paired_effect_rows": len(effects),
        "methods": list(PHASE9_METHODS),
        "scenarios": list(PHASE9_SCENARIOS),
        "primary_metric": "connected_pair_pdr",
        "metric_policy": (
            "PDR remains primary; proxy metrics are secondary and disclosed."
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
