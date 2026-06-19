"""Paper-ready Phase 8 optimization reports and proxy-metric disclosures."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from ..scenarios.generalization_suite import phase8_evaluation_scenarios
from .generalization import GENERALIZATION_METRICS
from .reporting import write_csv
from .statistics import summarize_values


PHASE8_SCENARIOS = tuple(
    scenario.name for scenario in phase8_evaluation_scenarios(0)
)
PHASE8_METHODS = (
    "GPSR",
    "Phase 7 KD-only",
    "Untrained Geo-Residual",
    "Geo-Residual KD",
    "Global Teacher",
    "Shortest-path Oracle",
)


def aggregate_phase8(
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
    for scenario in PHASE8_SCENARIOS:
        for method in PHASE8_METHODS:
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
    lines = [
        "# Phase 8 Optimized Routing Results",
        "",
        "| Method | Scenario | Connected PDR | Delay p95 | Path stretch | "
        "ETX proxy | Energy proxy |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    metrics = (
        "connected_pair_pdr",
        "p95_success_delay",
        "mean_path_stretch",
        "mean_expected_transmissions_proxy",
        "mean_transmission_energy_proxy",
    )
    for method in PHASE8_METHODS:
        for scenario in PHASE8_SCENARIOS:
            values = [
                _estimate(lookup[(scenario, method, metric)])
                for metric in metrics
            ]
            lines.append(
                f"| {method} | {scenario} | " + " | ".join(values) + " |"
            )
    lines.extend(
        [
            "",
            "`ETX proxy`는 독립 link-loss 가정의 기대 송신 횟수 대용치다.",
            "`Energy proxy`는 각 hop의 `(거리/통신반경)^2` 합이며 실제 Joule이 아니다.",
            "실제 routing-control overhead와 PHY energy는 ns-3/무선 모델 전에는 "
            "주장하지 않는다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _efficiency_table(rows: list[dict[str, Any]]) -> str:
    lookup = _lookup(rows)
    lines = [
        "# Phase 8 Efficiency and Reliability Results",
        "",
        "| Method | Scenario | Mean delay | Link lifetime proxy | "
        "Policy input bytes | Local-state bytes | Agent drop | TTL drop | "
        "Invalid drop |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    metrics = (
        "mean_success_delay",
        "mean_minimum_link_lifetime_steps",
        "mean_policy_input_bytes",
        "mean_local_observation_bytes",
        "agent_drop_rate",
        "ttl_drop_rate",
        "invalid_action_drop_rate",
    )
    for method in PHASE8_METHODS:
        for scenario in PHASE8_SCENARIOS:
            values = [
                _estimate(lookup[(scenario, method, metric)])
                for metric in metrics
            ]
            lines.append(
                f"| {method} | {scenario} | " + " | ".join(values) + " |"
            )
    lines.extend(
        [
            "",
            "`Policy input bytes`는 episode 동안 각 정책이 실제 참조한 배열의 "
            "누적 크기다. `Local-state bytes`는 환경이 획득한 전체 local state "
            "크기이며 둘 다 네트워크 제어 패킷 overhead와 동일하지 않다.",
            "`Link lifetime proxy`는 현재 상대 위치·속도가 유지된다는 가정의 "
            "선택 링크 잔여 수명 최솟값이다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    lookup = _lookup(rows)
    figure, axis = plt.subplots(figsize=(11, 6))
    for method in PHASE8_METHODS:
        axis.plot(
            PHASE8_SCENARIOS,
            [
                float(
                    lookup[
                        (scenario, method, "connected_pair_pdr")
                    ]["mean"]
                )
                for scenario in PHASE8_SCENARIOS
            ],
            marker="o",
            label=method,
        )
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("Connected-pair PDR")
    axis.set_title("Phase 8 Geo-Residual Distillation")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    for extension in ("png", "pdf", "svg"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=200)
    plt.close(figure)


def write_phase8_artifacts(
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
    aggregates = aggregate_phase8(summary_rows)
    write_csv(raw / "episodes.csv", episode_rows)
    write_csv(raw / "seed_summaries.csv", summary_rows)
    write_csv(raw / "training_metrics.csv", training_rows)
    write_csv(summaries / "statistics.csv", aggregates)
    (summaries / "statistics.json").write_text(
        json.dumps(aggregates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (tables / "optimized_results.md").write_text(
        _main_table(aggregates), encoding="utf-8"
    )
    (tables / "efficiency_results.md").write_text(
        _efficiency_table(aggregates), encoding="utf-8"
    )
    _plot(aggregates, figures / "optimized_pdr")
    manifest = {
        **metadata,
        "episode_rows": len(episode_rows),
        "seed_summary_rows": len(summary_rows),
        "statistics_rows": len(aggregates),
        "methods": list(PHASE8_METHODS),
        "scenarios": list(PHASE8_SCENARIOS),
        "proxy_metric_warning": (
            "ETX and energy are simulator-level proxies, not PHY measurements."
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
