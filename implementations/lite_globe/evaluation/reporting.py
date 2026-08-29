"""Paper-ready Phase 6 result serialization, tables, and figures."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
import tempfile
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "lite_globe_matplotlib"),
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .records import SUMMARY_METRICS
from .statistics import summarize_values


METHOD_ORDER = (
    "Random",
    "GPSR",
    "Untrained Student",
    "PPO-only Student",
    "KD-only Student",
    "KD+PPO Student",
    "Global Teacher",
)
SCENARIO_ORDER = (
    "routing_hole",
    "routing_hole_link_loss",
    "mobile_dense",
    "mobile_sparse",
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_seed_summaries(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    combinations = {
        (row["scenario"], row["method"])
        for row in rows
    }
    for row in rows:
        for metric in SUMMARY_METRICS:
            if metric == "mean_delay_steps" and int(row["delivered"]) == 0:
                continue
            grouped[(row["scenario"], row["method"], metric)].append(
                float(row[metric])
            )
    aggregated: list[dict[str, Any]] = []
    for scenario, method in sorted(combinations):
        for metric in SUMMARY_METRICS:
            values = grouped.get((scenario, method, metric), [])
            if values:
                statistics: dict[str, int | float | None] = (
                    summarize_values(values).to_dict()
                )
            else:
                statistics = {
                    "count": 0,
                    "mean": None,
                    "standard_deviation": None,
                    "ci95_low": None,
                    "ci95_high": None,
                }
            aggregated.append(
                {
                    "scenario": scenario,
                    "method": method,
                    "metric": metric,
                    **statistics,
                }
            )
    return aggregated


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


def render_main_markdown(rows: list[dict[str, Any]]) -> str:
    lookup = _lookup(rows)
    lines = [
        "# Phase 6 In-Distribution Results",
        "",
        "| Method | PDR | Delay | Throughput | Reward |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHOD_ORDER:
        values = [
            _estimate(lookup[("routing_hole", method, metric)])
            for metric in (
                "packet_delivery_ratio",
                "mean_delay_steps",
                "throughput_packets_per_step",
                "mean_episode_reward",
            )
        ]
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "값은 학습 seed 평균 ± 95% t 신뢰구간 반폭이다.",
            "전달 실패 episode는 delay 평균에서 제외되며 PDR과 함께 해석해야 한다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _latex_escape(text: str) -> str:
    return text.replace("+", r"\texttt{+}").replace("_", r"\_")


def render_main_latex(rows: list[dict[str, Any]]) -> str:
    lookup = _lookup(rows)
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & PDR & Delay & Throughput & Reward \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        values = [
            _estimate(lookup[("routing_hole", method, metric)]).replace(
                "±", r"$\pm$"
            )
            for metric in (
                "packet_delivery_ratio",
                "mean_delay_steps",
                "throughput_packets_per_step",
                "mean_episode_reward",
            )
        ]
        lines.append(
            f"{_latex_escape(method)} & " + " & ".join(values) + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_ood_markdown(rows: list[dict[str, Any]]) -> str:
    lookup = _lookup(rows)
    available = {row["scenario"] for row in rows}
    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in available]
    lines = [
        "# Phase 6 OOD PDR",
        "",
        "| Method | " + " | ".join(scenarios) + " |",
        "| --- | " + " | ".join("---:" for _ in scenarios) + " |",
    ]
    for method in METHOD_ORDER:
        values = [
            _estimate(
                lookup[(scenario, method, "packet_delivery_ratio")]
            )
            for scenario in scenarios
        ]
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _plot_metric(
    rows: list[dict[str, Any]],
    *,
    scenario: str,
    metric: str,
    ylabel: str,
    path: Path,
) -> None:
    lookup = _lookup(rows)
    selected = [lookup[(scenario, method, metric)] for method in METHOD_ORDER]
    means = [float(row["mean"]) for row in selected]
    errors = [
        (float(row["ci95_high"]) - float(row["ci95_low"])) / 2
        for row in selected
    ]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(METHOD_ORDER, means, yerr=errors, capsize=4, color="#4776b4")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{scenario}: {ylabel}")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    for extension in ("png", "pdf", "svg"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=200)
    plt.close(figure)


def _plot_ood(rows: list[dict[str, Any]], path: Path) -> None:
    lookup = _lookup(rows)
    available = {row["scenario"] for row in rows}
    scenarios = [scenario for scenario in SCENARIO_ORDER if scenario in available]
    figure, axis = plt.subplots(figsize=(10, 5))
    for method in METHOD_ORDER:
        axis.plot(
            scenarios,
            [
                float(
                    lookup[
                        (scenario, method, "packet_delivery_ratio")
                    ]["mean"]
                )
                for scenario in scenarios
            ],
            marker="o",
            label=method,
        )
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("Packet Delivery Ratio")
    axis.set_title("In-distribution and OOD PDR")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    for extension in ("png", "pdf", "svg"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=200)
    plt.close(figure)


def write_phase6_artifacts(
    output_dir: Path,
    *,
    episode_rows: list[dict[str, Any]],
    seed_summary_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    cost_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    raw = output_dir / "raw"
    summaries = output_dir / "summaries"
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    for directory in (raw, summaries, tables, figures):
        directory.mkdir(parents=True, exist_ok=True)

    aggregate_rows = aggregate_seed_summaries(seed_summary_rows)
    write_csv(raw / "episodes.csv", episode_rows)
    write_csv(raw / "seed_summaries.csv", seed_summary_rows)
    write_csv(raw / "training_metrics.csv", training_rows)
    write_csv(raw / "policy_costs.csv", cost_rows)
    write_csv(summaries / "statistics.csv", aggregate_rows)
    (summaries / "statistics.json").write_text(
        json.dumps(aggregate_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (tables / "main_results.md").write_text(
        render_main_markdown(aggregate_rows), encoding="utf-8"
    )
    (tables / "main_results.tex").write_text(
        render_main_latex(aggregate_rows), encoding="utf-8"
    )
    (tables / "ood_results.md").write_text(
        render_ood_markdown(aggregate_rows), encoding="utf-8"
    )
    _plot_metric(
        aggregate_rows,
        scenario="routing_hole",
        metric="packet_delivery_ratio",
        ylabel="Packet Delivery Ratio",
        path=figures / "pdr_comparison",
    )
    _plot_metric(
        aggregate_rows,
        scenario="routing_hole",
        metric="mean_episode_reward",
        ylabel="Mean Episode Reward",
        path=figures / "reward_comparison",
    )
    _plot_ood(aggregate_rows, figures / "ood_pdr")
    manifest = {
        **metadata,
        "episode_rows": len(episode_rows),
        "seed_summary_rows": len(seed_summary_rows),
        "statistics_rows": len(aggregate_rows),
        "methods": list(METHOD_ORDER),
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
