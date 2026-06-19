"""Phase 7 tables and figures centered on honest generalization metrics."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .generalization import GENERALIZATION_METRICS
from .reporting import write_csv
from .statistics import summarize_values
import matplotlib.pyplot as plt


PHASE7_METHODS = (
    "Random",
    "GPSR",
    "Untrained Student",
    "PPO-only Student",
    "KD-only Student",
    "KD+PPO Student",
    "Global Teacher",
    "Shortest-path Oracle",
)

PHASE7_SCENARIOS = (
    "heldout_medium",
    "ood_link_loss",
    "ood_fast_mobility",
    "ood_sparse",
    "ood_nodes_10",
    "unconditional_sparse",
)


def aggregate_generalization(
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
    for scenario in PHASE7_SCENARIOS:
        for method in PHASE7_METHODS:
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


def _lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict]:
    return {
        (row["scenario"], row["method"], row["metric"]): row
        for row in rows
    }


def _estimate(row: dict[str, Any]) -> str:
    if row["mean"] is None:
        return "N/A"
    half = (float(row["ci95_high"]) - float(row["ci95_low"])) / 2
    return f'{float(row["mean"]):.3f} ± {half:.3f}'


def _table(rows: list[dict[str, Any]]) -> str:
    lookup = _lookup(rows)
    lines = [
        "# Phase 7 Generalization Results",
        "",
        "| Method | "
        + " | ".join(PHASE7_SCENARIOS)
        + " |",
        "| --- | " + " | ".join("---:" for _ in PHASE7_SCENARIOS) + " |",
    ]
    for method in PHASE7_METHODS:
        values = [
            _estimate(
                lookup[(scenario, method, "connected_pair_pdr")]
            )
            for scenario in PHASE7_SCENARIOS
        ]
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "표의 값은 초기 연결 가능한 endpoint pair에 대한 조건부 PDR이다.",
            "`unconditional_sparse`의 전체 PDR은 availability와 함께 별도 CSV에서 확인한다.",
        ]
    )
    return "\n".join(lines) + "\n"


def _latex_table(rows: list[dict[str, Any]]) -> str:
    lookup = _lookup(rows)
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Method & ID & Loss & Fast & Sparse & Nodes-10 & Unconditional \\",
        r"\midrule",
    ]
    for method in PHASE7_METHODS:
        values = [
            _estimate(
                lookup[(scenario, method, "connected_pair_pdr")]
            ).replace("±", r"$\pm$")
            for scenario in PHASE7_SCENARIOS
        ]
        escaped = method.replace("+", r"\texttt{+}")
        lines.append(f"{escaped} & " + " & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _unconditional_table(rows: list[dict[str, Any]]) -> str:
    lookup = _lookup(rows)
    scenario = "unconditional_sparse"
    lines = [
        "# Phase 7 Unconditional Sparse Results",
        "",
        "| Method | Endpoint availability | Overall PDR | Connected-pair PDR |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in PHASE7_METHODS:
        values = [
            _estimate(lookup[(scenario, method, metric)])
            for metric in (
                "endpoint_availability",
                "overall_pdr",
                "connected_pair_pdr",
            )
        ]
        lines.append(f"| {method} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _plot(rows: list[dict[str, Any]], path: Path) -> None:
    lookup = _lookup(rows)
    figure, axis = plt.subplots(figsize=(12, 6))
    for method in PHASE7_METHODS:
        axis.plot(
            PHASE7_SCENARIOS,
            [
                float(
                    lookup[
                        (scenario, method, "connected_pair_pdr")
                    ]["mean"]
                )
                for scenario in PHASE7_SCENARIOS
            ],
            marker="o",
            label=method,
        )
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("Connected-pair PDR")
    axis.set_title("Phase 7 Held-out and OOD Generalization")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    for extension in ("png", "pdf", "svg"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=200)
    plt.close(figure)


def write_phase7_artifacts(
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
    aggregates = aggregate_generalization(summary_rows)
    write_csv(raw / "episodes.csv", episode_rows)
    write_csv(raw / "seed_summaries.csv", summary_rows)
    write_csv(raw / "training_metrics.csv", training_rows)
    write_csv(summaries / "statistics.csv", aggregates)
    (summaries / "statistics.json").write_text(
        json.dumps(aggregates, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (tables / "generalization_results.md").write_text(
        _table(aggregates), encoding="utf-8"
    )
    (tables / "generalization_results.tex").write_text(
        _latex_table(aggregates), encoding="utf-8"
    )
    (tables / "unconditional_results.md").write_text(
        _unconditional_table(aggregates), encoding="utf-8"
    )
    _plot(aggregates, figures / "generalization_pdr")
    manifest = {
        **metadata,
        "episode_rows": len(episode_rows),
        "seed_summary_rows": len(summary_rows),
        "statistics_rows": len(aggregates),
        "methods": list(PHASE7_METHODS),
        "scenarios": list(PHASE7_SCENARIOS),
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
