"""Merge per-seed Colab outputs and produce paper-ready model-selection artifacts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluation.reporting import write_csv
from .evaluation.statistics import summarize_values
from .experiments.ablation_campaign import (
    FAST_SWITCHGLOBE, GEO_RESIDUAL, PREDICTIVE_NO_SWITCH,
    PREDICTIVE_PRIOR_ONLY, SWITCHGLOBE_EXACT,
)
from .run_model_selection_ablation import (
    GLOBAL_TEACHER, HISTORICAL_KD_ONLY, HISTORICAL_KD_PPO,
    HISTORICAL_NO_KD, HISTORICAL_UNTRAINED, METHODS, SEEDS,
)


METRICS = (
    "connected_pair_pdr", "deadline_delivery_ratio", "p95_success_delay",
    "mean_transmission_energy_proxy", "energy_per_delivered_packet",
)
LOWER = {"p95_success_delay", "mean_transmission_energy_proxy", "energy_per_delivered_packet"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--latency-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _summarize_episode_group(rows: list[dict[str, str]]) -> dict[str, float]:
    connected = [row for row in rows if int(row["initially_connected"])]
    delivered = [row for row in rows if int(row["delivered"])]
    energy = [float(row["transmission_energy_proxy"]) for row in rows]
    delays = [float(row["steps"]) for row in delivered]
    return {
        "connected_pair_pdr": sum(int(row["delivered"]) for row in connected) / max(len(connected), 1),
        "deadline_delivery_ratio": sum(int(row["deadline_met"]) for row in rows) / len(rows),
        "p95_success_delay": float(np.percentile(delays, 95)) if delays else math.nan,
        "mean_transmission_energy_proxy": float(np.mean(energy)),
        "energy_per_delivered_packet": sum(energy) / max(len(delivered), 1),
    }


def _stats(values: list[float]) -> dict[str, object]:
    finite = [value for value in values if math.isfinite(value)]
    return summarize_values(finite).to_dict() if finite else {
        "count": 0, "mean": None, "standard_deviation": None,
        "ci95_low": None, "ci95_high": None,
    }


def _save_figure(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(base.with_suffix(f".{suffix}"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    episodes: list[dict[str, str]] = []
    manifests = []
    for seed in SEEDS:
        directory = args.input_dir / f"seed_{seed}"
        episodes.extend(_read_csv(directory / "raw_episodes.csv"))
        manifests.append(json.loads((directory / "manifest.json").read_text(encoding="utf-8")))
    expected = len(SEEDS) * len(METHODS) * 14 * 200
    if len(episodes) != expected:
        raise ValueError(f"expected {expected} raw episode rows, got {len(episodes)}")
    keys = {(row["method"], row["scenario"], int(row["training_seed"]), int(row["evaluation_seed"])) for row in episodes}
    if len(keys) != expected:
        raise ValueError("duplicate episode key detected")
    scenarios = tuple(dict.fromkeys(row["scenario"] for row in episodes))

    groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in episodes:
        groups[(row["method"], int(row["training_seed"]), row["scenario"])].append(row)
    scenario_seed: list[dict[str, object]] = []
    overall_seed: list[dict[str, object]] = []
    for method in METHODS:
        for seed in SEEDS:
            combined = []
            for scenario in scenarios:
                selected = groups[(method, seed, scenario)]
                combined.extend(selected)
                scenario_seed.append({
                    "method": method, "training_seed": seed, "scenario": scenario,
                    **_summarize_episode_group(selected),
                })
            overall_seed.append({
                "method": method, "training_seed": seed, "scenario": "ALL_POOLED",
                **_summarize_episode_group(combined),
            })
    scenario_lookup = {
        (str(row["method"]), int(row["training_seed"]), str(row["scenario"])): row
        for row in scenario_seed
    }
    overall_lookup = {
        (str(row["method"]), int(row["training_seed"])): row for row in overall_seed
    }

    aggregate: list[dict[str, object]] = []
    for method in METHODS:
        for scenario in (*scenarios, "ALL_POOLED"):
            source = scenario_seed if scenario != "ALL_POOLED" else overall_seed
            selected = [row for row in source if row["method"] == method and row["scenario"] == scenario]
            for metric in METRICS:
                aggregate.append({
                    "method": method, "scenario": scenario, "metric": metric,
                    **_stats([float(row[metric]) for row in selected]),
                })

    paired: list[dict[str, object]] = []
    worst: list[dict[str, object]] = []
    for method in METHODS:
        if method == SWITCHGLOBE_EXACT:
            continue
        for metric in METRICS:
            deltas = []
            worst_per_seed = []
            for seed in SEEDS:
                candidate = float(overall_lookup[(method, seed)][metric])
                exact = float(overall_lookup[(SWITCHGLOBE_EXACT, seed)][metric])
                deltas.append((exact - candidate) if metric in LOWER else (candidate - exact))
                scenario_advantages = []
                for scenario in scenarios:
                    candidate_s = float(scenario_lookup[(method, seed, scenario)][metric])
                    exact_s = float(scenario_lookup[(SWITCHGLOBE_EXACT, seed, scenario)][metric])
                    if math.isfinite(candidate_s) and math.isfinite(exact_s):
                        scenario_advantages.append((exact_s - candidate_s) if metric in LOWER else (candidate_s - exact_s))
                worst_per_seed.append(min(scenario_advantages))
            paired.append({
                "variant": method, "baseline": SWITCHGLOBE_EXACT, "scope": "ALL_POOLED",
                "metric": metric, "positive_means_variant_better": 1, **_stats(deltas),
            })
            worst.append({
                "variant": method, "baseline": SWITCHGLOBE_EXACT,
                "metric": metric, "definition": "minimum direction-aligned scenario advantage per seed",
                "positive_means_variant_better": 1, **_stats(worst_per_seed),
            })

    indexed = {
        (row["method"], row["scenario"], int(row["training_seed"]), int(row["evaluation_seed"])): row
        for row in episodes
    }
    agreement_seed_scenario: list[dict[str, object]] = []
    for seed in SEEDS:
        for scenario in scenarios:
            strict_action, strict_trajectory, prefix_ratios = [], [], []
            for evaluation_seed in range(
                1_100_000 + scenarios.index(scenario) * 10_000,
                1_100_000 + scenarios.index(scenario) * 10_000 + 200,
            ):
                pred = indexed[(PREDICTIVE_PRIOR_ONLY, scenario, seed, evaluation_seed)]
                exact = indexed[(SWITCHGLOBE_EXACT, scenario, seed, evaluation_seed)]
                pred_actions = json.loads(pred["action_sequence"])
                exact_actions = json.loads(exact["action_sequence"])
                pred_path = json.loads(pred["node_trajectory"])
                exact_path = json.loads(exact["node_trajectory"])
                strict_action.append(float(pred_actions == exact_actions))
                strict_trajectory.append(float(pred_path == exact_path))
                common = 0
                for left, right in zip(pred_actions, exact_actions):
                    if left != right:
                        break
                    common += 1
                prefix_ratios.append(common / max(len(pred_actions), len(exact_actions), 1))
            agreement_seed_scenario.append({
                "training_seed": seed, "scenario": scenario,
                "exact_action_sequence_agreement": float(np.mean(strict_action)),
                "exact_node_trajectory_agreement": float(np.mean(strict_trajectory)),
                "mean_common_prefix_ratio": float(np.mean(prefix_ratios)),
            })
    agreement_aggregate: list[dict[str, object]] = []
    for metric in (
        "exact_action_sequence_agreement", "exact_node_trajectory_agreement",
        "mean_common_prefix_ratio",
    ):
        for scenario in (*scenarios, "ALL_POOLED"):
            per_seed = []
            for seed in SEEDS:
                selected = [row for row in agreement_seed_scenario if row["training_seed"] == seed and (scenario == "ALL_POOLED" or row["scenario"] == scenario)]
                per_seed.append(float(np.mean([float(row[metric]) for row in selected])))
            agreement_aggregate.append({"scenario": scenario, "metric": metric, **_stats(per_seed)})

    kd_pairs = (
        (HISTORICAL_KD_ONLY, HISTORICAL_NO_KD),
        (HISTORICAL_KD_PPO, HISTORICAL_NO_KD),
        (HISTORICAL_KD_PPO, HISTORICAL_KD_ONLY),
    )
    kd_effects: list[dict[str, object]] = []
    for candidate, baseline in kd_pairs:
        for metric in METRICS:
            differences = []
            for seed in SEEDS:
                left = float(overall_lookup[(candidate, seed)][metric])
                right = float(overall_lookup[(baseline, seed)][metric])
                differences.append((right - left) if metric in LOWER else (left - right))
            kd_effects.append({
                "variant": candidate, "baseline": baseline, "metric": metric,
                "positive_means_variant_better": 1, **_stats(differences),
            })

    latency = _read_csv(args.latency_dir / "seed_runtime_benchmarks.csv")
    latency_aggregate = _read_csv(args.latency_dir / "aggregate_latency_ci.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw" / "episodes.csv", episodes)
    write_csv(args.output_dir / "summaries" / "scenario_seed_metrics.csv", scenario_seed)
    write_csv(args.output_dir / "summaries" / "overall_seed_metrics.csv", overall_seed)
    write_csv(args.output_dir / "summaries" / "aggregate_95ci.csv", aggregate)
    write_csv(args.output_dir / "summaries" / "paired_effects_vs_switchglobe.csv", paired)
    write_csv(args.output_dir / "summaries" / "worst_scenario_effects.csv", worst)
    write_csv(args.output_dir / "summaries" / "predictive_vs_switch_agreement.csv", agreement_seed_scenario)
    write_csv(args.output_dir / "summaries" / "predictive_vs_switch_agreement_95ci.csv", agreement_aggregate)
    write_csv(args.output_dir / "summaries" / "distillation_audit_paired_95ci.csv", kd_effects)
    write_csv(args.output_dir / "latency" / "seed_runtime_benchmarks.csv", latency)
    write_csv(args.output_dir / "latency" / "aggregate_latency_ci.csv", latency_aggregate)

    agg_lookup = {(row["method"], row["scenario"], row["metric"]): row for row in aggregate}
    display_methods = (PREDICTIVE_PRIOR_ONLY, PREDICTIVE_NO_SWITCH, SWITCHGLOBE_EXACT, FAST_SWITCHGLOBE)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for axis, metric, title in zip(axes.flat, METRICS[:4], (
        "Connected-pair PDR", "Deadline delivery", "p95 successful delay", "Mean energy proxy",
    )):
        means = [float(agg_lookup[(m, "ALL_POOLED", metric)]["mean"]) for m in display_methods]
        errors = [(float(agg_lookup[(m, "ALL_POOLED", metric)]["ci95_high"]) - float(agg_lookup[(m, "ALL_POOLED", metric)]["ci95_low"])) / 2 for m in display_methods]
        axis.bar(range(len(display_methods)), means, yerr=errors, capsize=4)
        axis.set_xticks(range(len(display_methods)), ("Predictive-only", "Predictive full", "SwitchGLOBE", "Fast"), rotation=15, ha="right")
        axis.set_title(title); axis.grid(axis="y", alpha=.25)
    _save_figure(fig, args.output_dir / "figures" / "final_model_metrics")

    fig, axis = plt.subplots(figsize=(14, 6))
    x = np.arange(len(scenarios))
    for method, marker in ((PREDICTIVE_PRIOR_ONLY, "o"), (SWITCHGLOBE_EXACT, "s")):
        means = [float(agg_lookup[(method, scenario, "connected_pair_pdr")]["mean"]) for scenario in scenarios]
        axis.plot(x, means, marker=marker, label=method)
    axis.set_xticks(x, scenarios, rotation=30, ha="right"); axis.set_ylabel("Connected-pair PDR")
    axis.legend(); axis.grid(alpha=.25)
    _save_figure(fig, args.output_dir / "figures" / "predictive_vs_switch_scenario_pdr")

    agree_lookup = {(row["scenario"], row["metric"]): row for row in agreement_aggregate}
    fig, axis = plt.subplots(figsize=(14, 4.8))
    values = np.array([[float(agree_lookup[(scenario, metric)]["mean"]) for scenario in scenarios] for metric in (
        "exact_action_sequence_agreement", "exact_node_trajectory_agreement", "mean_common_prefix_ratio",
    )])
    image = axis.imshow(values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    axis.set_yticks(range(3), ("Exact action sequence", "Exact trajectory", "Common-prefix ratio"))
    axis.set_xticks(range(len(scenarios)), scenarios, rotation=30, ha="right")
    fig.colorbar(image, ax=axis, label="Agreement")
    _save_figure(fig, args.output_dir / "figures" / "predictive_switch_agreement")

    lat_lookup = {(row["device"], row["method"]): row for row in latency_aggregate}
    fig, axis = plt.subplots(figsize=(11, 5.5))
    width = .38; xx = np.arange(len(display_methods))
    for offset, device in ((-width/2, "cpu"), (width/2, "cuda")):
        means = [float(lat_lookup[(device, method)]["mean"]) for method in display_methods]
        axis.bar(xx + offset, means, width, label=device.upper())
    axis.set_xticks(xx, ("Predictive-only", "Predictive full", "SwitchGLOBE", "Fast"))
    axis.set_yscale("log"); axis.set_ylabel("Seed-level p95 batch-1 latency (ms, log scale)")
    axis.legend(); axis.grid(axis="y", alpha=.25)
    _save_figure(fig, args.output_dir / "figures" / "cpu_a100_batch1_latency")

    paired_lookup = {(row["variant"], row["metric"]): row for row in paired}
    primary = paired_lookup[(PREDICTIVE_PRIOR_ONLY, "connected_pair_pdr")]
    deadline = paired_lookup[(PREDICTIVE_PRIOR_ONLY, "deadline_delivery_ratio")]
    delay = paired_lookup[(PREDICTIVE_PRIOR_ONLY, "p95_success_delay")]
    cpu_pred = lat_lookup[("cpu", PREDICTIVE_PRIOR_ONLY)]
    cpu_exact = lat_lookup[("cpu", SWITCHGLOBE_EXACT)]
    selected = (
        PREDICTIVE_PRIOR_ONLY
        if float(primary["ci95_low"]) >= -0.005 and float(deadline["ci95_low"]) >= -0.005
        and float(cpu_pred["mean"]) < float(cpu_exact["mean"])
        else SWITCHGLOBE_EXACT
    )
    lines = [
        "# SwitchGLOBE confirmatory ablation and model selection", "",
        f"- Protocol: 5 training seeds × 14 scenarios × 200 episodes × {len(METHODS)} methods = {expected:,} episode rows.",
        "- All outcome contrasts use the training seed as the inferential unit and two-sided 95% Student-t intervals.",
        "- Positive paired effects mean that the named variant is better than SwitchGLOBE Exact.",
        "- Energy is a simulator transmission-energy proxy, not measured Joules.",
        "- The Phase-7 KD comparison is a historical matched-architecture audit, not a causal retraining ablation of the final architecture.",
        "", "## Primary predictive-only vs risk-switch result", "",
        f"- Connected PDR effect: {float(primary['mean']):+.6f} (95% CI {float(primary['ci95_low']):+.6f}, {float(primary['ci95_high']):+.6f}).",
        f"- Deadline-delivery effect: {float(deadline['mean']):+.6f} (95% CI {float(deadline['ci95_low']):+.6f}, {float(deadline['ci95_high']):+.6f}).",
        f"- p95-success-delay direction-aligned effect: {float(delay['mean']):+.6f} steps (95% CI {float(delay['ci95_low']):+.6f}, {float(delay['ci95_high']):+.6f}); positive favors predictive-only.",
        f"- CPU p95 latency: predictive-only {float(cpu_pred['mean']):.4f} ms vs SwitchGLOBE {float(cpu_exact['mean']):.4f} ms.",
        "", "## Predeclared selection rule", "",
        "Reliability-first rule: predictive-only is selected only when its lower 95% CI is no worse than -0.5 percentage points for both connected PDR and deadline delivery, and it reduces CPU p95 latency; otherwise SwitchGLOBE Exact is retained.",
        f"", f"**Selected model under this rule: {selected}.**", "",
        "The selection is an engineering conclusion under the stated margin, not proof of statistical equivalence. Review the worst-scenario table before publication.",
    ]
    (args.output_dir / "MODEL_SELECTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "complete": True,
        "suite": "switchglobe_confirmatory_model_selection_merged",
        "methods": list(METHODS), "training_seeds": list(SEEDS),
        "scenarios": list(scenarios), "episodes_per_scenario": 200,
        "episode_rows": len(episodes), "selected_model": selected,
        "source_manifests": manifests,
        "definitions": {
            "action_agreement": "fraction of paired episodes with identical complete action sequences",
            "trajectory_agreement": "fraction of paired episodes with identical complete node paths",
            "worst_scenario": "per-seed minimum direction-aligned scenario advantage, summarized across seeds",
            "ci": "two-sided Student-t interval across five paired training-seed effects",
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True, "selected_model": selected, "episode_rows": len(episodes)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
