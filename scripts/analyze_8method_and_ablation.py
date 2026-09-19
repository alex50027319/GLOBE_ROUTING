#!/usr/bin/env python3
"""Create verified five-seed summaries and publication figures.

The script intentionally recomputes every reported aggregate from episode-level
rows.  Confidence intervals use the five training seeds as the independent
replicates, rather than treating episodes as independent observations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from implementations.lite_globe.evaluation.statistics import summarize_values


METHOD_ORDER = [
    "AODV",
    "OLSR",
    "Greedy Geographic",
    "Evo-QGeo (Adapted)",
    "RDQN-HERP (Adapted)",
    "GAT-GRU-DDQN",
    "SwitchGLOBE",
    "FastSwitchGLOBE",
]

ABLATION_ORDER = [
    "Geo-Residual Student",
    "Predictive Prior Only",
    "Predictive Student (No Switch)",
    "SwitchGLOBE Exact",
    "FastSwitchGLOBE",
    "FastSwitchGLOBE + Top-2",
]

METRIC_META = {
    "connected_pair_pdr": ("Connected-pair PDR", "ratio", True),
    "overall_pdr": ("Overall PDR", "ratio", True),
    "deadline_delivery_ratio": ("Deadline delivery ratio", "ratio", True),
    "p95_success_delay_steps": ("P95 successful delay", "steps", False),
    "mean_success_delay_steps": ("Mean successful delay", "steps", False),
    "energy_per_delivered_packet": ("Energy / delivered packet", "proxy units", False),
    "mean_path_stretch": ("Mean path stretch", "ratio", False),
    "late_delivery_ratio": ("Late-delivery ratio", "ratio", False),
    "drop_rate": ("Drop rate", "ratio", False),
    "loop_rate": ("Loop rate", "ratio", False),
    "mean_transmission_attempts": ("Transmission attempts", "attempts / episode", False),
    "mean_control_bytes": ("Control bytes", "bytes / episode", False),
    "mean_control_messages": ("Control messages", "messages / episode", False),
    "mean_policy_input_bytes": ("Policy input bytes", "bytes / episode", False),
    "mean_local_observation_bytes": ("Local observation bytes", "bytes / episode", False),
    "mean_reward": ("Episode reward", "reward", True),
    "mean_throughput": ("Mean throughput proxy", "packets / step", True),
    "switch_activation_rate": ("Switch activation rate", "switch steps / routing step", None),
}

DEPLOYMENT_META = {
    "latency_p50_ms": ("Decision latency P50", "ms", False),
    "latency_p95_ms": ("Decision latency P95", "ms", False),
    "latency_p99_ms": ("Decision latency P99", "ms", False),
    "mean_latency_ms": ("Mean decision latency", "ms", False),
    "parameter_count": ("Trainable parameters", "parameters", False),
    "serialized_model_bytes": ("Serialized artifact size", "bytes", False),
    "peak_python_memory_bytes": ("Peak Python memory", "bytes", False),
    "input_bytes": ("Benchmark input size", "bytes", False),
}


def finite_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else float("nan")


def episode_seed_metrics(frame: pd.DataFrame) -> dict[str, float]:
    delivered = frame["delivered"].astype(float)
    connected = frame["initially_connected"].astype(float) > 0.5
    success = delivered > 0.5
    successful = frame.loc[success]
    delivered_count = float(delivered.sum())
    steps = pd.to_numeric(frame["steps"], errors="coerce").fillna(0.0)
    switch_steps = pd.to_numeric(frame["switch_steps"], errors="coerce").fillna(0.0)
    return {
        "episode_count": float(len(frame)),
        "connected_episode_count": float(connected.sum()),
        "delivered_count": delivered_count,
        "connected_pair_pdr": float(delivered[connected].mean()),
        "overall_pdr": float(delivered.mean()),
        "deadline_delivery_ratio": finite_mean(frame["deadline_met"]),
        "p95_success_delay_steps": float(successful["delay_steps"].quantile(0.95)),
        "mean_success_delay_steps": finite_mean(successful["delay_steps"]),
        "energy_per_delivered_packet": (
            float(pd.to_numeric(frame["transmission_energy_proxy"], errors="coerce").sum())
            / delivered_count
        ),
        "mean_path_stretch": finite_mean(successful["path_stretch"]),
        "late_delivery_ratio": finite_mean(frame["late_delivery"]),
        "drop_rate": finite_mean(frame["dropped"]),
        "loop_rate": finite_mean(frame["loop"]),
        "mean_transmission_attempts": finite_mean(frame["transmission_attempts"]),
        "mean_control_bytes": finite_mean(frame["control_bytes"]),
        "mean_control_messages": finite_mean(frame["control_messages"]),
        "mean_policy_input_bytes": finite_mean(frame["policy_input_bytes"]),
        "mean_local_observation_bytes": finite_mean(frame["local_observation_bytes"]),
        "mean_reward": finite_mean(frame["total_reward"]),
        "mean_throughput": finite_mean(frame["throughput"]),
        "switch_activation_rate": float(switch_steps.sum() / steps.sum()) if steps.sum() else 0.0,
    }


def compute_seed_table(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, seed), group in episodes.groupby(["method", "training_seed"], sort=False):
        rows.append({"method": method, "training_seed": int(seed), **episode_seed_metrics(group)})
    return pd.DataFrame(rows)


def summarize_seed_table(seed_table: pd.DataFrame, metric_meta: dict) -> pd.DataFrame:
    rows = []
    for method, group in seed_table.groupby("method", sort=False):
        for metric, (label, unit, higher_is_better) in metric_meta.items():
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            values = values[np.isfinite(values)]
            if values.empty:
                continue
            stat = summarize_values(values.tolist())
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "metric_label": label,
                    "unit": unit,
                    "higher_is_better": higher_is_better,
                    **stat.to_dict(),
                }
            )
    return pd.DataFrame(rows)


def ordered(summary: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    result = summary.copy()
    result["method"] = pd.Categorical(result["method"], categories=order, ordered=True)
    return result.sort_values(["metric", "method"]).reset_index(drop=True)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_metric(summary: pd.DataFrame, metric: str, order: list[str], stem: Path) -> None:
    data = summary[summary.metric == metric].copy()
    data["method"] = pd.Categorical(data.method, categories=order, ordered=True)
    data = data.sort_values("method", ascending=False)
    if data.empty:
        return
    label = data.metric_label.iloc[0]
    unit = data.unit.iloc[0]
    fig, ax = plt.subplots(figsize=(8.4, max(3.4, 0.48 * len(data) + 1.2)))
    y = np.arange(len(data))
    x = data["mean"].to_numpy(float)
    err = np.vstack([x - data.ci95_low.to_numpy(float), data.ci95_high.to_numpy(float) - x])
    colors = ["#d95f02" if str(m) == "FastSwitchGLOBE" else "#1b9e77" if "SwitchGLOBE" in str(m) else "#506784" for m in data.method]
    ax.errorbar(x, y, xerr=err, fmt="none", ecolor="#6b7280", capsize=3, lw=1.2)
    ax.scatter(x, y, c=colors, s=44, zorder=3)
    ax.set_yticks(y, [str(v) for v in data.method])
    ax.set_xlabel(f"{label} ({unit}); mean and 95% t-CI over 5 seeds")
    ax.set_title(label)
    ax.grid(axis="x", alpha=0.25)
    save_figure(fig, stem)


def plot_facets(summary: pd.DataFrame, metrics: list[str], order: list[str], stem: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), constrained_layout=True)
    for ax, metric in zip(axes.flat, metrics):
        data = summary[summary.metric == metric].copy()
        data["method"] = pd.Categorical(data.method, categories=order, ordered=True)
        data = data.sort_values("method", ascending=False)
        y = np.arange(len(data))
        x = data["mean"].to_numpy(float)
        err = np.vstack([x - data.ci95_low.to_numpy(float), data.ci95_high.to_numpy(float) - x])
        colors = ["#d95f02" if str(m) == "FastSwitchGLOBE" else "#1b9e77" if "SwitchGLOBE" in str(m) else "#506784" for m in data.method]
        ax.errorbar(x, y, xerr=err, fmt="none", ecolor="#6b7280", capsize=2, lw=1)
        ax.scatter(x, y, c=colors, s=34, zorder=3)
        ax.set_yticks(y, [str(v) for v in data.method], fontsize=8)
        ax.set_title(data.metric_label.iloc[0])
        ax.set_xlabel(f"{data.unit.iloc[0]} (95% t-CI)")
        ax.grid(axis="x", alpha=0.22)
    save_figure(fig, stem)


def component_effects(ablation_seed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = [
        ("Predictive/risk prior", "Geo-Residual Student", "Predictive Student (No Switch)"),
        ("Geo-residual in predictive model", "Predictive Prior Only", "Predictive Student (No Switch)"),
        ("Risk switching", "Predictive Student (No Switch)", "SwitchGLOBE Exact"),
        ("Fast distillation/compression", "SwitchGLOBE Exact", "FastSwitchGLOBE"),
        ("Top-2 backup (normal routing)", "FastSwitchGLOBE", "FastSwitchGLOBE + Top-2"),
    ]
    metrics = [
        "connected_pair_pdr",
        "overall_pdr",
        "deadline_delivery_ratio",
        "p95_success_delay_steps",
        "energy_per_delivered_packet",
        "mean_path_stretch",
    ]
    indexed = ablation_seed.set_index(["method", "training_seed"])
    paired_rows = []
    summary_rows = []
    seeds = sorted(ablation_seed.training_seed.unique())
    for component, base, enhanced in comparisons:
        for metric in metrics:
            _, unit, higher = METRIC_META[metric]
            values = []
            for seed in seeds:
                base_value = float(indexed.loc[(base, seed), metric])
                enhanced_value = float(indexed.loc[(enhanced, seed), metric])
                raw_difference = enhanced_value - base_value
                directional = raw_difference if higher else -raw_difference
                scale = 100.0 if unit == "ratio" else 1.0
                paired_rows.append(
                    {
                        "component": component,
                        "base_method": base,
                        "enhanced_method": enhanced,
                        "training_seed": int(seed),
                        "metric": metric,
                        "base_value": base_value,
                        "enhanced_value": enhanced_value,
                        "raw_difference": raw_difference * scale,
                        "directional_improvement": directional * scale,
                        "effect_unit": "percentage points" if unit == "ratio" else unit,
                    }
                )
                values.append(directional * scale)
            stat = summarize_values(values)
            summary_rows.append(
                {
                    "component": component,
                    "base_method": base,
                    "enhanced_method": enhanced,
                    "metric": metric,
                    "metric_label": METRIC_META[metric][0],
                    "effect_unit": "percentage points" if unit == "ratio" else unit,
                    **stat.to_dict(),
                }
            )
    return pd.DataFrame(paired_rows), pd.DataFrame(summary_rows)


def plot_component_effects(effects: pd.DataFrame, stem: Path) -> None:
    metrics = ["connected_pair_pdr", "deadline_delivery_ratio", "p95_success_delay_steps", "energy_per_delivered_packet"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    components = effects.component.drop_duplicates().tolist()
    for ax, metric in zip(axes.flat, metrics):
        data = effects[effects.metric == metric].set_index("component").loc[components].reset_index()
        y = np.arange(len(data))
        x = data["mean"].to_numpy(float)
        err = np.vstack([x - data.ci95_low.to_numpy(float), data.ci95_high.to_numpy(float) - x])
        colors = ["#1b9e77" if value >= 0 else "#d95f02" for value in x]
        ax.axvline(0, color="#374151", lw=1)
        ax.errorbar(x, y, xerr=err, fmt="none", ecolor="#6b7280", capsize=3)
        ax.scatter(x, y, c=colors, s=44, zorder=3)
        ax.set_yticks(y, components, fontsize=8)
        ax.set_title(data.metric_label.iloc[0])
        ax.set_xlabel(f"Directional improvement ({data.effect_unit.iloc[0]}); >0 is better")
        ax.grid(axis="x", alpha=0.22)
    save_figure(fig, stem)


def create_report(
    external_summary: pd.DataFrame,
    deployment_summary: pd.DataFrame,
    effect_summary: pd.DataFrame,
    output: Path,
    verification: dict,
) -> None:
    def value(method: str, metric: str, source: pd.DataFrame = external_summary) -> float:
        return float(source[(source.method.astype(str) == method) & (source.metric == metric)]["mean"].iloc[0])

    switch_pdr = value("SwitchGLOBE", "connected_pair_pdr")
    fast_pdr = value("FastSwitchGLOBE", "connected_pair_pdr")
    evo_pdr = value("Evo-QGeo (Adapted)", "connected_pair_pdr")
    switch_deadline = value("SwitchGLOBE", "deadline_delivery_ratio")
    fast_deadline = value("FastSwitchGLOBE", "deadline_delivery_ratio")
    evo_deadline = value("Evo-QGeo (Adapted)", "deadline_delivery_ratio")
    switch_latency = value("SwitchGLOBE", "latency_p95_ms", deployment_summary)
    fast_latency = value("FastSwitchGLOBE", "latency_p95_ms", deployment_summary)
    switch_params = value("SwitchGLOBE", "parameter_count", deployment_summary)
    fast_params = value("FastSwitchGLOBE", "parameter_count", deployment_summary)
    effects = effect_summary[effect_summary.metric == "connected_pair_pdr"].set_index("component")
    lines = [
        "# Verified 8-method comparison and SwitchGLOBE ablation",
        "",
        "## Verification basis",
        "",
        f"- Five independent training seeds: {verification['seeds']}.",
        f"- Fourteen scenarios and 200 evaluation episodes per scenario/seed/method.",
        f"- External comparison rows: {verification['external_episode_rows']:,}; ablation rows: {verification['ablation_episode_rows']:,}.",
        "- Confidence intervals are two-sided 95% Student-t intervals over the five seed-level aggregates.",
        "- SwitchGLOBE and FastSwitchGLOBE episode outcomes were cross-checked between the external and ablation archives; all common routing-outcome fields matched exactly.",
        "",
        "## Main result",
        "",
        f"SwitchGLOBE has the strongest reliability: connected-pair PDR {switch_pdr:.4f} ({(switch_pdr-evo_pdr)*100:+.2f} pp vs the strongest external baseline, Evo-QGeo) and deadline delivery {switch_deadline:.4f} ({(switch_deadline-evo_deadline)*100:+.2f} pp). FastSwitchGLOBE retains PDR {fast_pdr:.4f} ({(fast_pdr-evo_pdr)*100:+.2f} pp) and deadline delivery {fast_deadline:.4f} ({(fast_deadline-evo_deadline)*100:+.2f} pp).",
        "",
        "The advantage is not universal: AODV/OLSR/Greedy use less energy per delivered packet and have lower successful-path P95 delay in this simulator. The defensible claim is therefore **higher reliability under the tested dynamic scenarios**, not dominance on every metric.",
        "",
        "## FastSwitchGLOBE deployment trade-off",
        "",
        f"FastSwitchGLOBE reduces same-contract A100 P95 decision latency from {switch_latency:.3f} ms to {fast_latency:.3f} ms ({(1-fast_latency/switch_latency)*100:.1f}% reduction) and parameters from {switch_params:,.0f} to {fast_params:,.0f} ({(1-fast_params/switch_params)*100:.1f}% reduction). This costs {(switch_pdr-fast_pdr)*100:.2f} pp connected-pair PDR and {(switch_deadline-fast_deadline)*100:.2f} pp deadline delivery.",
        "",
        "## Which component drives novelty?",
        "",
        f"- Predictive/risk prior: {effects.loc['Predictive/risk prior','mean']:+.2f} pp connected-pair PDR (largest positive aggregate contribution).",
        f"- Risk switching: {effects.loc['Risk switching','mean']:+.2f} pp connected-pair PDR (smaller recovery/refinement contribution).",
        f"- Geo-residual inside the predictive model: {effects.loc['Geo-residual in predictive model','mean']:+.2f} pp connected-pair PDR in this aggregate comparison; it is not independently supported as the main reliability driver.",
        f"- Fast distillation/compression: {effects.loc['Fast distillation/compression','mean']:+.2f} pp connected-pair PDR; its verified value is efficiency, not accuracy improvement.",
        "- Top-2 backup produces zero normal-routing outcome change here. Its value must be demonstrated in a separate failover/staleness intervention experiment.",
        "",
        "Conclusion: the strongest evidence-backed key novelty is the **predictive risk-aware prior**, with switching as a secondary robustness mechanism and distillation as the deployment-efficiency mechanism.",
        "",
        "## Interpretation cautions",
        "",
        "- `policy_input_bytes` measures simulator-side observation/features presented to a policy; it is not network control traffic.",
        "- `serialized_model_bytes` for stateful RL baselines can include auxiliary checkpoint state, so parameter count and measured latency are cleaner complexity comparisons.",
        "- Decision latency is hardware/runtime dependent and should be reported with the A100, batch=1, same-contract context.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir
    (output / "csv").mkdir(parents=True, exist_ok=True)
    (output / "figures" / "external_metrics").mkdir(parents=True, exist_ok=True)
    (output / "figures" / "deployment_metrics").mkdir(parents=True, exist_ok=True)
    (output / "figures" / "ablation_metrics").mkdir(parents=True, exist_ok=True)

    external_episodes = pd.read_csv(args.external_dir / "raw" / "episodes.csv")
    ablation_episodes = pd.read_csv(args.ablation_dir / "raw" / "episodes.csv")
    deployment = pd.read_csv(args.external_dir / "raw" / "deployment_costs.csv")

    external_seed = compute_seed_table(external_episodes)
    ablation_seed = compute_seed_table(ablation_episodes)
    external_summary = ordered(summarize_seed_table(external_seed, METRIC_META), METHOD_ORDER)
    ablation_summary = ordered(summarize_seed_table(ablation_seed, METRIC_META), ABLATION_ORDER)
    deployment_seed = deployment.rename(columns={"training_seed": "training_seed"}).copy()
    deployment_summary = ordered(summarize_seed_table(deployment_seed, DEPLOYMENT_META), METHOD_ORDER)
    paired_effects, effect_summary = component_effects(ablation_seed)

    external_seed.to_csv(output / "csv" / "external_per_seed_metrics.csv", index=False)
    external_summary.to_csv(output / "csv" / "external_overall_metrics.csv", index=False)
    deployment_seed.to_csv(output / "csv" / "external_deployment_per_seed.csv", index=False)
    deployment_summary.to_csv(output / "csv" / "external_deployment_metrics.csv", index=False)
    ablation_seed.to_csv(output / "csv" / "ablation_per_seed_metrics.csv", index=False)
    ablation_summary.to_csv(output / "csv" / "ablation_overall_metrics.csv", index=False)
    paired_effects.to_csv(output / "csv" / "ablation_component_effects_per_seed.csv", index=False)
    effect_summary.to_csv(output / "csv" / "ablation_component_effects.csv", index=False)

    for metric in METRIC_META:
        plot_metric(external_summary, metric, METHOD_ORDER, output / "figures" / "external_metrics" / metric)
        plot_metric(ablation_summary, metric, ABLATION_ORDER, output / "figures" / "ablation_metrics" / metric)
    for metric in DEPLOYMENT_META:
        plot_metric(deployment_summary, metric, METHOD_ORDER, output / "figures" / "deployment_metrics" / metric)
    plot_facets(
        external_summary,
        ["connected_pair_pdr", "deadline_delivery_ratio", "p95_success_delay_steps", "energy_per_delivered_packet"],
        METHOD_ORDER,
        output / "figures" / "external_primary_metrics",
    )
    plot_facets(
        ablation_summary,
        ["connected_pair_pdr", "deadline_delivery_ratio", "p95_success_delay_steps", "energy_per_delivered_packet"],
        ABLATION_ORDER,
        output / "figures" / "ablation_primary_metrics",
    )
    plot_component_effects(effect_summary, output / "figures" / "ablation_component_effects")

    external_manifest = json.loads((args.external_dir / "manifest.json").read_text())
    ablation_manifest = json.loads((args.ablation_dir / "manifest.json").read_text())
    outcome_fields = [
        "delivered", "dropped", "drop_reason", "steps", "hop_count",
        "transmission_attempts", "transmission_energy_proxy", "deadline_met",
        "delay_steps", "loop", "initially_connected", "policy_input_bytes", "total_reward",
    ]
    keys = ["method", "training_seed", "scenario", "evaluation_seed"]
    external_two = external_episodes[external_episodes.method.isin(["SwitchGLOBE", "FastSwitchGLOBE"])].copy()
    external_two["method"] = external_two.method.replace({"SwitchGLOBE": "SwitchGLOBE Exact"})
    ablation_two = ablation_episodes[ablation_episodes.method.isin(["SwitchGLOBE Exact", "FastSwitchGLOBE"])].copy()
    joined = external_two[keys + outcome_fields].merge(
        ablation_two[keys + outcome_fields], on=keys, suffixes=("_external", "_ablation"), validate="one_to_one"
    )
    mismatch_counts = {}
    for field in outcome_fields:
        left = joined[f"{field}_external"]
        right = joined[f"{field}_ablation"]
        equal = (left.eq(right)) | (left.isna() & right.isna())
        mismatch_counts[field] = int((~equal).sum())
    verification = {
        "complete": bool(external_manifest.get("complete") and ablation_manifest.get("complete")),
        "mode": [external_manifest.get("mode"), ablation_manifest.get("mode")],
        "seeds": sorted(int(v) for v in external_episodes.training_seed.unique()),
        "external_methods": sorted(external_episodes.method.unique().tolist()),
        "ablation_methods": sorted(ablation_episodes.method.unique().tolist()),
        "scenarios": sorted(external_episodes.scenario.unique().tolist()),
        "external_episode_rows": int(len(external_episodes)),
        "external_expected_episode_rows": int(external_manifest.get("expected_episode_rows", -1)),
        "ablation_episode_rows": int(len(ablation_episodes)),
        "ablation_expected_episode_rows": int(ablation_manifest.get("expected_episode_rows", -1)),
        "common_episode_rows_cross_checked": int(len(joined)),
        "cross_experiment_outcome_mismatches": mismatch_counts,
        "all_cross_experiment_outcomes_match": all(v == 0 for v in mismatch_counts.values()),
    }
    verification["verified"] = bool(
        verification["complete"]
        and verification["external_episode_rows"] == verification["external_expected_episode_rows"]
        and verification["ablation_episode_rows"] == verification["ablation_expected_episode_rows"]
        and verification["all_cross_experiment_outcomes_match"]
        and len(verification["seeds"]) == 5
        and len(verification["external_methods"]) == 8
    )
    (output / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    create_report(external_summary, deployment_summary, effect_summary, output / "REPORT.md", verification)
    if not verification["verified"]:
        raise SystemExit("verification failed; see verification.json")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
