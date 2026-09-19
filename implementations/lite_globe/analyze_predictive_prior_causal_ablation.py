"""Merge five causal-ablation seeds into CSVs, figures, and a Korean report."""

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
from .run_predictive_prior_causal_ablation import (
    FULL, LEGACY, MANUAL, METHODS, NO_TEACHER, RISK_ONLY, SEEDS,
    SHORTEST_ONLY, TEACHER_ONLY,
)


METRICS = (
    "connected_pair_pdr", "deadline_delivery_ratio", "p95_success_delay",
    "mean_transmission_energy_proxy", "energy_per_delivered_packet",
)
LOWER = {"p95_success_delay", "mean_transmission_energy_proxy", "energy_per_delivered_packet"}
CONTRASTS = (
    (RISK_ONLY, MANUAL, "risk supervision vs fixed manual"),
    (SHORTEST_ONLY, MANUAL, "shortest supervision vs fixed manual"),
    (NO_TEACHER, RISK_ONLY, "add shortest label without teacher"),
    (TEACHER_ONLY, MANUAL, "teacher distillation vs fixed manual"),
    (FULL, NO_TEACHER, "primary teacher contribution"),
    (FULL, TEACHER_ONLY, "add oracle labels to teacher"),
    (LEGACY, FULL, "historical Phase11 reference vs matched full"),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _stats(values):
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return summarize_values(finite).to_dict()


def _episode_metrics(rows):
    connected = [row for row in rows if int(row["initially_connected"])]
    delivered = [row for row in rows if int(row["delivered"])]
    delays = [float(row["steps"]) for row in delivered]
    energy = [float(row["transmission_energy_proxy"]) for row in rows]
    return {
        "connected_pair_pdr": sum(int(row["delivered"]) for row in connected) / max(len(connected), 1),
        "deadline_delivery_ratio": sum(int(row["deadline_met"]) for row in rows) / len(rows),
        "p95_success_delay": float(np.percentile(delays, 95)) if delays else math.nan,
        "mean_transmission_energy_proxy": float(np.mean(energy)),
        "energy_per_delivered_packet": sum(energy) / max(len(delivered), 1),
    }


def _save(fig, base):
    base.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(base.with_suffix(f".{suffix}"), dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args(); episodes = []; parameters = []; curves = []; latency = []; manifests = []
    for seed in SEEDS:
        directory = args.input_dir / f"seed_{seed}"
        episodes.extend(_read(directory / "raw_episodes.csv"))
        parameters.extend(_read(directory / "learned_parameters_and_test_agreement.csv"))
        curves.extend(_read(directory / "training_curves.csv"))
        latency.extend(_read(directory / "latency_seed_summary.csv"))
        manifests.append(json.loads((directory / "manifest.json").read_text(encoding="utf-8")))
    expected = len(SEEDS) * len(METHODS) * 14 * 200
    if len(episodes) != expected:
        raise ValueError(f"episode rows {len(episodes)} != {expected}")
    key_count = len({(r["method"], r["scenario"], r["training_seed"], r["evaluation_seed"]) for r in episodes})
    if key_count != expected:
        raise ValueError("duplicate or missing episode keys")
    scenarios = tuple(dict.fromkeys(row["scenario"] for row in episodes))
    grouped = defaultdict(list)
    for row in episodes:
        grouped[(row["method"], int(row["training_seed"]), row["scenario"])].append(row)
    seed_scenario = []; seed_overall = []
    for method in METHODS:
        for seed in SEEDS:
            combined = []
            for scenario in scenarios:
                selected = grouped[(method, seed, scenario)]; combined.extend(selected)
                seed_scenario.append({"method": method, "training_seed": seed, "scenario": scenario, **_episode_metrics(selected)})
            seed_overall.append({"method": method, "training_seed": seed, "scenario": "ALL_POOLED", **_episode_metrics(combined)})
    lookup = {(r["method"], int(r["training_seed"])): r for r in seed_overall}
    scenario_lookup = {(r["method"], int(r["training_seed"]), r["scenario"]): r for r in seed_scenario}
    aggregate = []
    for method in METHODS:
        for metric in METRICS:
            aggregate.append({"method": method, "scope": "ALL_POOLED", "metric": metric, **_stats([lookup[(method, seed)][metric] for seed in SEEDS])})
    paired = []
    for variant, baseline, question in CONTRASTS:
        for metric in METRICS:
            values = []
            for seed in SEEDS:
                left, right = float(lookup[(variant, seed)][metric]), float(lookup[(baseline, seed)][metric])
                values.append((right - left) if metric in LOWER else (left - right))
            paired.append({
                "variant": variant, "baseline": baseline, "question": question,
                "metric": metric, "positive_means_variant_better": 1,
                **_stats(values),
            })
    worst = []
    for variant, baseline, question in CONTRASTS:
        for metric in METRICS:
            per_seed = []
            for seed in SEEDS:
                effects = []
                for scenario in scenarios:
                    left = float(scenario_lookup[(variant, seed, scenario)][metric])
                    right = float(scenario_lookup[(baseline, seed, scenario)][metric])
                    if math.isfinite(left) and math.isfinite(right):
                        effects.append((right - left) if metric in LOWER else (left - right))
                per_seed.append(min(effects))
            worst.append({
                "variant": variant, "baseline": baseline, "question": question,
                "metric": metric, "definition": "per-seed minimum direction-aligned scenario effect",
                **_stats(per_seed),
            })
    parameter_fields = (
        "geographic_strength", "forwardability_binary_strength", "forwardability_count_strength",
        "margin_strength", "lifetime_strength", "queue_headroom_strength",
        "onward_lifetime_strength", "break_penalty", "test_teacher_action_agreement",
        "test_shortest_action_agreement", "test_risk_action_agreement",
    )
    parameter_ci = []
    for method in METHODS:
        selected = [row for row in parameters if row["method"] == method]
        for metric in parameter_fields:
            parameter_ci.append({"method": method, "metric": metric, **_stats([row[metric] for row in selected])})
    latency_ci = []
    for device in ("cpu", "cuda"):
        for method in METHODS:
            selected = [float(row["p95_ms"]) for row in latency if row["device"] == device and row["variant"] == method]
            latency_ci.append({"device": device, "method": method, "metric": "seed_p95_ms", **_stats(selected)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "raw" / "episodes.csv", episodes)
    write_csv(args.output_dir / "raw" / "training_curves.csv", curves)
    write_csv(args.output_dir / "raw" / "latency_seed_summary.csv", latency)
    write_csv(args.output_dir / "summaries" / "seed_scenario_metrics.csv", seed_scenario)
    write_csv(args.output_dir / "summaries" / "seed_overall_metrics.csv", seed_overall)
    write_csv(args.output_dir / "summaries" / "aggregate_95ci.csv", aggregate)
    write_csv(args.output_dir / "summaries" / "paired_causal_effects_95ci.csv", paired)
    write_csv(args.output_dir / "summaries" / "worst_scenario_effects_95ci.csv", worst)
    write_csv(args.output_dir / "summaries" / "learned_parameters_95ci.csv", parameter_ci)
    write_csv(args.output_dir / "summaries" / "latency_95ci.csv", latency_ci)

    agg = {(r["method"], r["metric"]): r for r in aggregate}
    short = ("Manual", "Shortest", "Risk", "No teacher", "Teacher", "Full", "Legacy")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for axis, metric, title in zip(axes.flat, METRICS[:4], ("Connected PDR", "Deadline delivery", "p95 successful delay", "Mean energy proxy")):
        means = [float(agg[(method, metric)]["mean"]) for method in METHODS]
        errors = [(float(agg[(method, metric)]["ci95_high"])-float(agg[(method, metric)]["ci95_low"]))/2 for method in METHODS]
        axis.bar(range(len(METHODS)), means, yerr=errors, capsize=3)
        axis.set_xticks(range(len(METHODS)), short, rotation=20, ha="right"); axis.set_title(title); axis.grid(axis="y", alpha=.25)
    _save(fig, args.output_dir / "figures" / "predictive_prior_causal_metrics")

    primary = [row for row in paired if row["question"] == "primary teacher contribution"]
    fig, axis = plt.subplots(figsize=(9, 5))
    labels = ["Connected PDR", "Deadline", "p95 delay", "Mean energy", "Energy/delivered"]
    means = [float(row["mean"]) for row in primary]
    lower = [float(row["mean"])-float(row["ci95_low"]) for row in primary]
    upper = [float(row["ci95_high"])-float(row["mean"]) for row in primary]
    axis.errorbar(range(len(primary)), means, yerr=[lower, upper], fmt="o", capsize=5)
    axis.axhline(0, color="black", linewidth=1); axis.set_xticks(range(len(primary)), labels, rotation=20, ha="right")
    axis.set_ylabel("Direction-aligned effect: Full − No Teacher"); axis.grid(axis="y", alpha=.25)
    _save(fig, args.output_dir / "figures" / "primary_teacher_contribution")

    lat = {(row["device"], row["method"]): row for row in latency_ci}
    fig, axis = plt.subplots(figsize=(12, 5.5)); x = np.arange(len(METHODS)); width=.38
    for offset, device in ((-width/2, "cpu"), (width/2, "cuda")):
        axis.bar(x+offset, [float(lat[(device,m)]["mean"]) for m in METHODS], width, label=device.upper())
    axis.set_xticks(x, short, rotation=20, ha="right"); axis.set_yscale("log"); axis.set_ylabel("p95 batch-1 latency (ms, log)")
    axis.legend(); axis.grid(axis="y", alpha=.25)
    _save(fig, args.output_dir / "figures" / "predictive_prior_cpu_a100_latency")

    fig, axis = plt.subplots(figsize=(14, 6)); x=np.arange(len(scenarios))
    for method, marker in ((NO_TEACHER,"o"),(FULL,"s"),(LEGACY,"^")):
        means = [float(np.mean([float(scenario_lookup[(method, seed, scenario)]["connected_pair_pdr"]) for seed in SEEDS])) for scenario in scenarios]
        axis.plot(x, means, marker=marker, label=method)
    axis.set_xticks(x, scenarios, rotation=30, ha="right"); axis.set_ylabel("Connected-pair PDR"); axis.legend(); axis.grid(alpha=.25)
    _save(fig, args.output_dir / "figures" / "teacher_ablation_scenario_pdr")

    teacher_effect = {(row["metric"]): row for row in primary}
    pdr = teacher_effect["connected_pair_pdr"]; deadline = teacher_effect["deadline_delivery_ratio"]
    full_pdr, no_teacher_pdr = agg[(FULL,"connected_pair_pdr")], agg[(NO_TEACHER,"connected_pair_pdr")]
    full_latency_cpu, no_teacher_latency_cpu = lat[("cpu",FULL)], lat[("cpu",NO_TEACHER)]
    evidence = (
        "supported" if float(pdr["ci95_low"]) > 0 or float(deadline["ci95_low"]) > 0
        else "not_supported"
    )
    lines = [
        "# Predictive Prior policy-distillation causal ablation", "",
        "## 실험 설계", "",
        f"- 5 training seeds × 14 evaluation scenarios × 200 episodes × {len(METHODS)} methods = {expected:,} raw episodes.",
        "- 모든 신규 모델은 같은 초기값, residual=0, 동일 oracle/risk-oracle rollout 상태, Adam, batch 512, 120 epochs를 사용했다.",
        "- 차이는 학습 label/objective뿐이다. Teacher가 no-teacher 모델의 rollout 경로를 결정하지 않았다.",
        "- 통계 단위는 training seed이며 paired effect에 양측 95% Student-t CI를 적용했다.",
        "- 양의 direction-aligned effect는 variant가 baseline보다 우수함을 뜻한다.", "",
        "## 핵심 질문: teacher distillation이 필요한가?", "",
        f"- Full connected PDR: {float(full_pdr['mean']):.6f} [{float(full_pdr['ci95_low']):.6f}, {float(full_pdr['ci95_high']):.6f}]",
        f"- No-teacher connected PDR: {float(no_teacher_pdr['mean']):.6f} [{float(no_teacher_pdr['ci95_low']):.6f}, {float(no_teacher_pdr['ci95_high']):.6f}]",
        f"- Teacher 추가 paired PDR 효과: {float(pdr['mean']):+.6f} [{float(pdr['ci95_low']):+.6f}, {float(pdr['ci95_high']):+.6f}]",
        f"- Teacher 추가 deadline 효과: {float(deadline['mean']):+.6f} [{float(deadline['ci95_low']):+.6f}, {float(deadline['ci95_high']):+.6f}]",
        f"- CPU p95 latency: Full {float(full_latency_cpu['mean']):.4f} ms, No-teacher {float(no_teacher_latency_cpu['mean']):.4f} ms.", "",
        f"**Teacher의 독립적 신뢰성 향상 판정: {evidence}.**", "",
        "95% CI가 0을 포함하면 teacher가 쓸모없다는 증명이 아니라, 이 프로토콜에서 독립적 개선을 입증하지 못했다는 뜻이다. 반대로 CI 하한이 0보다 크면 동일구조·동일상태·동일예산 조건에서 teacher supervision의 추가 가치를 지지한다.", "",
        "## 해석 원칙", "",
        "- Fixed Manual은 학습 없는 hand-set coefficient 기준이다.",
        "- Risk-only 및 Shortest-only는 supervision source의 단독 효과를 보여준다.",
        "- Shortest+Risk (No Teacher) 대 Full이 distillation novelty의 1차 인과 비교다.",
        "- Existing Phase11은 historical reference이며 상태분포와 학습 절차가 달라 1차 causal contrast가 아니다.",
        "- energy는 simulator proxy이지 실제 Joule 측정값이 아니다.",
        "- 최종 논문 주장은 paired CI와 worst-scenario 표를 함께 보고 결정해야 한다.",
    ]
    (args.output_dir / "POLICY_DISTILLATION_CAUSAL_ANALYSIS_KO.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    manifest = {
        "schema_version": 1, "complete": True,
        "suite": "predictive_prior_causal_ablation_merged",
        "methods": list(METHODS), "seeds": list(SEEDS), "scenarios": list(scenarios),
        "episode_rows": len(episodes), "primary_teacher_evidence": evidence,
        "source_manifests": manifests,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"complete": True, "episode_rows": len(episodes), "primary_teacher_evidence": evidence}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
