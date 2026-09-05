"""Generate the 20-figure SwitchGLOBE latency optimization evidence pack."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from collections import defaultdict
import math

os.environ.setdefault("MPLCONFIGDIR", "/tmp/switchglobe-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "artifacts/switchglobe_latency_optimization/globev2_local_full_20260905"
GATE = ROOT / "artifacts/gated_switchglobe/calibration_guarded_20260905"
EARLY = ROOT / "artifacts/gated_switchglobe/early_exit_full_validation_20260905"
ABL = ROOT / "artifacts/final_paper_simulation/full/ablation"
ANALYSIS = ROOT / "artifacts/final_paper_simulation/full/analysis_8method_and_ablation/csv"
A100_LEGACY = ROOT / "artifacts/final_paper_simulation/full/final_latency_verified"
A100_CURRENT = ROOT / "artifacts/switchglobe_latency_optimization/globev2_colab_cli_20260905/results"
A100_PROFILE = ROOT / "artifacts/switchglobe_latency_optimization/globev2_colab_cli_20260905/profile/results"
VERIFIED = ROOT / "artifacts/switchglobe_latency_optimization/verified_candidate_1_2"

COLORS = {
    "Exact": "#173F5F", "Early Exit": "#F6AE2D", "Fast": "#2A9D8F",
    "Fast + Top-2": "#6A4C93", "Legacy": "#9B2226", "Buffered": "#577590",
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def mean(values) -> float:
    vals = list(values); return float(np.mean(vals)) if vals else math.nan


def ci95(values) -> tuple[float, float, float]:
    vals = np.asarray(list(values), dtype=float)
    center = float(vals.mean())
    if len(vals) < 2:
        return center, center, center
    half = 2.776445 * float(vals.std(ddof=1)) / math.sqrt(len(vals))
    return center, center - half, center + half


def save(fig, figures: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(figures / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(figures / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def style(axis, title: str, ylabel: str = "") -> None:
    axis.set_title(title, loc="left", fontweight="bold")
    if ylabel: axis.set_ylabel(ylabel)
    axis.grid(axis="y", alpha=.22)


def label(variant: str) -> str:
    return {
        "exact_eager_cpu": "Exact", "early_exit_cpu": "Early Exit",
        "fast_eager_cpu": "Fast", "fast_top2_eager_cpu": "Fast + Top-2",
        "legacy_repeated_cpu": "Legacy", "exact_buffered_cpu": "Buffered",
    }.get(variant, variant)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    figures = args.output_dir / "figures"; metrics = args.output_dir / "metrics"
    figures.mkdir(parents=True, exist_ok=True); metrics.mkdir(parents=True, exist_ok=True)
    captions: list[tuple[str, str]] = []

    local = read(LOCAL / "runtime_benchmarks.csv")
    end = [r for r in local if r["component"] == "end_to_end_policy"]
    variants = ["exact_eager_cpu", "early_exit_cpu", "fast_eager_cpu", "fast_top2_eager_cpu"]

    # 01: independently timed component decomposition.
    components = ["preprocess", "model", "action_extraction"]
    data = []
    fig, ax = plt.subplots(figsize=(10, 5.6)); bottom = np.zeros(len(variants))
    for component in components:
        vals = np.array([mean(float(r["p95_ms"]) for r in local if r["variant"] == v and r["component"] == component) for v in variants])
        ax.bar(range(len(variants)), vals, bottom=bottom, label=component); bottom += vals
        data.extend({"variant": label(v), "component": component, "mean_seed_p95_ms": x} for v, x in zip(variants, vals))
    e2e = [mean(float(r["p95_ms"]) for r in end if r["variant"] == v) for v in variants]
    ax.scatter(range(len(variants)), e2e, marker="D", color="black", label="measured end-to-end p95", zorder=5)
    ax.set_xticks(range(len(variants)), [label(v) for v in variants]); ax.legend(ncol=2)
    style(ax, "01  Local component profile (independent timers)", "Latency (ms, lower is better)")
    save(fig, figures, "01_baseline_component_latency")
    write(metrics / "figure_01.csv", data)
    captions.append(("01_baseline_component_latency", "각 구성요소는 독립 타이머라 합이 end-to-end와 정확히 같지 않다. batch=1, 5 seeds, local CPU."))

    # 02: central and tail metrics.
    measures = ["mean_ms", "p50_ms", "p95_ms", "p99_ms"]
    data=[]; fig, ax = plt.subplots(figsize=(10.5, 5.6)); width=.19; x=np.arange(len(variants))
    for i, m in enumerate(measures):
        vals=[mean(float(r[m]) for r in end if r["variant"]==v) for v in variants]
        ax.bar(x+(i-1.5)*width, vals, width, label=m.replace("_ms","")); data += [{"variant":label(v),"metric":m,"ms":val} for v,val in zip(variants,vals)]
    ax.set_xticks(x,[label(v) for v in variants]); ax.legend(ncol=4)
    style(ax,"02  Local decision latency summary","Latency (ms, lower is better)")
    save(fig,figures,"02_latency_summary"); write(metrics/"figure_02.csv",data)
    captions.append(("02_latency_summary","동일 checkpoint와 5 seeds에서 mean/P50/P95/P99를 비교한다. Fast 계열은 낮지만 reliability gate는 별도 판단한다."))

    # 03: ECDF over seed-level p95 (statistical unit is seed).
    data=[]; fig,ax=plt.subplots(figsize=(8.5,5.5))
    for v in variants:
        vals=sorted(float(r["p95_ms"]) for r in end if r["variant"]==v); y=np.arange(1,len(vals)+1)/len(vals)
        ax.step(vals,y,where="post",label=label(v),color=COLORS[label(v)])
        data += [{"variant":label(v),"seed_p95_ms":a,"ecdf":b} for a,b in zip(vals,y)]
    ax.legend(); ax.set_xlabel("Seed-level p95 latency (ms)"); style(ax,"03  Seed-level p95 ECDF","Cumulative fraction")
    save(fig,figures,"03_latency_ecdf"); write(metrics/"figure_03.csv",data)
    captions.append(("03_latency_ecdf","개별 decision을 독립 표본으로 부풀리지 않고 seed별 p95 5개로 그린 ECDF다."))

    # 04: paired p95 reduction forest.
    exact={int(r["training_seed"]):float(r["p95_ms"]) for r in end if r["variant"]=="exact_eager_cpu"}
    data=[]
    for v in ["early_exit_cpu","fast_eager_cpu","fast_top2_eager_cpu","exact_buffered_cpu","legacy_repeated_cpu"]:
        other={int(r["training_seed"]):float(r["p95_ms"]) for r in end if r["variant"]==v}
        reductions=[100*(exact[s]-other[s])/exact[s] for s in exact]; m,lo,hi=ci95(reductions)
        data.append({"variant":label(v),"mean_reduction_percent":m,"ci95_low":lo,"ci95_high":hi})
    fig,ax=plt.subplots(figsize=(8.5,5.5)); y=np.arange(len(data)); m=np.array([r["mean_reduction_percent"] for r in data]); lo=np.array([r["ci95_low"] for r in data]); hi=np.array([r["ci95_high"] for r in data])
    ax.errorbar(m,y,xerr=[m-lo,hi-m],fmt="o",capsize=5,color="#173F5F"); ax.axvline(0,color="black",lw=1); ax.set_yticks(y,[r["variant"] for r in data]); ax.set_xlabel("Paired p95 reduction vs Exact (%)")
    style(ax,"04  Paired seed-level p95 reduction (95% t-CI)")
    save(fig,figures,"04_latency_forest"); write(metrics/"figure_04.csv",data)
    captions.append(("04_latency_forest","양수는 Exact 대비 빠름. Early Exit과 buffering은 CI가 0을 가로질러 개선으로 판정하지 않는다."))

    quality = read(ABL / "validation/fast_vs_exact_overall.csv")
    q={r["metric"]:r for r in quality}
    a100rows=read(A100_CURRENT/"runtime_benchmarks.csv")
    a100end=[r for r in a100rows if r["component"]=="end_to_end_policy"]
    def a100p(v): return mean(float(r["p95_ms"]) for r in a100end if r["variant"]==v)
    exact_a100=a100p("exact_eager_cuda"); fast_a100=a100p("fast_eager_cuda")

    # 05/06 Pareto figures using the current A100 run and full network simulation.
    for num,metric,title in [(5,"connected_pair_pdr","Connected-pair PDR"),(6,"deadline_delivery_ratio","Deadline delivery ratio")]:
        row=q[metric]; pts=[("Exact",exact_a100,float(row["exact_mean"])),("Fast",fast_a100,float(row["fast_mean"]))]
        fig,ax=plt.subplots(figsize=(7.8,5.5))
        for name,xv,yv in pts: ax.scatter(xv,yv,s=100,color=COLORS[name]); ax.annotate(name,(xv,yv),xytext=(7,5),textcoords="offset points")
        ax.set_xlabel("A100 p95 decision latency (ms, lower better)"); style(ax,f"{num:02d}  Latency–reliability Pareto",f"{title} (higher better)")
        save(fig,figures,f"{num:02d}_pareto_{metric}"); write(metrics/f"figure_{num:02d}.csv",[{"method":n,"a100_p95_ms":x,metric:y} for n,x,y in pts])
        captions.append((f"{num:02d}_pareto_{metric}","A100 latency는 2026-09-05 colab-cli 동일 세션, reliability는 5×14×200 full 결과다."))

    seedsum=read(ABL/"raw/seed_summaries.csv")
    methods=["SwitchGLOBE Exact","FastSwitchGLOBE"]
    scenarios=sorted({r["scenario"] for r in seedsum if r["method"] in methods})
    # 07 reliability heatmap.
    mat=np.array([[mean(float(r["connected_pair_pdr"]) for r in seedsum if r["scenario"]==s and r["method"]==m) for m in methods] for s in scenarios])
    fig,ax=plt.subplots(figsize=(7.5,8)); im=ax.imshow(mat,aspect="auto",vmin=.45,vmax=1,cmap="viridis"); fig.colorbar(im,ax=ax,label="Connected-pair PDR")
    ax.set_xticks(range(2),["Exact","Fast"]); ax.set_yticks(range(len(scenarios)),scenarios,fontsize=8); style(ax,"07  Scenario × method reliability heatmap")
    save(fig,figures,"07_reliability_heatmap"); write(metrics/"figure_07.csv",[{"scenario":s,"method":methods[j],"connected_pair_pdr":mat[i,j]} for i,s in enumerate(scenarios) for j in range(2)])
    captions.append(("07_reliability_heatmap","5-seed scenario 평균. Fast의 열세가 OOD 대규모 노드 등 특정 조건에 집중되는지 보여준다."))

    # 08 latency heatmap from full exact calibration and early-exit execution (not same session).
    exactep=read(GATE/"raw/episodes.csv"); earlyep=read(EARLY/"early_exit_episodes.csv")
    mats=[]
    for src in (exactep,earlyep): mats.append([mean(float(r["decision_latency_p95_ms"]) for r in src if r["scenario"]==s) for s in scenarios])
    mat=np.array(mats).T; fig,ax=plt.subplots(figsize=(7.5,8)); im=ax.imshow(mat,aspect="auto",cmap="magma"); fig.colorbar(im,ax=ax,label="Episode p95 decision latency (ms)")
    ax.set_xticks(range(2),["Exact replay","Early Exit"]); ax.set_yticks(range(len(scenarios)),scenarios,fontsize=8); style(ax,"08  Scenario latency diagnostic (separate runs)")
    save(fig,figures,"08_latency_heatmap"); write(metrics/"figure_08.csv",[{"scenario":s,"method":["Exact","Early Exit"][j],"mean_episode_p95_ms":mat[i,j]} for i,s in enumerate(scenarios) for j in range(2)])
    captions.append(("08_latency_heatmap","14,000 episode 전수 실행 결과이나 same-session randomized benchmark가 아니므로 진단용이다."))

    # 09 seed variability.
    fig,ax=plt.subplots(figsize=(9,5.5)); data=[]
    for v in variants:
        xs=[int(r["training_seed"]) for r in end if r["variant"]==v]; ys=[float(r["p95_ms"]) for r in end if r["variant"]==v]
        ax.plot(xs,ys,"o-",label=label(v),color=COLORS[label(v)]); data += [{"seed":x,"variant":label(v),"p95_ms":y} for x,y in zip(xs,ys)]
    ax.set_xticks(sorted(exact)); ax.legend(ncol=2); ax.set_xlabel("Training seed"); style(ax,"09  Seed-level latency variability","p95 latency (ms)")
    save(fig,figures,"09_seed_variability"); write(metrics/"figure_09.csv",data); captions.append(("09_seed_variability","모든 후보를 동일 5개 training seed에서 비교한다."))

    # 10 switch activation by scenario.
    vals=[mean(float(r["switch_activation_rate"]) for r in seedsum if r["scenario"]==s and r["method"]=="SwitchGLOBE Exact") for s in scenarios]
    fig,ax=plt.subplots(figsize=(10,5.5)); ax.bar(range(len(scenarios)),vals,color=COLORS["Exact"]); ax.set_xticks(range(len(scenarios)),scenarios,rotation=55,ha="right",fontsize=8); style(ax,"10  Exact switch activation by scenario","Activation rate")
    save(fig,figures,"10_switch_activation"); write(metrics/"figure_10.csv",[{"scenario":s,"switch_activation_rate":v} for s,v in zip(scenarios,vals)]); captions.append(("10_switch_activation","Switch가 켜지는 빈도와 predictive branch 계산 필요 빈도는 동일하지 않다."))

    gate=read(GATE/"summaries/gate_margin_statistics.csv"); overall=sorted([r for r in gate if r["scope"]=="overall"],key=lambda r:float(r["gate_margin"]))
    margins=np.array([float(r["gate_margin"]) for r in overall]); cdf=np.array([float(r["skip_rate"]) for r in overall])
    # 11 approximate bins from empirical CDF.
    bins=np.diff(np.r_[0,cdf]); names=[f"≤{m:.2f}" if i==0 else f"({margins[i-1]:.2f},{m:.2f}]" for i,m in enumerate(margins)]
    fig,ax=plt.subplots(figsize=(9,5.5)); ax.bar(range(len(bins)),bins,color="#577590"); ax.set_xticks(range(len(bins)),names,rotation=30,ha="right"); style(ax,"11  Normal danger-score empirical bins","Fraction of decisions")
    save(fig,figures,"11_gate_score_distribution"); write(metrics/"figure_11.csv",[{"bin":n,"fraction":v} for n,v in zip(names,bins)]); captions.append(("11_gate_score_distribution","sweep의 누적 skip count 차이로 재구성한 구간 질량이며 원시 continuous score histogram은 아니다."))

    # 12 gate calibration.
    div=np.array([float(r["outcome_divergence_rate_of_all_steps"]) for r in overall])
    fig,ax=plt.subplots(figsize=(8.5,5.5)); ax.plot(margins,cdf,"o-",label="skip rate"); ax.plot(margins,div,"s-",label="action divergence / all steps"); ax.axvline(0,color="black",lw=1,ls="--"); ax.legend(); ax.set_xlabel("Absolute danger margin"); style(ax,"12  Early-exit calibration","Rate")
    save(fig,figures,"12_gate_calibration"); write(metrics/"figure_12.csv",[{"margin":m,"skip_rate":s,"divergence_rate":d} for m,s,d in zip(margins,cdf,div)]); captions.append(("12_gate_calibration","margin=0만 전수 관찰에서 divergence 0이다. 양의 margin은 divergence를 유발해 채택하지 않는다."))

    # 13 branch usage vs latency.
    branch=read(EARLY/"branch_usage.csv"); data=[]
    for s in scenarios:
        usage=mean(float(r["early_exit_rate"]) for r in branch if r["scenario"]==s); lat=mean(float(r["decision_latency_p95_ms"]) for r in earlyep if r["scenario"]==s); data.append({"scenario":s,"early_exit_rate":usage,"mean_episode_p95_ms":lat})
    fig,ax=plt.subplots(figsize=(8,5.5)); ax.scatter([r["early_exit_rate"] for r in data],[r["mean_episode_p95_ms"] for r in data],color=COLORS["Early Exit"]); ax.set_xlabel("Early-exit branch fraction"); style(ax,"13  Branch usage vs observed latency","Episode p95 latency (ms)")
    save(fig,figures,"13_branch_usage_latency"); write(metrics/"figure_13.csv",data); captions.append(("13_branch_usage_latency","높은 skip 비율만으로 실제 p95 개선이 보장되지 않으며 runtime control overhead가 중요함을 보여준다."))

    costs=read(LOCAL/"deployment_costs.csv")
    # 14 parameter and serialized size.
    names=["Exact","Early Exit","Fast","Fast + Top-2"]; vs=variants
    params=[mean(float(r["parameter_count"]) for r in costs if r["variant"]==v) for v in vs]; sizes=[mean(float(r["serialized_model_bytes"])/1024 for r in costs if r["variant"]==v) for v in vs]
    fig,ax=plt.subplots(figsize=(9,5.5)); x=np.arange(4); ax.bar(x-.18,params,.36,label="parameters"); ax2=ax.twinx(); ax2.bar(x+.18,sizes,.36,color="#F6AE2D",label="checkpoint KiB"); ax.set_xticks(x,names); ax.set_ylabel("Parameter count"); ax2.set_ylabel("Serialized checkpoint (KiB)"); style(ax,"14  Model footprint")
    save(fig,figures,"14_model_footprint"); write(metrics/"figure_14.csv",[{"method":n,"parameters":p,"checkpoint_kib":s} for n,p,s in zip(names,params,sizes)]); captions.append(("14_model_footprint","Early Exit은 Exact 가중치를 그대로 사용해 크기가 같고 Fast는 7,011 parameters다."))

    # 15 input bytes.
    inputs=[mean(float(r["input_bytes"]) for r in costs if r["variant"]==v) for v in vs]
    fig,ax=plt.subplots(figsize=(8.5,5.5)); ax.bar(names,inputs,color=[COLORS[n] for n in names]); style(ax,"15  Policy input bytes","Bytes per benchmark decision")
    save(fig,figures,"15_policy_input_bytes"); write(metrics/"figure_15.csv",[{"method":n,"input_bytes":v} for n,v in zip(names,inputs)]); captions.append(("15_policy_input_bytes","정책 입력 바이트이며 무선 control overhead와 동일한 지표가 아니다."))

    # 16 ablation contribution.
    effects=read(ANALYSIS/"ablation_component_effects.csv"); data=[r for r in effects if r["metric"]=="connected_pair_pdr"]
    fig,ax=plt.subplots(figsize=(9,5.5)); y=np.arange(len(data)); m=np.array([float(r["mean"]) for r in data]); lo=np.array([float(r["ci95_low"]) for r in data]); hi=np.array([float(r["ci95_high"]) for r in data]); ax.errorbar(m,y,xerr=[m-lo,hi-m],fmt="o",capsize=4); ax.axvline(0,color="black",lw=1); ax.set_yticks(y,[r["component"] for r in data]); ax.set_xlabel("Connected-pair PDR contribution (percentage points)"); style(ax,"16  Reliability ablation contributions")
    save(fig,figures,"16_ablation_contribution"); write(metrics/"figure_16.csv",data); captions.append(("16_ablation_contribution","SwitchGLOBE 성능을 만든 구성요소의 seed-level 95% CI다."))

    # 17 device/code-version comparison.
    localp={label(v):mean(float(r["p95_ms"]) for r in end if r["variant"]==v) for v in ["exact_eager_cpu","fast_eager_cpu"]}
    points=[]
    a100agg=read(A100_LEGACY/"aggregate_statistics.csv")
    for dev in ["cpu","cuda"]:
        for vn,n in [("SwitchGLOBE Exact","Exact"),("FastSwitchGLOBE","Fast")]:
            points.append({"environment":f"2026-08-29 bundle {dev}","method":n,"p95_ms":float(next(r for r in a100agg if r["device"]==dev and r["variant"]==vn and r["metric"]=="p95_ms")["mean"])})
    for dev in ["cpu","cuda"]:
        for vn,n in [(f"exact_eager_{dev}","Exact"),(f"fast_eager_{dev}","Fast")]:
            points.append({"environment":f"2026-09-05 colab-cli {dev}","method":n,"p95_ms":a100p(vn)})
    points += [{"environment":"2026-09-05 current local CPU","method":n,"p95_ms":v} for n,v in localp.items()]
    envs=list(dict.fromkeys(r["environment"] for r in points)); fig,ax=plt.subplots(figsize=(10,5.5)); x=np.arange(len(envs));
    for j,n in enumerate(["Exact","Fast"]): ax.bar(x+(j-.5)*.36,[next(r["p95_ms"] for r in points if r["environment"]==e and r["method"]==n) for e in envs],.36,label=n,color=COLORS[n])
    ax.set_xticks(x,envs,rotation=20,ha="right"); ax.legend(); style(ax,"17  CPU/GPU/A100 comparison (code version separated)","p95 latency (ms)")
    save(fig,figures,"17_device_comparison"); write(metrics/"figure_17.csv",points); captions.append(("17_device_comparison","2026-09-05 colab-cli 결과는 commit 92d17df의 현재 구현이다. 2026-08-29 번들은 독립 재현성 참고값으로만 병기한다."))

    # 18 worst-case scenario Fast reliability differences.
    byscenario=read(ABL/"validation/fast_vs_exact_by_scenario.csv"); byscenario=sorted(byscenario,key=lambda r:float(r["fast_minus_exact_connected_pdr_mean"]))
    fig,ax=plt.subplots(figsize=(10,6)); vals=[100*float(r["fast_minus_exact_connected_pdr_mean"]) for r in byscenario]; ax.bar(range(len(vals)),vals,color=["#9B2226" if v<-.5 else "#2A9D8F" for v in vals]); ax.axhline(-.5,color="black",ls="--",label="−0.5 pp gate"); ax.set_xticks(range(len(vals)),[r["scenario"] for r in byscenario],rotation=55,ha="right",fontsize=8); ax.legend(); style(ax,"18  Fast worst-case reliability by scenario","Fast − Exact connected PDR (pp)")
    save(fig,figures,"18_worst_case_scenario"); write(metrics/"figure_18.csv",byscenario); captions.append(("18_worst_case_scenario","Fast 손실이 OOD node-scale 조건에서 크게 나타나 최종 대체안으로 부적합하다."))

    # 19 paired local Early Exit latency difference.
    early={int(r["training_seed"]):float(r["p95_ms"]) for r in end if r["variant"]=="early_exit_cpu"}; data=[{"seed":s,"early_minus_exact_ms":early[s]-exact[s]} for s in sorted(exact)]
    fig,ax=plt.subplots(figsize=(8.5,5.5)); ax.bar([str(r["seed"]) for r in data],[r["early_minus_exact_ms"] for r in data],color=["#2A9D8F" if r["early_minus_exact_ms"]<0 else "#9B2226" for r in data]); ax.axhline(0,color="black",lw=1); style(ax,"19  Early Exit paired p95 difference","Early Exit − Exact (ms; lower better)")
    save(fig,figures,"19_early_exit_paired_difference"); write(metrics/"figure_19.csv",data); captions.append(("19_early_exit_paired_difference","5 seeds 중 일관된 개선이 아니며 평균적으로 악화했다."))

    # 20 candidate decision map.
    verified=read(VERIFIED/"aggregate_runtime.csv"); compiled=float(next(r for r in verified if r["variant"]=="exact_compiled_cpu")["p95_ms"]); oldexact=float(next(r for r in verified if r["variant"]=="exact_eager_cpu")["p95_ms"]); buffered=float(next(r for r in verified if r["variant"]=="exact_buffered_cpu")["p95_ms"])
    a100_exact=exact_a100
    a100_legacy=a100p("legacy_repeated_cuda")
    a100_early=a100p("early_exit_cuda")
    a100_fast_top2=a100p("fast_top2_eager_cuda")
    a100_buffered=a100p("exact_buffered_cuda")
    candidates=[
        ("Fused Exact",100*(a100_legacy-a100_exact)/a100_legacy,0.0,"adopt"),
        ("Buffered",100*(a100_exact-a100_buffered)/a100_exact,0.0,"hold"),
        ("torch.compile (old)",100*(oldexact-compiled)/oldexact,0.0,"reject"),
        ("Early Exit",100*(a100_exact-a100_early)/a100_exact,0.0,"reject"),
        ("Fast",100*(a100_exact-fast_a100)/a100_exact,100*float(q["connected_pair_pdr"]["directional_difference_mean"]),"trade-off"),
        ("Fast + Top-2",100*(a100_exact-a100_fast_top2)/a100_exact,100*float(q["connected_pair_pdr"]["directional_difference_mean"]),"trade-off"),
    ]
    fig,ax=plt.subplots(figsize=(9,5.8)); cmap={"adopt":"#2A9D8F","reject":"#9B2226","hold":"#F6AE2D","trade-off":"#6A4C93"}
    offsets={"Fused Exact":(-65,-25),"Buffered":(8,-27),"torch.compile (old)":(-12,-30),"Early Exit":(-30,16),"Fast":(-20,14),"Fast + Top-2":(-15,-28)}
    for n,xv,yv,status in candidates: ax.scatter(xv,yv,s=100,color=cmap[status]); ax.annotate(n,(xv,yv),xytext=offsets[n],textcoords="offset points")
    ax.axvline(0,color="black",lw=1); ax.axhline(-.5,color="black",ls="--",lw=1); ax.set_ylim(-2.0,.14); ax.set_xlabel("p95 latency reduction (%)"); style(ax,"20  Candidate decision map","Connected-pair PDR change (pp)")
    save(fig,figures,"20_failed_candidate_tradeoff"); write(metrics/"figure_20.csv",[{"candidate":n,"p95_reduction_percent":x,"connected_pdr_change_pp":y,"decision":s} for n,x,y,s in candidates]); captions.append(("20_failed_candidate_tradeoff","오른쪽 위가 바람직하다. Fused Exact는 동작 동일성을 보존하면서 legacy 중복-forward 대비 2026-09-05 A100에서 개선됐다. Fast는 reliability gate를 통과하지 못했다."))

    index=["# Figure index","","> 모든 그림은 PNG(240 dpi)와 SVG를 함께 제공한다. Smoke, local full, legacy A100 evidence는 캡션에서 구분한다.",""]
    for name,caption in captions: index += [f"## {name}","",caption,"",f"- Data: `metrics/figure_{name[:2]}.csv`",""]
    (args.output_dir/"FIGURE_INDEX.md").write_text("\n".join(index),encoding="utf-8")
    manifest={"complete":True,"figure_count":len(captions),"formats":["png","svg"],"sources":[str(LOCAL),str(GATE),str(EARLY),str(ABL),str(A100_CURRENT),str(A100_PROFILE),str(A100_LEGACY),str(VERIFIED)]}
    (args.output_dir/"figure_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
