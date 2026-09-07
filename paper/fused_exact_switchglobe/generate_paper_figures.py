from __future__ import annotations

import ast
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
DATA = Path(__file__).resolve().parent / "figure_data"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

COLORS = {
    "navy": "#12355B",
    "blue": "#2F6BFF",
    "cyan": "#2AA6B8",
    "orange": "#F28E2B",
    "red": "#D64550",
    "green": "#2B8A66",
    "gray": "#64748B",
    "light": "#EEF4FA",
    "ink": "#172033",
}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, wh, text, color, subtitle=None, dashed=False):
    x, y = xy
    w, h = wh
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.4,
        edgecolor=color,
        facecolor="white",
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h * 0.60, text, ha="center", va="center", weight="bold", color=COLORS["ink"], fontsize=10)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.29, subtitle, ha="center", va="center", color=COLORS["gray"], fontsize=7.7)


def arrow(ax, a, b, color=None, label=None):
    color = color or COLORS["gray"]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color=color))
    if label:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2 + 0.025, label, ha="center", va="bottom", color=color, fontsize=7.5)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.95, "Training with privileged topology", weight="bold", color=COLORS["navy"], fontsize=12)
    ax.text(0.56, 0.95, "Decentralized batch-1 forwarding", weight="bold", color=COLORS["navy"], fontsize=12)
    ax.axvline(0.51, color="#CBD5E1", lw=1.2, ls="--")
    box(ax, (0.04, 0.65), (0.18, 0.18), "Global graph state", COLORS["blue"], "topology, mobility, packet")
    box(ax, (0.29, 0.65), (0.17, 0.18), "PPO graph teacher", COLORS["blue"], "masked global preferences")
    arrow(ax, (0.22, 0.74), (0.29, 0.74), COLORS["blue"])
    box(ax, (0.15, 0.31), (0.22, 0.19), "Masked distillation", COLORS["cyan"], "KL + local supervision")
    arrow(ax, (0.375, 0.65), (0.28, 0.50), COLORS["cyan"])
    box(ax, (0.56, 0.66), (0.18, 0.18), "Strict-local input", COLORS["green"], "relay + one-hop neighbors")
    box(ax, (0.79, 0.70), (0.17, 0.14), "Normal branch", COLORS["green"], "geo prior + residual")
    box(ax, (0.79, 0.47), (0.17, 0.14), "Predictive branch", COLORS["orange"], "risk-aware local prior")
    arrow(ax, (0.74, 0.75), (0.79, 0.77), COLORS["green"])
    arrow(ax, (0.74, 0.73), (0.79, 0.54), COLORS["orange"])
    box(ax, (0.62, 0.26), (0.24, 0.16), "Exact risk switch", COLORS["red"], "DROP | high risk | safer candidate")
    arrow(ax, (0.87, 0.70), (0.76, 0.42), COLORS["green"])
    arrow(ax, (0.87, 0.47), (0.80, 0.42), COLORS["orange"])
    box(ax, (0.68, 0.06), (0.18, 0.13), "Masked next hop", COLORS["navy"], "or explicit DROP")
    arrow(ax, (0.74, 0.26), (0.76, 0.19), COLORS["navy"])
    ax.text(0.07, 0.14, "Teacher unavailable at deployment", color=COLORS["gray"], fontsize=8.5)
    ax.text(0.07, 0.09, "No global graph or controller messages", color=COLORS["gray"], fontsize=8.5)
    rows = [
        {"phase": "training", "source": "global graph state", "destination": "PPO graph teacher"},
        {"phase": "training", "source": "PPO graph teacher", "destination": "masked distillation"},
        {"phase": "deployment", "source": "strict-local input", "destination": "normal branch"},
        {"phase": "deployment", "source": "strict-local input", "destination": "predictive branch"},
        {"phase": "deployment", "source": "normal/predictive logits", "destination": "exact risk switch"},
        {"phase": "deployment", "source": "exact risk switch", "destination": "masked next hop or DROP"},
    ]
    write_csv(DATA / "figure_01_architecture.csv", ["phase", "source", "destination"], rows)
    save(fig, "figure_01_architecture")


def fusion() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7))
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    axes[0].set_title("Legacy repeated execution", color=COLORS["red"], weight="bold")
    box(axes[0], (0.06, 0.70), (0.30, 0.14), "Normal forward #1", COLORS["red"])
    box(axes[0], (0.58, 0.70), (0.30, 0.14), "Predictive forward #1", COLORS["red"])
    box(axes[0], (0.06, 0.38), (0.30, 0.14), "Normal forward #2", COLORS["red"], "diagnostics")
    box(axes[0], (0.58, 0.38), (0.30, 0.14), "Predictive forward #2", COLORS["red"], "diagnostics")
    box(axes[0], (0.33, 0.10), (0.30, 0.14), "Action + metadata", COLORS["navy"])
    for x in (0.21, 0.73):
        arrow(axes[0], (x, 0.70), (x, 0.52), COLORS["red"])
    arrow(axes[0], (0.21, 0.38), (0.41, 0.24), COLORS["red"])
    arrow(axes[0], (0.73, 0.38), (0.57, 0.24), COLORS["red"])
    axes[0].text(0.5, 0.91, "four branch forwards per routing decision", ha="center", color=COLORS["gray"], fontsize=8)
    axes[1].set_title("Fused Exact SwitchGLOBE", color=COLORS["green"], weight="bold")
    box(axes[1], (0.06, 0.64), (0.34, 0.17), "Normal forward", COLORS["green"], "computed once")
    box(axes[1], (0.60, 0.64), (0.34, 0.17), "Predictive forward", COLORS["orange"], "computed once")
    box(axes[1], (0.32, 0.34), (0.36, 0.17), "Tensor fusion + switch", COLORS["blue"], "reuse logits and intermediates")
    box(axes[1], (0.32, 0.08), (0.36, 0.15), "Action + diagnostics", COLORS["navy"], "exactly shared computation")
    arrow(axes[1], (0.23, 0.64), (0.41, 0.51), COLORS["green"])
    arrow(axes[1], (0.77, 0.64), (0.59, 0.51), COLORS["orange"])
    arrow(axes[1], (0.50, 0.34), (0.50, 0.23), COLORS["blue"])
    axes[1].text(0.5, 0.91, "same branches, thresholds, mask, and action semantics", ha="center", color=COLORS["gray"], fontsize=8)
    write_csv(DATA / "figure_02_fusion.csv", ["execution", "normal_branch_calls", "predictive_branch_calls", "total_branch_calls"], [
        {"execution": "legacy", "normal_branch_calls": 2, "predictive_branch_calls": 2, "total_branch_calls": 4},
        {"execution": "fused_exact", "normal_branch_calls": 1, "predictive_branch_calls": 1, "total_branch_calls": 2},
    ])
    save(fig, "figure_02_fusion")


def external_results() -> None:
    p = ROOT / "artifacts/final_paper_simulation/synthesis/combined_comparison/summaries/statistics_overall.csv"
    records = read_csv(p)
    order = ["AODV", "OLSR", "Greedy Geographic", "Evo-QGeo (Adapted)", "RDQN-HERP (Adapted)", "GAT-GRU-DDQN", "FastSwitchGLOBE", "SwitchGLOBE Exact"]
    # Adapt gracefully to exact stored labels.
    available = list(dict.fromkeys(r["method"] for r in records))
    order = [m for m in order if m in available] + [m for m in available if m not in order]
    metrics = [
        ("connected_pair_pdr", "Connected-pair PDR", True),
        ("deadline_delivery_ratio", "Deadline delivery ratio", True),
        ("p95_success_delay", "p95 successful delay (steps)", False),
        ("mean_policy_input_bytes", "Policy input (bytes)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    colors = [COLORS["blue"] if m == "SwitchGLOBE Exact" else COLORS["orange"] if "Fast" in m else "#A9B7C6" for m in order]
    rows = []
    for ax, (metric, label, _) in zip(axes.ravel(), metrics):
        sub = {r["method"]: r for r in records if r["metric"] == metric}
        y = np.arange(len(order))
        vals = np.array([float(sub[m]["mean"]) for m in order])
        low = np.array([float(sub[m]["ci95_low"]) for m in order])
        high = np.array([float(sub[m]["ci95_high"]) for m in order])
        ax.barh(y, vals, color=colors, alpha=.95)
        ax.errorbar(vals, y, xerr=np.vstack([vals-low, high-vals]), fmt="none", ecolor=COLORS["ink"], capsize=2, lw=.8)
        ax.set_yticks(y, [m.replace(" (Adapted)", "\n(Adapted)") for m in order], fontsize=7.2)
        ax.invert_yaxis(); ax.set_title(label, loc="left", weight="bold")
        ax.grid(axis="x", alpha=.18); ax.spines[["top", "right", "left"]].set_visible(False)
        for m, v, lo, hi in zip(order, vals, low, high): rows.append({"method": m, "metric": metric, "mean": v, "ci95_low": lo, "ci95_high": hi})
    fig.suptitle("Common-simulator comparison: five-seed scenario-macro estimates", weight="bold", color=COLORS["navy"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    write_csv(DATA / "figure_03_external_results.csv", ["method", "metric", "mean", "ci95_low", "ci95_high"], rows)
    save(fig, "figure_03_external_results")


def protocol() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stages = [
        (0.02, "5 training seeds", "42, 77, 123, 314, 2718"),
        (0.22, "14 scenarios", "held-out, stress, OOD"),
        (0.42, "200 episodes", "per seed and scenario"),
        (0.62, "Seed-level inference", "paired effects + t-CI"),
        (0.82, "Acceptance gate", "speed and reliability"),
    ]
    for i, (x, title, sub) in enumerate(stages):
        box(ax, (x, .38), (.16, .26), title, COLORS["blue"] if i < 4 else COLORS["green"], sub)
        if i < len(stages)-1: arrow(ax, (x+.16, .51), (stages[i+1][0], .51), COLORS["gray"])
    ax.text(.5, .82, "Reliability experiment: 5 × 14 × 200 = 14,000 episodes per method", ha="center", weight="bold", color=COLORS["navy"], fontsize=12)
    ax.text(.5, .17, "Latency benchmark: batch 1 · warm-up 50 · 2,000 repeats · synchronized end-to-end timing", ha="center", color=COLORS["gray"], fontsize=9)
    write_csv(DATA / "figure_04_protocol.csv", ["x", "stage", "detail"], [{"x": x, "stage": stage, "detail": detail} for x, stage, detail in stages])
    save(fig, "figure_04_protocol")


def literature_coverage() -> None:
    p = Path(__file__).resolve().parent / "literature_comparison.csv"
    raw = read_csv(p)
    families = ["PDR", "Delay", "Throughput", "Energy", "Routing Overhead", "Decision latency"]
    counts = {k: 0 for k in families}
    for record in raw:
        s = record.get("metrics", "")
        try:
            vals = ast.literal_eval(s)
        except Exception:
            vals = [s]
        joined = " ".join(map(str, vals)).lower()
        for k in families[:-1]:
            if k.lower() in joined: counts[k] += 1
    # Direct compute-latency reporting was manually audited in the associated evidence matrix.
    counts["Decision latency"] = 4
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    labels = list(counts); values = [counts[k] for k in labels]
    cols = ["#A9B7C6"] * 5 + [COLORS["red"]]
    ax.barh(np.arange(len(labels)), values, color=cols)
    ax.set_yticks(np.arange(len(labels)), labels); ax.invert_yaxis()
    ax.set_xlabel("Papers in the 45-record project database")
    ax.set_title("Metric coverage in the project literature corpus", loc="left", weight="bold", color=COLORS["navy"])
    ax.grid(axis="x", alpha=.2); ax.spines[["top", "right", "left"]].set_visible(False)
    for i, v in enumerate(values): ax.text(v+.4, i, str(v), va="center", fontsize=9)
    ax.text(0, len(labels)-.15, "Direct decision-time count is a manual scope audit; generic end-to-end delay is not inference latency.", fontsize=7.5, color=COLORS["gray"])
    write_csv(DATA / "figure_05_literature_coverage.csv", ["metric_family", "paper_count", "corpus_size"], [{"metric_family": label, "paper_count": value, "corpus_size": len(raw)} for label, value in zip(labels, values)])
    save(fig, "figure_05_literature_coverage")


if __name__ == "__main__":
    setup()
    architecture()
    fusion()
    external_results()
    protocol()
    literature_coverage()
    print(f"generated figures in {OUT}")
