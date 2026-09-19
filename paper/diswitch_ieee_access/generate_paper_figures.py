from __future__ import annotations

import ast
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diswitch-matplotlib")
import matplotlib
matplotlib.use("Agg")
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


def box(
    ax,
    xy,
    wh,
    text,
    color,
    subtitle=None,
    dashed=False,
    title_size=10,
    subtitle_size=7.7,
):
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
    ax.text(
        x + w / 2,
        y + h * 0.60,
        text,
        ha="center",
        va="center",
        weight="bold",
        color=COLORS["ink"],
        fontsize=title_size,
        linespacing=1.15,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.27,
            subtitle,
            ha="center",
            va="center",
            color=COLORS["gray"],
            fontsize=subtitle_size,
            linespacing=1.15,
        )


def arrow(ax, a, b, color=None, label=None):
    color = color or COLORS["gray"]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, linewidth=1.3, color=color))
    if label:
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2 + 0.025, label, ha="center", va="bottom", color=color, fontsize=7.5)


def architecture_training() -> None:
    """Training-only path rendered as an independent full-width figure."""
    fig, ax = plt.subplots(figsize=(11.6, 3.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.02,
        0.91,
        "Privileged training: global knowledge is distilled into local policies",
        weight="bold",
        color=COLORS["navy"],
        fontsize=12.5,
    )
    stages = [
        (0.025, 0.18, "Global graph state", "topology + mobility + packet", COLORS["blue"]),
        (0.285, 0.18, "PPO graph teacher", "masked global preferences", COLORS["blue"]),
        (0.545, 0.19, "Policy distillation", "KL + local supervision", COLORS["cyan"]),
        (0.815, 0.16, "Strict-local students", "normal + predictive policies", COLORS["green"]),
    ]
    y, h = 0.36, 0.27
    for i, (x, w, title, subtitle, color) in enumerate(stages):
        box(ax, (x, y), (w, h), title, color, subtitle, title_size=9.5, subtitle_size=7.5)
        if i < len(stages) - 1:
            arrow(ax, (x + w, y + h / 2), (stages[i + 1][0], y + h / 2), color)
    ax.text(
        0.50,
        0.13,
        "The teacher and global graph are removed after training; deployment receives no controller messages.",
        ha="center",
        color=COLORS["gray"],
        fontsize=8.5,
    )
    rows = [
        {"phase": "training", "source": "global graph state", "destination": "PPO graph teacher"},
        {"phase": "training", "source": "PPO graph teacher", "destination": "policy distillation"},
        {"phase": "training", "source": "policy distillation", "destination": "strict-local students"},
    ]
    write_csv(DATA / "figure_01_training.csv", ["phase", "source", "destination"], rows)
    save(fig, "figure_01_training")


def architecture_deployment() -> None:
    """Strict-local deployment path rendered as a separate horizontal figure."""
    fig, ax = plt.subplots(figsize=(11.6, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.02,
        0.92,
        "Decentralized batch-1 forwarding: local evidence branches and reconverges at the switch",
        weight="bold",
        color=COLORS["navy"],
        fontsize=12.5,
    )
    box(ax, (0.025, 0.34), (0.18, 0.28), "Strict-local input", COLORS["blue"], "relay + one-hop neighbors", title_size=9.5)
    box(ax, (0.30, 0.58), (0.20, 0.21), "Normal policy", COLORS["green"], "geographic prior + residual", title_size=9.4, subtitle_size=7.3)
    box(ax, (0.30, 0.17), (0.20, 0.21), "Predictive policy", COLORS["orange"], "risk-aware local prior", title_size=9.4, subtitle_size=7.3)
    box(ax, (0.61, 0.34), (0.17, 0.28), "Risk switch", COLORS["red"], "DROP | high risk | safer route", title_size=9.7, subtitle_size=7.1)
    box(ax, (0.86, 0.34), (0.12, 0.28), "Masked\nnext hop", COLORS["navy"], "or DROP", title_size=9.0, subtitle_size=7.2)
    arrow(ax, (0.205, 0.48), (0.30, 0.685), COLORS["green"])
    arrow(ax, (0.205, 0.48), (0.30, 0.275), COLORS["orange"])
    arrow(ax, (0.50, 0.685), (0.61, 0.51), COLORS["green"])
    arrow(ax, (0.50, 0.275), (0.61, 0.45), COLORS["orange"])
    arrow(ax, (0.78, 0.48), (0.86, 0.48), COLORS["navy"])
    ax.text(
        0.50,
        0.055,
        "All switch inputs are computed from strict-local observations and the two policy outputs.",
        ha="center",
        color=COLORS["gray"],
        fontsize=8.5,
    )
    rows = [
        {"phase": "deployment", "source": "strict-local input", "destination": "normal policy"},
        {"phase": "deployment", "source": "strict-local input", "destination": "predictive policy"},
        {"phase": "deployment", "source": "normal policy", "destination": "risk switch"},
        {"phase": "deployment", "source": "predictive policy", "destination": "risk switch"},
        {"phase": "deployment", "source": "risk switch", "destination": "masked next hop or DROP"},
    ]
    write_csv(DATA / "figure_02_deployment.csv", ["phase", "source", "destination"], rows)
    save(fig, "figure_02_deployment")


def fusion() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), gridspec_kw={"wspace": 0.22})
    for ax in axes:
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    axes[0].set_title("Legacy repeated execution", color=COLORS["red"], weight="bold")
    box(axes[0], (0.03, 0.70), (0.38, 0.14), "Normal policy forward #1", COLORS["red"], title_size=7.3)
    box(axes[0], (0.56, 0.70), (0.38, 0.14), "Predictive policy forward #1", COLORS["red"], title_size=7.0)
    box(axes[0], (0.03, 0.38), (0.38, 0.14), "Normal policy forward #2", COLORS["red"], "diagnostics", title_size=7.3, subtitle_size=6.9)
    box(axes[0], (0.56, 0.38), (0.38, 0.14), "Predictive policy forward #2", COLORS["red"], "diagnostics", title_size=7.0, subtitle_size=6.9)
    box(axes[0], (0.31, 0.10), (0.36, 0.14), "Action + metadata", COLORS["navy"], title_size=9.0)
    for x in (0.22, 0.75):
        arrow(axes[0], (x, 0.70), (x, 0.52), COLORS["red"])
    arrow(axes[0], (0.22, 0.38), (0.40, 0.24), COLORS["red"])
    arrow(axes[0], (0.75, 0.38), (0.58, 0.24), COLORS["red"])
    axes[0].text(0.5, 0.91, "four branch forwards per routing decision", ha="center", color=COLORS["gray"], fontsize=8)
    axes[1].set_title("DiSwitch", color=COLORS["green"], weight="bold")
    box(axes[1], (0.03, 0.64), (0.39, 0.17), "Normal policy forward", COLORS["green"], "computed once", title_size=8.7, subtitle_size=7.2)
    box(axes[1], (0.56, 0.64), (0.39, 0.17), "Predictive policy forward", COLORS["orange"], "computed once", title_size=8.4, subtitle_size=7.2)
    box(axes[1], (0.30, 0.34), (0.40, 0.17), "Tensor reuse + risk switch", COLORS["blue"], "reuse logits and intermediates", title_size=8.5, subtitle_size=7.0)
    box(axes[1], (0.30, 0.08), (0.40, 0.15), "Action + diagnostics", COLORS["navy"], "shared computation path", title_size=8.8, subtitle_size=7.0)
    arrow(axes[1], (0.23, 0.64), (0.41, 0.51), COLORS["green"])
    arrow(axes[1], (0.77, 0.64), (0.59, 0.51), COLORS["orange"])
    arrow(axes[1], (0.50, 0.34), (0.50, 0.23), COLORS["blue"])
    axes[1].text(0.5, 0.91, "same branches, thresholds, mask, and action semantics", ha="center", color=COLORS["gray"], fontsize=8)
    write_csv(DATA / "figure_02_fusion.csv", ["execution", "normal_branch_calls", "predictive_branch_calls", "total_branch_calls"], [
        {"execution": "legacy", "normal_branch_calls": 2, "predictive_branch_calls": 2, "total_branch_calls": 4},
        {"execution": "DiSwitch", "normal_branch_calls": 1, "predictive_branch_calls": 1, "total_branch_calls": 2},
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
    display_name = {"SwitchGLOBE Exact": "DiSwitch", "FastSwitchGLOBE": "Fast-DiSwitch"}
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
        ax.set_yticks(y, [display_name.get(m, m).replace(" (Adapted)", "\n(Adapted)") for m in order], fontsize=7.2)
        ax.invert_yaxis(); ax.set_title(label, loc="left", weight="bold")
        ax.grid(axis="x", alpha=.18); ax.spines[["top", "right", "left"]].set_visible(False)
        for m, v, lo, hi in zip(order, vals, low, high): rows.append({"method": display_name.get(m, m), "metric": metric, "mean": v, "ci95_low": lo, "ci95_high": hi})
    fig.suptitle("Common-simulator comparison: five-seed scenario-macro estimates", weight="bold", color=COLORS["navy"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    write_csv(DATA / "figure_03_external_results.csv", ["method", "metric", "mean", "ci95_low", "ci95_high"], rows)
    save(fig, "figure_03_external_results")


def protocol() -> None:
    fig, ax = plt.subplots(figsize=(12.0, 3.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    stages = [
        (0.025, "5 training seeds", "42, 77, 123, 314, 2718"),
        (0.225, "14 scenarios", "held-out + stress + OOD"),
        (0.425, "200 episodes", "per seed and scenario"),
        (0.625, "Seed-level\ninference", "paired effects + t-CI"),
        (0.825, "Acceptance\ngate", "speed + reliability"),
    ]
    for i, (x, title, sub) in enumerate(stages):
        box(
            ax,
            (x, .36),
            (.15, .28),
            title,
            COLORS["blue"] if i < 4 else COLORS["green"],
            sub,
            title_size=8.8,
            subtitle_size=6.9,
        )
        if i < len(stages)-1: arrow(ax, (x+.15, .50), (stages[i+1][0], .50), COLORS["gray"])
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
    architecture_training()
    architecture_deployment()
    fusion()
    external_results()
    protocol()
    literature_coverage()
    print(f"generated figures in {OUT}")
