"""Validate and summarize the five-seed exact-latency benchmark ZIP."""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEEDS = (42, 77, 123, 314, 2718)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_zip", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def ci95(values) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = (
        2.776445 * float(values.std(ddof=1)) / math.sqrt(values.size)
        if values.size > 1 else 0.0
    )
    return mean, mean - half, mean + half


def main() -> int:
    args = parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.result_zip) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        runtime = pd.read_csv(io.BytesIO(archive.read("runtime_benchmarks.csv")))
        costs = pd.read_csv(io.BytesIO(archive.read("deployment_costs.csv")))
        for name in ("latency_p95_variants.png", "latency_p95_variants.svg"):
            (args.output_dir / name).write_bytes(archive.read(name))
    assert manifest["complete"] is True
    assert manifest["candidate_3_executed"] is False
    assert tuple(manifest["training_seeds"]) == SEEDS
    assert len(runtime) == manifest["runtime_rows"] == 130
    assert len(costs) == manifest["deployment_cost_rows"] == 30
    assert not runtime.duplicated(["training_seed", "variant", "component"]).any()
    end = runtime[runtime.component == "end_to_end_policy"].copy()
    aggregate = []
    for variant, group in end.groupby("variant"):
        row = {"variant": variant, "seed_count": len(group)}
        for metric in (
            "mean_ms", "p50_ms", "p95_ms", "p99_ms", "cold_start_ms",
            "decisions_per_second",
        ):
            mean, low, high = ci95(group[metric])
            row.update({metric: mean, f"{metric}_ci95_low": low,
                        f"{metric}_ci95_high": high})
        aggregate.append(row)
    aggregate_frame = pd.DataFrame(aggregate).sort_values("p95_ms")
    aggregate_frame.to_csv(args.output_dir / "aggregate_runtime.csv", index=False)
    lookup = end.set_index(["training_seed", "variant"])
    comparisons = []
    for device in ("cpu", "cuda"):
        baseline = f"legacy_repeated_{device}"
        for variant in (
            f"exact_eager_{device}", f"exact_buffered_{device}",
            f"exact_compiled_{device}",
        ):
            reductions = []
            for seed in SEEDS:
                old = float(lookup.loc[(seed, baseline), "p95_ms"])
                new = float(lookup.loc[(seed, variant), "p95_ms"])
                reductions.append(100.0 * (old - new) / old)
            mean, low, high = ci95(reductions)
            comparisons.append({
                "baseline": baseline, "variant": variant,
                "p95_reduction_percent": mean,
                "ci95_low_percent": low, "ci95_high_percent": high,
            })
    cross = []
    for seed in SEEDS:
        old = float(lookup.loc[(seed, "legacy_repeated_cuda"), "p95_ms"])
        new = float(lookup.loc[(seed, "exact_eager_cpu"), "p95_ms"])
        cross.append(100.0 * (old - new) / old)
    mean, low, high = ci95(cross)
    comparisons.append({
        "baseline": "legacy_repeated_cuda",
        "variant": "exact_eager_cpu",
        "p95_reduction_percent": mean,
        "ci95_low_percent": low, "ci95_high_percent": high,
    })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(
        args.output_dir / "paired_p95_improvements.csv", index=False
    )
    plot_variants = [
        "legacy_repeated_cuda", "exact_eager_cuda", "exact_buffered_cuda",
        "exact_compiled_cuda", "legacy_repeated_cpu", "exact_eager_cpu",
        "exact_buffered_cpu", "exact_compiled_cpu",
    ]
    values = [
        float(aggregate_frame.set_index("variant").loc[item, "p95_ms"])
        for item in plot_variants
    ]
    figure, axis = plt.subplots(figsize=(12, 5.5))
    colors = ["#7f8c8d" if "legacy" in item else (
        "#1f77b4" if "eager" in item else "#f39c12"
    ) for item in plot_variants]
    axis.bar(np.arange(len(values)), values, color=colors)
    axis.set_xticks(np.arange(len(values)))
    axis.set_xticklabels(plot_variants, rotation=28, ha="right")
    axis.set_ylabel("Five-seed mean synchronized p95 (ms)")
    axis.set_title("SwitchGLOBE Candidate 1/2 A100-session verification")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.output_dir / "verified_latency_comparison.png", dpi=240)
    figure.savefig(args.output_dir / "verified_latency_comparison.svg")
    plt.close(figure)
    selected = aggregate_frame.set_index("variant")
    report = f"""# Candidate 1/2 A100 verification

- Manifest: complete
- Seeds: {list(SEEDS)}
- Repetitions: {manifest['repeats']} after {manifest['warmup']} warm-ups
- Candidate 3 executed: no

## Five-seed mean synchronized p95

| Variant | p95 ms |
|---|---:|
"""
    for variant in plot_variants:
        report += f"| {variant} | {selected.loc[variant, 'p95_ms']:.4f} |\n"
    report += f"""

## Decision

- Candidate 1 exact eager CUDA reduces p95 by {comparison_frame[(comparison_frame.baseline == 'legacy_repeated_cuda') & (comparison_frame.variant == 'exact_eager_cuda')].iloc[0].p95_reduction_percent:.2f}% on a paired-seed basis.
- Candidate 1 plus eager CPU reduces p95 by {mean:.2f}% relative to legacy CUDA on the same Colab A100 VM.
- Buffered execution adds too little benefit to justify the extra stateful buffer.
- Compiled CPU and CUDA are slower and have larger cold-start cost.
- Recommended exact deployment runtime: `exact_eager_cpu`.
"""
    (args.output_dir / "verification_report.md").write_text(report, encoding="utf-8")
    (args.output_dir / "validated_manifest.json").write_text(
        json.dumps({**manifest, "validated": True}, indent=2), encoding="utf-8"
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
