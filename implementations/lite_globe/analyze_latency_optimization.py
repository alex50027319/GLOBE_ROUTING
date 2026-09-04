"""Seed-paired inference-latency statistics for local and Colab A100 runs."""

from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np


T_975_DF = {
    1: 12.706205,
    2: 4.302653,
    3: 3.182446,
    4: 2.776445,
    5: 2.570582,
    6: 2.446912,
    7: 2.364624,
    8: 2.306004,
    9: 2.262157,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-csv", type=Path, required=True)
    parser.add_argument("--a100-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--random-seed", type=int, default=20260905)
    return parser.parse_args()


def load_end_to_end(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row["component"] == "end_to_end_policy"
        ]


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = abs(float(np.mean(differences)))
    values = []
    for signs in product((-1.0, 1.0), repeat=len(differences)):
        values.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def holm_adjust(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.zeros(len(pvalues), dtype=float)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * pvalues[int(index)])
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def analyze(
    rows: list[dict[str, str]], *, environment: str,
    resamples: int, random_seed: int,
) -> list[dict[str, object]]:
    variants = sorted({row["variant"] for row in rows})
    output: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed)
    for device in sorted({row["device"] for row in rows}):
        baseline_name = f"exact_eager_{device}"
        baseline = {
            int(row["training_seed"]): float(row["p95_ms"])
            for row in rows if row["variant"] == baseline_name
        }
        if not baseline:
            continue
        device_results: list[dict[str, object]] = []
        for variant in variants:
            if variant == baseline_name or not variant.endswith(f"_{device}"):
                continue
            candidate = {
                int(row["training_seed"]): float(row["p95_ms"])
                for row in rows if row["variant"] == variant
            }
            seeds = sorted(set(baseline) & set(candidate))
            if not seeds:
                continue
            base = np.asarray([baseline[seed] for seed in seeds])
            cand = np.asarray([candidate[seed] for seed in seeds])
            delta_ms = cand - base
            reduction = 100.0 * (base - cand) / base
            n = len(seeds)
            se = float(np.std(reduction, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            tcrit = T_975_DF.get(n - 1, 1.959964)
            center = float(np.mean(reduction))
            samples = rng.choice(reduction, size=(resamples, n), replace=True).mean(axis=1)
            device_results.append({
                "environment": environment,
                "device": device,
                "baseline": baseline_name,
                "candidate": variant,
                "n_seeds": n,
                "seeds": ";".join(map(str, seeds)),
                "baseline_p95_ms_seed_mean": float(np.mean(base)),
                "candidate_p95_ms_seed_mean": float(np.mean(cand)),
                "paired_delta_ms_seed_mean": float(np.mean(delta_ms)),
                "paired_reduction_percent_seed_mean": center,
                "paired_reduction_t95_low": center - tcrit * se,
                "paired_reduction_t95_high": center + tcrit * se,
                "paired_reduction_bootstrap95_low": float(np.percentile(samples, 2.5)),
                "paired_reduction_bootstrap95_high": float(np.percentile(samples, 97.5)),
                "exact_sign_flip_p": exact_sign_flip_pvalue(delta_ms),
            })
        adjusted = holm_adjust([
            float(row["exact_sign_flip_p"]) for row in device_results
        ])
        for row, pvalue in zip(device_results, adjusted, strict=True):
            row["holm_adjusted_p"] = pvalue
        output.extend(device_results)
    return output


def main() -> int:
    args = parse_args()
    results = analyze(
        load_end_to_end(args.local_csv), environment="local_mac",
        resamples=args.bootstrap_resamples, random_seed=args.random_seed,
    )
    results.extend(analyze(
        load_end_to_end(args.a100_csv), environment="colab_a100",
        resamples=args.bootstrap_resamples, random_seed=args.random_seed + 1,
    ))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {len(results)} paired comparisons to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
