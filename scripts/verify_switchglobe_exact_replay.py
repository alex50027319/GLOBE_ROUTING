"""Replay optimized SwitchGLOBE and compare non-latency rows to verified ZIPs."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.evaluation import episode_row, evaluate_policy_results
from implementations.lite_globe.experiments.external_comparison_campaign import load_switchglobe
from implementations.lite_globe.scenarios import phase9_evaluation_scenarios


SEEDS = (42, 77, 123, 314, 2718)
COLUMNS = (
    "delivered", "dropped", "drop_reason", "steps", "hop_count",
    "transmission_attempts", "transmission_energy_proxy", "deadline_met",
    "policy_input_bytes", "switch_steps", "branch_disagreement_steps",
    "switch_danger_reduction", "false_switch_steps", "missed_risk_steps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args(); report = []
    for seed in SEEDS:
        scenarios = phase9_evaluation_scenarios(seed)
        policy = load_switchglobe(
            args.checkpoint_dir, seed=seed,
            max_nodes=scenarios[0].config.max_nodes, hidden_dim=64,
            device="cpu",
        )
        rows = []
        for index, scenario in enumerate(scenarios):
            evaluation_seeds = list(range(
                1_100_000 + index * 10_000,
                1_100_000 + index * 10_000 + 200,
            ))
            results = evaluate_policy_results(
                FanetRoutingEnv(scenario.config), policy, evaluation_seeds,
                reset_options=scenario.reset_options,
            )
            rows.extend(episode_row(
                result, method="SwitchGLOBE", scenario=scenario.name,
                training_seed=seed,
            ) for result in results)
        current = pd.DataFrame(rows).sort_values(
            ["scenario", "evaluation_seed"]
        ).reset_index(drop=True)
        with zipfile.ZipFile(args.zip_dir / f"seeds_{seed}.zip") as archive:
            reference = pd.read_csv(io.BytesIO(archive.read("raw/episodes.csv")))
        reference = reference[reference.method == "SwitchGLOBE"].sort_values(
            ["scenario", "evaluation_seed"]
        ).reset_index(drop=True)
        seed_result = {"seed": seed, "rows": len(current), "complete": True}
        for column in COLUMNS:
            try:
                left = pd.to_numeric(reference[column])
                right = pd.to_numeric(current[column])
                equal = bool(np.allclose(
                    left, right, rtol=0, atol=1e-7, equal_nan=True
                ))
            except (TypeError, ValueError):
                equal = reference[column].fillna("").astype(str).equals(
                    current[column].fillna("").astype(str)
                )
            seed_result[column] = equal
            seed_result["complete"] = seed_result["complete"] and equal
        report.append(seed_result)
        print(seed, seed_result["complete"], flush=True)
    payload = {
        "complete": all(item["complete"] for item in report),
        "seeds": list(SEEDS), "episode_rows": sum(item["rows"] for item in report),
        "columns": list(COLUMNS), "per_seed": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
