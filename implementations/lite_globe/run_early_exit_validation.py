"""Full paired trajectory validation for calibrated SwitchGLOBE early exit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import sys

import torch

from .env.fanet_env import FanetRoutingEnv
from .evaluation.evaluator import run_episode
from .evaluation.records import episode_row
from .evaluation.reporting import write_csv
from .experiments.external_comparison_campaign import _switchglobe_path
from .provenance import checkpoint_sha256_map, config_sha256, git_provenance
from .run_latency_benchmark import _load_early_exit
from .scenarios import phase9_evaluation_scenarios


SEEDS = (42, 77, 123, 314, 2718)
OUTCOME_FIELDS = (
    "delivered",
    "dropped",
    "drop_reason",
    "steps",
    "hop_count",
    "transmission_attempts",
    "deadline_met",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--exact-episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=200)
    return parser.parse_args()


def _read_exact(path: Path) -> dict[tuple[int, str, int], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        (int(row["training_seed"]), row["scenario"], int(row["evaluation_seed"])): row
        for row in rows
    }


def main() -> int:
    args = parse_args()
    if args.episodes != 200:
        raise ValueError("full validation requires exactly 200 episodes per scenario")
    exact = _read_exact(args.exact_episodes)
    early_rows: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    for training_seed in SEEDS:
        scenarios = phase9_evaluation_scenarios(training_seed)
        policy = _load_early_exit(
            args.checkpoint_dir,
            seed=training_seed,
            max_nodes=scenarios[0].config.max_nodes,
            device=torch.device("cpu"),
        )
        for scenario_index, scenario in enumerate(scenarios):
            env = FanetRoutingEnv(scenario.config)
            for evaluation_seed in range(
                1_100_000 + scenario_index * 10_000,
                1_100_000 + scenario_index * 10_000 + args.episodes,
            ):
                result = run_episode(
                    env,
                    policy,
                    seed=evaluation_seed,
                    reset_options=scenario.reset_options,
                )
                row: dict[str, object] = episode_row(
                    result,
                    method="SwitchGLOBE Early Exit",
                    scenario=scenario.name,
                    training_seed=training_seed,
                )
                diagnostics = policy.episode_diagnostics()
                row["early_exit_steps"] = diagnostics.get("early_exit_steps", 0.0)
                row["predictive_branch_steps"] = diagnostics.get(
                    "predictive_branch_steps", 0.0
                )
                early_rows.append(row)
                key = (training_seed, scenario.name, evaluation_seed)
                baseline = exact.get(key)
                if baseline is None:
                    mismatches.append({
                        "training_seed": training_seed,
                        "scenario": scenario.name,
                        "evaluation_seed": evaluation_seed,
                        "field": "missing_exact_row",
                        "exact": "",
                        "early_exit": "",
                    })
                    continue
                for field in OUTCOME_FIELDS:
                    if str(row[field]) != str(baseline[field]):
                        mismatches.append({
                            "training_seed": training_seed,
                            "scenario": scenario.name,
                            "evaluation_seed": evaluation_seed,
                            "field": field,
                            "exact": baseline[field],
                            "early_exit": row[field],
                        })
            scenario_rows = [
                row for row in early_rows
                if row["training_seed"] == training_seed
                and row["scenario"] == scenario.name
            ]
            early_steps = sum(float(row["early_exit_steps"]) for row in scenario_rows)
            predictive_steps = sum(
                float(row["predictive_branch_steps"]) for row in scenario_rows
            )
            total = early_steps + predictive_steps
            branch_rows.append({
                "training_seed": training_seed,
                "scenario": scenario.name,
                "decision_steps": total,
                "early_exit_steps": early_steps,
                "predictive_branch_steps": predictive_steps,
                "early_exit_rate": early_steps / total if total else 0.0,
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "early_exit_episodes.csv", early_rows)
    write_csv(args.output_dir / "outcome_mismatches.csv", mismatches or [{
        "training_seed": "", "scenario": "", "evaluation_seed": "",
        "field": "", "exact": "", "early_exit": "",
    }])
    write_csv(args.output_dir / "branch_usage.csv", branch_rows)
    checkpoint_paths = {
        f"switchglobe_exact_seed_{seed}": _switchglobe_path(args.checkpoint_dir, seed)
        for seed in SEEDS
    }
    total_early = sum(float(row["early_exit_steps"]) for row in early_rows)
    total_predictive = sum(float(row["predictive_branch_steps"]) for row in early_rows)
    effective_config = {
        "seeds": list(SEEDS),
        "scenario_count_per_seed": 14,
        "episodes_per_scenario": args.episodes,
        "outcome_fields": list(OUTCOME_FIELDS),
    }
    manifest = {
        "schema_version": 1,
        "complete": True,
        "suite": "switchglobe_early_exit_full_paired_validation",
        "episode_rows": len(early_rows),
        "paired_outcome_mismatches": len(mismatches),
        "decision_steps": total_early + total_predictive,
        "early_exit_steps": total_early,
        "predictive_branch_steps": total_predictive,
        "early_exit_rate": (
            total_early / (total_early + total_predictive)
            if total_early + total_predictive else 0.0
        ),
        "exact_episode_source": str(args.exact_episodes),
        "config": effective_config,
        "config_sha256": config_sha256(effective_config),
        "checkpoint_sha256": checkpoint_sha256_map(checkpoint_paths),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        **git_provenance(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
