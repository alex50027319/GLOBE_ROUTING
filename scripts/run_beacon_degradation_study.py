#!/usr/bin/env python3
"""Beacon-degradation sensitivity study for the local risk features (E-1).

Research question
-----------------
The deployed policy is driven by ``m_v`` (link margin), ``l_v`` (link lifetime)
and ``o_v`` (best onward lifetime). In the current simulator those are read from
exact simulator state, so the published reliability numbers assume a relay with
perfect, instantaneous knowledge of neighbour and neighbour-of-neighbour
kinematics. This study measures how the methods degrade when that assumption is
relaxed, and in particular whether the risk switch earns its keep once the
signal it switches on becomes unreliable.

Methods compared
----------------
``SwitchGLOBE``            two-branch policy with the calibrated risk switch
``Predictive Prior Only``  the predictive branch alone (no normal branch, no
                           switch); statistically equivalent to SwitchGLOBE
                           under exact signals
``Geo-Residual Student``   the normal branch alone; uses no risk features and
                           therefore serves as the degradation-immune reference

Outputs
-------
``episodes.csv``          one row per episode
``seed_summaries.csv``    one row per (method, degradation cell, scenario, seed)
``degradation_slopes.csv``  paired SwitchGLOBE - Prior Only effects per cell
``manifest.json``         configuration, checkpoint hashes and row counts

The proposed-method checkpoints are read-only; this script never writes to them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.evaluation import (
    episode_row,
    evaluate_policy_results,
    generalization_summary,
)
from implementations.lite_globe.models import (
    GeographicResidualStudentPolicy,
    LiteGlobePStudentPolicy,
    SwitchGlobePolicy,
)
from implementations.lite_globe.models.policy_adapter import StudentPolicyAdapter
from implementations.lite_globe.scenarios import phase9_evaluation_scenarios
from implementations.lite_globe.scenarios.degradation_suite import (
    DegradationCell,
    apply_degradation,
    degradation_grid,
)
from implementations.lite_globe.utils import load_checkpoint

DEFAULT_SEEDS = (42, 77, 123, 314, 2718)
PRIMARY_METRICS = (
    "connected_pair_pdr",
    "deadline_delivery_ratio",
    "overall_pdr",
    "p95_success_delay",
    "energy_per_delivered_packet",
    "total_drop_rate",
)


@dataclass(frozen=True)
class StudyConfig:
    seeds: tuple[int, ...]
    episodes: int
    hidden_dim: int
    device: torch.device
    include_combined: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(path: Path, description: str) -> Path:
    if not path.is_file():
        raise SystemExit(
            f"missing {description}: {path}\n"
            "The Phase 8 / Phase 11 checkpoints are not committed to this "
            "repository. Point --phase8-checkpoint-dir and "
            "--phase11-checkpoint-dir at the directories that hold them, or "
            "regenerate them with scripts/train_switchglobe_pipeline.py."
        )
    return path


def _build_methods(
    *,
    training_seed: int,
    phase8_dir: Path,
    phase11_dir: Path,
    config: StudyConfig,
    max_nodes: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load the three comparison policies for one training seed."""

    phase8_path = _require(
        phase8_dir / f"seed_{training_seed}" / "geo_residual_kd.pt",
        "Phase 8 (Geo-Residual Student) checkpoint",
    )
    phase11_path = _require(
        phase11_dir / f"seed_{training_seed}" / "lite_globe_p.pt",
        "Phase 11 (Predictive Student) checkpoint",
    )

    normal = GeographicResidualStudentPolicy(max_nodes, hidden_dim=config.hidden_dim)
    load_checkpoint(phase8_path, normal, map_location=config.device)

    predictive_source = LiteGlobePStudentPolicy(max_nodes, hidden_dim=config.hidden_dim)
    load_checkpoint(phase11_path, predictive_source, map_location=config.device)

    # The predictive branch inside SwitchGLOBE has its learned residual disabled,
    # so "Predictive Prior Only" is the identical scoring function without the
    # normal branch and without the switch.
    prior_only = LiteGlobePStudentPolicy(max_nodes, hidden_dim=config.hidden_dim)
    prior_only.load_state_dict(predictive_source.state_dict())
    prior_only.set_residual_weight(0.0)

    switch_predictive = LiteGlobePStudentPolicy(max_nodes, hidden_dim=config.hidden_dim)
    switch_predictive.load_state_dict(predictive_source.state_dict())
    switchglobe = SwitchGlobePolicy(normal, switch_predictive)

    models = {
        "SwitchGLOBE": switchglobe,
        "Predictive Prior Only": prior_only,
        "Geo-Residual Student": normal,
    }
    hashes = {
        "phase8": _sha256(phase8_path),
        "phase11": _sha256(phase11_path),
    }
    return models, hashes


def _evaluation_seeds(training_seed: int, cell_index: int, scenario_index: int,
                      episodes: int) -> list[int]:
    """Return a disjoint, deterministic evaluation-seed block.

    Every method sees the same block for a given (cell, scenario, seed) so the
    comparison stays paired at the episode level.
    """

    start = (
        training_seed
        + 4_100_000
        + cell_index * 100_000
        + scenario_index * 1_000
    )
    return list(range(start, start + episodes))


def run_study(
    *,
    phase8_dir: Path,
    phase11_dir: Path,
    output_dir: Path,
    config: StudyConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = degradation_grid(include_combined=config.include_combined)
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, dict[str, str]] = {}

    for training_seed in config.seeds:
        base_scenarios = phase9_evaluation_scenarios(training_seed)
        max_nodes = base_scenarios[0].config.max_nodes
        models, hashes = _build_methods(
            training_seed=training_seed,
            phase8_dir=phase8_dir,
            phase11_dir=phase11_dir,
            config=config,
            max_nodes=max_nodes,
        )
        checkpoint_hashes[str(training_seed)] = hashes

        for cell_index, cell in enumerate(cells):
            scenarios = apply_degradation(base_scenarios, cell)
            for scenario_index, scenario in enumerate(scenarios):
                seeds = _evaluation_seeds(
                    training_seed, cell_index, scenario_index, config.episodes
                )
                for method_name, model in models.items():
                    model.eval()
                    results = evaluate_policy_results(
                        FanetRoutingEnv(scenario.config),
                        StudentPolicyAdapter(model, device=config.device),
                        seeds,
                        reset_options=scenario.reset_options,
                    )
                    for result in results:
                        row = episode_row(
                            result,
                            method=method_name,
                            scenario=scenario.name,
                            training_seed=training_seed,
                        )
                        row.update(
                            base_scenario=scenario.name.split("__")[0],
                            degradation_axis=cell.axis,
                            degradation_label=cell.label,
                            beacon_period=cell.settings.beacon_period,
                            beacon_loss=cell.settings.beacon_loss,
                            position_noise_std=cell.settings.position_noise_std,
                            velocity_noise_std=cell.settings.velocity_noise_std,
                            degrade_topology=cell.settings.degrade_topology,
                        )
                        episode_rows.append(row)
                    summary = generalization_summary(
                        results,
                        method=method_name,
                        scenario=scenario.name,
                        training_seed=training_seed,
                    )
                    summary_rows.append(
                        {
                            **summary,
                            "base_scenario": scenario.name.split("__")[0],
                            "degradation_axis": cell.axis,
                            "degradation_label": cell.label,
                            "beacon_period": cell.settings.beacon_period,
                            "beacon_loss": cell.settings.beacon_loss,
                            "position_noise_std": cell.settings.position_noise_std,
                            "velocity_noise_std": cell.settings.velocity_noise_std,
                            "degrade_topology": cell.settings.degrade_topology,
                        }
                    )
                print(
                    f"seed {training_seed} | {cell.label:>18} | "
                    f"{scenario.name}",
                    flush=True,
                )

    _write_csv(output_dir / "episodes.csv", episode_rows)
    _write_csv(output_dir / "seed_summaries.csv", summary_rows)
    slopes = _degradation_slopes(summary_rows, config.seeds)
    _write_csv(output_dir / "degradation_slopes.csv", slopes)

    manifest = {
        "study": "beacon_degradation_sensitivity",
        "purpose": (
            "Measure how SwitchGLOBE and Predictive Prior Only degrade when the "
            "local risk features are acquired from periodic, lossy, noisy "
            "beacons instead of exact simulator state."
        ),
        "seeds": list(config.seeds),
        "episodes_per_cell": config.episodes,
        "hidden_dim": config.hidden_dim,
        "device": str(config.device),
        "degradation_cells": [
            {
                "axis": cell.axis,
                "label": cell.label,
                "beacon_period": cell.settings.beacon_period,
                "beacon_loss": cell.settings.beacon_loss,
                "position_noise_std": cell.settings.position_noise_std,
                "velocity_noise_std": cell.settings.velocity_noise_std,
                "dead_reckon": cell.settings.dead_reckon,
                "degrade_topology": cell.settings.degrade_topology,
            }
            for cell in cells
        ],
        "checkpoint_sha256": checkpoint_hashes,
        "episode_rows": len(episode_rows),
        "summary_rows": len(summary_rows),
        "complete": True,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {len(episode_rows)} episode rows to {output_dir}")


def _degradation_slopes(
    summary_rows: list[dict[str, Any]], seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    """Return seed-paired SwitchGLOBE - Prior Only effects per degradation cell.

    A non-zero, growing effect as degradation increases would be evidence that
    the risk switch earns its keep once the signal becomes unreliable. A flat
    zero effect would confirm the switch is redundant at every operating point.
    """

    by_key: dict[tuple[str, str, int], list[float]] = {}
    for row in summary_rows:
        for metric in PRIMARY_METRICS:
            value = row.get(metric)
            if value is None:
                continue
            key = (row["method"], row["degradation_label"], row["training_seed"])
            by_key.setdefault((*key, metric), []).append(float(value))  # type: ignore[arg-type]

    labels = sorted({row["degradation_label"] for row in summary_rows})
    out: list[dict[str, Any]] = []
    for label in labels:
        for metric in PRIMARY_METRICS:
            paired = []
            for seed in seeds:
                a = by_key.get(("Predictive Prior Only", label, seed, metric))
                b = by_key.get(("SwitchGLOBE", label, seed, metric))
                if not a or not b:
                    continue
                paired.append(st.mean(b) - st.mean(a))
            if len(paired) < 2:
                continue
            mean = st.mean(paired)
            sd = st.stdev(paired)
            half = 2.776 * sd / (len(paired) ** 0.5) if len(paired) == 5 else float("nan")
            out.append(
                {
                    "degradation_label": label,
                    "metric": metric,
                    "contrast": "SwitchGLOBE - Predictive Prior Only",
                    "n_pairs": len(paired),
                    "mean": mean,
                    "sd": sd,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                }
            )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase8-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--phase11-checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/beacon_degradation/full"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Run only the one-factor-at-a-time axes.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One seed, 10 episodes per cell. Never mix with full results.",
    )
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    episodes = args.episodes
    output_dir = args.output_dir
    if args.smoke:
        seeds = (seeds[0],)
        episodes = 10
        if output_dir == Path("artifacts/beacon_degradation/full"):
            output_dir = Path("artifacts/beacon_degradation/smoke")

    run_study(
        phase8_dir=args.phase8_checkpoint_dir,
        phase11_dir=args.phase11_checkpoint_dir,
        output_dir=output_dir,
        config=StudyConfig(
            seeds=seeds,
            episodes=episodes,
            hidden_dim=args.hidden_dim,
            device=torch.device(args.device),
            include_combined=not args.no_combined,
        ),
    )


if __name__ == "__main__":
    main()
