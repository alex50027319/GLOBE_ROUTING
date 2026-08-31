"""Calibrate and gate the compositional-curriculum SwitchGLOBE candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .evaluation.reporting import write_csv
from .experiments.evo_globe_campaign import (
    CompositionalCurriculumConfig,
    evaluate_cost_to_go_candidate,
    train_compositional_switchglobe,
)
from .run_evo_globe_candidate2 import (
    _build_reference,
    _checkpoint_payload,
    _latency_rows,
)
from .scenarios import (
    phase9_compositional_predictive_calibration_scenarios,
    phase9_curriculum,
    phase9_evaluation_scenarios,
    phase9_hole_calibration_scenarios,
)
from .utils.checkpoint import save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-archive", type=Path)
    parser.add_argument("--archive-member")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-episodes", type=int, default=50)
    parser.add_argument("--heldout-episodes", type=int, default=50)
    parser.add_argument("--dataset-episodes-per-scenario", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--link-loss-scale", action="append", type=float)
    parser.add_argument("--latency-rounds", type=int, default=5)
    parser.add_argument("--latency-warmup", type=int, default=20)
    parser.add_argument("--latency-repeats", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/evo_globe/candidate3_gate/seed_42"),
    )
    return parser.parse_args()


def _evaluate_groups(
    model,
    groups: dict[str, list],
    *,
    episodes: int,
    device: torch.device,
) -> dict[str, dict]:
    bases = {
        "generic": 2_000_000,
        "holes": 2_100_000,
        "composite": 2_200_000,
    }
    return {
        name: evaluate_cost_to_go_candidate(
            model,
            scenarios,
            episode_seed_base=bases[name],
            episodes_per_scenario=episodes,
            device=device,
        )
        for name, scenarios in groups.items()
    }


def _calibration_row(
    *,
    scale: float,
    training: dict,
    evaluation: dict[str, dict],
) -> dict:
    row = {
        "link_loss_scale": scale,
        "completed_epochs": training["completed_epochs"],
        "validation_kl": training["validation_kl"],
        "validation_reference_agreement": training[
            "validation_reference_agreement"
        ],
        "validation_risk_oracle_agreement": training[
            "validation_risk_oracle_agreement"
        ],
    }
    for group, result in evaluation.items():
        for metric in (
            "connected_pair_pdr",
            "deadline_delivery_ratio",
            "energy_per_delivered_packet",
        ):
            row[f"{group}_{metric}"] = result[metric]
    return row


def _select_candidate(rows: list[dict], exact: dict[str, dict]) -> dict:
    feasible = [
        row
        for row in rows
        if row["generic_connected_pair_pdr"]
        >= exact["generic"]["connected_pair_pdr"] - 0.004
        and row["holes_connected_pair_pdr"]
        >= exact["holes"]["connected_pair_pdr"] - 0.01
    ]
    pool = feasible or rows
    return max(
        pool,
        key=lambda row: (
            row["composite_connected_pair_pdr"],
            row["composite_deadline_delivery_ratio"],
            row["generic_connected_pair_pdr"],
            row["holes_connected_pair_pdr"],
            -row["composite_energy_per_delivered_packet"],
            -row["link_loss_scale"],
        ),
    )


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    scales = tuple(args.link_loss_scale or (0.5, 1.0, 1.5))
    if len(set(scales)) != len(scales):
        raise ValueError("link-loss scales must be unique")
    payload, provenance = _checkpoint_payload(args)
    heldout_scenarios = phase9_evaluation_scenarios(args.seed)
    reference = _build_reference(
        payload,
        max_nodes=heldout_scenarios[0].config.max_nodes,
    )
    groups = {
        "generic": phase9_curriculum(args.seed),
        "holes": phase9_hole_calibration_scenarios(args.seed),
        "composite": (
            phase9_compositional_predictive_calibration_scenarios(args.seed)
        ),
    }
    exact_calibration = _evaluate_groups(
        reference,
        groups,
        episodes=args.calibration_episodes,
        device=device,
    )
    candidates = {}
    training_by_scale = {}
    calibration_rows = []
    for scale in scales:
        config = CompositionalCurriculumConfig(
            dataset_episodes_per_scenario=(
                args.dataset_episodes_per_scenario
            ),
            epochs=args.epochs,
            link_loss_scale=scale,
        )
        candidate, training = train_compositional_switchglobe(
            reference,
            config,
            seed=args.seed,
            device=device,
        )
        evaluation = _evaluate_groups(
            candidate,
            groups,
            episodes=args.calibration_episodes,
            device=device,
        )
        candidates[scale] = candidate
        training_by_scale[scale] = training
        calibration_rows.append(
            _calibration_row(
                scale=scale,
                training=training,
                evaluation=evaluation,
            )
        )
    selected_row = _select_candidate(calibration_rows, exact_calibration)
    selected_scale = selected_row["link_loss_scale"]
    selected = candidates[selected_scale]
    exact_heldout = evaluate_cost_to_go_candidate(
        reference,
        heldout_scenarios,
        episode_seed_base=1_100_000,
        episodes_per_scenario=args.heldout_episodes,
        device=device,
    )
    candidate_heldout = evaluate_cost_to_go_candidate(
        selected,
        heldout_scenarios,
        episode_seed_base=1_100_000,
        episodes_per_scenario=args.heldout_episodes,
        device=device,
    )
    latency = _latency_rows(
        reference.cpu(),
        selected.cpu(),
        heldout_scenarios[0],
        rounds=args.latency_rounds,
        warmup=args.latency_warmup,
        repeats=args.latency_repeats,
        candidate_method="SwitchGLOBE_compositional",
    )
    exact_p95 = np.asarray([
        row["latency_p95_ms"] for row in latency
        if row["method"] == "SwitchGLOBE_exact"
    ])
    candidate_p95 = np.asarray([
        row["latency_p95_ms"] for row in latency
        if row["method"] == "SwitchGLOBE_compositional"
    ])
    heldout_delta = {
        metric: candidate_heldout[metric] - exact_heldout[metric]
        for metric in (
            "connected_pair_pdr",
            "deadline_delivery_ratio",
            "energy_per_delivered_packet",
        )
    }
    summary = {
        "seed": args.seed,
        "calibration_episodes_per_scenario": args.calibration_episodes,
        "heldout_episodes_per_scenario": args.heldout_episodes,
        "scenario_count": len(heldout_scenarios),
        **provenance,
        "selected_link_loss_scale": selected_scale,
        "selected_training": training_by_scale[selected_scale],
        "exact_calibration": {
            group: {
                key: value
                for key, value in result.items()
                if key != "scenario_rows"
            }
            for group, result in exact_calibration.items()
        },
        "calibration_candidates": calibration_rows,
        "exact_heldout": {
            key: value
            for key, value in exact_heldout.items()
            if key != "scenario_rows"
        },
        "candidate_heldout": {
            key: value
            for key, value in candidate_heldout.items()
            if key != "scenario_rows"
        },
        "heldout_delta": heldout_delta,
        "latency": {
            "exact_median_p95_ms": float(np.median(exact_p95)),
            "candidate_median_p95_ms": float(np.median(candidate_p95)),
            "candidate_over_exact_ratio": float(
                np.median(candidate_p95) / np.median(exact_p95)
            ),
        },
    }
    scenario_rows = []
    for exact_row, candidate_row in zip(
        exact_heldout["scenario_rows"],
        candidate_heldout["scenario_rows"],
        strict=True,
    ):
        if exact_row["scenario"] != candidate_row["scenario"]:
            raise RuntimeError("paired scenario order changed")
        row = {"scenario": exact_row["scenario"]}
        for metric in (
            "connected_pair_pdr",
            "deadline_delivery_ratio",
            "energy_per_delivered_packet",
        ):
            row[f"exact_{metric}"] = exact_row[metric]
            row[f"candidate_{metric}"] = candidate_row[metric]
            row[f"delta_{metric}"] = candidate_row[metric] - exact_row[metric]
        scenario_rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "calibration_candidates.csv", calibration_rows)
    write_csv(args.output_dir / "scenario_metrics.csv", scenario_rows)
    write_csv(args.output_dir / "latency_rounds.csv", latency)
    save_checkpoint(
        args.output_dir / "compositional_switchglobe.pt",
        selected,
        metadata={
            "selected_link_loss_scale": selected_scale,
            "training": training_by_scale[selected_scale],
            **provenance,
        },
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
