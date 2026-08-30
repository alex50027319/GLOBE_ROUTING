"""Validation and serialization for FastSwitchGLOBE-only Colab chunks."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from ..experiments.fast_external_comparison_campaign import FAST_METHOD
from .external_comparison_reporting import PRIMARY_METRICS, SCENARIOS
from .reporting import write_csv


FAST_METHOD_CONTRACT = {
    "name": FAST_METHOD,
    "slug": "fast_switchglobe",
    "source": "distilled from the seed-matched SwitchGLOBE Exact checkpoint",
    "fidelity": "repository implementation; single-pass, no Top-2 failover, no cache",
    "trainable": True,
    "control_plane": False,
    "observation_fields": [
        "self_features",
        "neighbor_features",
        "edge_features",
        "packet_features",
        "action_mask",
        "candidate_forwardability",
        "candidate_risk_features",
    ],
    "hop_radius": "1-hop",
    "privileged_information": False,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def write_fast_external_chunk(
    output_dir: Path,
    *,
    episode_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    training_rows: list[dict[str, Any]],
    deployment_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    seeds = tuple(int(seed) for seed in metadata["config"]["training_seeds"])
    expected_summary_keys = {
        (FAST_METHOD, scenario, seed) for scenario in SCENARIOS for seed in seeds
    }
    actual_summary_keys: set[tuple[str, str, int]] = set()
    for row in summary_rows:
        key = (str(row["method"]), str(row["scenario"]), int(row["training_seed"]))
        if key in actual_summary_keys:
            raise ValueError(f"duplicate FastSwitchGLOBE summary row: {key}")
        actual_summary_keys.add(key)
        for metric in PRIMARY_METRICS:
            value = row[metric]
            if (
                value is None
                and metadata["mode"] == "smoke"
                and metric in {"p95_success_delay", "energy_per_delivered_packet"}
            ):
                continue
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"non-finite {metric} for {key}")
    if actual_summary_keys != expected_summary_keys:
        raise ValueError(
            "FastSwitchGLOBE summary contract mismatch: "
            f"missing={sorted(expected_summary_keys - actual_summary_keys)[:3]}, "
            f"extra={sorted(actual_summary_keys - expected_summary_keys)[:3]}"
        )

    expected_episodes = (
        len(SCENARIOS)
        * len(seeds)
        * int(metadata["config"]["evaluation_episodes"])
    )
    if len(episode_rows) != expected_episodes:
        raise ValueError(
            f"FastSwitchGLOBE episode row count {len(episode_rows)} != {expected_episodes}"
        )
    episode_keys = {
        (
            str(row["method"]),
            str(row["scenario"]),
            int(row["training_seed"]),
            int(row["evaluation_seed"]),
        )
        for row in episode_rows
    }
    if len(episode_keys) != len(episode_rows):
        raise ValueError("duplicate FastSwitchGLOBE episode keys")
    if {row["method"] for row in episode_rows} != {FAST_METHOD}:
        raise ValueError("Fast chunk contains a method other than FastSwitchGLOBE")

    write_csv(output_dir / "raw" / "episodes.csv", episode_rows)
    write_csv(output_dir / "raw" / "seed_summaries.csv", summary_rows)
    write_csv(output_dir / "raw" / "training.csv", training_rows)
    write_csv(output_dir / "raw" / "deployment_costs.csv", deployment_rows)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "suite": "fast_switchglobe_external_comparison_chunk",
        "mode": metadata["mode"],
        "methods": [FAST_METHOD],
        "scenarios": list(SCENARIOS),
        "training_seeds": list(seeds),
        "episode_rows": len(episode_rows),
        "expected_episode_rows": expected_episodes,
        "seed_summary_rows": len(summary_rows),
        "expected_seed_summary_rows": len(SCENARIOS) * len(seeds),
        "training_rows": len(training_rows),
        "deployment_cost_rows": len(deployment_rows),
        "method_contract": FAST_METHOD_CONTRACT,
        "metadata": metadata,
    }
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest
