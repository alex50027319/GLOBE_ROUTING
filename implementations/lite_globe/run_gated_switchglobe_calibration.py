"""Offline early-exit gate calibration replayed on SwitchGLOBE Exact checkpoints.

This does not train anything and does not change SwitchGLOBE Exact's actions.
It requires already-trained, already-calibrated SwitchGLOBE Exact checkpoints
(``<checkpoint-dir>/seed_<seed>/switchglobe.pt`` or the historical
``risk_switch_lite_globe_p.pt`` filename); see ``docs/method_history.md`` and
the SwitchGLOBE Experiment Contract for how those are produced. Output is a
divergence/skip-rate report to help choose a safety margin for a proposed
predictive-branch early-exit gate -- it is not a full-run PDR or latency
result and must not be merged with those artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys

import torch

from .evaluation.gated_switchglobe_calibration import (
    DEFAULT_GATE_MARGINS,
    GateCalibrationConfig,
    run_gate_calibration,
    write_gate_calibration_artifacts,
)
from .experiments.external_comparison_campaign import _switchglobe_path
from .provenance import checkpoint_sha256_map, git_provenance


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--switchglobe-checkpoint-dir",
        type=Path,
        default=Path("artifacts/switchglobe/final/checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/gated_switchglobe/calibration"),
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--evaluation-episodes", type=int, default=200)
    parser.add_argument(
        "--gate-margin",
        action="append",
        type=float,
        help=(
            "Candidate absolute danger-score cutoff(s) for skipping the "
            "predictive branch (skip iff normal_danger <= margin); "
            "repeatable. Defaults to a fixed sweep."
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = tuple(args.seed) if args.seed else (42, 77, 123, 314, 2718)
    if args.smoke:
        seeds = tuple(args.seed) if args.seed else (42,)
    if len(seeds) != len(set(seeds)):
        raise ValueError("training seeds must be unique")
    gate_margins = tuple(args.gate_margin) if args.gate_margin else DEFAULT_GATE_MARGINS
    config = GateCalibrationConfig(
        training_seeds=seeds,
        evaluation_episodes=3 if args.smoke else args.evaluation_episodes,
        hidden_dim=args.hidden_dim,
        gate_margins=gate_margins,
    )
    device = _device(args.device)
    rows = run_gate_calibration(
        config,
        switchglobe_checkpoint_dir=args.switchglobe_checkpoint_dir,
        device=device,
    )
    checkpoint_paths: dict[str, Path] = {}
    for seed in seeds:
        try:
            checkpoint_paths[f"switchglobe_exact_seed_{seed}"] = _switchglobe_path(
                args.switchglobe_checkpoint_dir, seed
            )
        except FileNotFoundError:
            pass
    manifest = write_gate_calibration_artifacts(
        args.output_dir,
        episode_rows=rows["episodes"],
        gate_margins=gate_margins,
        metadata={
            "mode": "smoke" if args.smoke else "full",
            "device": str(device),
            "training_seeds": list(seeds),
            "evaluation_episodes": config.evaluation_episodes,
            "hidden_dim": config.hidden_dim,
            "switchglobe_checkpoint_dir": str(args.switchglobe_checkpoint_dir),
            "checkpoint_sha256": checkpoint_sha256_map(checkpoint_paths),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            **git_provenance(),
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
