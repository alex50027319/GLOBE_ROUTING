"""Run external baselines and compare them with final SwitchGLOBE."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

import torch
import yaml

from .evaluation.baseline_reporting import write_baseline_artifacts
from .experiments import BaselineConfig, run_baseline_campaign


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    package = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=package / "config" / "baselines.yaml",
    )
    parser.add_argument(
        "--switchglobe-checkpoint-dir",
        type=Path,
        default=Path("artifacts/switchglobe/final/checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/baselines"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _config(raw: dict, smoke: bool) -> BaselineConfig:
    campaign = raw["campaign"]
    training = raw["training"]
    if smoke:
        return BaselineConfig(
            training_seeds=(42,),
            evaluation_episodes=3,
            hidden_dim=int(campaign["hidden_dim"]),
            tabular_episodes_per_stage=2,
            drama_episodes_per_stage=2,
            drama_batch_size=8,
            drama_replay_capacity=256,
            drama_learning_rate=float(training["drama_learning_rate"]),
            drama_auxiliary_coefficient=float(
                training["drama_auxiliary_coefficient"]
            ),
        )
    return BaselineConfig(
        training_seeds=tuple(
            int(seed) for seed in campaign["training_seeds"]
        ),
        evaluation_episodes=int(campaign["evaluation_episodes"]),
        hidden_dim=int(campaign["hidden_dim"]),
        tabular_episodes_per_stage=int(
            training["tabular_episodes_per_stage"]
        ),
        drama_episodes_per_stage=int(training["drama_episodes_per_stage"]),
        drama_batch_size=int(training["drama_batch_size"]),
        drama_replay_capacity=int(training["drama_replay_capacity"]),
        drama_learning_rate=float(training["drama_learning_rate"]),
        drama_auxiliary_coefficient=float(
            training["drama_auxiliary_coefficient"]
        ),
    )


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = _config(raw, args.smoke)
    device = _device(args.device or raw["runtime"]["device"])
    rows = run_baseline_campaign(
        config,
        switchglobe_checkpoint_dir=args.switchglobe_checkpoint_dir,
        output_checkpoint_dir=args.output_dir / "checkpoints",
        device=device,
        resume=args.resume,
    )
    manifest = write_baseline_artifacts(
        args.output_dir,
        episode_rows=rows["episodes"],
        summary_rows=rows["seed_summaries"],
        training_rows=rows["training"],
        metadata={
            "suite": "external_baselines",
            "mode": "smoke" if args.smoke else "full",
            "device": str(device),
            "config": asdict(config),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "switchglobe_checkpoint_dir": str(
                args.switchglobe_checkpoint_dir
            ),
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
