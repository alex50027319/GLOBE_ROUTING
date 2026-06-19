"""Run Phase 10 external RL baselines and proposed-method comparison."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

import torch
import yaml

from .evaluation import write_phase10_artifacts
from .experiments import Phase10Config, run_phase10_campaign


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
        default=package / "config" / "phase10.yaml",
    )
    parser.add_argument(
        "--phase8-checkpoint-dir",
        type=Path,
        default=Path("artifacts/lite_globe/phase8/checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/lite_globe/phase10"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _config(raw: dict, smoke: bool) -> Phase10Config:
    campaign = raw["campaign"]
    training = raw["training"]
    if smoke:
        return Phase10Config(
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
    return Phase10Config(
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
    rows = run_phase10_campaign(
        config,
        phase8_checkpoint_dir=args.phase8_checkpoint_dir,
        output_checkpoint_dir=args.output_dir / "checkpoints",
        device=device,
        resume=args.resume,
    )
    manifest = write_phase10_artifacts(
        args.output_dir,
        episode_rows=rows["episodes"],
        summary_rows=rows["seed_summaries"],
        training_rows=rows["training"],
        metadata={
            "phase": 10,
            "mode": "smoke" if args.smoke else "full",
            "device": str(device),
            "config": asdict(config),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "phase8_checkpoint_dir": str(args.phase8_checkpoint_dir),
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
