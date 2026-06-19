"""Run topology-held-out Phase 7 generalization validation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

import torch
import yaml

from .evaluation import write_phase7_artifacts
from .experiments import Phase7Config, run_phase7_campaign


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
        default=package / "config" / "phase7.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/lite_globe/phase7"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _config(raw: dict, smoke: bool) -> Phase7Config:
    campaign = raw["campaign"]
    training = raw["training"]
    if smoke:
        return Phase7Config(
            training_seeds=(42,),
            evaluation_episodes=6,
            hidden_dim=32,
            teacher_updates_per_stage=1,
            teacher_episodes_per_update=3,
            dataset_episodes_per_stage=4,
            distillation_epochs=3,
            student_updates_per_stage=1,
            student_episodes_per_update=3,
            kd_lambda_initial=0.1,
        )
    return Phase7Config(
        training_seeds=tuple(int(seed) for seed in campaign["training_seeds"]),
        evaluation_episodes=int(campaign["evaluation_episodes"]),
        hidden_dim=int(campaign["hidden_dim"]),
        teacher_updates_per_stage=int(
            training["teacher_updates_per_stage"]
        ),
        teacher_episodes_per_update=int(
            training["teacher_episodes_per_update"]
        ),
        dataset_episodes_per_stage=int(
            training["dataset_episodes_per_stage"]
        ),
        distillation_epochs=int(training["distillation_epochs"]),
        student_updates_per_stage=int(
            training["student_updates_per_stage"]
        ),
        student_episodes_per_update=int(
            training["student_episodes_per_update"]
        ),
        kd_lambda_initial=float(training["kd_lambda_initial"]),
    )


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = _config(raw, args.smoke)
    device = _device(args.device or raw["runtime"]["device"])
    rows = run_phase7_campaign(
        config,
        device=device,
        checkpoint_dir=args.output_dir / "checkpoints",
        resume=args.resume,
    )
    manifest = write_phase7_artifacts(
        args.output_dir,
        episode_rows=rows["episodes"],
        summary_rows=rows["seed_summaries"],
        training_rows=rows["training"],
        metadata={
            "phase": 7,
            "mode": "smoke" if args.smoke else "full",
            "device": str(device),
            "config": asdict(config),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
