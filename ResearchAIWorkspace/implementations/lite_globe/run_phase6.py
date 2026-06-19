"""Run the Lite-GLOBE multi-seed evaluation and paper artifact pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

import torch
import yaml

from .evaluation import write_phase6_artifacts
from .experiments import Phase6Config, run_phase6_campaign


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
        default=package / "config" / "phase6.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/lite_globe/phase6"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one small seed for local integration verification.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-seed Phase 6 checkpoints.",
    )
    return parser.parse_args()


def _config(raw: dict, smoke: bool) -> Phase6Config:
    campaign = raw["campaign"]
    training = raw["training"]
    cost = raw["cost"]
    if smoke:
        return Phase6Config(
            training_seeds=(42,),
            evaluation_episodes=8,
            hidden_dim=32,
            teacher_updates=2,
            teacher_episodes_per_update=4,
            dataset_episodes=9,
            distillation_epochs=3,
            student_updates=2,
            student_episodes_per_update=4,
            kd_lambda_initial=0.1,
            cost_warmup=2,
            cost_repeats=5,
        )
    return Phase6Config(
        training_seeds=tuple(int(seed) for seed in campaign["training_seeds"]),
        evaluation_episodes=int(campaign["evaluation_episodes"]),
        hidden_dim=int(campaign["hidden_dim"]),
        teacher_updates=int(training["teacher_updates"]),
        teacher_episodes_per_update=int(
            training["teacher_episodes_per_update"]
        ),
        dataset_episodes=int(training["dataset_episodes"]),
        distillation_epochs=int(training["distillation_epochs"]),
        student_updates=int(training["student_updates"]),
        student_episodes_per_update=int(
            training["student_episodes_per_update"]
        ),
        kd_lambda_initial=float(training["kd_lambda_initial"]),
        cost_warmup=int(cost["warmup"]),
        cost_repeats=int(cost["repeats"]),
    )


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = _config(raw, args.smoke)
    device = _device(args.device or raw["runtime"]["device"])
    rows = run_phase6_campaign(
        config,
        device=device,
        checkpoint_dir=args.output_dir / "checkpoints",
        resume=args.resume,
    )
    manifest = write_phase6_artifacts(
        args.output_dir,
        episode_rows=rows["episodes"],
        seed_summary_rows=rows["seed_summaries"],
        training_rows=rows["training"],
        cost_rows=rows["costs"],
        metadata={
            "phase": 6,
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
