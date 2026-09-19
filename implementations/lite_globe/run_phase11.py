"""Run Phase 11 Lite-GLOBE-P predictive residual optimization."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

import torch
import yaml

from .evaluation import write_phase11_artifacts
from .experiments import Phase11Config, run_phase11_campaign


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
        default=package / "config" / "phase11.yaml",
    )
    parser.add_argument(
        "--phase7-checkpoint-dir",
        type=Path,
        default=Path("artifacts/lite_globe/phase7/checkpoints"),
    )
    parser.add_argument(
        "--phase8-checkpoint-dir",
        type=Path,
        default=Path("artifacts/lite_globe/phase8/checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/lite_globe/phase11"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _config(raw: dict, smoke: bool) -> Phase11Config:
    campaign = raw["campaign"]
    training = raw["training"]
    if smoke:
        return Phase11Config(
            training_seeds=(42,),
            evaluation_episodes=5,
            hidden_dim=int(campaign["hidden_dim"]),
            dataset_episodes_per_stage=9,
            distillation_epochs=5,
            oracle_coefficient=float(training["oracle_coefficient"]),
            risk_oracle_coefficient=float(
                training["risk_oracle_coefficient"]
            ),
            teacher_action_coefficient=float(
                training["teacher_action_coefficient"]
            ),
            early_stopping_patience=0,
            initial_prior_strength=float(
                training["initial_prior_strength"]
            ),
            initial_predictive_strength=tuple(
                float(value)
                for value in training["initial_predictive_strength"]
            ),
            initial_break_penalty=float(
                training["initial_break_penalty"]
            ),
            initial_residual_bound=float(
                training["initial_residual_bound"]
            ),
            calibration_episodes_per_stage=5,
            structural_hole_episodes_per_variant=9,
            calibration_pdr_tolerance=float(
                training["calibration_pdr_tolerance"]
            ),
            predictive_replay_multiplier=max(
                1, int(training["predictive_replay_multiplier"])
            ),
            predictive_pretraining_epochs=5,
            include_link_loss_training=bool(
                training.get("include_link_loss_training", False)
            ),
        )
    return Phase11Config(
        training_seeds=tuple(
            int(seed) for seed in campaign["training_seeds"]
        ),
        evaluation_episodes=int(campaign["evaluation_episodes"]),
        hidden_dim=int(campaign["hidden_dim"]),
        dataset_episodes_per_stage=int(
            training["dataset_episodes_per_stage"]
        ),
        distillation_epochs=int(training["distillation_epochs"]),
        oracle_coefficient=float(training["oracle_coefficient"]),
        risk_oracle_coefficient=float(
            training["risk_oracle_coefficient"]
        ),
        teacher_action_coefficient=float(
            training["teacher_action_coefficient"]
        ),
        early_stopping_patience=int(
            training["early_stopping_patience"]
        ),
        initial_prior_strength=float(
            training["initial_prior_strength"]
        ),
        initial_predictive_strength=tuple(
            float(value)
            for value in training["initial_predictive_strength"]
        ),
        initial_break_penalty=float(training["initial_break_penalty"]),
        initial_residual_bound=float(training["initial_residual_bound"]),
        calibration_episodes_per_stage=int(
            training["calibration_episodes_per_stage"]
        ),
        structural_hole_episodes_per_variant=int(
            training["structural_hole_episodes_per_variant"]
        ),
        calibration_pdr_tolerance=float(
            training["calibration_pdr_tolerance"]
        ),
        predictive_replay_multiplier=int(
            training["predictive_replay_multiplier"]
        ),
        predictive_pretraining_epochs=int(
            training["predictive_pretraining_epochs"]
        ),
        include_link_loss_training=bool(
            training.get("include_link_loss_training", False)
        ),
    )


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = _config(raw, args.smoke)
    device = _device(args.device or raw["runtime"]["device"])
    rows = run_phase11_campaign(
        config,
        phase7_checkpoint_dir=args.phase7_checkpoint_dir,
        phase8_checkpoint_dir=args.phase8_checkpoint_dir,
        output_checkpoint_dir=args.output_dir / "checkpoints",
        device=device,
        resume=args.resume,
    )
    manifest = write_phase11_artifacts(
        args.output_dir,
        episode_rows=rows["episodes"],
        summary_rows=rows["seed_summaries"],
        training_rows=rows["training"],
        metadata={
            "phase": 11,
            "mode": "smoke" if args.smoke else "full",
            "device": str(device),
            "config": asdict(config),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "phase7_checkpoint_dir": str(args.phase7_checkpoint_dir),
            "phase8_checkpoint_dir": str(args.phase8_checkpoint_dir),
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
