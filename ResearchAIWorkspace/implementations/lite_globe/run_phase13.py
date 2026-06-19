"""Run Phase 13 Risk-Switch Lite-GLOBE-P+ calibration and evaluation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

import torch
import yaml

from .evaluation import write_phase13_artifacts
from .experiments import Phase13Config, run_phase13_campaign


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
        default=package / "config" / "phase13.yaml",
    )
    parser.add_argument(
        "--phase8-checkpoint-dir",
        type=Path,
        default=Path("artifacts/lite_globe/phase8/checkpoints"),
    )
    parser.add_argument(
        "--phase11-checkpoint-dir",
        type=Path,
        default=Path("artifacts/lite_globe/phase11/checkpoints"),
    )
    parser.add_argument(
        "--phase12-checkpoint-dir",
        type=Path,
        default=Path("artifacts/lite_globe/phase12/checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/lite_globe/phase13"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _config(raw: dict, smoke: bool) -> Phase13Config:
    campaign = raw["campaign"]
    calibration = raw["calibration"]
    if smoke:
        return Phase13Config(
            training_seeds=(42,),
            evaluation_episodes=5,
            hidden_dim=int(campaign["hidden_dim"]),
            calibration_episodes_per_stage=5,
            calibration_pdr_tolerance=float(
                calibration["calibration_pdr_tolerance"]
            ),
            switch_thresholds=tuple(
                float(value)
                for value in calibration["switch_thresholds"][:2]
            ),
            margin_gates=tuple(
                float(value) for value in calibration["margin_gates"][:1]
            ),
            lifetime_gates=tuple(
                float(value)
                for value in calibration["lifetime_gates"][:2]
            ),
            onward_gates=tuple(
                float(value) for value in calibration["onward_gates"][:2]
            ),
            topk_onward_gates=tuple(
                float(value)
                for value in calibration["topk_onward_gates"][:1]
            ),
            redundancy_gates=tuple(
                float(value)
                for value in calibration["redundancy_gates"][:1]
            ),
            loss_keep_gates=tuple(
                float(value)
                for value in calibration["loss_keep_gates"][:2]
            ),
            predictive_margins=tuple(
                float(value)
                for value in calibration["predictive_margins"][:1]
            ),
            energy_tie_weights=tuple(
                float(value)
                for value in calibration["energy_tie_weights"][:1]
            ),
            drop_suppression_bonuses=tuple(
                float(value)
                for value in calibration["drop_suppression_bonuses"][:1]
            ),
        )
    return Phase13Config(
        training_seeds=tuple(
            int(seed) for seed in campaign["training_seeds"]
        ),
        evaluation_episodes=int(campaign["evaluation_episodes"]),
        hidden_dim=int(campaign["hidden_dim"]),
        calibration_episodes_per_stage=int(
            calibration["calibration_episodes_per_stage"]
        ),
        calibration_pdr_tolerance=float(
            calibration["calibration_pdr_tolerance"]
        ),
        switch_thresholds=tuple(
            float(value) for value in calibration["switch_thresholds"]
        ),
        margin_gates=tuple(
            float(value) for value in calibration["margin_gates"]
        ),
        lifetime_gates=tuple(
            float(value) for value in calibration["lifetime_gates"]
        ),
        onward_gates=tuple(
            float(value) for value in calibration["onward_gates"]
        ),
        topk_onward_gates=tuple(
            float(value) for value in calibration["topk_onward_gates"]
        ),
        redundancy_gates=tuple(
            float(value) for value in calibration["redundancy_gates"]
        ),
        loss_keep_gates=tuple(
            float(value) for value in calibration["loss_keep_gates"]
        ),
        predictive_margins=tuple(
            float(value) for value in calibration["predictive_margins"]
        ),
        energy_tie_weights=tuple(
            float(value) for value in calibration["energy_tie_weights"]
        ),
        drop_suppression_bonuses=tuple(
            float(value)
            for value in calibration["drop_suppression_bonuses"]
        ),
    )


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = _config(raw, args.smoke)
    device = _device(args.device or raw["runtime"]["device"])
    rows = run_phase13_campaign(
        config,
        phase8_checkpoint_dir=args.phase8_checkpoint_dir,
        phase11_checkpoint_dir=args.phase11_checkpoint_dir,
        phase12_checkpoint_dir=args.phase12_checkpoint_dir,
        output_checkpoint_dir=args.output_dir / "checkpoints",
        device=device,
        resume=args.resume,
    )
    manifest = write_phase13_artifacts(
        args.output_dir,
        episode_rows=rows["episodes"],
        summary_rows=rows["seed_summaries"],
        training_rows=rows["training"],
        metadata={
            "phase": 13,
            "mode": "smoke" if args.smoke else "full",
            "device": str(device),
            "config": asdict(config),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "phase8_checkpoint_dir": str(args.phase8_checkpoint_dir),
            "phase11_checkpoint_dir": str(args.phase11_checkpoint_dir),
            "phase12_checkpoint_dir": str(args.phase12_checkpoint_dir),
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
