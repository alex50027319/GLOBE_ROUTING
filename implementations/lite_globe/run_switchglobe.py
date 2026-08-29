"""Calibrate and evaluate the final SwitchGLOBE routing policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

import torch
import yaml

from .evaluation import write_switchglobe_artifacts
from .experiments import SwitchGlobeConfig, run_switchglobe_campaign
from .run_phase12 import _config, _device


def parse_args() -> argparse.Namespace:
    package = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=package / "config" / "switchglobe.yaml",
    )
    parser.add_argument(
        "--phase8-checkpoint-dir",
        type=Path,
        default=Path("artifacts/switchglobe/training/geo_residual/checkpoints"),
    )
    parser.add_argument(
        "--phase11-checkpoint-dir",
        type=Path,
        default=Path("artifacts/switchglobe/training/predictive/checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/switchglobe/final"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config: SwitchGlobeConfig = _config(raw, args.smoke)
    device = _device(args.device or raw["runtime"]["device"])
    rows = run_switchglobe_campaign(
        config,
        phase8_checkpoint_dir=args.phase8_checkpoint_dir,
        phase11_checkpoint_dir=args.phase11_checkpoint_dir,
        output_checkpoint_dir=args.output_dir / "checkpoints",
        device=device,
        resume=args.resume,
    )
    manifest = write_switchglobe_artifacts(
        args.output_dir,
        episode_rows=rows["episodes"],
        summary_rows=rows["seed_summaries"],
        training_rows=rows["training"],
        metadata={
            "algorithm": "SwitchGLOBE",
            "historical_phase": 12,
            "mode": "smoke" if args.smoke else "full",
            "device": str(device),
            "config": asdict(config),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "phase8_checkpoint_dir": str(args.phase8_checkpoint_dir),
            "phase11_checkpoint_dir": str(args.phase11_checkpoint_dir),
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
