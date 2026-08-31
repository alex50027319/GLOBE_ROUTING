"""Run the five-seed SwitchGLOBE leave-one-component-out ablation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import sys

import torch

from .evaluation.novelty_ablation_reporting import (
    write_novelty_ablation_artifacts,
)
from .experiments.novelty_ablation_campaign import (
    NoveltyAblationConfig,
    run_novelty_ablation_campaign,
)
from .experiments.external_comparison_campaign import _switchglobe_path
from .provenance import checkpoint_sha256_map, config_sha256, git_provenance


SEEDS = (42, 77, 123, 314, 2718)


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
        "--phase7-checkpoint-dir",
        type=Path,
        default=Path("ResearchAIWorkspace/artifacts/lite_globe/phase7/checkpoints"),
    )
    parser.add_argument(
        "--phase8-checkpoint-dir",
        type=Path,
        default=Path("ResearchAIWorkspace/artifacts/lite_globe/phase8/checkpoints"),
    )
    parser.add_argument(
        "--switchglobe-checkpoint-dir",
        type=Path,
        default=Path("ResearchAIWorkspace/artifacts/lite_globe/phase12/checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final_paper_simulation/full/novelty_ablation"),
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--zip-results", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = tuple(args.seed) if args.seed else SEEDS
    if len(seeds) != len(set(seeds)):
        raise ValueError("training seeds must be unique")
    episodes = 3 if args.smoke else args.episodes
    config = NoveltyAblationConfig(
        training_seeds=seeds,
        evaluation_episodes=episodes,
    )
    device = _device(args.device)
    rows = run_novelty_ablation_campaign(
        config,
        phase7_checkpoint_dir=args.phase7_checkpoint_dir,
        phase8_checkpoint_dir=args.phase8_checkpoint_dir,
        switchglobe_checkpoint_dir=args.switchglobe_checkpoint_dir,
        device=device,
    )
    checkpoints: dict[str, Path] = {}
    for seed in seeds:
        checkpoints[f"phase7_kd_only_seed_{seed}"] = (
            args.phase7_checkpoint_dir / f"seed_{seed}" / "kd_only_student.pt"
        )
        checkpoints[f"phase8_geo_residual_seed_{seed}"] = (
            args.phase8_checkpoint_dir / f"seed_{seed}" / "geo_residual_kd.pt"
        )
        checkpoints[f"switchglobe_full_seed_{seed}"] = _switchglobe_path(
            args.switchglobe_checkpoint_dir,
            seed,
        )
    metadata = {
        "mode": "smoke" if args.smoke else "full",
        "device": str(device),
        "config": asdict(config),
        "config_sha256": config_sha256(config),
        "checkpoint_sha256": checkpoint_sha256_map(checkpoints),
        "phase7_checkpoint_dir": str(args.phase7_checkpoint_dir),
        "phase8_checkpoint_dir": str(args.phase8_checkpoint_dir),
        "switchglobe_checkpoint_dir": str(args.switchglobe_checkpoint_dir),
        "ablation_definitions": {
            "w/o Risk-Switch": (
                "seed-matched Phase 8 normal branch only"
            ),
            "w/o Geo-Residual": (
                "Phase 7 KD-only shared MLP normal branch plus the exact "
                "predictive branch and exact switch gates"
            ),
            "w/o Distillation": (
                "untrained analytic geographic/forwardability and predictive "
                "priors plus the exact switch gates; no Teacher-trained weights"
            ),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        **git_provenance(),
    }
    manifest = write_novelty_ablation_artifacts(
        args.output_dir,
        episode_rows=rows["episodes"],
        summary_rows=rows["seed_summaries"],
        metadata=metadata,
    )
    if args.zip_results:
        manifest["result_zip"] = shutil.make_archive(
            str(args.output_dir),
            "zip",
            root_dir=args.output_dir,
        )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
