"""Evaluate the six SwitchGLOBE ablation variants in one standardized schema."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import sys

import torch
import yaml

from .evaluation.ablation_reporting import write_ablation_artifacts
from .experiments.ablation_campaign import AblationConfig, run_ablation_campaign
from .experiments.external_comparison_campaign import _switchglobe_path
from .experiments.latency_optimization_campaign import checkpoint_path as fast_checkpoint_path
from .provenance import checkpoint_sha256_map, config_sha256, git_provenance


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
    parser.add_argument("--config", type=Path, default=package / "config" / "ablation.yaml")
    parser.add_argument(
        "--phase8-checkpoint-dir", type=Path,
        default=Path("ResearchAIWorkspace/artifacts/lite_globe/phase8/checkpoints"),
    )
    parser.add_argument(
        "--phase11-checkpoint-dir", type=Path,
        default=Path("ResearchAIWorkspace/artifacts/lite_globe/phase11/checkpoints"),
    )
    parser.add_argument(
        "--switchglobe-checkpoint-dir", type=Path,
        default=Path("ResearchAIWorkspace/artifacts/lite_globe/phase12/checkpoints"),
    )
    parser.add_argument(
        "--fast-checkpoint-dir", type=Path,
        default=Path("artifacts/switchglobe_latency_optimization/fast_switchglobe/checkpoints"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/lite_globe/ablation"))
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--zip-results", action="store_true")
    return parser.parse_args()


def _config(raw: dict, *, smoke: bool, seeds: list[int] | None) -> AblationConfig:
    campaign = raw["campaign"]
    selected = tuple(seeds) if seeds else tuple(int(seed) for seed in campaign["training_seeds"])
    if smoke:
        selected = tuple(seeds) if seeds else (42,)
    if len(selected) != len(set(selected)):
        raise ValueError("training seeds must be unique")
    return AblationConfig(
        training_seeds=selected,
        evaluation_episodes=3 if smoke else int(campaign["evaluation_episodes"]),
        hidden_dim=int(campaign["hidden_dim"]),
        fast_hidden_dim=int(campaign["fast_hidden_dim"]),
    )


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = _config(raw, smoke=args.smoke, seeds=args.seed)
    device = _device(args.device or raw["runtime"]["device"])
    rows = run_ablation_campaign(
        config,
        phase8_checkpoint_dir=args.phase8_checkpoint_dir,
        phase11_checkpoint_dir=args.phase11_checkpoint_dir,
        switchglobe_checkpoint_dir=args.switchglobe_checkpoint_dir,
        fast_checkpoint_dir=args.fast_checkpoint_dir,
        device=device,
    )
    checkpoint_paths: dict[str, Path] = {}
    for seed in config.training_seeds:
        checkpoint_paths[f"geo_residual_seed_{seed}"] = (
            args.phase8_checkpoint_dir / f"seed_{seed}" / "geo_residual_kd.pt"
        )
        checkpoint_paths[f"predictive_seed_{seed}"] = (
            args.phase11_checkpoint_dir / f"seed_{seed}" / "lite_globe_p.pt"
        )
        try:
            checkpoint_paths[f"switchglobe_exact_seed_{seed}"] = _switchglobe_path(
                args.switchglobe_checkpoint_dir, seed
            )
        except FileNotFoundError:
            pass
        checkpoint_paths[f"fast_switchglobe_seed_{seed}"] = fast_checkpoint_path(
            args.fast_checkpoint_dir, seed
        )
    manifest = write_ablation_artifacts(
        args.output_dir,
        episode_rows=rows["episodes"], summary_rows=rows["seed_summaries"],
        metadata={
            "mode": "smoke" if args.smoke else "full",
            "device": str(device), "config": asdict(config),
            "config_sha256": config_sha256(config),
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "phase8_checkpoint_dir": str(args.phase8_checkpoint_dir),
            "phase11_checkpoint_dir": str(args.phase11_checkpoint_dir),
            "switchglobe_checkpoint_dir": str(args.switchglobe_checkpoint_dir),
            "fast_checkpoint_dir": str(args.fast_checkpoint_dir),
            "checkpoint_sha256": checkpoint_sha256_map(checkpoint_paths),
            **git_provenance(),
        },
    )
    if args.zip_results:
        manifest["result_zip"] = shutil.make_archive(str(args.output_dir), "zip", root_dir=args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
