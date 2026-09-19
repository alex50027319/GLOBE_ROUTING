"""Run resumable external routing comparison chunks."""

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

from .evaluation.external_comparison_reporting import write_external_comparison_artifacts
from .experiments.external_comparison_campaign import ExternalComparisonConfig, run_external_comparison


def parse_args() -> argparse.Namespace:
    package = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=package / "config" / "external_comparison.yaml")
    parser.add_argument("--switchglobe-checkpoint-dir", type=Path,
                        default=Path("ResearchAIWorkspace/artifacts/lite_globe/phase12/checkpoints"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/external_comparison"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", action="append", type=int, help="Run one or more independent training-seed chunks")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--zip-results", action="store_true")
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def config_from_yaml(raw: dict, *, smoke: bool, seeds: list[int] | None) -> ExternalComparisonConfig:
    campaign, training = raw["campaign"], raw["training"]
    selected = tuple(seeds) if seeds else tuple(int(value) for value in campaign["training_seeds"])
    if smoke:
        selected = tuple(seeds) if seeds else (42,)
    if len(selected) != len(set(selected)):
        raise ValueError("training seeds must be unique")
    return ExternalComparisonConfig(
        training_seeds=selected,
        evaluation_episodes=3 if smoke else int(campaign["evaluation_episodes"]),
        hidden_dim=int(campaign["hidden_dim"]),
        tabular_episodes_per_stage=2 if smoke else int(training["tabular_episodes_per_stage"]),
        neural_episodes_per_stage=2 if smoke else int(training["neural_episodes_per_stage"]),
        neural_batch_size=8 if smoke else int(training["neural_batch_size"]),
    )


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = config_from_yaml(raw, smoke=args.smoke, seeds=args.seed)
    mode = "smoke" if args.smoke else "full"
    seed_slug = "seeds_" + "_".join(str(seed) for seed in config.training_seeds)
    output_dir = args.output_dir / mode / seed_slug
    manifest_path = output_dir / "manifest.json"
    if args.resume and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        normalized_config = json.loads(json.dumps(asdict(config)))
        if existing.get("complete") and existing.get("metadata", {}).get("config") == normalized_config:
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return 0
    device = _device(args.device)
    rows = run_external_comparison(
        config,
        switchglobe_checkpoint_dir=args.switchglobe_checkpoint_dir,
        checkpoint_dir=output_dir / "checkpoints",
        device=device,
        resume=args.resume,
    )
    manifest = write_external_comparison_artifacts(
        output_dir,
        episode_rows=rows["episodes"], summary_rows=rows["seed_summaries"],
        training_rows=rows["training"], deployment_rows=rows["deployment_costs"],
        method_contracts=rows["method_contracts"],
        metadata={
            "mode": mode, "device": str(device), "config": asdict(config),
            "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "switchglobe_checkpoint_dir": str(args.switchglobe_checkpoint_dir),
        },
    )
    if args.zip_results:
        archive = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir)
        manifest["result_zip"] = archive
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
