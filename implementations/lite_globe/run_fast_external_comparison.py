"""Train/resume and full-evaluate FastSwitchGLOBE in seed-sized Colab chunks."""

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

from .evaluation.fast_external_comparison_reporting import write_fast_external_chunk
from .experiments.external_comparison_campaign import _switchglobe_path
from .experiments.fast_external_comparison_campaign import (
    FastExternalComparisonConfig,
    run_fast_external_comparison,
)
from .experiments.latency_optimization_campaign import checkpoint_path
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
    parser.add_argument(
        "--config",
        type=Path,
        default=package / "config" / "fast_external_comparison.yaml",
    )
    parser.add_argument(
        "--switchglobe-checkpoint-dir",
        type=Path,
        default=Path("ResearchAIWorkspace/artifacts/lite_globe/phase12/checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/fast_external_comparison"),
    )
    parser.add_argument(
        "--fast-checkpoint-dir",
        type=Path,
        help="Defaults to <output-dir>/checkpoints.",
    )
    parser.add_argument(
        "--pretrained-fast-checkpoint-dir",
        type=Path,
        help="Optional verified full FastSwitchGLOBE checkpoints to evaluate without retraining.",
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--zip-results", action="store_true")
    return parser.parse_args()


def config_from_yaml(
    raw: dict, *, smoke: bool, seeds: list[int] | None
) -> FastExternalComparisonConfig:
    campaign, training = raw["campaign"], raw["training"]
    selected = (
        tuple(seeds)
        if seeds
        else tuple(int(seed) for seed in campaign["training_seeds"])
    )
    if smoke:
        selected = tuple(seeds) if seeds else (42,)
    if len(selected) != len(set(selected)):
        raise ValueError("training seeds must be unique")
    return FastExternalComparisonConfig(
        training_seeds=selected,
        evaluation_episodes=3 if smoke else int(campaign["evaluation_episodes"]),
        exact_hidden_dim=int(campaign["exact_hidden_dim"]),
        fast_hidden_dim=int(campaign["fast_hidden_dim"]),
        dataset_episodes_per_scenario=(
            3 if smoke else int(training["dataset_episodes_per_scenario"])
        ),
        epochs=2 if smoke else int(training["epochs"]),
        batch_size=32 if smoke else int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        temperature=float(training["temperature"]),
        action_coefficient=float(training["action_coefficient"]),
        switch_coefficient=float(training["switch_coefficient"]),
    )


def _archive_path(output_root: Path, *, mode: str, seed_slug: str) -> Path:
    return output_root / mode / f"fast_{seed_slug}.zip"


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = config_from_yaml(raw, smoke=args.smoke, seeds=args.seed)
    mode = "smoke" if args.smoke else "full"
    seed_slug = "seeds_" + "_".join(str(seed) for seed in config.training_seeds)
    output_dir = args.output_dir / mode / seed_slug
    # Smoke and full checkpoints must never share a resume directory because the
    # training config hash intentionally differs between the two modes.
    fast_checkpoint_dir = args.fast_checkpoint_dir or args.output_dir / mode / "checkpoints"
    archive_path = _archive_path(args.output_dir, mode=mode, seed_slug=seed_slug)
    normalized_config = json.loads(json.dumps(asdict(config)))

    manifest_path = output_dir / "manifest.json"
    if args.resume and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("complete")
            and existing.get("mode") == mode
            and existing.get("metadata", {}).get("config") == normalized_config
            and existing.get("episode_rows") == existing.get("expected_episode_rows")
        ):
            if args.zip_results and not archive_path.is_file():
                shutil.make_archive(
                    str(archive_path.with_suffix("")), "zip", root_dir=output_dir
                )
            print(json.dumps(existing, ensure_ascii=False, indent=2))
            return 0

    device = _device(args.device or raw["runtime"]["device"])
    rows = run_fast_external_comparison(
        config,
        switchglobe_checkpoint_dir=args.switchglobe_checkpoint_dir,
        fast_checkpoint_dir=fast_checkpoint_dir,
        pretrained_fast_checkpoint_dir=args.pretrained_fast_checkpoint_dir,
        device=device,
        resume=args.resume,
    )

    checkpoint_paths: dict[str, Path] = {}
    effective_fast_paths = {
        int(row["training_seed"]): Path(str(row["path"]))
        for row in rows["fast_checkpoint_paths"]
    }
    for seed in config.training_seeds:
        checkpoint_paths[f"switchglobe_exact_seed_{seed}"] = _switchglobe_path(
            args.switchglobe_checkpoint_dir, seed
        )
        fast_path = effective_fast_paths[seed]
        checkpoint_paths[f"fast_switchglobe_seed_{seed}"] = fast_path
        bundled_checkpoint = output_dir / "checkpoints" / f"seed_{seed}" / fast_path.name
        bundled_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fast_path, bundled_checkpoint)

    manifest = write_fast_external_chunk(
        output_dir,
        episode_rows=rows["episodes"],
        summary_rows=rows["seed_summaries"],
        training_rows=rows["training"],
        deployment_rows=rows["deployment_costs"],
        metadata={
            "mode": mode,
            "device": str(device),
            "config": asdict(config),
            "config_sha256": config_sha256(config),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "switchglobe_checkpoint_dir": str(args.switchglobe_checkpoint_dir),
            "fast_checkpoint_dir": str(fast_checkpoint_dir),
            "checkpoint_sha256": checkpoint_sha256_map(checkpoint_paths),
            **git_provenance(),
        },
    )
    if args.zip_results:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(
            str(archive_path.with_suffix("")), "zip", root_dir=output_dir
        )
        manifest["result_zip"] = str(archive_path)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
