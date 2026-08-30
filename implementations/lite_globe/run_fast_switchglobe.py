"""Train or resume FastSwitchGLOBE distillation checkpoints per training seed.

This is a thin CLI wrapper around the library functions already implemented
in ``experiments/latency_optimization_campaign.py``
(``train_or_load_fast``/``train_fast_policy``). It does not evaluate routing
performance or benchmark latency; see ``run_ablation.py`` and
``run_latency_benchmark.py`` for those.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys

import torch
import yaml

from .experiments.external_comparison_campaign import (
    _switchglobe_path,
    load_switchglobe,
)
from .experiments.latency_optimization_campaign import (
    LatencyOptimizationConfig,
    checkpoint_path as fast_checkpoint_path,
    train_or_load_fast,
)
from .evaluation.reporting import write_csv
from .provenance import checkpoint_sha256_map, config_sha256, git_provenance
from .scenarios import phase9_evaluation_scenarios


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
        "--config", type=Path, default=package / "config" / "fast_switchglobe.yaml"
    )
    parser.add_argument(
        "--switchglobe-checkpoint-dir",
        type=Path,
        default=Path("ResearchAIWorkspace/artifacts/lite_globe/phase12/checkpoints"),
        help="Read-only Exact SwitchGLOBE teacher checkpoints, one per seed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/switchglobe_latency_optimization/fast_switchglobe"),
    )
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _config(raw: dict, *, smoke: bool, seeds: list[int] | None) -> LatencyOptimizationConfig:
    training = raw["training"]
    selected = (
        tuple(seeds)
        if seeds
        else tuple(int(seed) for seed in training["training_seeds"])
    )
    if smoke:
        selected = tuple(seeds) if seeds else (42,)
    if len(selected) != len(set(selected)):
        raise ValueError("training seeds must be unique")
    return LatencyOptimizationConfig(
        training_seeds=selected,
        dataset_episodes_per_scenario=3 if smoke else int(training["dataset_episodes_per_scenario"]),
        evaluation_episodes=int(training["evaluation_episodes"]),
        epochs=2 if smoke else int(training["epochs"]),
        batch_size=32 if smoke else int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        temperature=float(training["temperature"]),
        action_coefficient=float(training["action_coefficient"]),
        switch_coefficient=float(training["switch_coefficient"]),
        hidden_dim=int(training["hidden_dim"]),
    )


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config = _config(raw, smoke=args.smoke, seeds=args.seed)
    device = _device(args.device or raw["runtime"]["device"])
    checkpoint_dir = args.output_dir / "checkpoints"
    started = platform.uname()
    training_rows: list[dict[str, object]] = []
    for seed in config.training_seeds:
        scenarios = phase9_evaluation_scenarios(seed)
        max_nodes = scenarios[0].config.max_nodes
        teacher = load_switchglobe(
            args.switchglobe_checkpoint_dir, seed=seed, max_nodes=max_nodes,
            hidden_dim=64, device=device,
        )
        _, training = train_or_load_fast(
            teacher, config=config, seed=seed, checkpoint_dir=checkpoint_dir,
            device=device, resume=args.resume,
        )
        training_rows.append({"training_seed": seed, **training})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "training_metrics.csv", training_rows)
    checkpoint_paths: dict[str, Path] = {}
    for seed in config.training_seeds:
        try:
            checkpoint_paths[f"switchglobe_exact_seed_{seed}"] = _switchglobe_path(
                args.switchglobe_checkpoint_dir, seed
            )
        except FileNotFoundError:
            pass
        checkpoint_paths[f"fast_switchglobe_seed_{seed}"] = fast_checkpoint_path(
            checkpoint_dir, seed
        )
    manifest = {
        "schema_version": 1,
        "complete": True,
        "suite": "fast_switchglobe_training",
        "mode": "smoke" if args.smoke else "full",
        "device": str(device),
        "config": asdict(config),
        "config_sha256": config_sha256(config),
        "training_seeds": list(config.training_seeds),
        "training_rows": len(training_rows),
        "switchglobe_checkpoint_dir": str(args.switchglobe_checkpoint_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_sha256": checkpoint_sha256_map(checkpoint_paths),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "os_uname": list(started),
        **git_provenance(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
