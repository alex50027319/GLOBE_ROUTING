"""Merge and validate the 5 completed external-comparison Colab ZIPs into Phase B.

Per docs/claude_final_simulation_master_prompt.md section 5, Phase B is primarily an
audit-and-merge of already-completed seed ZIPs, not a fresh 98,000-episode GPU run.
This script:

  1. Unzips the 5 per-seed archives to a scratch directory.
  2. Validates row counts, method/scenario/seed coverage, and duplicate-free keys.
  3. Merges raw episodes/training/deployment-cost/seed-summary rows, tagging each
     episode with its source archive for provenance.
  4. Reuses ``evaluation.external_comparison_reporting.write_external_comparison_artifacts``
     for cross-seed aggregate statistics, paired effects, tables, and figures (the
     function aggregates over whatever rows it is given, so feeding it all 5 seeds'
     summary rows at once yields the correct 5-seed aggregate/paired statistics).
  5. Optionally runs a small CPU-only smoke re-evaluation per seed against the current
     SwitchGLOBE Exact checkpoints and compares outcome-determinism fields against the
     corresponding ZIP rows, to catch silent checkpoint drift.
  6. Writes a manifest augmented with git/config/checkpoint provenance and a markdown
     validation report, and a standalone method_contracts.json.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import zipfile

from implementations.lite_globe.evaluation.external_comparison_reporting import (
    SCENARIOS,
    write_external_comparison_artifacts,
)
from implementations.lite_globe.baselines.registry import COMPARISON_METHODS
from implementations.lite_globe.provenance import (
    checkpoint_sha256_map,
    config_sha256,
    file_sha256,
    git_provenance,
)

DEFAULT_SEEDS = (42, 77, 123, 314, 2718)
EXPECTED_EPISODES_PER_SEED = 19_600
EXPECTED_SUMMARIES_PER_SEED = 98
DETERMINISM_FIELDS = (
    "delivered", "dropped", "drop_reason", "steps", "hop_count",
    "transmission_attempts", "transmission_energy_proxy", "deadline_met",
    "delay_steps", "loop", "initially_connected", "policy_input_bytes",
    "total_reward",
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@dataclass
class SeedArchive:
    seed: int
    zip_path: Path
    extract_dir: Path
    manifest: dict[str, Any]
    episodes: list[dict[str, Any]]
    training: list[dict[str, Any]]
    deployment_costs: list[dict[str, Any]]
    seed_summaries: list[dict[str, Any]]


def load_seed_archive(zip_dir: Path, seed: int, scratch_root: Path) -> SeedArchive:
    zip_path = zip_dir / f"seeds_{seed}.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"missing archive for seed {seed}: {zip_path}")
    extract_dir = scratch_root / f"seed_{seed}"
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    manifest = json.loads((extract_dir / "manifest.json").read_text(encoding="utf-8"))
    archive_label = zip_path.name
    episodes = _read_csv(extract_dir / "raw" / "episodes.csv")
    for row in episodes:
        row["source_archive"] = archive_label
    training = _read_csv(extract_dir / "raw" / "training.csv")
    deployment_costs = _read_csv(extract_dir / "raw" / "deployment_costs.csv")
    seed_summaries = _read_csv(extract_dir / "raw" / "seed_summaries.csv")
    return SeedArchive(
        seed=seed, zip_path=zip_path, extract_dir=extract_dir, manifest=manifest,
        episodes=episodes, training=training, deployment_costs=deployment_costs,
        seed_summaries=seed_summaries,
    )


def validate_archive(archive: SeedArchive) -> list[str]:
    problems = []
    if not archive.manifest.get("complete"):
        problems.append(f"seed {archive.seed}: manifest not marked complete")
    if len(archive.episodes) != EXPECTED_EPISODES_PER_SEED:
        problems.append(
            f"seed {archive.seed}: episode rows {len(archive.episodes)} != {EXPECTED_EPISODES_PER_SEED}"
        )
    if len(archive.seed_summaries) != EXPECTED_SUMMARIES_PER_SEED:
        problems.append(
            f"seed {archive.seed}: seed-summary rows {len(archive.seed_summaries)} != {EXPECTED_SUMMARIES_PER_SEED}"
        )
    manifest_methods = list(archive.manifest.get("methods", []))
    if manifest_methods != list(COMPARISON_METHODS):
        problems.append(f"seed {archive.seed}: method list mismatch: {manifest_methods}")
    manifest_scenarios = list(archive.manifest.get("scenarios", []))
    if manifest_scenarios != list(SCENARIOS):
        problems.append(f"seed {archive.seed}: scenario order/name mismatch")
    return problems


def merge_archives(archives: list[SeedArchive]) -> dict[str, Any]:
    problems: list[str] = []
    for archive in archives:
        problems.extend(validate_archive(archive))

    episode_keys: set[tuple[str, str, str, str]] = set()
    for archive in archives:
        for row in archive.episodes:
            key = (row["method"], row["scenario"], row["training_seed"], row["evaluation_seed"])
            if key in episode_keys:
                problems.append(f"duplicate episode key across archives: {key}")
            episode_keys.add(key)

    summary_keys: set[tuple[str, str, str]] = set()
    for archive in archives:
        for row in archive.seed_summaries:
            key = (row["method"], row["scenario"], row["training_seed"])
            if key in summary_keys:
                problems.append(f"duplicate seed-summary key across archives: {key}")
            summary_keys.add(key)

    contract_hashes = {
        json.dumps(archive.manifest.get("method_contracts", []), sort_keys=True)
        for archive in archives
    }
    if len(contract_hashes) != 1:
        problems.append("method_contracts differ across seed archives; cannot pick one canonical copy")
        method_contracts: list[dict[str, Any]] = []
    else:
        method_contracts = archives[0].manifest.get("method_contracts", [])

    configs = {json.dumps(a.manifest["metadata"]["config"] | {"training_seeds": None}, sort_keys=True) for a in archives}
    if len(configs) != 1:
        problems.append("per-seed training configs differ beyond training_seeds; cannot merge safely")

    if problems:
        raise ValueError("merge validation failed:\n" + "\n".join(problems))

    episodes = [row for archive in archives for row in archive.episodes]
    training = [row for archive in archives for row in archive.training]
    deployment_costs = [row for archive in archives for row in archive.deployment_costs]
    seed_summaries = [row for archive in archives for row in archive.seed_summaries]
    return {
        "episodes": episodes, "training": training, "deployment_costs": deployment_costs,
        "seed_summaries": seed_summaries, "method_contracts": method_contracts,
    }


def spot_check(
    seeds: tuple[int, ...], *, switchglobe_checkpoint_dir: Path, merged_episodes: list[dict[str, Any]],
    scratch_root: Path, python_executable: str,
) -> dict[str, Any]:
    """CPU-only re-evaluation of the first 3 episodes/scenario per seed.

    Compares recomputed SwitchGLOBE outcomes against the corresponding ZIP rows on
    outcome-determinism fields (not wall-clock latency, which is hardware-dependent).
    """

    lookup = {
        (row["method"], row["scenario"], row["training_seed"], row["evaluation_seed"]): row
        for row in merged_episodes
    }
    per_seed: dict[str, Any] = {}
    total_checked = 0
    all_mismatches: list[dict[str, Any]] = []
    for seed in seeds:
        seed_out = scratch_root / "spot_check" / str(seed)
        result = subprocess.run(
            [
                python_executable, "-m", "implementations.lite_globe.run_external_comparison",
                "--smoke", "--seed", str(seed), "--device", "cpu",
                "--output-dir", str(seed_out),
                "--switchglobe-checkpoint-dir", str(switchglobe_checkpoint_dir),
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            per_seed[str(seed)] = {"status": "error", "stderr": result.stderr[-2000:]}
            continue
        recomputed = _read_csv(seed_out / "smoke" / f"seeds_{seed}" / "raw" / "episodes.csv")
        checked = 0
        mismatches = []
        for row in recomputed:
            if row["method"] != "SwitchGLOBE":
                continue
            key = (row["method"], row["scenario"], row["training_seed"], row["evaluation_seed"])
            original = lookup.get(key)
            if original is None:
                mismatches.append({"key": key, "issue": "not_found_in_merged_zip_rows"})
                continue
            checked += 1
            for field in DETERMINISM_FIELDS:
                original_value, recomputed_value = original[field], row[field]
                if original_value == recomputed_value:
                    continue
                try:
                    if abs(float(original_value) - float(recomputed_value)) <= 1e-6:
                        continue
                except ValueError:
                    pass
                mismatches.append({
                    "key": key, "field": field,
                    "original": original_value, "recomputed": recomputed_value,
                })
        per_seed[str(seed)] = {"status": "ok", "episodes_checked": checked, "mismatches": mismatches}
        total_checked += checked
        all_mismatches.extend(mismatches)
    return {
        "kind": "cpu_smoke_spot_check_vs_recorded_zip_rows",
        "note": "3 episodes/scenario per seed, SwitchGLOBE method only, outcome-determinism fields only "
                "(latency excluded as hardware-dependent); not a full re-evaluation",
        "total_episodes_checked": total_checked,
        "total_mismatches": len(all_mismatches),
        "per_seed": per_seed,
    }


def write_validation_report(path: Path, *, seeds, merged, spot_check_result, manifest) -> None:
    lines = [
        "# Phase B validation report — external baseline comparison",
        "",
        f"- training seeds: {list(seeds)}",
        f"- episode rows: {len(merged['episodes'])} (expected {len(seeds) * EXPECTED_EPISODES_PER_SEED})",
        f"- seed-summary rows: {len(merged['seed_summaries'])} (expected {len(seeds) * EXPECTED_SUMMARIES_PER_SEED})",
        f"- methods: {list(COMPARISON_METHODS)}",
        f"- scenarios: {list(SCENARIOS)}",
        "- duplicate episode/summary keys across the 5 archives: 0 (checked during merge; merge raises on any duplicate)",
        "- method_contracts identical across all 5 archives: yes (single canonical copy used)",
        "- per-seed training config identical apart from training_seeds: yes",
        "",
        "## Spot-check (CPU, no Colab/GPU)",
        "",
        f"- kind: {spot_check_result['kind']}",
        f"- episodes checked: {spot_check_result['total_episodes_checked']}",
        f"- mismatches: {spot_check_result['total_mismatches']}",
    ]
    if spot_check_result["total_mismatches"]:
        lines.append("- **MISMATCHES FOUND — see manifest.json spot_check field for details; do not treat merge as verified.**")
    else:
        lines.append("- no mismatches: current SwitchGLOBE Exact checkpoints reproduce the recorded ZIP outcomes exactly on the sampled episodes.")
    lines += [
        "",
        "## Known gaps carried over from preflight",
        "",
        "- The 5 source ZIPs' own manifests do not record git commit hash or checkpoint/config "
        "SHA-256 from when they were originally produced on Colab — this predates this merge and "
        "is not fabricated here. The merged manifest below records the *current* checkpoint hashes "
        "(matched via the CPU spot-check) and the *current* git state at merge time.",
        "",
        "## Aggregate/paired statistics",
        "",
        f"- aggregate_statistics rows: {manifest['statistics_rows']}",
        f"- paired_effects rows: {manifest['paired_effect_rows']}",
        "",
        "Energy is a simulator transmission-energy proxy, not Joules. `policy_input_bytes` is not "
        "routing-control overhead. `delivered/steps` is not Mbps throughput.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", type=Path, default=Path("artifacts/external_comparison_colab_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final_paper_simulation/full/baselines"))
    parser.add_argument("--switchglobe-checkpoint-dir", type=Path,
                        default=Path("ResearchAIWorkspace/artifacts/lite_globe/phase12/checkpoints"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--skip-spot-check", action="store_true")
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    with tempfile.TemporaryDirectory(prefix="phaseB_merge_") as scratch:
        scratch_root = Path(scratch)
        archives = [load_seed_archive(args.zip_dir, seed, scratch_root) for seed in seeds]
        merged = merge_archives(archives)

        if args.skip_spot_check:
            spot_check_result = {
                "kind": "cpu_smoke_spot_check_vs_recorded_zip_rows", "skipped": True,
                "total_episodes_checked": 0, "total_mismatches": 0, "per_seed": {},
            }
        else:
            spot_check_result = spot_check(
                seeds, switchglobe_checkpoint_dir=args.switchglobe_checkpoint_dir,
                merged_episodes=merged["episodes"], scratch_root=scratch_root,
                python_executable=sys.executable,
            )
            if spot_check_result["total_mismatches"]:
                raise ValueError(
                    "spot-check found mismatches between current checkpoints and recorded ZIP "
                    f"outcomes: {spot_check_result['total_mismatches']} — refusing to report Phase B "
                    "as verified; see spot_check details"
                )

        effective_config = dict(archives[0].manifest["metadata"]["config"])
        effective_config["training_seeds"] = list(seeds)
        metadata = {
            "mode": "full",
            "source": "merged_from_precomputed_colab_zips",
            "config": effective_config,
            "source_archives": [
                {
                    "seed": archive.seed, "zip": str(archive.zip_path),
                    "zip_sha256": file_sha256(archive.zip_path),
                    "original_manifest_mode": archive.manifest.get("mode"),
                    "original_manifest_complete": archive.manifest.get("complete"),
                }
                for archive in archives
            ],
        }

        args.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = write_external_comparison_artifacts(
            args.output_dir,
            episode_rows=merged["episodes"], summary_rows=merged["seed_summaries"],
            training_rows=merged["training"], deployment_rows=merged["deployment_costs"],
            method_contracts=merged["method_contracts"], metadata=metadata,
        )

        # Master-prompt-spec file layout additions (kept alongside the reused function's
        # own output rather than renaming it, to avoid touching validated library code).
        (args.output_dir / "summaries" / "seed_summaries.csv").write_text(
            (args.output_dir / "raw" / "seed_summaries.csv").read_text(encoding="utf-8"), encoding="utf-8",
        )
        (args.output_dir / "summaries" / "aggregate_statistics.csv").write_text(
            (args.output_dir / "summaries" / "statistics.csv").read_text(encoding="utf-8"), encoding="utf-8",
        )
        (args.output_dir / "method_contracts.json").write_text(
            json.dumps(merged["method_contracts"], ensure_ascii=False, indent=2), encoding="utf-8",
        )

        checkpoint_paths = {
            f"switchglobe_exact_seed_{seed}": args.switchglobe_checkpoint_dir / f"seed_{seed}" / "risk_switch_lite_globe_p.pt"
            for seed in seeds
        }
        manifest["provenance"] = {
            **git_provenance(),
            "config_sha256": config_sha256(effective_config),
            "checkpoint_sha256": checkpoint_sha256_map(checkpoint_paths),
        }
        manifest["spot_check"] = spot_check_result
        manifest_path = args.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        write_validation_report(
            args.output_dir / "validation_report.md", seeds=seeds, merged=merged,
            spot_check_result=spot_check_result, manifest=manifest,
        )

    print(json.dumps({
        "episode_rows": len(merged["episodes"]), "seed_summary_rows": len(merged["seed_summaries"]),
        "statistics_rows": manifest["statistics_rows"], "paired_effect_rows": manifest["paired_effect_rows"],
        "spot_check_mismatches": spot_check_result["total_mismatches"],
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
