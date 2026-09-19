"""Merge five verified FastSwitchGLOBE chunks with the existing 7-method full run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
from typing import Any
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from implementations.lite_globe.baselines.registry import (
    COMPARISON_METHODS,
    EXTERNAL_METHODS,
    PROPOSED_METHOD,
)
from implementations.lite_globe.evaluation.external_comparison_reporting import (
    SCENARIOS,
    write_external_comparison_artifacts,
)
from implementations.lite_globe.evaluation.fast_external_comparison_reporting import (
    FAST_METHOD_CONTRACT,
)
from implementations.lite_globe.experiments.fast_external_comparison_campaign import (
    FAST_METHOD,
)
from implementations.lite_globe.provenance import file_sha256, git_provenance


DEFAULT_SEEDS = (42, 77, 123, 314, 2718)
ALL_METHODS = (*COMPARISON_METHODS, FAST_METHOD)
BASE_EPISODES_PER_SEED = len(COMPARISON_METHODS) * len(SCENARIOS) * 200
FAST_EPISODES_PER_SEED = len(SCENARIOS) * 200


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _extract(zip_path: Path, destination: Path) -> dict[str, Any]:
    if not zip_path.is_file():
        raise FileNotFoundError(f"missing result archive: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        unsafe = [name for name in archive.namelist() if Path(name).is_absolute() or ".." in Path(name).parts]
        if unsafe:
            raise ValueError(f"unsafe paths in {zip_path}: {unsafe[:3]}")
        archive.extractall(destination)
    return json.loads((destination / "manifest.json").read_text(encoding="utf-8"))


def _validate_base(manifest: dict[str, Any], *, seed: int) -> None:
    config = manifest.get("metadata", {}).get("config", {})
    problems = []
    if not manifest.get("complete") or manifest.get("mode") != "full":
        problems.append("manifest must be complete full mode")
    if manifest.get("methods") != list(COMPARISON_METHODS):
        problems.append("7-method roster mismatch")
    if manifest.get("scenarios") != list(SCENARIOS):
        problems.append("scenario roster mismatch")
    if list(config.get("training_seeds", [])) != [seed]:
        problems.append("training seed mismatch")
    if int(config.get("evaluation_episodes", -1)) != 200:
        problems.append("evaluation_episodes must equal 200")
    if int(manifest.get("episode_rows", -1)) != BASE_EPISODES_PER_SEED:
        problems.append("base episode row count mismatch")
    if int(manifest.get("seed_summary_rows", -1)) != len(COMPARISON_METHODS) * len(SCENARIOS):
        problems.append("base summary row count mismatch")
    if problems:
        raise ValueError(f"seed {seed} base archive invalid: " + "; ".join(problems))


def _validate_fast(manifest: dict[str, Any], *, seed: int) -> None:
    config = manifest.get("metadata", {}).get("config", {})
    problems = []
    if not manifest.get("complete") or manifest.get("mode") != "full":
        problems.append("manifest must be complete full mode")
    if manifest.get("suite") != "fast_switchglobe_external_comparison_chunk":
        problems.append("suite mismatch")
    if manifest.get("methods") != [FAST_METHOD]:
        problems.append("method roster mismatch")
    if manifest.get("scenarios") != list(SCENARIOS):
        problems.append("scenario roster mismatch")
    if list(config.get("training_seeds", [])) != [seed]:
        problems.append("training seed mismatch")
    if int(config.get("evaluation_episodes", -1)) != 200:
        problems.append("evaluation_episodes must equal 200")
    if int(manifest.get("episode_rows", -1)) != FAST_EPISODES_PER_SEED:
        problems.append("Fast episode row count mismatch")
    if int(manifest.get("episode_rows", -1)) != int(manifest.get("expected_episode_rows", -2)):
        problems.append("Fast episode_rows/expected_episode_rows mismatch")
    if int(manifest.get("seed_summary_rows", -1)) != len(SCENARIOS):
        problems.append("Fast summary row count mismatch")
    if problems:
        raise ValueError(f"seed {seed} Fast archive invalid: " + "; ".join(problems))


def _episode_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row["scenario"]),
        int(row["training_seed"]),
        int(row["evaluation_seed"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-zip-dir",
        type=Path,
        default=Path("artifacts/external_comparison_colab_results"),
    )
    parser.add_argument(
        "--fast-zip-dir",
        type=Path,
        default=Path("artifacts/fast_external_comparison_colab_results"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/final_paper_simulation/full/baselines_with_fast"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    seeds = tuple(args.seeds)
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")

    episodes: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    deployment: list[dict[str, Any]] = []
    source_archives: list[dict[str, Any]] = []
    base_contracts: list[dict[str, Any]] | None = None

    with tempfile.TemporaryDirectory(prefix="fast_external_merge_") as scratch:
        scratch_root = Path(scratch)
        for seed in seeds:
            base_zip = args.baseline_zip_dir / f"seeds_{seed}.zip"
            fast_zip = args.fast_zip_dir / f"fast_seeds_{seed}.zip"
            base_dir, fast_dir = scratch_root / f"base_{seed}", scratch_root / f"fast_{seed}"
            base_manifest = _extract(base_zip, base_dir)
            fast_manifest = _extract(fast_zip, fast_dir)
            _validate_base(base_manifest, seed=seed)
            _validate_fast(fast_manifest, seed=seed)

            current_contracts = list(base_manifest.get("method_contracts", []))
            if base_contracts is None:
                base_contracts = current_contracts
            elif current_contracts != base_contracts:
                raise ValueError("external method contracts differ across base archives")

            base_episodes = _read_csv(base_dir / "raw" / "episodes.csv")
            fast_episodes = _read_csv(fast_dir / "raw" / "episodes.csv")
            reference_keys = {
                _episode_key(row)
                for row in base_episodes
                if row["method"] == PROPOSED_METHOD
            }
            fast_keys = {_episode_key(row) for row in fast_episodes}
            if fast_keys != reference_keys:
                raise ValueError(
                    f"seed {seed}: Fast and SwitchGLOBE evaluation keys differ; "
                    f"missing={sorted(reference_keys-fast_keys)[:3]}, "
                    f"extra={sorted(fast_keys-reference_keys)[:3]}"
                )

            for row in base_episodes:
                row["source_archive"] = base_zip.name
            for row in fast_episodes:
                row["source_archive"] = fast_zip.name
            episodes.extend(base_episodes)
            episodes.extend(fast_episodes)
            summaries.extend(_read_csv(base_dir / "raw" / "seed_summaries.csv"))
            summaries.extend(_read_csv(fast_dir / "raw" / "seed_summaries.csv"))
            training.extend(_read_csv(base_dir / "raw" / "training.csv"))
            training.extend(_read_csv(fast_dir / "raw" / "training.csv"))
            deployment.extend(_read_csv(base_dir / "raw" / "deployment_costs.csv"))
            deployment.extend(_read_csv(fast_dir / "raw" / "deployment_costs.csv"))
            source_archives.extend(
                [
                    {"kind": "base", "seed": seed, "path": str(base_zip), "sha256": file_sha256(base_zip)},
                    {"kind": "fast", "seed": seed, "path": str(fast_zip), "sha256": file_sha256(fast_zip)},
                ]
            )

    expected_rows = len(ALL_METHODS) * len(SCENARIOS) * len(seeds) * 200
    if len(episodes) != expected_rows:
        raise ValueError(f"combined episode rows {len(episodes)} != {expected_rows}")
    unique_keys = {
        (row["method"], row["scenario"], row["training_seed"], row["evaluation_seed"])
        for row in episodes
    }
    if len(unique_keys) != len(episodes):
        raise ValueError("duplicate combined episode keys")

    effective_config = {
        "training_seeds": list(seeds),
        "evaluation_episodes": 200,
        "comparison_contract": "identical scenario/reset/evaluation seeds",
    }
    method_contracts = [*(base_contracts or []), FAST_METHOD_CONTRACT]
    manifest = write_external_comparison_artifacts(
        args.output_dir,
        episode_rows=episodes,
        summary_rows=summaries,
        training_rows=training,
        deployment_rows=deployment,
        method_contracts=method_contracts,
        metadata={
            "mode": "full",
            "source": "verified merge of existing 7-method and FastSwitchGLOBE-only Colab chunks",
            "config": effective_config,
            "source_archives": source_archives,
            **git_provenance(),
        },
        comparison_methods=ALL_METHODS,
        proposed_methods=(PROPOSED_METHOD, FAST_METHOD),
        external_methods=EXTERNAL_METHODS,
    )
    manifest["validation"] = {
        "paired_episode_keys_match": True,
        "duplicate_episode_keys": 0,
        "full_seed_count": len(seeds),
        "base_episode_rows_per_seed": BASE_EPISODES_PER_SEED,
        "fast_episode_rows_per_seed": FAST_EPISODES_PER_SEED,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "validation_report.md").write_text(
        "\n".join(
            [
                "# SwitchGLOBE / FastSwitchGLOBE external comparison validation",
                "",
                f"- seeds: {list(seeds)}",
                f"- methods: {list(ALL_METHODS)}",
                f"- scenarios: {len(SCENARIOS)}",
                f"- episode rows: {len(episodes)} / {expected_rows}",
                f"- seed summary rows: {len(summaries)} / {len(ALL_METHODS) * len(SCENARIOS) * len(seeds)}",
                "- Fast/SwitchGLOBE scenario, training-seed, evaluation-seed keys: exact match",
                "- duplicate episode keys: 0",
                "- complete full manifests required for all 10 source ZIPs",
                "",
                "Energy is a simulator transmission-energy proxy, not Joules. Policy input bytes are not routing-control overhead.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
