"""Build the combined FastSwitchGLOBE vs SwitchGLOBE Exact vs external-baseline
comparison by merging Phase B (external baseline full comparison) and Phase C
(ablation) seed-summary/episode rows.

No new Colab/GPU run: Phase C's ``FastSwitchGLOBE`` rows were produced under the
exact same protocol as Phase B's baseline rows (same 14 scenarios in the same
order, same evaluation-seed formula, same 5 training seeds, same 200
episodes/scenario), verified by direct inspection before this script was
written. This script only merges already-validated local CSVs; it does not
touch the Colab/GPU workflow, the original Phase B/Phase C output files, or
existing checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from implementations.lite_globe.baselines.registry import EXTERNAL_METHODS
from implementations.lite_globe.evaluation.combined_comparison_reporting import (
    COMBINED_METHODS,
    FAST_SWITCHGLOBE,
    SWITCHGLOBE_EXACT,
    write_combined_comparison_artifacts,
)
from implementations.lite_globe.provenance import config_sha256, file_sha256, git_provenance

DEFAULT_SEEDS = (42, 77, 123, 314, 2718)
DEFAULT_BASELINE_DIR = Path("artifacts/final_paper_simulation/full/baselines")
DEFAULT_ABLATION_DIR = Path("artifacts/final_paper_simulation/full/ablation")
DEFAULT_OUTPUT_DIR = Path("artifacts/final_paper_simulation/synthesis/combined_comparison")

BASELINE_KEEP_METHODS = (*EXTERNAL_METHODS, "SwitchGLOBE")
BASELINE_RENAME = {"SwitchGLOBE": SWITCHGLOBE_EXACT}
ABLATION_KEEP_METHODS = (FAST_SWITCHGLOBE,)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_and_rename(rows: list[dict[str, Any]], *, keep_methods: tuple[str, ...],
                       rename: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Keep only rows whose ``method`` is in ``keep_methods``, applying ``rename``."""

    rename = rename or {}
    output = []
    for row in rows:
        method = str(row["method"])
        if method not in keep_methods:
            continue
        new_row = dict(row)
        new_row["method"] = rename.get(method, method)
        output.append(new_row)
    return output


def tag_source(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [{**row, "source_dataset": source} for row in rows]


def validate_episode_keys(rows: list[dict[str, Any]]) -> int:
    """Raise if any (method, scenario, training_seed, evaluation_seed) repeats."""

    seen: set[tuple[str, str, int, int]] = set()
    for row in rows:
        key = (str(row["method"]), str(row["scenario"]), int(row["training_seed"]), int(row["evaluation_seed"]))
        if key in seen:
            raise ValueError(f"duplicate episode key in combined comparison: {key}")
        seen.add(key)
    return len(seen)


def common_columns(baseline_header: list[str], ablation_header: list[str]) -> list[str]:
    return sorted(set(baseline_header) & set(ablation_header))


def project_columns(rows: list[dict[str, Any]], columns: list[str]) -> list[dict[str, Any]]:
    return [{column: row[column] for column in columns} for row in rows]


def build_metadata(*, baseline_manifest: dict[str, Any], ablation_manifest: dict[str, Any],
                    baseline_manifest_path: Path, ablation_manifest_path: Path,
                    training_seeds: tuple[int, ...] = DEFAULT_SEEDS) -> dict[str, Any]:
    config = {
        "training_seeds": list(training_seeds),
        "evaluation_episodes": 200,
        "methods": list(COMBINED_METHODS),
    }
    return {
        "training_seeds": list(training_seeds),
        "config": config,
        "config_sha256": config_sha256(config),
        **git_provenance(),
        "source_baseline": {
            "manifest_path": str(baseline_manifest_path),
            "manifest_sha256": file_sha256(baseline_manifest_path),
            "recorded_git_commit_hash": baseline_manifest.get("metadata", {}).get("git_commit_hash"),
            "episode_rows": baseline_manifest.get("episode_rows"),
        },
        "source_ablation": {
            "manifest_path": str(ablation_manifest_path),
            "manifest_sha256": file_sha256(ablation_manifest_path),
            "recorded_git_commit_hash": ablation_manifest.get("metadata", {}).get("git_commit_hash"),
            "episode_rows": ablation_manifest.get("episode_rows"),
        },
    }


def merge(*, baseline_dir: Path = DEFAULT_BASELINE_DIR, ablation_dir: Path = DEFAULT_ABLATION_DIR,
          output_dir: Path = DEFAULT_OUTPUT_DIR, training_seeds: tuple[int, ...] = DEFAULT_SEEDS) -> dict[str, Any]:
    baseline_manifest = json.loads((baseline_dir / "manifest.json").read_text(encoding="utf-8"))
    ablation_manifest = json.loads((ablation_dir / "manifest.json").read_text(encoding="utf-8"))
    if baseline_manifest.get("complete") is not True:
        raise ValueError("baseline manifest is not complete")
    if ablation_manifest.get("complete") is not True:
        raise ValueError("ablation manifest is not complete")

    baseline_summaries = select_and_rename(
        _read_csv(baseline_dir / "raw" / "seed_summaries.csv"),
        keep_methods=BASELINE_KEEP_METHODS, rename=BASELINE_RENAME,
    )
    ablation_summaries = select_and_rename(
        _read_csv(ablation_dir / "raw" / "seed_summaries.csv"),
        keep_methods=ABLATION_KEEP_METHODS,
    )
    summary_rows = tag_source(baseline_summaries, "phase_b_baseline") + tag_source(ablation_summaries, "phase_c_ablation")

    baseline_episode_path = baseline_dir / "raw" / "episodes.csv"
    ablation_episode_path = ablation_dir / "raw" / "episodes.csv"
    with baseline_episode_path.open(encoding="utf-8") as handle:
        baseline_header = next(csv.reader(handle))
    with ablation_episode_path.open(encoding="utf-8") as handle:
        ablation_header = next(csv.reader(handle))
    columns = common_columns(baseline_header, ablation_header)

    baseline_episodes = select_and_rename(
        _read_csv(baseline_episode_path), keep_methods=BASELINE_KEEP_METHODS, rename=BASELINE_RENAME,
    )
    ablation_episodes = select_and_rename(
        _read_csv(ablation_episode_path), keep_methods=ABLATION_KEEP_METHODS,
    )
    episode_rows = (
        tag_source(project_columns(baseline_episodes, columns), "phase_b_baseline")
        + tag_source(project_columns(ablation_episodes, columns), "phase_c_ablation")
    )
    validate_episode_keys(episode_rows)

    expected_episode_rows = len(COMBINED_METHODS) * 14 * len(training_seeds) * 200
    if len(episode_rows) != expected_episode_rows:
        raise ValueError(f"combined episode row count {len(episode_rows)} != {expected_episode_rows}")

    metadata = build_metadata(
        baseline_manifest=baseline_manifest, ablation_manifest=ablation_manifest,
        baseline_manifest_path=baseline_dir / "manifest.json", ablation_manifest_path=ablation_dir / "manifest.json",
        training_seeds=training_seeds,
    )

    manifest = write_combined_comparison_artifacts(
        output_dir, seed_summary_rows=summary_rows, episode_rows=episode_rows, metadata=metadata,
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--ablation-dir", type=Path, default=DEFAULT_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = merge(baseline_dir=args.baseline_dir, ablation_dir=args.ablation_dir, output_dir=args.output_dir)
    print(json.dumps({
        key: manifest[key] for key in (
            "complete", "methods", "seed_summary_rows", "expected_seed_summary_rows",
            "episode_rows", "statistics_overall_rows", "paired_effect_overall_rows",
        )
    }, indent=2))


if __name__ == "__main__":
    main()
