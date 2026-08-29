"""Merge chunked Phase 13 artifact directories into one report directory."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from implementations.lite_globe.evaluation import write_phase13_artifacts
from implementations.lite_globe.evaluation.phase13_reporting import (
    PHASE13_SCENARIOS,
)
from implementations.lite_globe.experiments.phase13_campaign import (
    PHASE13_METHODS,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help=(
            "Phase13 artifact directories or zip files. Each input must contain "
            "raw/episodes.csv, raw/seed_summaries.csv, and raw/training_metrics.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "lite_globe" / "phase13_merged",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=ROOT / "artifacts" / "lite_globe" / "_phase13_merge_work",
    )
    return parser.parse_args()


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _extract_if_needed(input_path: Path, work_dir: Path) -> Path:
    if input_path.is_dir():
        return input_path
    if input_path.suffix != ".zip":
        raise ValueError(f"unsupported input path: {input_path}")
    target = work_dir / input_path.stem
    if target.exists():
        return target
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(input_path, "r") as archive:
        archive.extractall(target)
    return target


def _find_artifact_root(path: Path) -> Path:
    if (path / "raw" / "episodes.csv").is_file():
        return path
    matches = [
        candidate
        for candidate in path.rglob("raw/episodes.csv")
        if (candidate.parent / "seed_summaries.csv").is_file()
        and (candidate.parent / "training_metrics.csv").is_file()
    ]
    if not matches:
        raise FileNotFoundError(f"no Phase13 raw artifacts found under {path}")
    return matches[0].parents[1]


def _dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    output: list[dict[str, str]] = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def _validate_complete_rows(
    episode_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    training_rows: list[dict[str, str]],
) -> dict[str, object]:
    """Reject a merge that is missing any seed/scenario/method unit."""

    episode_counts = Counter(
        (int(row["training_seed"]), row["scenario"], row["method"])
        for row in episode_rows
    )
    episode_identity = [
        (
            int(row["training_seed"]),
            row["scenario"],
            row["method"],
            int(row["evaluation_seed"]),
        )
        for row in episode_rows
    ]
    if len(episode_identity) != len(set(episode_identity)):
        raise ValueError("duplicate Phase 13 episode identities remain after merge")

    summary_keys = [
        (int(row["training_seed"]), row["scenario"], row["method"])
        for row in summary_rows
    ]
    if len(summary_keys) != len(set(summary_keys)):
        raise ValueError("duplicate Phase 13 seed summaries remain after merge")

    seeds = sorted({key[0] for key in episode_counts})
    expected_keys = {
        (seed, scenario, method)
        for seed in seeds
        for scenario in PHASE13_SCENARIOS
        for method in PHASE13_METHODS
    }
    if set(episode_counts) != expected_keys:
        missing = sorted(expected_keys - set(episode_counts))
        unexpected = sorted(set(episode_counts) - expected_keys)
        raise ValueError(
            "incomplete Phase 13 episode grid: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )
    if set(summary_keys) != expected_keys:
        missing = sorted(expected_keys - set(summary_keys))
        unexpected = sorted(set(summary_keys) - expected_keys)
        raise ValueError(
            "incomplete Phase 13 summary grid: "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    summary_by_key = {
        (int(row["training_seed"]), row["scenario"], row["method"]): row
        for row in summary_rows
    }
    for key, count in episode_counts.items():
        declared = int(float(summary_by_key[key]["episodes"]))
        if count != declared:
            raise ValueError(
                f"episode count mismatch for {key}: raw={count}, summary={declared}"
            )
    training_seeds = [int(row["training_seed"]) for row in training_rows]
    if sorted(training_seeds) != seeds or len(training_seeds) != len(set(training_seeds)):
        raise ValueError(
            "training metrics must contain exactly one row for every merged seed"
        )
    per_unit_counts = sorted(set(episode_counts.values()))
    if len(per_unit_counts) != 1:
        raise ValueError(
            f"inconsistent evaluation episode counts: {per_unit_counts}"
        )
    return {
        "validated": True,
        "training_seeds": seeds,
        "scenarios_per_seed": len(PHASE13_SCENARIOS),
        "methods_per_scenario": len(PHASE13_METHODS),
        "evaluation_episodes_per_unit": per_unit_counts[0],
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    work_dir = args.work_dir if args.work_dir.is_absolute() else ROOT / args.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)

    episode_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    training_rows: list[dict[str, str]] = []
    sources: list[str] = []

    for raw_input in args.inputs:
        input_path = _resolve(raw_input)
        extracted = _extract_if_needed(input_path, work_dir)
        artifact_root = _find_artifact_root(extracted)
        sources.append(str(artifact_root.relative_to(ROOT) if artifact_root.is_relative_to(ROOT) else artifact_root))
        episode_rows.extend(_read_csv(artifact_root / "raw" / "episodes.csv"))
        summary_rows.extend(_read_csv(artifact_root / "raw" / "seed_summaries.csv"))
        training_rows.extend(_read_csv(artifact_root / "raw" / "training_metrics.csv"))

    episode_rows = _dedupe(episode_rows)
    summary_rows = _dedupe(summary_rows)
    training_rows = _dedupe(training_rows)
    validation = _validate_complete_rows(
        episode_rows,
        summary_rows,
        training_rows,
    )
    training_seeds = validation["training_seeds"]

    manifest = write_phase13_artifacts(
        output_dir,
        episode_rows=episode_rows,
        summary_rows=summary_rows,
        training_rows=training_rows,
        metadata={
            "phase": 13,
            "mode": "merged",
            "device": "mixed",
            "merged_sources": sources,
            "training_seeds": training_seeds,
            "merge_validation": validation,
        },
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
