"""Run Phase 13 Colab evaluations in short resumable seed chunks.

Each session downloads an atomic progress ZIP and is released. The next session
restores that ZIP, so calibration candidates and scenario-method evaluations
are never intentionally repeated. After all seed ZIPs exist, they are merged
into one Phase 13 report directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 77, 123, 314, 2718)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 13 as short resumable Colab jobs by seed."
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated seeds to run. Default: 42,77,123,314,2718.",
    )
    parser.add_argument("--gpu", default="A100", help="Colab GPU type.")
    parser.add_argument(
        "--exec-timeout",
        default="86400",
        help="Per-chunk Colab exec timeout in seconds.",
    )
    parser.add_argument(
        "--log-dir",
        default="artifacts/lite_globe/phase13_seed_runs",
        help="Directory for runner and per-seed logs.",
    )
    parser.add_argument(
        "--merge-output-dir",
        default="artifacts/lite_globe/phase13_merged",
        help="Merged Phase 13 report output directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run seeds even if their result zip already exists.",
    )
    parser.add_argument(
        "--single-session",
        action="store_true",
        help="Use the legacy one-long-session-per-seed behavior.",
    )
    parser.add_argument(
        "--calibration-candidates-per-chunk",
        type=int,
        default=32,
        help="New calibration candidates per short Colab session (default: 32).",
    )
    parser.add_argument(
        "--evaluation-units-per-chunk",
        type=int,
        default=20,
        help="New scenario-method units per short session (default: 20).",
    )
    parser.add_argument(
        "--max-chunks-per-seed",
        type=int,
        default=60,
        help="Safety bound for resumable sessions per seed (default: 60).",
    )
    parser.add_argument(
        "--detach",
        action="store_true",
        help="Launch this runner in the background and return immediately.",
    )
    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def log(message: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def launch_detached(args: argparse.Namespace) -> int:
    log_dir = ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    runner_log = log_dir / "runner.log"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--seeds",
        args.seeds,
        "--gpu",
        args.gpu,
        "--exec-timeout",
        str(args.exec_timeout),
        "--log-dir",
        args.log_dir,
        "--merge-output-dir",
        args.merge_output_dir,
        "--calibration-candidates-per-chunk",
        str(args.calibration_candidates_per_chunk),
        "--evaluation-units-per-chunk",
        str(args.evaluation_units_per_chunk),
        "--max-chunks-per-seed",
        str(args.max_chunks_per_seed),
    ]
    if args.force:
        cmd.append("--force")
    if args.single_session:
        cmd.append("--single-session")
    with runner_log.open("ab") as log_file:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(f"detached runner pid={process.pid}")
    log(f"runner log: {runner_log}")
    return 0


def _zip_has(path: Path, suffix: str) -> bool:
    with zipfile.ZipFile(path, "r") as archive:
        return any(name.endswith(suffix) for name in archive.namelist())


def _run_seed_single_session(seed: int, args: argparse.Namespace) -> None:
    log_dir = ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    seed_log = log_dir / f"seed_{seed}.log"
    result_zip = ROOT / "artifacts" / "lite_globe" / f"phase13_seeds_{seed}_results.zip"
    if result_zip.exists() and not args.force:
        log(f"seed {seed}: result exists, skipping {result_zip}")
        return

    session = f"globe-phase13-seed-{seed}"
    cmd = [
        sys.executable,
        "scripts/colab_run.py",
        "--phase",
        "13",
        "--gpu",
        args.gpu,
        "--session",
        session,
        "--seeds",
        str(seed),
        "--exec-timeout",
        str(args.exec_timeout),
    ]
    log(f"seed {seed}: starting session {session}")
    log(f"seed {seed}: log file {seed_log}")
    with seed_log.open("ab") as log_file:
        subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )
    log(f"seed {seed}: completed {result_zip}")


def _latest_chunk(log_dir: Path, seed: int) -> tuple[int, Path] | None:
    pattern = f"phase13_seeds_{seed}_chunk_*.zip"
    artifact_dir = ROOT / "artifacts" / "lite_globe"
    matches = list(log_dir.glob(pattern)) + list(artifact_dir.glob(pattern))
    def chunk_number(path: Path) -> int:
        match = re.search(r"_chunk_(\d+)_results\.zip$", path.name)
        if match is None:
            raise ValueError(f"invalid Phase 13 chunk filename: {path.name}")
        return int(match.group(1))

    ordered = sorted(matches, key=chunk_number, reverse=True)
    for candidate in ordered:
        try:
            if _zip_has(candidate, "/partial_manifest.json") or _zip_has(
                candidate, "/manifest.json"
            ):
                return chunk_number(candidate), candidate
        except zipfile.BadZipFile:
            continue
    return None


def _run_seed_chunked(seed: int, args: argparse.Namespace) -> None:
    log_dir = ROOT / args.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    canonical_zip = (
        ROOT / "artifacts" / "lite_globe" / f"phase13_seeds_{seed}_results.zip"
    )
    if canonical_zip.exists() and not args.force:
        log(f"seed {seed}: final result exists, skipping {canonical_zip}")
        return

    previous: Path | None = None
    start_chunk = 1
    if not args.force:
        latest = _latest_chunk(log_dir, seed)
        if latest is not None:
            latest_id, previous = latest
            start_chunk = latest_id + 1
            log(f"seed {seed}: resuming after chunk {latest_id}: {previous}")

    for chunk_number in range(start_chunk, args.max_chunks_per_seed + 1):
        chunk_id = f"{chunk_number:03d}"
        session = f"globe-phase13-seed-{seed}-chunk-{chunk_id}"
        seed_log = log_dir / f"seed_{seed}_chunk_{chunk_id}.log"
        downloaded = (
            ROOT
            / "artifacts"
            / "lite_globe"
            / f"phase13_seeds_{seed}_chunk_{chunk_id}_results.zip"
        )
        cmd = [
            sys.executable,
            "scripts/colab_run.py",
            "--phase",
            "13",
            "--gpu",
            args.gpu,
            "--session",
            session,
            "--seeds",
            str(seed),
            "--chunk-id",
            chunk_id,
            "--max-calibration-candidates",
            str(args.calibration_candidates_per_chunk),
            "--max-evaluation-units",
            str(args.evaluation_units_per_chunk),
            "--exec-timeout",
            str(args.exec_timeout),
            "--skip-package",
        ]
        if previous is not None:
            cmd.extend(["--resume-from", str(previous)])
        log(f"seed {seed}: starting short session chunk {chunk_id}")
        with seed_log.open("ab") as log_file:
            subprocess.run(
                cmd,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not downloaded.is_file():
            raise FileNotFoundError(
                f"chunk {chunk_id} did not download expected ZIP: {downloaded}"
            )
        archived = log_dir / downloaded.name
        shutil.copy2(downloaded, archived)
        previous = archived
        if _zip_has(archived, "/manifest.json") and not _zip_has(
            archived, "/partial_manifest.json"
        ):
            shutil.copy2(archived, canonical_zip)
            log(f"seed {seed}: final result completed: {canonical_zip}")
            return
        if not _zip_has(archived, "/partial_manifest.json"):
            raise ValueError(
                f"chunk {chunk_id} contains neither a final nor partial manifest"
            )
        log(f"seed {seed}: chunk {chunk_id} saved; session was released")
    raise RuntimeError(
        f"seed {seed} exceeded --max-chunks-per-seed={args.max_chunks_per_seed}"
    )


def run_seed(seed: int, args: argparse.Namespace) -> None:
    if args.single_session:
        _run_seed_single_session(seed, args)
    else:
        _run_seed_chunked(seed, args)


def merge_results(seeds: list[int], args: argparse.Namespace) -> None:
    inputs = [
        ROOT / "artifacts" / "lite_globe" / f"phase13_seeds_{seed}_results.zip"
        for seed in seeds
    ]
    missing = [path for path in inputs if not path.exists()]
    if missing:
        for path in missing:
            log(f"merge skipped, missing: {path}")
        return
    cmd = [
        sys.executable,
        "scripts/merge_phase13_artifacts.py",
        "--inputs",
        *(str(path) for path in inputs),
        "--output-dir",
        args.merge_output_dir,
    ]
    log(f"merging {len(inputs)} seed result zips")
    subprocess.run(cmd, cwd=ROOT, check=True)
    log(f"merged output: {ROOT / args.merge_output_dir}")


def main() -> int:
    args = parse_args()
    if args.detach:
        return launch_detached(args)

    seeds = parse_seeds(args.seeds)
    log(f"seed queue started: seeds={seeds}, gpu={args.gpu}")
    try:
        if not args.single_session:
            log("building one shared Phase 13 Colab bundle")
            subprocess.run(
                [sys.executable, "scripts/package_phase13_colab.py"],
                cwd=ROOT,
                check=True,
            )
        for seed in seeds:
            run_seed(seed, args)
        merge_results(seeds, args)
    except subprocess.CalledProcessError as error:
        log(f"command failed with exit code {error.returncode}: {error.cmd}")
        return error.returncode
    log("seed queue finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
