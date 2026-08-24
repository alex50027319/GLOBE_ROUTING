"""Run Phase 13 Colab evaluations one seed at a time.

This wrapper keeps long Colab runs recoverable: each training seed gets its own
session name, output directory, result zip, and log file. After all seed zips
exist, the wrapper merges them into one Phase 13 report directory.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 77, 123, 314, 2718)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase 13 Colab jobs sequentially by training seed."
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
        help="Per-seed Colab exec timeout in seconds.",
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
    ]
    if args.force:
        cmd.append("--force")
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


def run_seed(seed: int, args: argparse.Namespace) -> None:
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
