"""Run the complete training lineage required to reproduce SwitchGLOBE."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


STAGES = ("teacher", "geo_residual", "predictive", "switchglobe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts/switchglobe"))
    parser.add_argument("--start-at", choices=STAGES, default="teacher")
    parser.add_argument("--stop-after", choices=STAGES, default="switchglobe")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _run(module: str, arguments: list[str]) -> None:
    command = [sys.executable, "-m", module, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    root = args.artifacts_root
    common = ["--device", args.device]
    if args.smoke:
        common.append("--smoke")
    if args.resume:
        common.append("--resume")

    teacher = root / "training" / "teacher"
    geo = root / "training" / "geo_residual"
    predictive = root / "training" / "predictive"
    final = root / "final"
    commands = {
        "teacher": (
            "implementations.lite_globe.run_phase7",
            [*common, "--output-dir", str(teacher)],
        ),
        "geo_residual": (
            "implementations.lite_globe.run_phase8",
            [
                *common,
                "--phase7-checkpoint-dir", str(teacher / "checkpoints"),
                "--output-dir", str(geo),
            ],
        ),
        "predictive": (
            "implementations.lite_globe.run_phase11",
            [
                *common,
                "--phase7-checkpoint-dir", str(teacher / "checkpoints"),
                "--phase8-checkpoint-dir", str(geo / "checkpoints"),
                "--output-dir", str(predictive),
            ],
        ),
        "switchglobe": (
            "implementations.lite_globe.run_switchglobe",
            [
                *common,
                "--phase8-checkpoint-dir", str(geo / "checkpoints"),
                "--phase11-checkpoint-dir", str(predictive / "checkpoints"),
                "--output-dir", str(final),
            ],
        ),
    }
    first = STAGES.index(args.start_at)
    last = STAGES.index(args.stop_after)
    if first > last:
        raise ValueError("--start-at must not come after --stop-after")
    for stage in STAGES[first : last + 1]:
        module, arguments = commands[stage]
        _run(module, arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
