"""Create a minimal Colab bundle for the Phase 10 external RL campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 77, 123, 314, 2718)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts"
        / "lite_globe"
        / "phase10_colab_bundle.zip",
    )
    return parser.parse_args()


def required_paths() -> list[Path]:
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "requirements-lite-globe.txt",
        ROOT / "README_PHASE10_COLAB.md",
    ]
    paths.extend(
        path
        for base in (
            ROOT / "implementations",
            ROOT / "tests" / "lite_globe",
        )
        for path in base.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name != ".DS_Store"
    )
    for seed in SEEDS:
        paths.extend(
            [
                ROOT
                / "artifacts"
                / "lite_globe"
                / "phase8"
                / "checkpoints"
                / f"seed_{seed}"
                / "geo_residual_kd.pt",
                ROOT
                / "artifacts"
                / "lite_globe"
                / "phase8"
                / "checkpoints"
                / f"seed_{seed}"
                / "training_metrics.json",
            ]
        )
    return sorted(set(paths))


def main() -> int:
    args = parse_args()
    paths = required_paths()
    missing = [path for path in paths if not path.is_file()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"required Phase 10 files are missing:\n{joined}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        args.output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Created {args.output.relative_to(ROOT)} ({size_mb:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
