"""Create a minimal Colab bundle for the Phase 11 Lite-GLOBE-P campaign."""

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
        / "phase11_colab_bundle.zip",
    )
    parser.add_argument(
        "--phase7-checkpoint-dir",
        type=Path,
        default=ROOT / "artifacts" / "lite_globe" / "phase7" / "checkpoints",
        help="Read-only source for global_teacher.pt per seed.",
    )
    parser.add_argument(
        "--phase8-checkpoint-dir",
        type=Path,
        default=ROOT / "artifacts" / "lite_globe" / "phase8" / "checkpoints",
        help="Read-only source for geo_residual_kd.pt/training_metrics.json per seed.",
    )
    return parser.parse_args()


def required_paths(
    *, phase7_checkpoint_dir: Path, phase8_checkpoint_dir: Path
) -> list[tuple[Path, Path]]:
    """Return (source_path, archive_relative_path) pairs."""

    pairs = [
        (path, path.relative_to(ROOT))
        for path in (
            ROOT / "pyproject.toml",
            ROOT / "requirements-lite-globe.txt",
            ROOT / "README_PHASE11_COLAB.md",
        )
    ]
    pairs.extend(
        (path, path.relative_to(ROOT))
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
        pairs.extend(
            [
                (
                    phase7_checkpoint_dir / f"seed_{seed}" / "global_teacher.pt",
                    Path("artifacts/lite_globe/phase7/checkpoints")
                    / f"seed_{seed}" / "global_teacher.pt",
                ),
                (
                    phase8_checkpoint_dir / f"seed_{seed}" / "geo_residual_kd.pt",
                    Path("artifacts/lite_globe/phase8/checkpoints")
                    / f"seed_{seed}" / "geo_residual_kd.pt",
                ),
                (
                    phase8_checkpoint_dir / f"seed_{seed}" / "training_metrics.json",
                    Path("artifacts/lite_globe/phase8/checkpoints")
                    / f"seed_{seed}" / "training_metrics.json",
                ),
            ]
        )
    return sorted(set(pairs), key=lambda pair: pair[1])


def main() -> int:
    args = parse_args()
    output = args.output
    if not output.is_absolute():
        output = ROOT / output
    pairs = required_paths(
        phase7_checkpoint_dir=args.phase7_checkpoint_dir,
        phase8_checkpoint_dir=args.phase8_checkpoint_dir,
    )
    missing = [source for source, _ in pairs if not source.is_file()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"required Phase 11 files are missing:\n{joined}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for source, arcname in pairs:
            archive.write(source, arcname)
    size_mb = output.stat().st_size / (1024 * 1024)
    try:
        display_path = output.relative_to(ROOT)
    except ValueError:
        display_path = output
    print(f"Created {display_path} ({size_mb:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
