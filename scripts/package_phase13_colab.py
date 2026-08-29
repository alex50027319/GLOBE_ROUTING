"""Create a minimal Colab bundle for Phase 13 Lite-GLOBE-P+."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 77, 123, 314, 2718)


def _artifact_root() -> Path:
    """Locate a complete checkpoint set in current or legacy layouts."""

    current = ROOT / "artifacts"
    legacy = ROOT / "ResearchAIWorkspace" / "artifacts"
    markers = (
        Path("lite_globe/phase8/checkpoints/seed_42/geo_residual_kd.pt"),
        Path("lite_globe/phase11/checkpoints/seed_42/lite_globe_p.pt"),
        Path(
            "lite_globe/phase12/checkpoints/seed_42/"
            "risk_switch_lite_globe_p.pt"
        ),
    )
    for candidate in (current, legacy):
        if all((candidate / marker).is_file() for marker in markers):
            return candidate
    return current


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "artifacts"
        / "lite_globe"
        / "phase13_colab_bundle.zip",
    )
    return parser.parse_args()


def required_paths() -> list[Path]:
    artifacts = _artifact_root()
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "requirements-lite-globe.txt",
        ROOT / "README_PHASE13_COLAB.md",
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
                artifacts
                / "lite_globe"
                / "phase8"
                / "checkpoints"
                / f"seed_{seed}"
                / "geo_residual_kd.pt",
                artifacts
                / "lite_globe"
                / "phase8"
                / "checkpoints"
                / f"seed_{seed}"
                / "training_metrics.json",
                artifacts
                / "lite_globe"
                / "phase11"
                / "checkpoints"
                / f"seed_{seed}"
                / "lite_globe_p.pt",
                artifacts
                / "lite_globe"
                / "phase11"
                / "checkpoints"
                / f"seed_{seed}"
                / "training_metrics.json",
                artifacts
                / "lite_globe"
                / "phase12"
                / "checkpoints"
                / f"seed_{seed}"
                / "risk_switch_lite_globe_p.pt",
                artifacts
                / "lite_globe"
                / "phase12"
                / "checkpoints"
                / f"seed_{seed}"
                / "training_metrics.json",
            ]
        )
    return sorted(set(paths))


def main() -> int:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    paths = required_paths()
    missing = [path for path in paths if not path.is_file()]
    if missing:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"required Phase 13 files are missing:\n{joined}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in paths:
            if path.is_relative_to(ROOT / "ResearchAIWorkspace"):
                arcname = path.relative_to(ROOT / "ResearchAIWorkspace")
            else:
                arcname = path.relative_to(ROOT)
            archive.write(path, arcname)
    size_mb = output.stat().st_size / (1024 * 1024)
    display_path = output.relative_to(ROOT) if output.is_relative_to(ROOT) else output
    print(f"Created {display_path} ({size_mb:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
