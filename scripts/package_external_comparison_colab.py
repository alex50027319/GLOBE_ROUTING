"""Package external-comparison code and read-only SwitchGLOBE checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 77, 123, 314, 2718)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path,
                        default=ROOT / "ResearchAIWorkspace" / "artifacts" / "lite_globe" / "phase12" / "checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "external_comparison_colab_bundle.zip")
    parser.add_argument(
        "--fast-checkpoint-dir",
        type=Path,
        help="Optionally include one verified FastSwitchGLOBE checkpoint per seed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    fast_checkpoint_dir = (
        args.fast_checkpoint_dir.resolve()
        if args.fast_checkpoint_dir is not None
        else None
    )
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "requirements-lite-globe.txt",
        ROOT / "README_BASELINES_COLAB.md",
        ROOT / "README_FASTSWITCHGLOBE_EXTERNAL_COLAB.md",
    ]
    for base in (ROOT / "implementations", ROOT / "scripts", ROOT / "tests" / "lite_globe"):
        paths.extend(path for path in base.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    for seed in SEEDS:
        checkpoint = checkpoint_dir / f"seed_{seed}" / "risk_switch_lite_globe_p.pt"
        if not checkpoint.is_file():
            checkpoint = checkpoint_dir / f"seed_{seed}" / "switchglobe.pt"
        paths.append(checkpoint)
        if fast_checkpoint_dir is not None:
            paths.append(fast_checkpoint_dir / f"seed_{seed}" / "fast_switchglobe.pt")
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing bundle inputs:\n" + "\n".join(map(str, missing)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(set(paths)):
            archive.write(path, path.relative_to(ROOT))
    print(f"Created {args.output} ({args.output.stat().st_size / 2**20:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
