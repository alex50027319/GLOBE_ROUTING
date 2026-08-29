"""Create a source-only Colab bundle for SwitchGLOBE training/evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "switchglobe_colab_bundle.zip",
    )
    return parser.parse_args()


def required_paths() -> list[Path]:
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "requirements-lite-globe.txt",
        ROOT / "README_SWITCHGLOBE_COLAB.md",
        ROOT / "scripts" / "train_switchglobe_pipeline.py",
    ]
    paths.extend(
        path
        for base in (ROOT / "implementations", ROOT / "tests" / "lite_globe")
        for path in base.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
        and path.name != ".DS_Store"
    )
    return sorted(set(paths))


def main() -> int:
    args = parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    paths = required_paths()
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "required SwitchGLOBE files are missing:\n"
            + "\n".join(str(path) for path in missing)
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in paths:
            archive.write(path, path.relative_to(ROOT))
    print(f"Created {output} ({output.stat().st_size / 1024**2:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
