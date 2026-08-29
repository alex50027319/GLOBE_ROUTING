"""Locate completed Phase 13 artifacts in an active Colab runtime."""

from __future__ import annotations

from pathlib import Path


def main() -> None:
    roots = (Path("/content"), Path.cwd())
    seen: set[Path] = set()
    for root in roots:
        print(f"ROOT {root} exists={root.exists()}")
        if not root.exists():
            continue
        for pattern in (
            "phase13*_results.zip",
            "**/phase13*_results.zip",
            "**/phase13*/manifest.json",
            "**/phase13*/raw/episodes.csv",
        ):
            for path in root.glob(pattern):
                resolved = path.resolve()
                if resolved in seen or not path.is_file():
                    continue
                seen.add(resolved)
                print(f"FOUND {resolved} size={path.stat().st_size}")


if __name__ == "__main__":
    main()
