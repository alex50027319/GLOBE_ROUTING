#!/usr/bin/env python3
"""Reject a Claude edit that leaves a Python file syntactically invalid."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0

    raw_path = payload.get("tool_input", {}).get("file_path", "")
    if not raw_path or not raw_path.endswith(".py"):
        return 0

    path = Path(raw_path)
    if not path.is_file():
        return 0
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as error:
        print(f"Python syntax validation failed for {path}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
