#!/usr/bin/env python3
"""PostToolUse hook: py_compile a .py file right after Edit/Write touches it.

Reads the standard Claude Code hook JSON payload from stdin. On a syntax
error, exits 2 so the error is fed back to Claude immediately instead of
surfacing later in a test run.
"""

from __future__ import annotations

import json
import py_compile
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    file_path = payload.get("tool_input", {}).get("file_path", "")
    if not file_path.endswith(".py"):
        return 0

    try:
        py_compile.compile(file_path, doraise=True)
    except py_compile.PyCompileError as error:
        print(f"Syntax error in {file_path}:\n{error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
