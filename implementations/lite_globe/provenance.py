"""Shared git/config/checkpoint provenance helpers for run manifests.

Every ``run_*.py`` entry point that writes a manifest should record, per
``docs/claude_final_simulation_master_prompt.md`` section 3: the git commit
hash and dirty file list, the effective config and its SHA-256, and the
SHA-256 of every checkpoint file actually loaded. This module centralizes
that logic so campaigns do not each hand-roll their own subset of it.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


def git_commit_hash(repo_root: Path | str | None = None) -> str | None:
    """Current HEAD commit hash, or ``None`` outside a git checkout."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def git_dirty_files(repo_root: Path | str | None = None) -> list[str]:
    """Paths reported as modified/added/deleted/untracked by ``git status``."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_root,
            capture_output=True, text=True, check=True, timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    return [line[3:].strip() for line in result.stdout.splitlines() if line.strip()]


def git_provenance(repo_root: Path | str | None = None) -> dict[str, Any]:
    """``{"git_commit_hash": ..., "dirty_files": [...]}`` for a manifest."""

    return {
        "git_commit_hash": git_commit_hash(repo_root),
        "dirty_files": git_dirty_files(repo_root),
    }


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_sha256_map(paths: Mapping[str, Path | str]) -> dict[str, str]:
    """Map a label (e.g. ``"switchglobe_exact_seed_42"``) to its file hash.

    Missing files are silently omitted rather than raising, since manifests
    are still worth writing when a caller wires up an optional checkpoint
    slot (e.g. Top-2 reusing the Fast weights) that a given invocation does
    not exercise.
    """

    hashes: dict[str, str] = {}
    for label, path in paths.items():
        candidate = Path(path)
        if candidate.is_file():
            hashes[label] = file_sha256(candidate)
    return hashes


def config_sha256(config: Any) -> str:
    """Hash an effective config (dataclass instance, dict, or plain value).

    Uses the same ``json.dumps(..., sort_keys=True)`` serialization already
    relied on by ``experiments.latency_optimization_campaign.config_sha256``
    for resume-drift detection, so hashes stay comparable across callers.
    """

    if is_dataclass(config) and not isinstance(config, type):
        payload: Any = asdict(config)
    else:
        payload = config
    data = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
