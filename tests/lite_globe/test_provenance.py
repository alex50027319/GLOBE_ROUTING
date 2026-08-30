"""Shared git/config/checkpoint provenance helpers used by run manifests."""

from __future__ import annotations

import subprocess

from implementations.lite_globe import provenance


def _init_repo(path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


def test_git_provenance_reports_commit_and_clean_dirty_state(tmp_path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    subprocess.run(["git", "add", "a.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)

    clean = provenance.git_provenance(tmp_path)
    assert clean["git_commit_hash"]
    assert len(clean["git_commit_hash"]) == 40
    assert clean["dirty_files"] == []

    (tmp_path / "b.txt").write_text("new", encoding="utf-8")
    dirty = provenance.git_provenance(tmp_path)
    assert dirty["git_commit_hash"] == clean["git_commit_hash"]
    assert dirty["dirty_files"] == ["b.txt"]


def test_git_provenance_outside_a_repo_is_none_and_empty(tmp_path) -> None:
    outside = provenance.git_provenance(tmp_path)
    assert outside["git_commit_hash"] is None
    assert outside["dirty_files"] == []


def test_file_sha256_matches_hashlib_and_detects_content_change(tmp_path) -> None:
    import hashlib

    path = tmp_path / "data.bin"
    path.write_bytes(b"switchglobe" * 1000)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert provenance.file_sha256(path) == expected

    path.write_bytes(b"different content")
    assert provenance.file_sha256(path) != expected


def test_checkpoint_sha256_map_hashes_existing_and_skips_missing(tmp_path) -> None:
    present = tmp_path / "seed_1.pt"
    present.write_bytes(b"checkpoint-bytes")
    missing = tmp_path / "seed_2.pt"

    hashes = provenance.checkpoint_sha256_map({
        "seed_1": present, "seed_2": missing,
    })
    assert set(hashes) == {"seed_1"}
    assert hashes["seed_1"] == provenance.file_sha256(present)


def test_config_sha256_is_deterministic_across_dataclass_and_dict_and_sensitive_to_change() -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class TinyConfig:
        seeds: tuple[int, ...] = (1, 2)
        epochs: int = 5

    base = TinyConfig()
    same = TinyConfig()
    changed = TinyConfig(epochs=6)
    assert provenance.config_sha256(base) == provenance.config_sha256(same)
    assert provenance.config_sha256(base) != provenance.config_sha256(changed)

    # A dict payload equal to the dataclass's field mapping hashes identically,
    # since both go through the same json.dumps(sort_keys=True) serialization.
    from dataclasses import asdict
    assert provenance.config_sha256(asdict(base)) == provenance.config_sha256(base)
