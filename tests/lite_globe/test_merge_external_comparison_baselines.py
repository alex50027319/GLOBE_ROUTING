"""Unit tests for the Phase B external-comparison ZIP merge/validation logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from implementations.lite_globe.baselines.registry import COMPARISON_METHODS
from implementations.lite_globe.evaluation.external_comparison_reporting import PRIMARY_METRICS, SCENARIOS
from scripts import merge_external_comparison_baselines as merge_mod
from scripts.merge_external_comparison_baselines import SeedArchive, merge_archives, validate_archive


def _summary_rows(seed: int, *, base: float = 0.5) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        for method in COMPARISON_METHODS:
            row = {"scenario": scenario, "method": method, "training_seed": str(seed)}
            for metric in PRIMARY_METRICS:
                row[metric] = base
            rows.append(row)
    return rows


def _episode_rows(seed: int) -> list[dict]:
    rows = []
    for scenario in SCENARIOS:
        for method in COMPARISON_METHODS:
            rows.append({
                "scenario": scenario, "method": method,
                "training_seed": str(seed), "evaluation_seed": f"{seed}-{scenario}-{method}-0",
            })
    return rows


def _archive(seed: int, *, contract=None, summary_value: float = 0.5) -> SeedArchive:
    contract = contract if contract is not None else [{"name": "AODV", "slug": "aodv"}]
    manifest = {
        "complete": True,
        "methods": list(COMPARISON_METHODS),
        "scenarios": list(SCENARIOS),
        "method_contracts": contract,
        "metadata": {"config": {"training_seeds": [seed], "evaluation_episodes": 1, "hidden_dim": 64}},
    }
    return SeedArchive(
        seed=seed, zip_path=Path(f"seeds_{seed}.zip"), extract_dir=Path("/tmp/unused"),
        manifest=manifest, episodes=_episode_rows(seed), training=[], deployment_costs=[],
        seed_summaries=_summary_rows(seed, base=summary_value),
    )


def test_validate_archive_accepts_well_formed_archive(monkeypatch):
    monkeypatch.setattr(merge_mod, "EXPECTED_EPISODES_PER_SEED", len(SCENARIOS) * len(COMPARISON_METHODS))
    archive = _archive(42)
    assert validate_archive(archive) == []


def test_validate_archive_flags_wrong_episode_count(monkeypatch):
    monkeypatch.setattr(merge_mod, "EXPECTED_EPISODES_PER_SEED", 999999)
    problems = validate_archive(_archive(42))
    assert any("episode rows" in p for p in problems)


def test_merge_archives_succeeds_for_two_consistent_seeds(monkeypatch):
    monkeypatch.setattr(merge_mod, "EXPECTED_EPISODES_PER_SEED", len(SCENARIOS) * len(COMPARISON_METHODS))
    merged = merge_archives([_archive(42, summary_value=0.4), _archive(77, summary_value=0.6)])
    assert len(merged["episodes"]) == 2 * len(SCENARIOS) * len(COMPARISON_METHODS)
    assert len(merged["seed_summaries"]) == 2 * len(SCENARIOS) * len(COMPARISON_METHODS)
    assert merged["method_contracts"] == [{"name": "AODV", "slug": "aodv"}]


def test_merge_archives_rejects_duplicate_episode_keys(monkeypatch):
    monkeypatch.setattr(merge_mod, "EXPECTED_EPISODES_PER_SEED", len(SCENARIOS) * len(COMPARISON_METHODS))
    same_seed_twice = [_archive(42), _archive(42)]
    with pytest.raises(ValueError, match="duplicate episode key"):
        merge_archives(same_seed_twice)


def test_merge_archives_rejects_divergent_method_contracts(monkeypatch):
    monkeypatch.setattr(merge_mod, "EXPECTED_EPISODES_PER_SEED", len(SCENARIOS) * len(COMPARISON_METHODS))
    archives = [
        _archive(42, contract=[{"name": "AODV", "slug": "aodv"}]),
        _archive(77, contract=[{"name": "AODV", "slug": "aodv-different"}]),
    ]
    with pytest.raises(ValueError, match="method_contracts differ"):
        merge_archives(archives)


def test_merge_archives_rejects_divergent_training_config(monkeypatch):
    monkeypatch.setattr(merge_mod, "EXPECTED_EPISODES_PER_SEED", len(SCENARIOS) * len(COMPARISON_METHODS))
    a, b = _archive(42), _archive(77)
    b.manifest["metadata"]["config"]["hidden_dim"] = 128
    with pytest.raises(ValueError, match="training configs differ"):
        merge_archives([a, b])
