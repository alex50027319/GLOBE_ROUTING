"""Unit tests for the combined-comparison merge helpers (Phase B + Phase C)."""

from __future__ import annotations

import pytest

from implementations.lite_globe.evaluation.combined_comparison_reporting import (
    FAST_SWITCHGLOBE,
    SWITCHGLOBE_EXACT,
)
from scripts.merge_combined_comparison import (
    ABLATION_KEEP_METHODS,
    BASELINE_KEEP_METHODS,
    BASELINE_RENAME,
    common_columns,
    project_columns,
    select_and_rename,
    tag_source,
    validate_episode_keys,
)


def test_select_and_rename_keeps_only_listed_methods_and_renames():
    rows = [
        {"method": "AODV", "value": 1},
        {"method": "SwitchGLOBE", "value": 2},
        {"method": "Not Kept", "value": 3},
    ]
    out = select_and_rename(rows, keep_methods=BASELINE_KEEP_METHODS, rename=BASELINE_RENAME)
    methods = {row["method"] for row in out}
    assert methods == {"AODV", SWITCHGLOBE_EXACT}
    # original rows are untouched (no mutation of source data)
    assert rows[1]["method"] == "SwitchGLOBE"


def test_select_and_rename_ablation_keeps_only_fast_switchglobe():
    rows = [{"method": "SwitchGLOBE Exact"}, {"method": FAST_SWITCHGLOBE}, {"method": "Geo-Residual Student"}]
    out = select_and_rename(rows, keep_methods=ABLATION_KEEP_METHODS)
    assert [row["method"] for row in out] == [FAST_SWITCHGLOBE]


def test_tag_source_adds_field_without_mutating_input():
    rows = [{"a": 1}]
    tagged = tag_source(rows, "phase_b_baseline")
    assert tagged == [{"a": 1, "source_dataset": "phase_b_baseline"}]
    assert "source_dataset" not in rows[0]


def test_validate_episode_keys_counts_unique_rows():
    rows = [
        {"method": "AODV", "scenario": "s1", "training_seed": "42", "evaluation_seed": "1"},
        {"method": "AODV", "scenario": "s1", "training_seed": "42", "evaluation_seed": "2"},
    ]
    assert validate_episode_keys(rows) == 2


def test_validate_episode_keys_rejects_duplicate():
    rows = [
        {"method": "AODV", "scenario": "s1", "training_seed": "42", "evaluation_seed": "1"},
        {"method": "AODV", "scenario": "s1", "training_seed": "42", "evaluation_seed": "1"},
    ]
    with pytest.raises(ValueError, match="duplicate episode key"):
        validate_episode_keys(rows)


def test_common_columns_is_intersection():
    assert common_columns(["a", "b", "c"], ["b", "c", "d"]) == ["b", "c"]


def test_project_columns_keeps_only_requested_keys():
    rows = [{"a": 1, "b": 2, "c": 3}]
    assert project_columns(rows, ["a", "c"]) == [{"a": 1, "c": 3}]
