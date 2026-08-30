"""FastSwitchGLOBE external-comparison chunk and 8-method report contracts."""

from __future__ import annotations

from dataclasses import asdict

from implementations.lite_globe.baselines.registry import (
    COMPARISON_METHODS,
    EXTERNAL_METHODS,
    PROPOSED_METHOD,
)
from implementations.lite_globe.evaluation.external_comparison_reporting import (
    PRIMARY_METRICS,
    SCENARIOS,
    write_external_comparison_artifacts,
)
from implementations.lite_globe.evaluation.fast_external_comparison_reporting import (
    FAST_METHOD_CONTRACT,
    write_fast_external_chunk,
)
from implementations.lite_globe.experiments.fast_external_comparison_campaign import (
    FAST_METHOD,
    FastExternalComparisonConfig,
)
from implementations.lite_globe.run_fast_external_comparison import config_from_yaml
from scripts.merge_fast_external_comparison import _validate_base, _validate_fast


def _summary(method: str, scenario: str, seed: int) -> dict:
    return {
        "method": method,
        "scenario": scenario,
        "training_seed": seed,
        **{metric: 0.5 for metric in PRIMARY_METRICS},
    }


def _episode(method: str, scenario: str, seed: int, evaluation_seed: int) -> dict:
    return {
        "method": method,
        "scenario": scenario,
        "training_seed": seed,
        "evaluation_seed": evaluation_seed,
        "delivered": 1,
    }


def test_smoke_config_is_single_seed_and_small() -> None:
    raw = {
        "campaign": {
            "training_seeds": [42, 77],
            "evaluation_episodes": 200,
            "exact_hidden_dim": 64,
            "fast_hidden_dim": 32,
        },
        "training": {
            "dataset_episodes_per_scenario": 100,
            "epochs": 60,
            "batch_size": 256,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "temperature": 1.0,
            "action_coefficient": 1.0,
            "switch_coefficient": 0.2,
        },
    }
    config = config_from_yaml(raw, smoke=True, seeds=[42])
    assert config.training_seeds == (42,)
    assert config.evaluation_episodes == 3
    assert config.dataset_episodes_per_scenario == 3
    assert config.epochs == 2


def test_fast_chunk_writer_enforces_expected_cardinality(tmp_path) -> None:
    seed, episodes_per_scenario = 42, 2
    config = FastExternalComparisonConfig(
        training_seeds=(seed,), evaluation_episodes=episodes_per_scenario
    )
    summaries = [_summary(FAST_METHOD, scenario, seed) for scenario in SCENARIOS]
    episodes = [
        _episode(FAST_METHOD, scenario, seed, 1_100_000 + index * 10_000 + offset)
        for index, scenario in enumerate(SCENARIOS)
        for offset in range(episodes_per_scenario)
    ]
    manifest = write_fast_external_chunk(
        tmp_path,
        episode_rows=episodes,
        summary_rows=summaries,
        training_rows=[{"method": FAST_METHOD, "training_seed": seed}],
        deployment_rows=[{"method": FAST_METHOD, "training_seed": seed}],
        metadata={"mode": "smoke", "config": asdict(config)},
    )
    assert manifest["complete"] is True
    assert manifest["episode_rows"] == manifest["expected_episode_rows"] == 28
    assert manifest["seed_summary_rows"] == 14


def test_generalized_report_accepts_exact_and_fast_as_proposed(tmp_path) -> None:
    methods = (*COMPARISON_METHODS, FAST_METHOD)
    seed = 42
    summaries = [
        _summary(method, scenario, seed)
        for scenario in SCENARIOS
        for method in methods
    ]
    episodes = [
        _episode(method, scenario, seed, 1_100_000 + index * 10_000)
        for index, scenario in enumerate(SCENARIOS)
        for method in methods
    ]
    manifest = write_external_comparison_artifacts(
        tmp_path,
        episode_rows=episodes,
        summary_rows=summaries,
        training_rows=[],
        deployment_rows=[],
        method_contracts=[FAST_METHOD_CONTRACT],
        metadata={
            "mode": "smoke",
            "config": {"training_seeds": [seed], "evaluation_episodes": 1},
        },
        comparison_methods=methods,
        proposed_methods=(PROPOSED_METHOD, FAST_METHOD),
        external_methods=EXTERNAL_METHODS,
    )
    assert manifest["methods"] == list(methods)
    assert manifest["proposed_methods"] == [PROPOSED_METHOD, FAST_METHOD]
    assert manifest["episode_rows"] == len(methods) * len(SCENARIOS)
    paired = (tmp_path / "summaries" / "paired_effects.csv").read_text()
    assert "proposed_method" in paired
    assert FAST_METHOD in paired and PROPOSED_METHOD in paired


def test_merge_manifest_validators_accept_full_contracts() -> None:
    base = {
        "complete": True,
        "mode": "full",
        "methods": list(COMPARISON_METHODS),
        "scenarios": list(SCENARIOS),
        "episode_rows": len(COMPARISON_METHODS) * len(SCENARIOS) * 200,
        "seed_summary_rows": len(COMPARISON_METHODS) * len(SCENARIOS),
        "metadata": {"config": {"training_seeds": [42], "evaluation_episodes": 200}},
    }
    fast = {
        "complete": True,
        "mode": "full",
        "suite": "fast_switchglobe_external_comparison_chunk",
        "methods": [FAST_METHOD],
        "scenarios": list(SCENARIOS),
        "episode_rows": len(SCENARIOS) * 200,
        "expected_episode_rows": len(SCENARIOS) * 200,
        "seed_summary_rows": len(SCENARIOS),
        "metadata": {"config": {"training_seeds": [42], "evaluation_episodes": 200}},
    }
    _validate_base(base, seed=42)
    _validate_fast(fast, seed=42)
