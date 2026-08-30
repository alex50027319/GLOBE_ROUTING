"""Phase 2 Local Student architecture guarantees."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from implementations.lite_globe.models.masking import masked_softmax
from implementations.lite_globe.models.policy_adapter import StudentPolicyAdapter
from implementations.lite_globe.models.student_policy import LocalStudentPolicy
from implementations.lite_globe.models.student_policy import FastSwitchGlobePolicy
from implementations.lite_globe.models.tensor_observation import (
    observation_to_tensors,
)


def _synthetic_observation(
    max_nodes: int,
    valid_nodes: list[int],
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(123)
    observation = {
        "self_features": rng.normal(size=6).astype(np.float32),
        "neighbor_features": rng.normal(size=(max_nodes, 7)).astype(np.float32),
        "edge_features": rng.uniform(size=(max_nodes, 2)).astype(np.float32),
        "packet_features": rng.normal(size=6).astype(np.float32),
        "action_mask": np.zeros(max_nodes + 1, dtype=np.int8),
    }
    observation["action_mask"][valid_nodes] = 1
    observation["action_mask"][max_nodes] = 1
    return observation


def _synthetic_fast_observation(
    max_nodes: int, valid_nodes: list[int]
) -> dict[str, np.ndarray]:
    observation = _synthetic_observation(max_nodes, valid_nodes)
    observation["candidate_forwardability"] = np.zeros(
        (max_nodes, 2), dtype=np.float32
    )
    observation["candidate_risk_features"] = np.zeros(
        (max_nodes, 4), dtype=np.float32
    )
    return observation


def test_action_probabilities_are_normalized_and_invalid_are_zero() -> None:
    torch.manual_seed(1)
    model = LocalStudentPolicy(max_nodes=4, hidden_dim=32)
    observation = observation_to_tensors(_synthetic_observation(4, [0, 2]))
    output = model(observation)

    torch.testing.assert_close(output.probabilities.sum(), torch.tensor(1.0))
    assert output.probabilities[1].item() == 0.0
    assert output.probabilities[3].item() == 0.0
    assert torch.isneginf(output.masked_logits[1])
    assert torch.isneginf(output.masked_logits[3])


def test_no_neighbor_assigns_all_probability_to_drop() -> None:
    model = LocalStudentPolicy(max_nodes=4, hidden_dim=32)
    observation = observation_to_tensors(_synthetic_observation(4, []))
    probabilities = model(observation).probabilities
    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])
    torch.testing.assert_close(probabilities, expected)


@pytest.mark.parametrize("valid_nodes", [[1], [0, 1, 2, 3]])
def test_one_and_maximum_neighbor_counts(valid_nodes: list[int]) -> None:
    model = LocalStudentPolicy(max_nodes=4, hidden_dim=64)
    observation = observation_to_tensors(
        _synthetic_observation(4, valid_nodes)
    )
    probabilities = model(observation).probabilities
    assert torch.all(torch.isfinite(probabilities))
    assert torch.count_nonzero(probabilities).item() == len(valid_nodes) + 1


def test_candidate_permutation_equivariance() -> None:
    torch.manual_seed(7)
    model = LocalStudentPolicy(max_nodes=5, hidden_dim=32).eval()
    original_np = _synthetic_observation(5, [0, 2, 4])
    permutation = np.array([3, 0, 4, 1, 2])
    permuted_np = {
        key: value.copy() for key, value in original_np.items()
    }
    permuted_np["neighbor_features"] = original_np["neighbor_features"][
        permutation
    ]
    permuted_np["edge_features"] = original_np["edge_features"][permutation]
    permuted_np["action_mask"][:5] = original_np["action_mask"][
        permutation
    ]

    original = model(observation_to_tensors(original_np)).probabilities
    permuted = model(observation_to_tensors(permuted_np)).probabilities
    torch.testing.assert_close(permuted[:5], original[:5][permutation])
    torch.testing.assert_close(permuted[5], original[5])


def test_batched_observation_output_shape() -> None:
    model = LocalStudentPolicy(max_nodes=4, hidden_dim=32)
    single = observation_to_tensors(_synthetic_observation(4, [0, 1]))
    batch = {
        key: torch.stack((value, value), dim=0)
        for key, value in single.items()
    }
    output = model(batch)
    assert output.probabilities.shape == (2, 5)
    torch.testing.assert_close(
        output.probabilities.sum(dim=-1), torch.ones(2)
    )


def test_masked_softmax_rejects_empty_support() -> None:
    with pytest.raises(ValueError, match="valid action"):
        masked_softmax(torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool))


def test_adapter_never_selects_invalid_action(line_env, line_positions) -> None:
    observation, _ = line_env.reset(
        seed=10,
        options={"positions": line_positions, "source": 0, "destination": 2},
    )
    torch.manual_seed(10)
    model = LocalStudentPolicy(
        max_nodes=line_env.config.max_nodes,
        hidden_dim=32,
    )
    adapter = StudentPolicyAdapter(model, deterministic=True)
    action = adapter.act(observation)
    assert observation["action_mask"][action] == 1


def test_fast_failover_uses_top2_without_second_forward() -> None:
    torch.manual_seed(21)
    model = LocalStudentPolicy(max_nodes=4, hidden_dim=32)
    adapter = StudentPolicyAdapter(
        model,
        deterministic=True,
        force_forward_if_available=True,
        enable_fast_failover=True,
    )
    observation = _synthetic_observation(4, [0, 1, 2])
    forwards = 0

    def count_forward(*_args) -> None:
        nonlocal forwards
        forwards += 1

    hook = model.register_forward_hook(count_forward)
    try:
        decision = adapter.act_with_metadata(observation)
        assert decision.backup_action is not None
        live_mask = observation["action_mask"].copy()
        live_mask[decision.action] = 0
        resolved = adapter.resolve_decision(decision, live_mask)
    finally:
        hook.remove()

    assert resolved == decision.backup_action
    assert resolved != decision.action
    assert forwards == 1
    assert adapter.episode_diagnostics()["fast_failover_steps"] == 1.0


def test_fast_failover_drops_when_primary_and_backup_are_stale() -> None:
    torch.manual_seed(22)
    model = LocalStudentPolicy(max_nodes=4, hidden_dim=32)
    adapter = StudentPolicyAdapter(
        model,
        force_forward_if_available=True,
        enable_fast_failover=True,
    )
    observation = _synthetic_observation(4, [0, 1])
    decision = adapter.act_with_metadata(observation)
    assert decision.backup_action is not None
    live_mask = observation["action_mask"].copy()
    live_mask[decision.action] = 0
    live_mask[decision.backup_action] = 0

    assert adapter.resolve_decision(decision, live_mask) == model.drop_action
    assert adapter.episode_diagnostics()["fast_failover_miss_steps"] == 1.0


def test_fast_failover_is_opt_in() -> None:
    model = LocalStudentPolicy(max_nodes=4, hidden_dim=32)
    observation = _synthetic_observation(4, [0, 1, 2])
    decision = StudentPolicyAdapter(
        model, force_forward_if_available=True
    ).act_with_metadata(observation)
    assert decision.backup_action is None


def test_freshness_cache_reuses_exact_decision_without_forward() -> None:
    torch.manual_seed(23)
    model = FastSwitchGlobePolicy(max_nodes=4, hidden_dim=32)
    adapter = StudentPolicyAdapter(
        model,
        force_forward_if_available=True,
        enable_fast_failover=True,
        enable_freshness_cache=True,
    )
    observation = _synthetic_fast_observation(4, [0, 1, 2])
    forwards = 0

    def count_forward(*_args) -> None:
        nonlocal forwards
        forwards += 1

    hook = model.register_forward_hook(count_forward)
    try:
        first = adapter.act_with_metadata(observation)
        second = adapter.act_with_metadata(observation)
        adapter.clear_freshness_cache()
        third = adapter.act_with_metadata(observation)
        diagnostics = adapter.episode_diagnostics()
        adapter.reset(seed=23)
        fourth = adapter.act_with_metadata(observation)
    finally:
        hook.remove()

    assert first == second
    assert second == third
    assert third == fourth
    assert forwards == 3
    assert diagnostics["freshness_cache_miss_steps"] == 2.0
    assert diagnostics["freshness_cache_hit_steps"] == 1.0
    assert adapter.episode_diagnostics()["freshness_cache_miss_steps"] == 1.0


def test_freshness_cache_invalidates_changed_mask_and_expired_entry() -> None:
    torch.manual_seed(24)
    model = FastSwitchGlobePolicy(max_nodes=4, hidden_dim=32)
    adapter = StudentPolicyAdapter(
        model,
        force_forward_if_available=True,
        enable_fast_failover=True,
        enable_freshness_cache=True,
        freshness_cache_ttl_ms=5.0,
    )
    clock = [1_000_000]
    adapter._cache_clock_ns = lambda: clock[0]
    observation = _synthetic_fast_observation(4, [0, 1, 2])
    forwards = 0

    def count_forward(*_args) -> None:
        nonlocal forwards
        forwards += 1

    hook = model.register_forward_hook(count_forward)
    try:
        adapter.act_with_metadata(observation)
        changed = {key: value.copy() for key, value in observation.items()}
        changed["action_mask"][2] = 0
        adapter.act_with_metadata(changed)
        changed_state = {
            key: value.copy() for key, value in observation.items()
        }
        changed_state["neighbor_features"][0, 0] += 1.0
        adapter.act_with_metadata(changed_state)
        clock[0] += 5_000_001
        adapter.act_with_metadata(observation)
    finally:
        hook.remove()

    assert forwards == 4
    diagnostics = adapter.episode_diagnostics()
    assert diagnostics["freshness_cache_miss_steps"] == 4.0
    assert diagnostics["freshness_cache_state_evictions"] == 1.0
    assert diagnostics["freshness_cache_stale_evictions"] == 1.0


def test_fast_switchglobe_diagnostics_report_switch_head_activation() -> None:
    torch.manual_seed(11)
    model = FastSwitchGlobePolicy(max_nodes=4, hidden_dim=32)
    observation = _synthetic_fast_observation(4, [0, 1, 2])
    tensors = observation_to_tensors(observation)

    _, switch_logit = model.forward_with_auxiliary(tensors)
    diagnostics = model.diagnostics(tensors)

    assert "switch_steps" in diagnostics
    expected = 1.0 if switch_logit.item() >= 0 else 0.0
    assert diagnostics["switch_steps"].item() == expected


def test_fast_switchglobe_adapter_populates_switch_steps_diagnostic() -> None:
    torch.manual_seed(12)
    model = FastSwitchGlobePolicy(max_nodes=4, hidden_dim=32)
    adapter = StudentPolicyAdapter(model)
    observation = _synthetic_fast_observation(4, [0, 1, 2])

    adapter.act_with_metadata(observation)

    diagnostics = adapter.episode_diagnostics()
    assert "switch_steps" in diagnostics
    assert diagnostics["switch_steps"] in (0.0, 1.0)


def test_freshness_cache_requires_deterministic_fast_policy() -> None:
    with pytest.raises(ValueError, match="deterministic"):
        StudentPolicyAdapter(
            FastSwitchGlobePolicy(max_nodes=4),
            deterministic=False,
            enable_freshness_cache=True,
        )
    with pytest.raises(ValueError, match="FastSwitchGLOBE"):
        StudentPolicyAdapter(
            LocalStudentPolicy(max_nodes=4),
            enable_freshness_cache=True,
        )


def test_initialization_is_seed_reproducible() -> None:
    observation = observation_to_tensors(_synthetic_observation(4, [0, 2]))
    torch.manual_seed(99)
    left = LocalStudentPolicy(max_nodes=4, hidden_dim=32)
    torch.manual_seed(99)
    right = LocalStudentPolicy(max_nodes=4, hidden_dim=32)
    torch.testing.assert_close(
        left(observation).probabilities,
        right(observation).probabilities,
    )
