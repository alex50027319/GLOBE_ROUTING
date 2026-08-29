"""Phase 2 Local Student architecture guarantees."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from implementations.lite_globe.models.masking import masked_softmax
from implementations.lite_globe.models.policy_adapter import StudentPolicyAdapter
from implementations.lite_globe.models.student_policy import LocalStudentPolicy
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
