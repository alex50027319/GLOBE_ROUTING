"""Phase 3 global Teacher architecture and training gate."""

from __future__ import annotations

import numpy as np
import torch

from implementations.lite_globe.algorithms.ppo import PpoConfig
from implementations.lite_globe.algorithms.teacher_trainer import train_teacher
from implementations.lite_globe.baselines import GpsrPolicy
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.evaluation import evaluate_policy
from implementations.lite_globe.models.teacher_adapter import TeacherPolicyAdapter
from implementations.lite_globe.models.teacher_gnn import (
    GlobalTeacherActorCritic,
)
from implementations.lite_globe.models.tensor_observation import (
    observation_to_tensors,
)
from implementations.lite_globe.scenarios import (
    routing_hole_config,
    routing_hole_options,
)
from implementations.lite_globe.utils import (
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)


def _teacher_observation():
    env = FanetRoutingEnv(routing_hole_config())
    local, _ = env.reset(seed=4, options=routing_hole_options())
    return env, local, env.global_observation()


def test_teacher_and_student_action_support_are_aligned() -> None:
    _, local, global_observation = _teacher_observation()
    np.testing.assert_array_equal(
        local["action_mask"], global_observation["action_mask"]
    )


def test_teacher_output_is_masked_and_value_is_scalar() -> None:
    torch.manual_seed(4)
    env, _, global_observation = _teacher_observation()
    model = GlobalTeacherActorCritic(env.config.max_nodes, hidden_dim=32)
    output = model(observation_to_tensors(global_observation))
    assert output.probabilities.shape == (env.config.max_nodes + 1,)
    assert output.value.ndim == 0
    torch.testing.assert_close(output.probabilities.sum(), torch.tensor(1.0))
    invalid = torch.as_tensor(global_observation["action_mask"] == 0)
    assert torch.all(output.probabilities[invalid] == 0)


def test_teacher_is_equivariant_to_global_node_permutation() -> None:
    torch.manual_seed(8)
    env, _, original_np = _teacher_observation()
    model = GlobalTeacherActorCritic(env.config.max_nodes, hidden_dim=32).eval()
    permutation = np.array([3, 0, 5, 2, 1, 4])
    permuted_np = {key: value.copy() for key, value in original_np.items()}
    permuted_np["node_features"] = original_np["node_features"][permutation]
    permuted_np["node_mask"] = original_np["node_mask"][permutation]
    permuted_np["adjacency"] = original_np["adjacency"][
        np.ix_(permutation, permutation)
    ]
    permuted_np["edge_features"] = original_np["edge_features"][
        permutation
    ][:, permutation]
    permuted_np["action_mask"][: env.config.max_nodes] = original_np[
        "action_mask"
    ][permutation]

    original = model(
        observation_to_tensors(original_np)
    ).probabilities
    permuted = model(
        observation_to_tensors(permuted_np)
    ).probabilities
    torch.testing.assert_close(
        permuted[: env.config.max_nodes],
        original[: env.config.max_nodes][permutation],
    )
    torch.testing.assert_close(permuted[-1], original[-1])


def test_teacher_checkpoint_round_trip(tmp_path) -> None:
    torch.manual_seed(12)
    env, _, observation = _teacher_observation()
    original = GlobalTeacherActorCritic(env.config.max_nodes, hidden_dim=32)
    expected = original(observation_to_tensors(observation)).probabilities
    path = tmp_path / "teacher.pt"
    save_checkpoint(path, original, metadata={"phase": 3, "seed": 12})

    restored = GlobalTeacherActorCritic(env.config.max_nodes, hidden_dim=32)
    metadata = load_checkpoint(path, restored)
    actual = restored(observation_to_tensors(observation)).probabilities
    assert metadata == {"phase": 3, "seed": 12}
    torch.testing.assert_close(actual, expected)


def test_teacher_ppo_beats_gpsr_on_routing_hole() -> None:
    seed = 42
    seed_everything(seed)
    config = routing_hole_config(seed)
    options = routing_hole_options()
    env = FanetRoutingEnv(config)
    model = GlobalTeacherActorCritic(config.max_nodes, hidden_dim=64)
    train_teacher(
        env,
        model,
        ppo_config=PpoConfig(
            learning_rate=1e-3,
            entropy_coefficient=0.05,
            update_epochs=4,
            minibatch_size=128,
        ),
        updates=15,
        episodes_per_update=16,
        seed=seed,
        reset_options=options,
    )
    seeds = list(range(10_000, 10_020))
    gpsr = evaluate_policy(
        env,
        GpsrPolicy(env.drop_action),
        seeds,
        reset_options=options,
    )
    teacher = evaluate_policy(
        env,
        TeacherPolicyAdapter(env, model),
        seeds,
        reset_options=options,
    )
    assert gpsr.packet_delivery_ratio == 0.0
    assert teacher.packet_delivery_ratio >= 0.8
