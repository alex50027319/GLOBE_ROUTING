"""Phase 4 offline dataset and forward-KL distillation guarantees."""

from __future__ import annotations

import numpy as np
import torch

from implementations.lite_globe.algorithms.distillation import (
    DistillationConfig,
    evaluate_distillation,
    forward_kl_loss,
    train_student_distillation,
)
from implementations.lite_globe.data import (
    DistillationDataset,
    discounted_returns_from_trajectories,
    generate_return_guided_dataset,
    generate_teacher_dataset,
    split_by_episode_group,
)
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.models.student_policy import LocalStudentPolicy
from implementations.lite_globe.models.teacher_gnn import (
    GlobalTeacherActorCritic,
)
from implementations.lite_globe.scenarios import (
    predictive_break_config,
    predictive_break_options,
    routing_hole_config,
    routing_hole_options,
)


def _synthetic_dataset(groups: int = 10) -> DistillationDataset:
    max_nodes = 4
    rng = np.random.default_rng(5)
    masks = np.zeros((groups, max_nodes + 1), dtype=np.int8)
    masks[:, [0, 2, max_nodes]] = 1
    logits = rng.normal(size=(groups, max_nodes + 1)).astype(np.float32)
    masked = np.where(masks, logits, -np.inf)
    shifted = masked - np.max(masked, axis=-1, keepdims=True)
    exp = np.where(masks, np.exp(shifted), 0.0)
    probabilities = exp / exp.sum(axis=-1, keepdims=True)
    arrays = {
        "self_features": rng.normal(size=(groups, 6)).astype(np.float32),
        "neighbor_features": rng.normal(
            size=(groups, max_nodes, 7)
        ).astype(np.float32),
        "edge_features": rng.uniform(
            size=(groups, max_nodes, 2)
        ).astype(np.float32),
        "packet_features": rng.normal(size=(groups, 6)).astype(np.float32),
        "action_mask": masks,
        "teacher_logits": logits,
        "teacher_probabilities": probabilities.astype(np.float32),
        "selected_actions": np.argmax(probabilities, axis=-1).astype(np.int64),
        "episode_seeds": np.arange(groups, dtype=np.int64),
        "episode_steps": np.zeros(groups, dtype=np.int64),
        "scenario_ids": np.asarray(["toy"] * groups, dtype=np.str_),
    }
    return DistillationDataset(arrays)


def test_identical_masked_policy_has_zero_forward_kl() -> None:
    logits = torch.tensor([[0.2, -0.3, 1.0, 5.0]])
    mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    kl, teacher, student = forward_kl_loss(
        logits, logits.clone(), mask, temperature=2.0
    )
    torch.testing.assert_close(kl, torch.tensor(0.0), atol=1e-7, rtol=0)
    torch.testing.assert_close(teacher, student)
    assert teacher[0, 3].item() == 0.0


def test_dataset_round_trip_excludes_global_state(tmp_path) -> None:
    dataset = _synthetic_dataset()
    path = tmp_path / "dataset.npz"
    dataset.save(path)
    restored = DistillationDataset.load(path)
    assert len(restored) == len(dataset)
    assert "adjacency" not in restored.arrays
    assert "node_features" not in restored.arrays
    for key in dataset.arrays:
        np.testing.assert_array_equal(restored.arrays[key], dataset.arrays[key])


def test_group_split_has_no_seed_or_scenario_leakage() -> None:
    split = split_by_episode_group(_synthetic_dataset(), seed=9)
    train = set(split.train.group_ids)
    validation = set(split.validation.group_ids)
    test = set(split.test.group_ids)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == {
        f"toy:{seed}" for seed in range(10)
    }


def test_student_learns_toy_teacher_distribution() -> None:
    torch.manual_seed(4)
    env = FanetRoutingEnv(routing_hole_config())
    teacher = GlobalTeacherActorCritic(env.config.max_nodes, hidden_dim=32)
    dataset = generate_teacher_dataset(
        env,
        teacher,
        episode_seeds=list(range(30)),
        scenario_id="routing_hole",
        reset_options=routing_hole_options(),
    )
    split = split_by_episode_group(dataset, seed=4)
    torch.manual_seed(4)
    student = LocalStudentPolicy(env.config.max_nodes, hidden_dim=32)
    config = DistillationConfig(
        epochs=40,
        batch_size=64,
        learning_rate=2e-3,
    )
    before = evaluate_distillation(student, split.test, config=config)
    train_student_distillation(
        student,
        split.train,
        split.validation,
        config=config,
        seed=4,
    )
    after = evaluate_distillation(student, split.test, config=config)
    assert np.isfinite(after.kl)
    assert after.kl < before.kl
    assert after.action_agreement >= 0.8


def test_discounted_returns_do_not_cross_episode_boundaries() -> None:
    returns = discounted_returns_from_trajectories(
        rewards=np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        dones=np.asarray([0, 1, 0, 1], dtype=np.int64),
        episode_seeds=np.asarray([10, 10, 11, 11], dtype=np.int64),
        episode_steps=np.asarray([0, 1, 0, 1], dtype=np.int64),
        scenario_ids=np.asarray(["s", "s", "s", "s"], dtype=np.str_),
        gamma=0.5,
    )
    np.testing.assert_allclose(returns, [2.0, 2.0, 5.0, 4.0])


def test_return_guided_dataset_contains_only_local_training_targets() -> None:
    config = predictive_break_config(42)
    reference = LocalStudentPolicy(config.max_nodes, hidden_dim=32)
    dataset = generate_return_guided_dataset(
        FanetRoutingEnv(config),
        reference,
        episode_seeds=[101, 102, 103],
        scenario_id="predictive_reference_rollout",
        reset_options=predictive_break_options(0.0),
        rollout_policy="reference",
        return_discount=0.85,
    )

    assert len(dataset) > 0
    assert "adjacency" not in dataset.arrays
    assert "node_features" not in dataset.arrays
    assert {
        "rollout_actions",
        "rollout_rewards",
        "rollout_dones",
        "discounted_returns",
    }.issubset(dataset.arrays)
    assert np.all(np.isfinite(dataset.arrays["discounted_returns"]))
    assert int(dataset.arrays["rollout_dones"].sum()) == 3


def test_return_guided_auxiliary_loss_adds_no_deployment_parameters() -> None:
    dataset = _synthetic_dataset(groups=12)
    arrays = {key: value.copy() for key, value in dataset.arrays.items()}
    arrays.update(
        {
            "rollout_actions": arrays["selected_actions"].copy(),
            "rollout_rewards": np.linspace(-1.0, 1.0, 12).astype(np.float32),
            "rollout_dones": np.ones(12, dtype=np.int64),
            "discounted_returns": np.linspace(-5.0, 10.0, 12).astype(
                np.float32
            ),
        }
    )
    guided = DistillationDataset(arrays)
    split = split_by_episode_group(guided, seed=17)
    model = LocalStudentPolicy(max_nodes=4, hidden_dim=32)
    keys_before = tuple(model.state_dict())
    parameters_before = sum(parameter.numel() for parameter in model.parameters())
    config = DistillationConfig(
        epochs=3,
        batch_size=8,
        return_action_coefficient=0.2,
        return_weight_temperature=5.0,
    )
    result = train_student_distillation(
        model,
        split.train,
        split.validation,
        config=config,
        seed=17,
    )

    assert result.validation.rollout_action_agreement is not None
    assert tuple(model.state_dict()) == keys_before
    assert sum(parameter.numel() for parameter in model.parameters()) == parameters_before
