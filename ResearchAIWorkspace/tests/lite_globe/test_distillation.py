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
    generate_teacher_dataset,
    split_by_episode_group,
)
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.models.student_policy import LocalStudentPolicy
from implementations.lite_globe.models.teacher_gnn import (
    GlobalTeacherActorCritic,
)
from implementations.lite_globe.scenarios import (
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
