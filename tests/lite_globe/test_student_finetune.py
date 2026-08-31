"""Phase 5 local PPO fine-tuning guarantees."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from implementations.lite_globe.algorithms.ppo import PpoConfig
from implementations.lite_globe.algorithms.student_finetune import (
    StudentFineTuneConfig,
    fine_tune_student,
    kd_lambda_at_update,
)
from implementations.lite_globe.data import generate_teacher_dataset
from implementations.lite_globe.env.fanet_env import FanetRoutingEnv
from implementations.lite_globe.models.student_actor_critic import (
    LocalStudentActorCritic,
)
from implementations.lite_globe.models.student_policy import LocalStudentPolicy
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


def test_local_actor_critic_shapes_and_mask() -> None:
    env = FanetRoutingEnv(routing_hole_config())
    observation, _ = env.reset(seed=3, options=routing_hole_options())
    model = LocalStudentActorCritic(
        LocalStudentPolicy(env.config.max_nodes, hidden_dim=32)
    )
    output = model(observation_to_tensors(observation))
    assert output.value.ndim == 0
    assert output.probabilities.shape == (env.config.max_nodes + 1,)
    assert torch.all(
        output.probabilities[
            torch.as_tensor(observation["action_mask"] == 0)
        ]
        == 0
    )


def test_kd_lambda_schedule_decays() -> None:
    assert kd_lambda_at_update(1.0, 0.1, 0) == 1.0
    assert kd_lambda_at_update(1.0, 0.1, 10) < 1.0
    assert kd_lambda_at_update(0.0, 0.1, 10) == 0.0
    with pytest.raises(ValueError):
        kd_lambda_at_update(-1.0, 0.1, 0)


def test_pure_ppo_never_accesses_global_observation() -> None:
    torch.manual_seed(5)
    env = FanetRoutingEnv(routing_hole_config())

    def forbidden():
        raise AssertionError("Phase 5 must not query global state")

    env.global_observation = forbidden  # type: ignore[method-assign]
    model = LocalStudentActorCritic(
        LocalStudentPolicy(env.config.max_nodes, hidden_dim=32)
    )
    result = fine_tune_student(
        env,
        model,
        ppo_config=PpoConfig(
            learning_rate=1e-3,
            update_epochs=1,
            minibatch_size=32,
        ),
        fine_tune_config=StudentFineTuneConfig(
            updates=2,
            episodes_per_update=4,
        ),
        seed=5,
        reset_options=routing_hole_options(),
    )
    assert result.episodes == 8
    assert result.transitions > 0
    assert np.isfinite(result.final_metrics.policy_loss)
    assert result.final_metrics.kd_loss == 0.0


def test_optional_offline_kd_runs_without_teacher_query() -> None:
    torch.manual_seed(7)
    env = FanetRoutingEnv(routing_hole_config())
    teacher = GlobalTeacherActorCritic(env.config.max_nodes, hidden_dim=32)
    dataset = generate_teacher_dataset(
        env,
        teacher,
        episode_seeds=list(range(6)),
        scenario_id="routing_hole",
        reset_options=routing_hole_options(),
    )

    def forbidden():
        raise AssertionError("fine-tuning cannot query the Teacher graph")

    env.global_observation = forbidden  # type: ignore[method-assign]
    model = LocalStudentActorCritic(
        LocalStudentPolicy(env.config.max_nodes, hidden_dim=32)
    )
    result = fine_tune_student(
        env,
        model,
        ppo_config=PpoConfig(
            learning_rate=1e-3,
            update_epochs=1,
            minibatch_size=32,
        ),
        fine_tune_config=StudentFineTuneConfig(
            updates=2,
            episodes_per_update=4,
            kd_lambda_initial=0.5,
            kd_decay_rate=0.2,
            kd_batch_size=16,
        ),
        seed=7,
        reset_options=routing_hole_options(),
        kd_dataset=dataset,
    )
    assert result.final_metrics.kd_lambda < 0.5
    assert np.isfinite(result.final_metrics.kd_loss)
