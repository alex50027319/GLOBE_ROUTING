"""Outcome-guided local rollout data for deployment-neutral distillation."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from ..baselines import risk_aware_shortest_path
from ..env.fanet_env import FanetRoutingEnv
from ..env.graph_utils import shortest_path
from ..models.student_policy import StudentPolicyOutput
from ..models.tensor_observation import observation_to_tensors
from .distillation_dataset import (
    DistillationDataset,
    LOCAL_KEYS,
    OPTIONAL_LOCAL_KEYS,
    discounted_returns_from_trajectories,
)


class LocalReferencePolicy(Protocol):
    def to(self, device: torch.device | str): ...
    def eval(self): ...
    def __call__(self, observation: dict[str, torch.Tensor]) -> StudentPolicyOutput: ...


@torch.inference_mode()
def generate_return_guided_dataset(
    env: FanetRoutingEnv,
    reference: LocalReferencePolicy,
    *,
    episode_seeds: list[int],
    scenario_id: str,
    reset_options: dict[str, Any] | None = None,
    rollout_policy: str = "reference",
    return_discount: float = 0.85,
    device: torch.device | str = "cpu",
) -> DistillationDataset:
    """Collect local trajectories with realized discounted returns.

    The reference policy supplies a KL anchor at every visited state.  The
    environment rollout may follow that policy, a shortest-path oracle, or the
    existing risk-aware oracle.  Only deployable local observations and scalar
    training targets are retained; global graph state is never serialized.
    """

    if not episode_seeds:
        raise ValueError("at least one episode seed is required")
    if rollout_policy not in {"reference", "oracle", "risk_oracle"}:
        raise ValueError(
            "rollout_policy must be 'reference', 'oracle', or 'risk_oracle'"
        )
    if rollout_policy == "risk_oracle" and not env.config.include_risk_features:
        raise ValueError("risk_oracle rollout requires risk features")
    device = torch.device(device)
    reference.to(device).eval()
    records: dict[str, list[np.ndarray | int | float | str]] = {
        key: [] for key in (*LOCAL_KEYS, *OPTIONAL_LOCAL_KEYS)
    }
    records.update(
        {
            "teacher_logits": [],
            "teacher_probabilities": [],
            "selected_actions": [],
            "oracle_actions": [],
            "risk_oracle_actions": [],
            "rollout_actions": [],
            "rollout_rewards": [],
            "rollout_dones": [],
            "episode_seeds": [],
            "episode_steps": [],
            "scenario_ids": [],
        }
    )
    for episode_seed in episode_seeds:
        local_observation, _ = env.reset(
            seed=episode_seed,
            options=reset_options,
        )
        terminated = False
        truncated = False
        step = 0
        while not (terminated or truncated):
            output = reference(
                observation_to_tensors(local_observation, device=device)
            )
            reference_action = int(torch.argmax(output.probabilities).item())
            adjacency = env.adjacency.copy()
            for visited in env.packet.path[:-1]:
                adjacency[visited, :] = False
                adjacency[:, visited] = False
            oracle_path = shortest_path(
                adjacency,
                env.packet.current,
                env.packet.destination,
            )
            oracle_action = (
                int(oracle_path[1])
                if oracle_path is not None and len(oracle_path) >= 2
                else env.drop_action
            )
            risk_path = (
                risk_aware_shortest_path(env)
                if env.config.include_risk_features
                else None
            )
            risk_oracle_action = (
                int(risk_path[1])
                if risk_path is not None and len(risk_path) >= 2
                else env.drop_action
            )
            if rollout_policy == "reference":
                rollout_action = reference_action
            elif rollout_policy == "oracle":
                rollout_action = oracle_action
            else:
                rollout_action = risk_oracle_action

            for key in LOCAL_KEYS:
                records[key].append(np.array(local_observation[key], copy=True))
            for key in OPTIONAL_LOCAL_KEYS:
                if key in local_observation:
                    records[key].append(
                        np.array(local_observation[key], copy=True)
                    )
            records["teacher_logits"].append(
                output.logits.detach().cpu().numpy().astype(np.float32)
            )
            records["teacher_probabilities"].append(
                output.probabilities.detach().cpu().numpy().astype(np.float32)
            )
            records["selected_actions"].append(reference_action)
            records["oracle_actions"].append(oracle_action)
            if env.config.include_risk_features:
                records["risk_oracle_actions"].append(risk_oracle_action)
            records["rollout_actions"].append(rollout_action)
            records["episode_seeds"].append(episode_seed)
            records["episode_steps"].append(step)
            records["scenario_ids"].append(scenario_id)

            local_observation, reward, terminated, truncated, _ = env.step(
                rollout_action
            )
            records["rollout_rewards"].append(float(reward))
            records["rollout_dones"].append(int(terminated or truncated))
            step += 1

    arrays: dict[str, NDArray[np.generic]] = {}
    integer_keys = {
        "selected_actions",
        "oracle_actions",
        "risk_oracle_actions",
        "rollout_actions",
        "rollout_dones",
        "episode_seeds",
        "episode_steps",
    }
    for key, values in records.items():
        if not values:
            continue
        if key == "scenario_ids":
            arrays[key] = np.asarray(values, dtype=np.str_)
        elif key in integer_keys:
            arrays[key] = np.asarray(values, dtype=np.int64)
        elif key == "action_mask":
            arrays[key] = np.stack(values).astype(np.int8)
        elif key == "rollout_rewards":
            arrays[key] = np.asarray(values, dtype=np.float32)
        else:
            arrays[key] = np.stack(values).astype(np.float32)
    arrays["discounted_returns"] = discounted_returns_from_trajectories(
        rewards=arrays["rollout_rewards"],
        dones=arrays["rollout_dones"],
        episode_seeds=arrays["episode_seeds"],
        episode_steps=arrays["episode_steps"],
        scenario_ids=arrays["scenario_ids"],
        gamma=return_discount,
    )
    return DistillationDataset(arrays)
