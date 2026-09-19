"""Generate local-only offline data from a gated privileged Teacher."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from ..baselines import risk_aware_shortest_path
from ..env.fanet_env import FanetRoutingEnv
from ..env.graph_utils import shortest_path
from ..models.teacher_gnn import GlobalTeacherActorCritic
from ..models.tensor_observation import observation_to_tensors
from .distillation_dataset import (
    DistillationDataset,
    LOCAL_KEYS,
    OPTIONAL_LOCAL_KEYS,
)


@torch.inference_mode()
def generate_teacher_dataset(
    env: FanetRoutingEnv,
    teacher: GlobalTeacherActorCritic,
    *,
    episode_seeds: list[int],
    scenario_id: str,
    reset_options: dict[str, Any] | None = None,
    rollout_policy: str = "teacher",
    device: torch.device | str = "cpu",
) -> DistillationDataset:
    """Collect deterministic Teacher trajectories without storing global state."""

    if not episode_seeds:
        raise ValueError("at least one episode seed is required")
    if rollout_policy not in {"teacher", "oracle", "risk_oracle"}:
        raise ValueError(
            "rollout_policy must be 'teacher', 'oracle', or 'risk_oracle'"
        )
    if rollout_policy == "risk_oracle" and not env.config.include_risk_features:
        raise ValueError("risk_oracle rollout requires risk features")
    device = torch.device(device)
    teacher.to(device).eval()
    records: dict[str, list[np.ndarray | int | str]] = {
        key: [] for key in (*LOCAL_KEYS, *OPTIONAL_LOCAL_KEYS)
    }
    records.update(
        {
            "teacher_logits": [],
            "teacher_probabilities": [],
            "selected_actions": [],
            "oracle_actions": [],
            "risk_oracle_actions": [],
            "episode_seeds": [],
            "episode_steps": [],
            "scenario_ids": [],
        }
    )
    for episode_seed in episode_seeds:
        local_observation, _ = env.reset(
            seed=episode_seed, options=reset_options
        )
        terminated = False
        truncated = False
        step = 0
        while not (terminated or truncated):
            global_tensors = observation_to_tensors(
                env.global_observation(), device=device
            )
            output = teacher(global_tensors)
            teacher_action = int(torch.argmax(output.probabilities).item())
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
            records["selected_actions"].append(teacher_action)
            records["oracle_actions"].append(oracle_action)
            if env.config.include_risk_features:
                records["risk_oracle_actions"].append(
                    risk_oracle_action
                )
            records["episode_seeds"].append(episode_seed)
            records["episode_steps"].append(step)
            records["scenario_ids"].append(scenario_id)
            if rollout_policy == "teacher":
                rollout_action = teacher_action
            elif rollout_policy == "oracle":
                rollout_action = oracle_action
            else:
                rollout_action = risk_oracle_action
            local_observation, _, terminated, truncated, _ = env.step(
                rollout_action
            )
            step += 1

    arrays: dict[str, NDArray[np.generic]] = {}
    for key, values in records.items():
        if not values:
            continue
        if key == "scenario_ids":
            arrays[key] = np.asarray(values, dtype=np.str_)
        elif key in {
            "selected_actions",
            "oracle_actions",
            "risk_oracle_actions",
            "episode_seeds",
            "episode_steps",
        }:
            arrays[key] = np.asarray(values, dtype=np.int64)
        elif key == "action_mask":
            arrays[key] = np.stack(values).astype(np.int8)
        else:
            arrays[key] = np.stack(values).astype(np.float32)
    return DistillationDataset(arrays)
