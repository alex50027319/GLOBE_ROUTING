"""Environment-facing adapter for the privileged Teacher."""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray

from ..env.fanet_env import FanetRoutingEnv
from .teacher_gnn import GlobalTeacherActorCritic
from .tensor_observation import observation_to_tensors


class TeacherPolicyAdapter:
    """Choose actions from the environment's full graph observation."""

    def __init__(
        self,
        env: FanetRoutingEnv,
        model: GlobalTeacherActorCritic,
        *,
        device: torch.device | str = "cpu",
        deterministic: bool = True,
    ) -> None:
        self.env = env
        self.model = model.to(device)
        self.device = torch.device(device)
        self.deterministic = deterministic
        self.generator = torch.Generator(device=self.device)

    def reset(self, seed: int | None = None) -> None:
        self.generator.manual_seed(0 if seed is None else seed)

    def observation_bytes(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> int:
        del observation
        return sum(
            int(value.nbytes)
            for value in self.env.global_observation().values()
        )

    @torch.inference_mode()
    def act(self, observation: dict[str, NDArray[np.generic]]) -> int:
        del observation
        tensors = observation_to_tensors(
            self.env.global_observation(), device=self.device
        )
        probabilities = self.model(tensors).probabilities
        if self.deterministic:
            return int(torch.argmax(probabilities).item())
        return int(
            torch.multinomial(
                probabilities,
                num_samples=1,
                generator=self.generator,
            ).item()
        )
