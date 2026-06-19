"""Strong non-learning local baseline using deployable predictive features."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class PredictiveGeographicPolicy:
    """Combine geographic progress, forwardability, and local link quality."""

    def __init__(
        self,
        drop_action: int,
        *,
        geographic_weight: float = 8.0,
        forwardability_weight: tuple[float, float] = (1.5, 0.25),
        risk_weight: tuple[float, float, float, float] = (
            0.50,
            0.75,
            0.20,
            20.0,
        ),
    ) -> None:
        self.drop_action = drop_action
        self.geographic_weight = geographic_weight
        self.forwardability_weight = np.asarray(
            forwardability_weight, dtype=np.float32
        )
        self.risk_weight = np.asarray(risk_weight, dtype=np.float32)

    def reset(self, seed: int | None = None) -> None:
        del seed

    def observation_bytes(
        self, observation: dict[str, NDArray[np.generic]]
    ) -> int:
        keys = (
            "neighbor_features",
            "packet_features",
            "action_mask",
            "candidate_forwardability",
            "candidate_risk_features",
        )
        return sum(
            int(observation[key].nbytes)
            for key in keys
            if key in observation
        )

    def act(self, observation: dict[str, NDArray[np.generic]]) -> int:
        mask = observation["action_mask"]
        candidates = np.flatnonzero(mask[: self.drop_action])
        if candidates.size == 0:
            return self.drop_action
        destination_delta = observation["packet_features"][:2]
        neighbor_delta = observation["neighbor_features"][candidates, :2]
        current_distance = float(np.linalg.norm(destination_delta))
        next_distance = np.linalg.norm(
            destination_delta[None, :] - neighbor_delta, axis=1
        )
        score = self.geographic_weight * (
            current_distance - next_distance
        )
        if "candidate_forwardability" in observation:
            score += (
                observation["candidate_forwardability"][candidates]
                @ self.forwardability_weight
            )
        if "candidate_risk_features" in observation:
            score += (
                observation["candidate_risk_features"][candidates]
                @ self.risk_weight
            )
        return int(candidates[int(np.argmax(score))])
