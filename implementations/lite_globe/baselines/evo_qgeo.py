"""Audited common-contract adaptation of Evo-QGeo."""

from __future__ import annotations

import numpy as np

from .common import Observation
from .external_rl import EvoQGeoPolicy as _LegacyEvoQGeoPolicy


class EvoQGeoAdaptedPolicy(_LegacyEvoQGeoPolicy):
    """Evo-QGeo proxy retaining geographic, link-evolution and hole terms.

    The original paper's PRR window and beacon-carried neighbor Q values are
    mapped to the simulator's normalized edge availability, predicted link
    lifetime, forwardability, and this policy's decentralized Q table.
    """

    source = "10.3390/drones10020150"
    fidelity = "common-contract adaptation"

    def reset(self, seed: int | None = None) -> None:
        super().reset(seed)
        self.control_messages = 0
        self.control_bytes = 0
        self.hole_bypass_steps = 0

    def act(self, observation: Observation) -> int:
        candidates = np.flatnonzero(observation["action_mask"][: self.drop_action])
        self.control_messages += int(candidates.size)
        # node id, link estimate, expected duration and scalar Q value
        self.control_bytes += int(candidates.size) * 16
        progress = self._link_state_scores(observation)[0][:, 2]
        if candidates.size and not np.any(progress[candidates] > 0.0):
            self.hole_bypass_steps += 1
        return super().act(observation)

    def episode_diagnostics(self) -> dict[str, float]:
        return {
            "control_messages": float(self.control_messages),
            "control_bytes": float(self.control_bytes),
            "hole_bypass_steps": float(self.hole_bypass_steps),
        }
