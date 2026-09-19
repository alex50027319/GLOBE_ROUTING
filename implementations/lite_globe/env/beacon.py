"""Beacon-driven observation degradation for local risk features.

The Lite-GLOBE observation builder normally reads exact simulator positions and
velocities, so ``m_v`` (link margin), ``l_v`` (link lifetime) and ``o_v`` (best
onward lifetime) are ground-truth quantities with zero staleness and zero
estimation error.  A deployed relay cannot obtain those quantities for free: it
learns neighbour kinematics from periodic beacons that arrive late, get lost,
and carry noisy GPS/IMU estimates.

This module models that acquisition cost so the risk features can be degraded
in a controlled way.  The default :class:`BeaconDegradation` is disabled and the
whole path is a bit-exact no-op, which keeps every existing checkpoint, test and
full-run artifact reproducible.

Modelling scope (stated explicitly so results are not over-claimed):

* One shared beacon cache per episode, indexed by *observed* node.  A real
  protocol keeps a per-observer table; the shared cache approximates a periodic
  network-wide beacon flood and is a first-order model, not a MAC simulation.
* Between beacons the cache dead-reckons with the last known velocity, which is
  what a link-lifetime estimator would do.
* Beacon transmission itself consumes no simulated airtime here.  The byte cost
  is reported separately by :meth:`BeaconCache.control_accounting`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]

# Bytes per beacon payload: node id (2) + position (2 x 4) + velocity (2 x 4)
# + queue occupancy (1) + sequence number (1).  Matches the accounting rule used
# by ``baselines/evo_qgeo.py`` closely enough to be compared with it.
BEACON_PAYLOAD_BYTES = 20


@dataclass(frozen=True)
class BeaconDegradation:
    """Beacon period, loss and sensor-noise settings for local observations.

    Attributes:
        beacon_period: Steps between beacon broadcasts. ``1`` means every step.
        beacon_loss: Probability that an individual broadcast is not received.
        position_noise_std: Gaussian position error, as a fraction of the
            communication radius.
        velocity_noise_std: Gaussian velocity error, as a fraction of
            ``max_speed``.
        dead_reckon: Extrapolate stale entries with the last known velocity.
            When ``False`` the last received position is used verbatim, which
            isolates staleness from extrapolation error.
        degrade_topology: Also derive the action mask from beacon-known
            adjacency. When ``False`` (default) only the *features* degrade and
            the mask stays exact, which isolates the value of the risk signal
            from the cost of not knowing who the neighbours are.
    """

    beacon_period: int = 1
    beacon_loss: float = 0.0
    position_noise_std: float = 0.0
    velocity_noise_std: float = 0.0
    dead_reckon: bool = True
    degrade_topology: bool = False

    def __post_init__(self) -> None:
        if self.beacon_period < 1:
            raise ValueError("beacon_period must be at least 1")
        if not 0.0 <= self.beacon_loss < 1.0:
            raise ValueError("beacon_loss must be in [0, 1)")
        if self.position_noise_std < 0.0 or self.velocity_noise_std < 0.0:
            raise ValueError("noise standard deviations must be non-negative")

    @property
    def enabled(self) -> bool:
        """Whether any degradation is active.

        When this is ``False`` the environment skips the beacon path entirely
        and observations are identical to the undegraded implementation.
        """

        return (
            self.beacon_period > 1
            or self.beacon_loss > 0.0
            or self.position_noise_std > 0.0
            or self.velocity_noise_std > 0.0
            or self.degrade_topology
        )

    def label(self) -> str:
        """Return a short, filename-safe description of the setting."""

        if not self.enabled:
            return "exact"
        return (
            f"T{self.beacon_period}"
            f"_L{self.beacon_loss:g}"
            f"_P{self.position_noise_std:g}"
            f"_V{self.velocity_noise_std:g}"
            + ("_topo" if self.degrade_topology else "")
        )


class BeaconCache:
    """Last-received neighbour kinematics with dead-reckoned extrapolation."""

    def __init__(
        self,
        settings: BeaconDegradation,
        *,
        num_nodes: int,
        communication_radius: float,
        max_speed: float,
        time_step: float,
    ) -> None:
        self.settings = settings
        self.num_nodes = num_nodes
        self.position_sigma = settings.position_noise_std * communication_radius
        self.velocity_sigma = settings.velocity_noise_std * max(max_speed, 1e-6)
        self.time_step = time_step
        self.step_index = 0
        self.beacons_sent = 0
        self.beacons_received = 0
        self._positions = np.zeros((num_nodes, 2), dtype=np.float32)
        self._velocities = np.zeros((num_nodes, 2), dtype=np.float32)
        self._last_update = np.zeros(num_nodes, dtype=np.int64)

    def reset(
        self,
        positions: FloatArray,
        velocities: FloatArray,
        rng: np.random.Generator,
    ) -> None:
        """Seed the cache with a successful beacon from every node at step 0."""

        self.step_index = 0
        self.beacons_sent = 0
        self.beacons_received = 0
        self._positions = self._noisy(positions, self.position_sigma, rng)
        self._velocities = self._noisy(velocities, self.velocity_sigma, rng)
        self._last_update = np.zeros(self.num_nodes, dtype=np.int64)
        self.beacons_sent += self.num_nodes
        self.beacons_received += self.num_nodes

    def step(
        self,
        positions: FloatArray,
        velocities: FloatArray,
        rng: np.random.Generator,
    ) -> None:
        """Advance one simulation step and refresh entries that beaconed."""

        self.step_index += 1
        if self.step_index % self.settings.beacon_period != 0:
            return
        self.beacons_sent += self.num_nodes
        if self.settings.beacon_loss > 0.0:
            received = rng.random(self.num_nodes) >= self.settings.beacon_loss
        else:
            received = np.ones(self.num_nodes, dtype=bool)
        if not np.any(received):
            return
        self.beacons_received += int(np.count_nonzero(received))
        fresh_positions = self._noisy(positions, self.position_sigma, rng)
        fresh_velocities = self._noisy(velocities, self.velocity_sigma, rng)
        self._positions[received] = fresh_positions[received]
        self._velocities[received] = fresh_velocities[received]
        self._last_update[received] = self.step_index

    def believed_state(self) -> tuple[FloatArray, FloatArray]:
        """Return the currently believed positions and velocities."""

        if not self.settings.dead_reckon:
            return self._positions.copy(), self._velocities.copy()
        age = (self.step_index - self._last_update).astype(np.float32)
        drift = self._velocities * (age * self.time_step)[:, None]
        return (
            (self._positions + drift).astype(np.float32),
            self._velocities.copy(),
        )

    def mean_age_steps(self) -> float:
        """Return the mean staleness of the cache in simulation steps."""

        return float(np.mean(self.step_index - self._last_update))

    def control_accounting(self) -> dict[str, float]:
        """Return beacon control-plane cost for the episode so far.

        This is the quantity the manuscript currently reports as ``0.0`` for the
        proposed method while charging AODV and OLSR for their control traffic.
        """

        return {
            "beacon_messages": float(self.beacons_sent),
            "beacon_bytes": float(self.beacons_sent * BEACON_PAYLOAD_BYTES),
            "beacon_delivery_ratio": (
                self.beacons_received / self.beacons_sent
                if self.beacons_sent
                else 1.0
            ),
            "mean_beacon_age_steps": self.mean_age_steps(),
        }

    @staticmethod
    def _noisy(
        values: FloatArray, sigma: float, rng: np.random.Generator
    ) -> FloatArray:
        if sigma <= 0.0:
            return np.asarray(values, dtype=np.float32).copy()
        noise = rng.normal(0.0, sigma, size=np.shape(values))
        return (np.asarray(values, dtype=np.float32) + noise).astype(np.float32)
