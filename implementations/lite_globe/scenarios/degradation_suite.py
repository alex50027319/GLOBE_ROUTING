"""Beacon-degradation sensitivity grids for the local risk features.

The proposed routing policy is driven by ``m_v`` (link margin), ``l_v`` (link
lifetime) and ``o_v`` (best onward lifetime). In the current simulator those are
ground-truth quantities, so the reliability results assume a relay that knows
neighbour and neighbour-of-neighbour kinematics exactly and instantly. This
module builds the scenario grids that measure what happens when that assumption
is relaxed.

The design is a one-factor-at-a-time sensitivity sweep anchored at the exact
setting, plus a small number of combined operating points. A full cross product
of every axis would be 36 cells per scenario and is not needed to establish the
degradation slope.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..env.beacon import BeaconDegradation
from ..env.config import FanetConfig
from .evaluation_suite import EvaluationScenario


@dataclass(frozen=True)
class DegradationCell:
    """One beacon setting together with the sweep axis it belongs to."""

    axis: str
    label: str
    settings: BeaconDegradation


EXACT = BeaconDegradation()

#: Beacon interval in simulation steps. Episodes are 12-16 steps long and link
#: lifetimes are roughly 10-20 steps, so this range spans "fresh" to "useless".
STALENESS_PERIODS = (1, 2, 3, 5, 8, 10)

#: Probability that an individual beacon broadcast is not received.
LOSS_PROBABILITIES = (0.0, 0.1, 0.2, 0.3)

#: Gaussian sensor error, as a fraction of communication radius and max speed.
NOISE_LEVELS = (0.0, 0.05, 0.10, 0.15)

#: Combined points chosen to represent plausible deployments rather than
#: worst-case stress: a well-provisioned link, and a congested lossy one.
COMBINED_POINTS = (
    ("moderate", BeaconDegradation(2, 0.1, 0.05, 0.05)),
    ("severe", BeaconDegradation(5, 0.3, 0.15, 0.15)),
)


def staleness_axis() -> list[DegradationCell]:
    """Vary only the beacon period."""

    return [
        DegradationCell("staleness", f"period_{period}", BeaconDegradation(beacon_period=period))
        for period in STALENESS_PERIODS
    ]


def loss_axis() -> list[DegradationCell]:
    """Vary only the beacon loss probability."""

    return [
        DegradationCell("loss", f"loss_{loss:g}", BeaconDegradation(beacon_loss=loss))
        for loss in LOSS_PROBABILITIES
    ]


def noise_axis() -> list[DegradationCell]:
    """Vary only the sensor noise applied to position and velocity."""

    return [
        DegradationCell(
            "noise",
            f"noise_{sigma:g}",
            BeaconDegradation(position_noise_std=sigma, velocity_noise_std=sigma),
        )
        for sigma in NOISE_LEVELS
    ]


def combined_axis() -> list[DegradationCell]:
    """Combined operating points, with and without topology degradation."""

    cells: list[DegradationCell] = []
    for name, settings in COMBINED_POINTS:
        cells.append(DegradationCell("combined", name, settings))
        cells.append(
            DegradationCell(
                "combined_topology",
                f"{name}_topology",
                replace(settings, degrade_topology=True),
            )
        )
    return cells


def degradation_grid(*, include_combined: bool = True) -> list[DegradationCell]:
    """Return the deduplicated sensitivity grid.

    The exact setting appears once as the shared anchor of every axis.
    """

    cells = [DegradationCell("exact", "exact", EXACT)]
    seen = {EXACT}
    for axis in (staleness_axis(), loss_axis(), noise_axis()):
        for cell in axis:
            if cell.settings in seen:
                continue
            seen.add(cell.settings)
            cells.append(cell)
    if include_combined:
        for cell in combined_axis():
            if cell.settings in seen:
                continue
            seen.add(cell.settings)
            cells.append(cell)
    return cells


def apply_degradation(
    scenarios: list[EvaluationScenario], cell: DegradationCell
) -> list[EvaluationScenario]:
    """Rebuild an evaluation suite under one beacon setting.

    Scenario names are suffixed with the degradation label so that raw rows from
    different cells never collide in a merged archive.
    """

    return [
        EvaluationScenario(
            f"{scenario.name}__{cell.label}",
            _with_beacon(scenario.config, cell.settings),
            scenario.reset_options,
            f"{scenario.distribution}__{cell.axis}",
        )
        for scenario in scenarios
    ]


def _with_beacon(config: FanetConfig, settings: BeaconDegradation) -> FanetConfig:
    return replace(config, beacon=settings)
