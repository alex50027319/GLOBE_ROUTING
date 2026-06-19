"""Seed-level statistics and 95% confidence intervals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import numpy as np


T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


@dataclass(frozen=True)
class Statistic:
    count: int
    mean: float
    standard_deviation: float
    ci95_low: float
    ci95_high: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def summarize_values(values: list[float]) -> Statistic:
    """Compute sample standard deviation and two-sided 95% t interval."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("statistics require at least one finite value")
    mean = float(np.mean(finite))
    if finite.size == 1:
        return Statistic(1, mean, 0.0, mean, mean)
    standard_deviation = float(np.std(finite, ddof=1))
    degrees = int(finite.size - 1)
    critical = T_CRITICAL_975.get(degrees, 1.96)
    half_width = critical * standard_deviation / sqrt(finite.size)
    return Statistic(
        count=int(finite.size),
        mean=mean,
        standard_deviation=standard_deviation,
        ci95_low=mean - half_width,
        ci95_high=mean + half_width,
    )
