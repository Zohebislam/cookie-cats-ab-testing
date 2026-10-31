"""Power, sample size and minimum detectable effect for a two-proportion test.

All formulas use the normal approximation, which is very accurate at the sample sizes here
(tens of thousands of players per group).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _z(alpha: float, power: float) -> tuple[float, float]:
    return stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)


def mde(p_base: float, n_a: int, n_b: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Smallest absolute difference the test detects with the given power (two-sided)."""
    z_a, z_b = _z(alpha, power)
    se = np.sqrt(p_base * (1 - p_base) * (1 / n_a + 1 / n_b))
    return float((z_a + z_b) * se)


def sample_size_per_arm(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Players needed in each group to detect an absolute change of `mde_abs`."""
    if mde_abs == 0:
        raise ValueError("mde_abs must be non-zero")
    p_alt = p_base + mde_abs
    if not 0 < p_alt < 1:
        raise ValueError("p_base + mde_abs must be between 0 and 1")
    z_a, z_b = _z(alpha, power)
    var = p_base * (1 - p_base) + p_alt * (1 - p_alt)
    return int(np.ceil((z_a + z_b) ** 2 * var / mde_abs**2))


def power_two_prop(p_a: float, p_b: float, n_a: int, n_b: int, alpha: float = 0.05) -> float:
    """Probability a two-sided z-test rejects when the true rates are p_a and p_b."""
    z_crit = stats.norm.ppf(1 - alpha / 2)
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    shift = abs(p_b - p_a) / se
    return float(stats.norm.cdf(shift - z_crit) + stats.norm.cdf(-shift - z_crit))
