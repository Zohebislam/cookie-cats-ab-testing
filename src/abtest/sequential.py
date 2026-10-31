"""Ways to look at an experiment early without inflating false positives.

1. Group sequential design (GSD) with Lan-DeMets alpha spending.
   You plan K looks in advance. At each look the z-statistic is compared with a boundary
   that is stricter than 1.96 early on, so the total false positive rate stays at alpha.
   Boundaries are computed exactly by numerical integration (Armitage, McPherson and Rowe
   recursion), not by simulation.

2. mixture Sequential Probability Ratio Test (mSPRT, Johari et al. 2017).
   Gives "always valid" p-values: you can look as often as you like, even after every
   single user, and the false positive rate stays at or below alpha.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

# ---------------------------------------------------------------------------
# Alpha spending functions: how much of the total alpha may be used by information time t.
# ---------------------------------------------------------------------------


def obf_spending(t, alpha: float = 0.05):
    """O'Brien-Fleming-type: spends almost nothing early, keeps the final look near 1.96.

    Defined per side with alpha/2 on each side (the convention used by gsDesign and ldbounds),
    then doubled for the two-sided total.
    """
    t = np.asarray(t, dtype=float)
    per_side = 2 * stats.norm.sf(stats.norm.isf(alpha / 4) / np.sqrt(t))
    return 2 * per_side


def pocock_spending(t, alpha: float = 0.05):
    """Pocock-type: spends alpha evenly, so early stopping is easier but the final bar is higher."""
    t = np.asarray(t, dtype=float)
    return alpha * np.log(1 + (np.e - 1) * t)


SPENDING = {"obf": obf_spending, "pocock": pocock_spending}


def gsd_boundaries(info_fracs, alpha: float = 0.05, spending: str = "obf", n_grid: int = 2001) -> np.ndarray:
    """Two-sided z-boundaries for a group sequential test.

    Parameters
    ----------
    info_fracs : increasing fractions of the planned sample at each look, ending at 1.
    alpha      : overall two-sided false positive rate.
    spending   : "obf" (O'Brien-Fleming-type) or "pocock" (Pocock-type).

    Returns
    -------
    Array of boundaries c_k. Stop and reject at look k if |Z_k| >= c_k.
    """
    t = np.asarray(info_fracs, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("info_fracs must be a non-empty 1-D sequence")
    if np.any(t <= 0) or np.any(t > 1) or np.any(np.diff(t) <= 0):
        raise ValueError("info_fracs must be strictly increasing values in (0, 1]")
    if spending not in SPENDING:
        raise ValueError(f"spending must be one of {sorted(SPENDING)}")

    cumulative = SPENDING[spending](t, alpha)
    if np.isclose(t[-1], 1.0):
        cumulative[-1] = alpha  # spend exactly alpha by the final look
    increments = np.diff(np.concatenate([[0.0], cumulative]))

    bounds = np.empty(t.size)

    # Work with the score S_k = Z_k * sqrt(t_k), which behaves like Brownian motion B(t_k).
    bounds[0] = stats.norm.isf(increments[0] / 2)
    edge = bounds[0] * np.sqrt(t[0])
    s = np.linspace(-edge, edge, n_grid)
    dens = stats.norm.pdf(s, scale=np.sqrt(t[0]))  # density of S_1 among paths that kept going

    for k in range(1, t.size):
        sd = np.sqrt(t[k] - t[k - 1])
        weights = np.full(n_grid, s[1] - s[0])
        weights[[0, -1]] *= 0.5  # trapezoid rule
        mass = dens * weights
        root_t = np.sqrt(t[k])

        def crossing_prob(c: float) -> float:
            b = c * root_t
            return float(np.sum(mass * (stats.norm.sf((b - s) / sd) + stats.norm.cdf((-b - s) / sd))))

        target = increments[k]
        if target <= 0:
            bounds[k] = np.inf
            continue
        bounds[k] = optimize.brentq(lambda c: crossing_prob(c) - target, 1e-8, 60.0, xtol=1e-12)

        edge = bounds[k] * root_t
        s_new = np.linspace(-edge, edge, n_grid)
        kernel = stats.norm.pdf((s_new[:, None] - s[None, :]) / sd) / sd
        dens = kernel @ mass
        s = s_new

    return bounds


# ---------------------------------------------------------------------------
# mSPRT for a difference in two proportions
# ---------------------------------------------------------------------------


def msprt_log_lambda(x_a, n_a, x_b, n_b, tau: float = 0.01):
    """Log of the mixture likelihood ratio for H0: p_A = p_B.

    Uses a normal mixing distribution N(0, tau^2) over the true difference. tau is the scale of
    effects you expect (0.01 = one percentage point). Works element-wise on arrays.
    """
    x_a, n_a, x_b, n_b = (np.asarray(v, dtype=float) for v in (x_a, n_a, x_b, n_b))
    pooled = (x_a + x_b) / (n_a + n_b)
    var = pooled * (1 - pooled) * (1 / n_a + 1 / n_b)
    theta = x_b / n_b - x_a / n_a
    tau2 = tau**2
    with np.errstate(divide="ignore", invalid="ignore"):
        log_lam = 0.5 * np.log(var / (var + tau2)) + tau2 * theta**2 / (2 * var * (var + tau2))
    return np.where(var > 0, log_lam, 0.0)


def msprt_always_valid_p(log_lambda_path) -> np.ndarray:
    """Always-valid p-value at each look: min(1, 1 / running max of the likelihood ratio)."""
    running = np.maximum.accumulate(np.asarray(log_lambda_path, dtype=float), axis=-1)
    return np.minimum(1.0, np.exp(-running))
