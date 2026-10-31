"""Bayesian analysis of a two-group conversion experiment (Beta-Binomial model)."""

from __future__ import annotations

import numpy as np


def posterior_params(x: int, n: int, prior: tuple[float, float] = (1.0, 1.0)) -> tuple[float, float]:
    """Beta posterior for a rate after x successes in n trials. Default prior is uniform."""
    return prior[0] + x, prior[1] + n - x


def summarize(
    x_a: int,
    n_a: int,
    x_b: int,
    n_b: int,
    prior: tuple[float, float] = (1.0, 1.0),
    draws: int = 400_000,
    seed: int = 0,
) -> dict:
    """Posterior summary of the difference p_B - p_A.

    expected_loss_* is how much retention you expect to give up, on average, if you ship that
    version and it turns out to be the worse one. It is the usual Bayesian decision metric.
    """
    rng = np.random.default_rng(seed)
    a_a, b_a = posterior_params(x_a, n_a, prior)
    a_b, b_b = posterior_params(x_b, n_b, prior)
    p_a = rng.beta(a_a, b_a, draws)
    p_b = rng.beta(a_b, b_b, draws)
    diff = p_b - p_a
    lift = diff / p_a
    lo, hi = np.quantile(diff, [0.025, 0.975])
    lift_lo, lift_hi = np.quantile(lift, [0.025, 0.975])
    return {
        "posterior_a": (a_a, b_a),
        "posterior_b": (a_b, b_b),
        "prob_b_better": float((diff > 0).mean()),
        "prob_a_better": float((diff < 0).mean()),
        "diff_mean": float(diff.mean()),
        "diff_cri_low": float(lo),
        "diff_cri_high": float(hi),
        "lift_cri_low": float(lift_lo),
        "lift_cri_high": float(lift_hi),
        "expected_loss_ship_b": float(np.maximum(p_a - p_b, 0).mean()),
        "expected_loss_ship_a": float(np.maximum(p_b - p_a, 0).mean()),
    }


def prob_b_better_normal(x_a, n_a, x_b, n_b, prior: tuple[float, float] = (1.0, 1.0)):
    """Fast normal approximation of P(p_B > p_A) from the Beta posteriors (vectorised).

    Accurate when each group has thousands of users, which is true at every look in the
    simulations. Used there because drawing posterior samples millions of times is slow.
    """
    from scipy import stats

    x_a, n_a, x_b, n_b = (np.asarray(v, dtype=float) for v in (x_a, n_a, x_b, n_b))
    a_a, b_a = prior[0] + x_a, prior[1] + n_a - x_a
    a_b, b_b = prior[0] + x_b, prior[1] + n_b - x_b
    mean_a, mean_b = a_a / (a_a + b_a), a_b / (a_b + b_b)
    var_a = a_a * b_a / ((a_a + b_a) ** 2 * (a_a + b_a + 1))
    var_b = a_b * b_b / ((a_b + b_b) ** 2 * (a_b + b_b + 1))
    return stats.norm.cdf((mean_b - mean_a) / np.sqrt(var_a + var_b))
