"""Classical (fixed-horizon) tests used in the main analysis."""

from __future__ import annotations

import numpy as np
from scipy import stats


def srm_test(n_a: int, n_b: int, expected_share_a: float = 0.5) -> dict:
    """Sample ratio mismatch check: chi-square goodness of fit against the planned split.

    A small p-value means the groups are not the size the randomization should produce,
    which can point to a bug in assignment or logging.
    """
    n = n_a + n_b
    expected = [n * expected_share_a, n * (1 - expected_share_a)]
    chi2, p = stats.chisquare([n_a, n_b], f_exp=expected)
    return {
        "n_a": n_a,
        "n_b": n_b,
        "share_a": n_a / n,
        "chi2": float(chi2),
        "p_value": float(p),
    }


def two_prop_ztest(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> dict:
    """Two-sided z-test for a difference in proportions (B minus A).

    The test statistic uses the pooled standard error (correct under the null hypothesis of
    no difference). The confidence interval uses the unpooled standard error, which does not
    assume the null is true.
    """
    p_a, p_b = x_a / n_a, x_b / n_b
    diff = p_b - p_a

    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool
    p_value = 2 * stats.norm.sf(abs(z))

    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    return {
        "p_a": p_a,
        "p_b": p_b,
        "diff": diff,
        "rel_lift": diff / p_a,
        "z": float(z),
        "p_value": float(p_value),
        "ci_low": float(diff - z_crit * se_unpooled),
        "ci_high": float(diff + z_crit * se_unpooled),
    }


def bootstrap_diff(
    x_a: int, n_a: int, x_b: int, n_b: int, n_boot: int = 100_000, seed: int = 0
) -> np.ndarray:
    """Bootstrap distribution of p_B - p_A.

    Resampling n yes/no outcomes with replacement gives a Binomial(n, p_hat) count, so the
    bootstrap can be drawn directly from the binomial. This is exactly the same as resampling
    rows, just much faster.
    """
    rng = np.random.default_rng(seed)
    boot_a = rng.binomial(n_a, x_a / n_a, size=n_boot) / n_a
    boot_b = rng.binomial(n_b, x_b / n_b, size=n_boot) / n_b
    return boot_b - boot_a


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def rounds_comparison(rounds_a: np.ndarray, rounds_b: np.ndarray, n_boot: int = 2_000, seed: int = 0) -> dict:
    """Compare game rounds played, a heavily skewed count.

    Uses the Mann-Whitney U test (rank based, so extreme players do not dominate) and a
    bootstrap confidence interval for the difference in medians.
    """
    u = stats.mannwhitneyu(rounds_b, rounds_a, alternative="two-sided")
    rng = np.random.default_rng(seed)
    med_diffs = np.empty(n_boot)
    for i in range(n_boot):
        med_diffs[i] = np.median(rng.choice(rounds_b, rounds_b.size)) - np.median(
            rng.choice(rounds_a, rounds_a.size)
        )
    lo, hi = percentile_ci(med_diffs)
    # Probability that a random B player played more rounds than a random A player (ties split).
    prob_superiority = u.statistic / (rounds_a.size * rounds_b.size)
    return {
        "median_a": float(np.median(rounds_a)),
        "median_b": float(np.median(rounds_b)),
        "median_diff": float(np.median(rounds_b) - np.median(rounds_a)),
        "median_diff_ci_low": lo,
        "median_diff_ci_high": hi,
        "mann_whitney_p": float(u.pvalue),
        "prob_b_plays_more": float(prob_superiority),
    }
