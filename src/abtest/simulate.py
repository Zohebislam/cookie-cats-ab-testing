"""Monte Carlo studies: what peeking does, and whether the fixes work.

The dataset has no timestamps, so the experiment's timeline is simulated: players are assumed
to arrive evenly over 14 days and the analyst checks results once a day. Two engines are used:

* Parametric: each day's retained players are Binomial draws at a chosen rate. Fast, so it is
  used for the large studies (tens of thousands of simulated experiments).
* Permutation: the real Cookie Cats outcomes are shuffled between groups and put in a random
  arrival order. Any difference is then pure chance (an A/A test on real data). Used to check
  that the parametric results hold on the actual data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .bayesian import prob_b_better_normal
from .sequential import gsd_boundaries, msprt_log_lambda

METHOD_LABELS = {
    "fixed": "Fixed horizon (look once at the end)",
    "naive_peeking": "Naive peeking (stop the first day p < 0.05)",
    "gsd_obf": "Group sequential, O'Brien-Fleming",
    "gsd_pocock": "Group sequential, Pocock",
    "msprt": "mSPRT (always-valid p-values)",
    "bayes_peeking": "Bayesian peeking (stop when P(better) > 97.5%)",
    "bayes_peeking_95": "Bayesian peeking (stop when P(better) > 95%)",
}


def split_sizes(n: int, k: int) -> np.ndarray:
    """Split n users into k nearly equal daily batches."""
    base, extra = divmod(n, k)
    return np.array([base + (i < extra) for i in range(k)])


def _pooled_z(xa, na, xb, nb):
    pooled = (xa + xb) / (na + nb)
    se = np.sqrt(pooled * (1 - pooled) * (1 / na + 1 / nb))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (xb / nb - xa / na) / se
    return np.where(se > 0, z, 0.0)


def _first_true(mask: np.ndarray) -> np.ndarray:
    """Index of the first True in each row, or -1 if none."""
    any_true = mask.any(axis=1)
    return np.where(any_true, mask.argmax(axis=1), -1)


def sequential_study(
    p_a: float,
    p_b: float,
    n_a: int,
    n_b: int,
    n_looks: int = 14,
    n_sims: int = 20_000,
    alpha: float = 0.05,
    tau: float = 0.01,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate many experiments and apply every decision rule to the same data.

    If p_a == p_b, the rejection rate is the false positive rate.
    If p_a != p_b, the rejection rate (in the true direction) is the power.

    The main Bayesian rule declares a winner when P(B better) > 97.5% or < 2.5%. With a
    single look at the end that matches a two-sided p < 0.05 test, so any extra false
    positives come from peeking alone. The popular 95% rule is also reported; note that it
    already allows about 10% false positives even with one look.
    """
    rng = np.random.default_rng(seed)
    inc_a, inc_b = split_sizes(n_a, n_looks), split_sizes(n_b, n_looks)
    xa = np.cumsum(rng.binomial(inc_a, p_a, size=(n_sims, n_looks)), axis=1)
    xb = np.cumsum(rng.binomial(inc_b, p_b, size=(n_sims, n_looks)), axis=1)
    na, nb = np.cumsum(inc_a), np.cumsum(inc_b)
    info = (na + nb) / (n_a + n_b)

    z = _pooled_z(xa, na, xb, nb)
    z_crit = stats.norm.isf(alpha / 2)
    b_obf = gsd_boundaries(info, alpha, "obf")
    b_poc = gsd_boundaries(info, alpha, "pocock")
    log_lam = msprt_log_lambda(xa, na, xb, nb, tau)
    p_better = prob_b_better_normal(xa, na, xb, nb)

    crossed = {
        "fixed": np.zeros_like(z, dtype=bool),
        "naive_peeking": np.abs(z) >= z_crit,
        "gsd_obf": np.abs(z) >= b_obf,
        "gsd_pocock": np.abs(z) >= b_poc,
        "msprt": log_lam >= np.log(1 / alpha),
        "bayes_peeking": (p_better >= 0.975) | (p_better <= 0.025),
        "bayes_peeking_95": (p_better >= 0.95) | (p_better <= 0.05),
    }
    crossed["fixed"][:, -1] = np.abs(z[:, -1]) >= z_crit

    true_sign = np.sign(p_b - p_a)
    rows = []
    for method, mask in crossed.items():
        stop = _first_true(mask)
        rejected = stop >= 0
        stop_idx = np.where(rejected, stop, n_looks - 1)
        z_at_stop = z[np.arange(n_sims), stop_idx]
        if true_sign == 0:
            hit = rejected
        else:
            hit = rejected & (np.sign(z_at_stop) == true_sign)
        rate = hit.mean()
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS[method],
                "reject_rate": float(rate),
                "reject_rate_se": float(np.sqrt(rate * (1 - rate) / n_sims)),
                "avg_sample_used": float(info[stop_idx].mean()),
                "median_stop_day_if_rejected": float(np.median(stop[rejected]) + 1) if rejected.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def fpr_vs_looks(
    p: float, n_per_arm: int, looks=(1, 2, 3, 5, 7, 10, 14, 21, 30, 50, 100), n_sims: int = 20_000, alpha: float = 0.05, seed: int = 0
) -> pd.DataFrame:
    """False positive rate of naive peeking as the number of equally spaced looks grows."""
    rows = []
    z_crit = stats.norm.isf(alpha / 2)
    for i, k in enumerate(looks):
        rng = np.random.default_rng(seed + i)
        inc = split_sizes(n_per_arm, k)
        xa = np.cumsum(rng.binomial(inc, p, size=(n_sims, k)), axis=1)
        xb = np.cumsum(rng.binomial(inc, p, size=(n_sims, k)), axis=1)
        n = np.cumsum(inc)
        z = _pooled_z(xa, n, xb, n)
        fpr = (np.abs(z) >= z_crit).any(axis=1).mean()
        rows.append({"looks": k, "false_positive_rate": float(fpr), "se": float(np.sqrt(fpr * (1 - fpr) / n_sims))})
    return pd.DataFrame(rows)


def permutation_aa(outcomes: np.ndarray, n_b: int, n_looks: int = 14, n_sims: int = 2_000, alpha: float = 0.05, seed: int = 0) -> dict:
    """A/A tests on the real outcomes: shuffle group labels and arrival order, then peek daily."""
    rng = np.random.default_rng(seed)
    y = np.asarray(outcomes, dtype=np.int64)
    n_total = y.size
    labels = np.zeros(n_total, dtype=bool)
    labels[:n_b] = True
    ends = np.cumsum(split_sizes(n_total, n_looks)) - 1
    z_crit = stats.norm.isf(alpha / 2)
    b_obf = None

    naive = np.empty(n_sims, dtype=bool)
    fixed = np.empty(n_sims, dtype=bool)
    obf = np.empty(n_sims, dtype=bool)
    for s in range(n_sims):
        is_b = rng.permutation(labels)       # random assignment, same group sizes as the real test
        yy = y[rng.permutation(n_total)]     # random arrival order
        nb = np.cumsum(is_b)[ends]
        na = (ends + 1) - nb
        xb = np.cumsum(yy * is_b)[ends]
        xa = np.cumsum(yy * ~is_b)[ends]
        z = _pooled_z(xa, na, xb, nb)
        if b_obf is None:
            b_obf = gsd_boundaries((ends + 1) / n_total, alpha, "obf")
        naive[s] = (np.abs(z) >= z_crit).any()
        fixed[s] = abs(z[-1]) >= z_crit
        obf[s] = (np.abs(z) >= b_obf).any()

    def rate(v):
        r = v.mean()
        return {"rate": float(r), "se": float(np.sqrt(r * (1 - r) / n_sims))}

    return {"n_sims": n_sims, "fixed": rate(fixed), "naive_peeking": rate(naive), "gsd_obf": rate(obf)}


def example_paths(p: float, n_per_arm: int, n_looks: int = 14, n_paths: int = 40, seed: int = 7) -> np.ndarray:
    """A handful of A/A z-statistic paths for the explanatory chart."""
    rng = np.random.default_rng(seed)
    inc = split_sizes(n_per_arm, n_looks)
    xa = np.cumsum(rng.binomial(inc, p, size=(n_paths, n_looks)), axis=1)
    xb = np.cumsum(rng.binomial(inc, p, size=(n_paths, n_looks)), axis=1)
    n = np.cumsum(inc)
    return _pooled_z(xa, n, xb, n)
