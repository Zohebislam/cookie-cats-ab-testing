"""End-to-end analysis. Returns plain dictionaries and tables so results can be saved and tested."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import bayesian, frequentist, power, sequential, simulate
from .data import CONTROL, METRICS, TREATMENT, group_counts, quality_report

ALPHA = 0.05
N_LOOKS = 14          # simulated plan: one look per day for two weeks
MSPRT_TAU = 0.01      # mSPRT mixing scale: effects around 1 percentage point
SEED = 2026


def main_analysis(df: pd.DataFrame, n_boot: int = 100_000) -> dict:
    """Data checks, the two retention tests, power, Bayesian view and the rounds metric."""
    out: dict = {"quality": quality_report(df)}

    q = out["quality"]
    out["srm"] = frequentist.srm_test(q["players_control"], q["players_treatment"])

    tests = {}
    for i, metric in enumerate(METRICS):
        c = group_counts(df, metric)
        z = frequentist.two_prop_ztest(c["x_a"], c["n_a"], c["x_b"], c["n_b"], ALPHA)
        boot = frequentist.bootstrap_diff(c["x_a"], c["n_a"], c["x_b"], c["n_b"], n_boot, SEED + i)
        lo, hi = frequentist.percentile_ci(boot)
        tests[metric] = {**c, **z, "boot_ci_low": lo, "boot_ci_high": hi, "boot_share_below_zero": float((boot < 0).mean())}
    adjusted = frequentist.holm([tests[m]["p_value"] for m in METRICS])
    for metric, p_adj in zip(METRICS, adjusted):
        tests[metric]["p_holm"] = p_adj
    out["tests"] = tests

    # Sample ratio mismatch sensitivity. gate_40 has more players than a 50/50 split predicts.
    # Worst case for the conclusion: every excess gate_40 player is a non-returning player who
    # should not have been counted. Remove them and re-test.
    excess = q["players_treatment"] - q["players_control"]
    sens = {}
    for metric in METRICS:
        t = tests[metric]
        worst = frequentist.two_prop_ztest(t["x_a"], t["n_a"], t["x_b"], t["n_b"] - excess, ALPHA)
        sens[metric] = {"removed_players": excess, "diff": worst["diff"], "p_value": worst["p_value"],
                        "ci_low": worst["ci_low"], "ci_high": worst["ci_high"]}
    out["srm_worst_case"] = sens

    pw = {}
    for metric in METRICS:
        t = tests[metric]
        pw[metric] = {
            "baseline": t["p_a"],
            "mde_80": power.mde(t["p_a"], t["n_a"], t["n_b"], ALPHA, 0.80),
            "power_at_observed_effect": power.power_two_prop(t["p_a"], t["p_b"], t["n_a"], t["n_b"], ALPHA),
            "n_per_arm_for_observed_effect": power.sample_size_per_arm(t["p_a"], t["diff"], ALPHA, 0.80),
        }
    out["power"] = pw

    out["bayes"] = {
        m: bayesian.summarize(tests[m]["x_a"], tests[m]["n_a"], tests[m]["x_b"], tests[m]["n_b"], seed=SEED + 10 + i)
        for i, m in enumerate(METRICS)
    }

    rounds_a = df.loc[df["version"] == CONTROL, "sum_gamerounds"].to_numpy()
    rounds_b = df.loc[df["version"] == TREATMENT, "sum_gamerounds"].to_numpy()
    out["rounds"] = frequentist.rounds_comparison(rounds_a, rounds_b, seed=SEED)

    # How the real result looks under the sequential methods, at the final look.
    t7 = tests["retention_7"]
    info = np.arange(1, N_LOOKS + 1) / N_LOOKS
    final_obf = float(sequential.gsd_boundaries(info, ALPHA, "obf")[-1])
    final_poc = float(sequential.gsd_boundaries(info, ALPHA, "pocock")[-1])
    log_lam = float(sequential.msprt_log_lambda(t7["x_a"], t7["n_a"], t7["x_b"], t7["n_b"], MSPRT_TAU))
    out["sequential_on_real_data"] = {
        "z_retention_7": t7["z"],
        "obf_final_boundary": final_obf,
        "pocock_final_boundary": final_poc,
        "passes_obf": abs(t7["z"]) >= final_obf,
        "passes_pocock": abs(t7["z"]) >= final_poc,
        "msprt_p_at_end": float(min(1.0, np.exp(-log_lam))),
    }
    return out


def peeking_studies(df: pd.DataFrame, main: dict, n_sims: int = 50_000, n_perm: int = 2_000) -> dict:
    """All simulation studies of peeking and the fixes."""
    t7 = main["tests"]["retention_7"]
    n_a, n_b = t7["n_a"], t7["n_b"]
    pooled = (t7["x_a"] + t7["x_b"]) / (n_a + n_b)

    info = (np.cumsum(simulate.split_sizes(n_a, N_LOOKS)) + np.cumsum(simulate.split_sizes(n_b, N_LOOKS))) / (n_a + n_b)
    boundaries = pd.DataFrame(
        {
            "day": np.arange(1, N_LOOKS + 1),
            "info_fraction": info,
            "naive": stats.norm.isf(ALPHA / 2),
            "obf": sequential.gsd_boundaries(info, ALPHA, "obf"),
            "pocock": sequential.gsd_boundaries(info, ALPHA, "pocock"),
        }
    )

    aa = simulate.sequential_study(pooled, pooled, n_a, n_b, N_LOOKS, n_sims, ALPHA, MSPRT_TAU, seed=SEED + 100)
    power_study = simulate.sequential_study(t7["p_a"], t7["p_b"], n_a, n_b, N_LOOKS, n_sims, ALPHA, MSPRT_TAU, seed=SEED + 200)
    looks = simulate.fpr_vs_looks(pooled, (n_a + n_b) // 2, n_sims=min(n_sims, 20_000), alpha=ALPHA, seed=SEED + 300)
    perm = simulate.permutation_aa(df["retention_7"].to_numpy(), n_b, N_LOOKS, n_perm, ALPHA, seed=SEED + 400)
    paths = simulate.example_paths(pooled, (n_a + n_b) // 2, N_LOOKS, seed=SEED + 500)

    return {
        "settings": {"n_sims": n_sims, "n_perm": n_perm, "looks": N_LOOKS, "alpha": ALPHA, "msprt_tau": MSPRT_TAU, "base_rate": pooled},
        "boundaries": boundaries,
        "aa": aa,
        "power": power_study,
        "fpr_vs_looks": looks,
        "permutation_aa": perm,
        "example_paths": paths,
    }
