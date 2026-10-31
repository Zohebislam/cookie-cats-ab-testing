"""Tests for the statistical code. Run with `python -m pytest` or `python -m unittest`."""

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from abtest import bayesian, frequentist, power, sequential, simulate  # noqa: E402
from abtest.data import CONTROL, TREATMENT, group_counts, load  # noqa: E402


class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load()

    def test_shape_and_groups(self):
        self.assertEqual(len(self.df), 90_189)
        counts = self.df["version"].value_counts()
        self.assertEqual(counts[CONTROL], 44_700)
        self.assertEqual(counts[TREATMENT], 45_489)

    def test_no_duplicates_or_missing(self):
        self.assertFalse(self.df["userid"].duplicated().any())
        self.assertEqual(self.df.isna().sum().sum(), 0)

    def test_metrics_are_boolean(self):
        self.assertEqual(self.df["retention_1"].dtype, bool)
        self.assertEqual(self.df["retention_7"].dtype, bool)

    def test_known_retention_rates(self):
        c = group_counts(self.df, "retention_7")
        self.assertAlmostEqual(c["x_a"] / c["n_a"], 0.190201, places=5)
        self.assertAlmostEqual(c["x_b"] / c["n_b"], 0.182000, places=5)


class TestFrequentist(unittest.TestCase):
    def test_ztest_matches_chi_square(self):
        # A two-sided pooled z-test is equivalent to a 2x2 chi-square test without continuity correction.
        x_a, n_a, x_b, n_b = 8502, 44700, 8279, 45489
        res = frequentist.two_prop_ztest(x_a, n_a, x_b, n_b)
        table = [[x_a, n_a - x_a], [x_b, n_b - x_b]]
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        self.assertAlmostEqual(res["z"] ** 2, chi2, places=6)
        self.assertAlmostEqual(res["p_value"], p, places=10)

    def test_ztest_ci_contains_diff_and_has_right_width(self):
        res = frequentist.two_prop_ztest(500, 1000, 550, 1000)
        self.assertLess(res["ci_low"], res["diff"])
        self.assertGreater(res["ci_high"], res["diff"])
        se = np.sqrt(0.5 * 0.5 / 1000 + 0.55 * 0.45 / 1000)
        self.assertAlmostEqual(res["ci_high"] - res["ci_low"], 2 * 1.959964 * se, places=5)

    def test_srm(self):
        res = frequentist.srm_test(44700, 45489)
        expected = 2 * ((45489 - 45094.5) ** 2) / 45094.5
        self.assertAlmostEqual(res["chi2"], expected, places=6)
        self.assertLess(res["p_value"], 0.01)
        self.assertGreater(frequentist.srm_test(5000, 5000)["p_value"], 0.99)

    def test_holm(self):
        adjusted = frequentist.holm([0.01, 0.04, 0.03, 0.005])
        # sorted: 0.005*4=0.02, 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> running max gives 0.06
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06, 0.02])

    def test_bootstrap_agrees_with_wald(self):
        boot = frequentist.bootstrap_diff(8502, 44700, 8279, 45489, n_boot=200_000, seed=1)
        lo, hi = frequentist.percentile_ci(boot)
        wald = frequentist.two_prop_ztest(8502, 44700, 8279, 45489)
        self.assertAlmostEqual(lo, wald["ci_low"], delta=0.0003)
        self.assertAlmostEqual(hi, wald["ci_high"], delta=0.0003)


class TestPower(unittest.TestCase):
    def test_power_at_mde_is_80_percent(self):
        m = power.mde(0.19, 45_000, 45_000)
        self.assertAlmostEqual(power.power_two_prop(0.19, 0.19 + m, 45_000, 45_000), 0.80, delta=0.01)

    def test_sample_size_gives_target_power(self):
        n = power.sample_size_per_arm(0.19, -0.01)
        self.assertAlmostEqual(power.power_two_prop(0.19, 0.18, n, n), 0.80, delta=0.005)

    def test_no_effect_means_power_equals_alpha(self):
        self.assertAlmostEqual(power.power_two_prop(0.2, 0.2, 10_000, 10_000), 0.05, places=6)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            power.sample_size_per_arm(0.2, 0)
        with self.assertRaises(ValueError):
            power.sample_size_per_arm(0.95, 0.1)


class TestSequential(unittest.TestCase):
    def test_obf_matches_published_values(self):
        # Lan-DeMets O'Brien-Fleming, 5 equal looks, two-sided alpha 0.05 (gsDesign / ldbounds).
        b = sequential.gsd_boundaries(np.arange(1, 6) / 5, 0.05, "obf")
        np.testing.assert_allclose(b, [4.8769, 3.3569, 2.6803, 2.2898, 2.0310], atol=2e-3)

    def test_single_look_is_fixed_test(self):
        for spending in ("obf", "pocock"):
            self.assertAlmostEqual(sequential.gsd_boundaries([1.0], 0.05, spending)[0], 1.959964, places=5)

    def test_type_one_error_by_simulation(self):
        rng = np.random.default_rng(0)
        info = np.array([0.1, 0.25, 0.4, 0.6, 0.8, 1.0])  # unequal spacing on purpose
        n = 300_000
        inc = rng.standard_normal((n, info.size)) * np.sqrt(np.diff(np.concatenate([[0], info])))
        z = np.cumsum(inc, axis=1) / np.sqrt(info)
        for spending in ("obf", "pocock"):
            b = sequential.gsd_boundaries(info, 0.05, spending)
            rate = (np.abs(z) >= b).any(axis=1).mean()
            self.assertAlmostEqual(rate, 0.05, delta=0.0025, msg=spending)

    def test_boundaries_shape(self):
        b = sequential.gsd_boundaries(np.arange(1, 15) / 14, 0.05, "obf")
        self.assertTrue(np.all(np.diff(b) < 0))  # OBF boundaries shrink over time
        self.assertGreater(b[-1], 1.96)          # final bar is a little above 1.96

    def test_invalid_info_fracs(self):
        for bad in ([0.5, 0.4, 1.0], [0.0, 1.0], [0.5, 1.2], []):
            with self.assertRaises(ValueError):
                sequential.gsd_boundaries(bad)

    def test_msprt_p_values_never_increase(self):
        log_lam = np.array([0.1, 1.5, 0.3, 2.5, 1.0])
        p = sequential.msprt_always_valid_p(log_lam)
        self.assertTrue(np.all(np.diff(p) <= 0))
        self.assertAlmostEqual(p[-1], np.exp(-2.5))

    def test_msprt_zero_when_no_difference(self):
        self.assertLess(sequential.msprt_log_lambda(100, 1000, 100, 1000), 0)


class TestBayesian(unittest.TestCase):
    def test_symmetric_case(self):
        res = bayesian.summarize(500, 1000, 500, 1000, draws=200_000)
        self.assertAlmostEqual(res["prob_b_better"], 0.5, delta=0.01)

    def test_normal_approximation_matches_sampling(self):
        res = bayesian.summarize(8502, 44700, 8279, 45489, draws=400_000)
        approx = float(bayesian.prob_b_better_normal(8502, 44700, 8279, 45489))
        self.assertAlmostEqual(res["prob_b_better"], approx, delta=0.0005)

    def test_posterior_params(self):
        self.assertEqual(bayesian.posterior_params(3, 10), (4.0, 8.0))


class TestSimulation(unittest.TestCase):
    def test_split_sizes(self):
        sizes = simulate.split_sizes(100, 7)
        self.assertEqual(sizes.sum(), 100)
        self.assertLessEqual(sizes.max() - sizes.min(), 1)

    def test_peeking_inflates_and_fixes_hold(self):
        res = simulate.sequential_study(0.2, 0.2, 20_000, 20_000, n_looks=10, n_sims=20_000, seed=3).set_index("method")
        se = 0.0016  # standard error of a 5% rate with 20,000 simulations
        self.assertAlmostEqual(res.loc["fixed", "reject_rate"], 0.05, delta=4 * se)
        self.assertAlmostEqual(res.loc["gsd_obf", "reject_rate"], 0.05, delta=4 * se)
        self.assertAlmostEqual(res.loc["gsd_pocock", "reject_rate"], 0.05, delta=4 * se)
        self.assertLessEqual(res.loc["msprt", "reject_rate"], 0.05)
        self.assertGreater(res.loc["naive_peeking", "reject_rate"], 0.15)

    def test_power_direction_only_counts_correct_sign(self):
        res = simulate.sequential_study(0.20, 0.19, 30_000, 30_000, n_looks=5, n_sims=5_000, seed=4).set_index("method")
        self.assertGreater(res.loc["fixed", "reject_rate"], 0.8)


if __name__ == "__main__":
    unittest.main()
