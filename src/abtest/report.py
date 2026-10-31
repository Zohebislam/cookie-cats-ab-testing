"""Write reports/FINDINGS.md. Every number comes from the analysis results."""

from __future__ import annotations


def _pct(x: float, d: int = 1) -> str:
    return f"{x * 100:.{d}f}%"


def _pts(x: float, d: int = 2) -> str:
    return f"{x * 100:+.{d}f} pts"


def _p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def build(main: dict, peek: dict) -> str:
    q, srm = main["quality"], main["srm"]
    t1, t7 = main["tests"]["retention_1"], main["tests"]["retention_7"]
    pw1, pw7 = main["power"]["retention_1"], main["power"]["retention_7"]
    b1, b7 = main["bayes"]["retention_1"], main["bayes"]["retention_7"]
    w7 = main["srm_worst_case"]["retention_7"]
    rounds = main["rounds"]
    seq = main["sequential_on_real_data"]
    aa = peek["aa"].set_index("method")
    pw = peek["power"].set_index("method")
    perm = peek["permutation_aa"]
    looks = peek["fpr_vs_looks"].set_index("looks")["false_positive_rate"]
    s = peek["settings"]
    obf_saving = 1 - pw.loc["gsd_obf", "avg_sample_used"]

    return f"""# Findings: Cookie Cats gate experiment

## Recommendation

**Keep the gate at level 30.** Moving it to level 40 lowered 7-day retention from
{_pct(t7['p_a'], 2)} to {_pct(t7['p_b'], 2)} ({_pts(t7['diff'])}, a {abs(t7['rel_lift']):.1%} relative drop,
p = {_p(t7['p_value'])}). Day-1 retention and game rounds moved in the same direction but were not
significant. Nothing in the data suggests level 40 is better on any metric.

One caveat: the groups are not the size a 50/50 split should produce (see Data checks). In the
worst case the day-7 result weakens to borderline. The direction of the decision does not change,
but the logging behind the split should be checked before the next experiment.

## Data checks

| Check | Result |
|---|---|
| Players | {q['rows']:,} ({q['players_control']:,} gate_30, {q['players_treatment']:,} gate_40) |
| Duplicate player IDs | {q['duplicate_userids']} |
| Missing values | {q['missing_values']} |
| Players with 0 rounds | {q['zero_round_players']:,} ({_pct(q['zero_round_share'])}) |
| Most rounds by one player | {q['rounds_max']:,} (next highest: {q['rounds_second_max']:,}) |
| Sample ratio (gate_30 share) | {_pct(srm['share_a'], 2)}, chi-square p = {_p(srm['p_value'])} |

**Sample ratio mismatch.** With a fair 50/50 coin, a gap this large happens less than
{max(1, round(srm['p_value'] * 100))}% of the time. It can mean some players were logged or assigned
incorrectly. Worst-case test: drop the {w7['removed_players']:,} extra gate_40 players and assume all of
them were non-returning players. Day-7 retention then differs by {_pts(w7['diff'])}
(p = {_p(w7['p_value'])}). That is the most favorable possible case for level 40, and it still does
not show level 40 doing better.

The player with {q['rounds_max']:,} rounds is about {q['rounds_max'] / q['rounds_second_max']:.0f}x the next highest and is probably a logging
error or a bot. It has no effect on retention and the rounds analysis below uses rank-based
methods, so it is kept.

## Main results

| Metric | gate_30 | gate_40 | Difference | 95% CI | p-value | Holm p |
|---|---|---|---|---|---|---|
| Day-1 retention | {_pct(t1['p_a'], 2)} | {_pct(t1['p_b'], 2)} | {_pts(t1['diff'])} | [{t1['ci_low'] * 100:.2f}, {t1['ci_high'] * 100:.2f}] | {_p(t1['p_value'])} | {_p(t1['p_holm'])} |
| Day-7 retention | {_pct(t7['p_a'], 2)} | {_pct(t7['p_b'], 2)} | {_pts(t7['diff'])} | [{t7['ci_low'] * 100:.2f}, {t7['ci_high'] * 100:.2f}] | {_p(t7['p_value'])} | {_p(t7['p_holm'])} |

- Two-sided two-proportion z-tests, alpha = 0.05. Holm correction because two metrics were tested.
- Bootstrap (100,000 resamples) agrees: day-7 CI [{t7['boot_ci_low'] * 100:.2f}, {t7['boot_ci_high'] * 100:.2f}] pts;
  {_pct(t7['boot_share_below_zero'], 2)} of resamples show a drop.
- Game rounds: median {rounds['median_a']:.0f} vs {rounds['median_b']:.0f}, Mann-Whitney p = {_p(rounds['mann_whitney_p'])}.
  A random gate_40 player out-plays a random gate_30 player {_pct(rounds['prob_b_plays_more'])} of the time
  (50% would mean no difference).

**Was the test big enough?** With about 45,000 players per group, the smallest change detectable
with 80% power is {pw7['mde_80'] * 100:.2f} pts for day-7 retention and {pw1['mde_80'] * 100:.2f} pts for day-1.
The day-7 effect is larger than its threshold (power {_pct(pw7['power_at_observed_effect'], 0)}). The day-1 effect is
smaller than its threshold (power {_pct(pw1['power_at_observed_effect'], 0)}), so its non-significant result means
"too small to see with this sample", not "no effect". Detecting it reliably would take about
{pw1['n_per_arm_for_observed_effect']:,} players per group.

## Bayesian view

With uniform priors, there is a **{_pct(b7['prob_a_better'], 2)}** probability that gate_30 has the higher
day-7 retention (95% credible interval for the difference [{b7['diff_cri_low'] * 100:.2f}, {b7['diff_cri_high'] * 100:.2f}] pts).
For day-1 it is {_pct(b1['prob_a_better'], 1)}.

Expected loss from shipping gate_40: {b7['expected_loss_ship_b'] * 100:.3f} pts of day-7 retention.
Expected loss from keeping gate_30: {b7['expected_loss_ship_a'] * 100:.5f} pts. Keeping gate_30 is the
low-risk choice.

## The peeking problem

The dataset has no timestamps, so a two-week test was simulated: players arrive evenly over
{s['looks']} days and the analyst checks once a day. Base rate = the real pooled day-7 retention
({_pct(s['base_rate'], 1)}); group sizes = the real ones. {s['n_sims']:,} simulated experiments per scenario.

**False positives when nothing changed (A/A tests):**

| Rule | False positive rate | Stops at day (median, when it stops) |
|---|---|---|
| Look once at the end | {_pct(aa.loc['fixed', 'reject_rate'])} | 14 |
| Check daily, stop at first p < 0.05 | **{_pct(aa.loc['naive_peeking', 'reject_rate'])}** | {aa.loc['naive_peeking', 'median_stop_day_if_rejected']:.0f} |
| Bayesian, stop when P(better) > 97.5% | **{_pct(aa.loc['bayes_peeking', 'reject_rate'])}** | {aa.loc['bayes_peeking', 'median_stop_day_if_rejected']:.0f} |
| Bayesian, stop when P(better) > 95% | **{_pct(aa.loc['bayes_peeking_95', 'reject_rate'])}** | {aa.loc['bayes_peeking_95', 'median_stop_day_if_rejected']:.0f} |
| Group sequential, O'Brien-Fleming | {_pct(aa.loc['gsd_obf', 'reject_rate'])} | {aa.loc['gsd_obf', 'median_stop_day_if_rejected']:.0f} |
| Group sequential, Pocock | {_pct(aa.loc['gsd_pocock', 'reject_rate'])} | {aa.loc['gsd_pocock', 'median_stop_day_if_rejected']:.0f} |
| mSPRT (always-valid p-values) | {_pct(aa.loc['msprt', 'reject_rate'])} | {aa.loc['msprt', 'median_stop_day_if_rejected']:.0f} |

- Checking daily for two weeks turns a promised 5% false positive rate into about
  {_pct(looks.loc[14], 0)}. Checking 100 times pushes it to {_pct(looks.loc[100], 0)}.
- **Bayesian methods are not immune.** A probability threshold used as a stopping rule behaves
  almost exactly like a p-value threshold. The common 95% rule is worse still, because it already
  allows about 10% false positives with a single look.
- **Check on the real data:** shuffling the real Cookie Cats outcomes between groups
  ({perm['n_sims']:,} times) gives {_pct(perm['naive_peeking']['rate'])} false positives for daily peeking,
  {_pct(perm['fixed']['rate'])} for one look and {_pct(perm['gsd_obf']['rate'])} for O'Brien-Fleming. The simulation holds.

**When the effect is real** (true drop = the observed {abs(t7['diff']) * 100:.2f} pts):

| Rule | Power | Share of planned sample used |
|---|---|---|
| Look once at the end | {_pct(pw.loc['fixed', 'reject_rate'])} | 100% |
| Group sequential, O'Brien-Fleming | {_pct(pw.loc['gsd_obf', 'reject_rate'])} | {_pct(pw.loc['gsd_obf', 'avg_sample_used'], 0)} |
| Group sequential, Pocock | {_pct(pw.loc['gsd_pocock', 'reject_rate'])} | {_pct(pw.loc['gsd_pocock', 'avg_sample_used'], 0)} |
| mSPRT | {_pct(pw.loc['msprt', 'reject_rate'])} | {_pct(pw.loc['msprt', 'avg_sample_used'], 0)} |

**Takeaway:** an O'Brien-Fleming plan keeps the 5% promise, loses about
{(pw.loc['fixed', 'reject_rate'] - pw.loc['gsd_obf', 'reject_rate']) * 100:.1f} points of power, and ends the
test about {obf_saving:.0%} sooner on average when the effect is real. mSPRT lets you look any time,
but that freedom costs power at this sample size. Naive and Bayesian peeking look faster and more
powerful only because they also "find" effects that are not there.

**The real result under these plans.** The full-data day-7 z-statistic is {seq['z_retention_7']:.2f}.
That clears the final O'Brien-Fleming boundary ({seq['obf_final_boundary']:.2f}) and the final Pocock
boundary ({seq['pocock_final_boundary']:.2f}). The mSPRT always-valid p-value at the end is
{seq['msprt_p_at_end']:.3f}. On the data as logged, the conclusion holds under every valid method
(the sample ratio caveat above still applies).

## Limitations

- No timestamps, so the timing of looks is simulated, not real.
- The sample ratio mismatch could not be traced to a cause with this data.
- Retention is the only outcome; there is no revenue or purchase data.
- The game rounds count is measured after the gate, so it cannot be used to check that the groups
  were similar before the experiment.
"""
