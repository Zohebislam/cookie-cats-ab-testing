# Findings: Cookie Cats gate experiment

## Recommendation

**Keep the gate at level 30.** Moving it to level 40 lowered 7-day retention from
19.02% to 18.20% (-0.82 pts, a 4.3% relative drop,
p = 0.0016). Day-1 retention and game rounds moved in the same direction but were not
significant. Nothing in the data suggests level 40 is better on any metric.

One caveat: the groups are not the size a 50/50 split should produce (see Data checks). In the
worst case the day-7 result weakens to borderline. The direction of the decision does not change,
but the logging behind the split should be checked before the next experiment.

## Data checks

| Check | Result |
|---|---|
| Players | 90,189 (44,700 gate_30, 45,489 gate_40) |
| Duplicate player IDs | 0 |
| Missing values | 0 |
| Players with 0 rounds | 3,994 (4.4%) |
| Most rounds by one player | 49,854 (next highest: 2,961) |
| Sample ratio (gate_30 share) | 49.56%, chi-square p = 0.0086 |

**Sample ratio mismatch.** With a fair 50/50 coin, a gap this large happens less than
1% of the time. It can mean some players were logged or assigned
incorrectly. Worst-case test: drop the 789 extra gate_40 players and assume all of
them were non-returning players. Day-7 retention then differs by -0.50 pts
(p = 0.0561). That is the most favorable possible case for level 40, and it still does
not show level 40 doing better.

The player with 49,854 rounds is about 17x the next highest and is probably a logging
error or a bot. It has no effect on retention and the rounds analysis below uses rank-based
methods, so it is kept.

## Main results

| Metric | gate_30 | gate_40 | Difference | 95% CI | p-value | Holm p |
|---|---|---|---|---|---|---|
| Day-1 retention | 44.82% | 44.23% | -0.59 pts | [-1.24, 0.06] | 0.0744 | 0.0744 |
| Day-7 retention | 19.02% | 18.20% | -0.82 pts | [-1.33, -0.31] | 0.0016 | 0.0031 |

- Two-sided two-proportion z-tests, alpha = 0.05. Holm correction because two metrics were tested.
- Bootstrap (100,000 resamples) agrees: day-7 CI [-1.33, -0.31] pts;
  99.92% of resamples show a drop.
- Game rounds: median 17 vs 16, Mann-Whitney p = 0.0502.
  A random gate_40 player out-plays a random gate_30 player 49.6% of the time
  (50% would mean no difference).

**Was the test big enough?** With about 45,000 players per group, the smallest change detectable
with 80% power is 0.73 pts for day-7 retention and 0.93 pts for day-1.
The day-7 effect is larger than its threshold (power 89%). The day-1 effect is
smaller than its threshold (power 43%), so its non-significant result means
"too small to see with this sample", not "no effect". Detecting it reliably would take about
111,188 players per group.

## Bayesian view

With uniform priors, there is a **99.92%** probability that gate_30 has the higher
day-7 retention (95% credible interval for the difference [-1.33, -0.31] pts).
For day-1 it is 96.3%.

Expected loss from shipping gate_40: 0.819 pts of day-7 retention.
Expected loss from keeping gate_30: 0.00005 pts. Keeping gate_30 is the
low-risk choice.

## The peeking problem

The dataset has no timestamps, so a two-week test was simulated: players arrive evenly over
14 days and the analyst checks once a day. Base rate = the real pooled day-7 retention
(18.6%); group sizes = the real ones. 50,000 simulated experiments per scenario.

**False positives when nothing changed (A/A tests):**

| Rule | False positive rate | Stops at day (median, when it stops) |
|---|---|---|
| Look once at the end | 5.1% | 14 |
| Check daily, stop at first p < 0.05 | **21.6%** | 4 |
| Bayesian, stop when P(better) > 97.5% | **21.6%** | 4 |
| Bayesian, stop when P(better) > 95% | **37.9%** | 3 |
| Group sequential, O'Brien-Fleming | 5.0% | 12 |
| Group sequential, Pocock | 5.0% | 6 |
| mSPRT (always-valid p-values) | 1.2% | 6 |

- Checking daily for two weeks turns a promised 5% false positive rate into about
  22%. Checking 100 times pushes it to 38%.
- **Bayesian methods are not immune.** A probability threshold used as a stopping rule behaves
  almost exactly like a p-value threshold. The common 95% rule is worse still, because it already
  allows about 10% false positives with a single look.
- **Check on the real data:** shuffling the real Cookie Cats outcomes between groups
  (2,000 times) gives 20.9% false positives for daily peeking,
  5.3% for one look and 5.5% for O'Brien-Fleming. The simulation holds.

**When the effect is real** (true drop = the observed 0.82 pts):

| Rule | Power | Share of planned sample used |
|---|---|---|
| Look once at the end | 88.8% | 100% |
| Group sequential, O'Brien-Fleming | 87.5% | 71% |
| Group sequential, Pocock | 80.8% | 61% |
| mSPRT | 62.2% | 75% |

**Takeaway:** an O'Brien-Fleming plan keeps the 5% promise, loses about
1.3 points of power, and ends the
test about 29% sooner on average when the effect is real. mSPRT lets you look any time,
but that freedom costs power at this sample size. Naive and Bayesian peeking look faster and more
powerful only because they also "find" effects that are not there.

**The real result under these plans.** The full-data day-7 z-statistic is -3.16.
That clears the final O'Brien-Fleming boundary (2.10) and the final Pocock
boundary (2.50). The mSPRT always-valid p-value at the end is
0.037. On the data as logged, the conclusion holds under every valid method
(the sample ratio caveat above still applies).

## Limitations

- No timestamps, so the timing of looks is simulated, not real.
- The sample ratio mismatch could not be traced to a cause with this data.
- Retention is the only outcome; there is no revenue or purchase data.
- The game rounds count is measured after the gate, so it cannot be used to check that the groups
  were similar before the experiment.
