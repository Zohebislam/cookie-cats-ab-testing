"""A/B testing toolkit for the Cookie Cats experiment.

Modules
-------
data          load and validate the raw experiment data
frequentist   sample ratio check, two-proportion z-test, bootstrap, Holm correction
power         minimum detectable effect, sample size and power
sequential    group sequential boundaries (alpha spending) and mSPRT
bayesian      Beta-Binomial posteriors and expected loss
simulate      Monte Carlo studies of peeking and of the fixes
"""

__version__ = "1.0.0"
