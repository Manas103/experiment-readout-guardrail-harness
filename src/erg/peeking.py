"""Continuous-peeking simulation: repeated Welch's t-tests on a null effect.

An analyst who checks the p-value after every new batch of data and stops
the first time p < 0.05 inflates the false positive rate far above the
nominal alpha, because each look is an additional chance to cross the
threshold by chance. This module simulates that behavior on data with NO
true treatment effect (both arms draw from the identical distribution) and
measures how often the analyst would be fooled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from erg.dgp import RevenueParams, simulate_customers


def look_schedule(n_per_arm: int, n_looks: int) -> np.ndarray:
    """Cumulative sample sizes per arm at each look, evenly spaced.

    The first look happens once 1/n_looks of the data has arrived, not at
    n=1, matching how a real team would batch daily/weekly data rather
    than test after every single customer.
    """
    sizes = np.linspace(n_per_arm / n_looks, n_per_arm, n_looks)
    return np.round(sizes).astype(int)


def welch_p_values(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorized two-sided Welch's t-test p-value along axis=1.

    a, b: shape (n_tests, n_customers_so_far). Welch's test (unequal
    variance) is used rather than the pooled-variance t-test because the
    heavy-tailed zero-inflated revenue distribution has very different
    variance under the null only by sampling noise per arm, and Welch is
    the standard robust default for metrics like this.
    """
    _, p = stats.ttest_ind(a, b, axis=1, equal_var=False)
    return p


@dataclass(frozen=True)
class PeekingConfig:
    n_tests: int = 2000
    n_per_arm: int = 4000
    n_looks: int = 20
    alpha: float = 0.05
    seed: int = 20260501


@dataclass
class PeekingResult:
    config: PeekingConfig
    pvalue_trajectory: np.ndarray  # shape (n_tests, n_looks)
    flagged_any_look: np.ndarray  # shape (n_tests,), bool
    flagged_final_look_only: np.ndarray  # shape (n_tests,), bool

    @property
    def peeking_fpr(self) -> float:
        return float(self.flagged_any_look.mean())

    @property
    def single_look_fpr(self) -> float:
        return float(self.flagged_final_look_only.mean())


def run_null_peeking_experiment(
    config: PeekingConfig = PeekingConfig(),
    params: RevenueParams = RevenueParams(),
) -> PeekingResult:
    """Run config.n_tests independent null A/B tests with sequential peeking.

    Both arms of every test draw from the identical revenue distribution
    (no true effect is injected anywhere), so any test flagged significant
    is by construction a false positive.
    """
    rng = np.random.default_rng(config.seed)
    sizes = look_schedule(config.n_per_arm, config.n_looks)

    control = simulate_customers(
        config.n_tests * config.n_per_arm, rng, params
    ).reshape(config.n_tests, config.n_per_arm)
    treatment = simulate_customers(
        config.n_tests * config.n_per_arm, rng, params
    ).reshape(config.n_tests, config.n_per_arm)

    pvals = np.empty((config.n_tests, config.n_looks))
    for k, m in enumerate(sizes):
        pvals[:, k] = welch_p_values(control[:, :m], treatment[:, :m])

    flagged_any = (pvals < config.alpha).any(axis=1)
    flagged_final = pvals[:, -1] < config.alpha

    return PeekingResult(
        config=config,
        pvalue_trajectory=pvals,
        flagged_any_look=flagged_any,
        flagged_final_look_only=flagged_final,
    )
