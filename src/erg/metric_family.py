"""Multi-metric family readout: what happens once a single experiment is
scored on 20 metrics instead of one.

A real experimentation program does not report a single p-value per
experiment. A product surface like a home feed is usually scored on a
family of related metrics (several engagement sub-metrics, a handful of
secondary and guardrail metrics) at once, and a team that ships the moment
ANY of those metrics crosses p < 0.05 is running 20 simultaneous chances to
be fooled by noise, not one. This module measures how often that "read a
false win" failure mode fires under a global null (no true effect on any
metric) with no correction at all, and how much Benjamini-Hochberg (BH)
false discovery rate control brings that back down.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.stats.multitest import multipletests


def simulate_correlated_null_pvalues(
    n_trials: int,
    n_metrics: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Two-sided p-values for `n_metrics` metrics, `n_trials` independent
    experiments, every metric drawn under its own null (no true effect).

    Metrics within one experiment are correlated at a constant pairwise
    `rho` (an equicorrelated / compound-symmetry structure), which is the
    realistic case for a metric family: several engagement sub-metrics on
    the same surface move together because they are driven by the same
    underlying user sessions, not independent coin flips. `rho=0` recovers
    independent metrics, used below only to cross-check this function
    against the closed-form independent-metrics formula.
    """
    if n_metrics < 1:
        raise ValueError("n_metrics must be >= 1")
    cov = np.full((n_metrics, n_metrics), rho) + np.eye(n_metrics) * (1.0 - rho)
    z = rng.multivariate_normal(
        mean=np.zeros(n_metrics), cov=cov, size=n_trials, method="cholesky"
    )
    # two-sided p-value from a standard normal z-statistic, each metric's
    # own null variance is exactly 1 by construction of `cov` (unit diagonal)
    from scipy import stats

    p = 2.0 * stats.norm.sf(np.abs(z))
    return p


def bh_reject_hand_rolled(pvals: np.ndarray, q: float) -> np.ndarray:
    """Independent, from-scratch implementation of the Benjamini-Hochberg
    step-up procedure, used as a reference oracle against
    `statsmodels.stats.multitest.multipletests(method="fdr_bh")`.

    Sort p-values ascending; find the largest rank k such that
    p_(k) <= (k/m) * q; reject all hypotheses with rank <= k.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * q
    eligible = sorted_p <= thresholds
    reject_sorted = np.zeros(m, dtype=bool)
    if eligible.any():
        k = np.max(np.nonzero(eligible)[0]) + 1  # largest eligible rank
        reject_sorted[:k] = True
    reject = np.zeros(m, dtype=bool)
    reject[order] = reject_sorted
    return reject


@dataclass(frozen=True)
class MetricFamilyConfig:
    n_metrics: int = 20
    n_trials: int = 20000
    alpha: float = 0.05
    fdr_q: float = 0.05
    rho: float = 0.3
    seed: int = 20260915


@dataclass
class MetricFamilyResult:
    config: MetricFamilyConfig
    pvals: np.ndarray  # shape (n_trials, n_metrics)
    naive_false_win: np.ndarray  # shape (n_trials,), bool
    bh_false_win: np.ndarray  # shape (n_trials,), bool

    @property
    def naive_false_win_rate(self) -> float:
        return float(self.naive_false_win.mean())

    @property
    def bh_false_win_rate(self) -> float:
        return float(self.bh_false_win.mean())


def run_metric_family_experiment(
    config: MetricFamilyConfig = MetricFamilyConfig(),
) -> MetricFamilyResult:
    """Run `config.n_trials` independent global-null experiments, each
    scored on `config.n_metrics` correlated metrics, under two decision
    rules: naive (ship if any metric's raw p < alpha) and BH-controlled
    (ship if any metric survives Benjamini-Hochberg at fdr_q).

    Every metric in every trial has NO true effect, so any "ship" decision
    under either rule is, by construction, a false win.
    """
    rng = np.random.default_rng(config.seed)
    pvals = simulate_correlated_null_pvalues(
        config.n_trials, config.n_metrics, config.rho, rng
    )

    naive_false_win = (pvals < config.alpha).any(axis=1)

    bh_false_win = np.empty(config.n_trials, dtype=bool)
    for i in range(config.n_trials):
        reject, _, _, _ = multipletests(
            pvals[i], alpha=config.fdr_q, method="fdr_bh"
        )
        bh_false_win[i] = reject.any()

    return MetricFamilyResult(
        config=config,
        pvals=pvals,
        naive_false_win=naive_false_win,
        bh_false_win=bh_false_win,
    )
