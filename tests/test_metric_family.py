import numpy as np
from statsmodels.stats.multitest import multipletests

from erg.metric_family import (
    MetricFamilyConfig,
    bh_reject_hand_rolled,
    run_metric_family_experiment,
    simulate_correlated_null_pvalues,
)


def test_bh_hand_rolled_matches_statsmodels_oracle():
    rng = np.random.default_rng(1)
    for _ in range(200):
        m = rng.integers(2, 40)
        pvals = rng.uniform(0, 1, size=m)
        q = rng.choice([0.05, 0.1, 0.2])
        ours = bh_reject_hand_rolled(pvals, q)
        theirs, _, _, _ = multipletests(pvals, alpha=q, method="fdr_bh")
        assert np.array_equal(ours, theirs)


def test_pvalues_uniform_under_null_single_metric():
    rng = np.random.default_rng(2)
    p = simulate_correlated_null_pvalues(50000, 1, rho=0.0, rng=rng)
    # a KS-style sanity check: under the null, p-values are Uniform(0,1),
    # so the fraction below alpha should be within sampling noise of alpha
    for alpha in (0.01, 0.05, 0.1, 0.5):
        rate = (p < alpha).mean()
        assert abs(rate - alpha) < 0.01


def test_independent_metrics_naive_false_win_matches_closed_form():
    """With rho=0 (independent metrics), P(at least one of k p-values below
    alpha under the global null) has the exact closed form
    1 - (1 - alpha)^k. This is the reference oracle for the realistic
    (rho=0.3) simulation below: if the simulator's independent-metrics case
    does not match the textbook formula, the correlated case cannot be
    trusted either.
    """
    rng = np.random.default_rng(3)
    config = MetricFamilyConfig(n_metrics=20, n_trials=40000, rho=0.0, seed=99)
    pvals = simulate_correlated_null_pvalues(
        config.n_trials, config.n_metrics, config.rho, rng
    )
    measured = (pvals < config.alpha).any(axis=1).mean()
    closed_form = 1.0 - (1.0 - config.alpha) ** config.n_metrics
    assert abs(measured - closed_form) < 0.01


def test_bh_never_rejects_more_than_naive():
    """BH is a strictly more conservative gate than "any raw p < alpha":
    anything BH rejects at level q <= alpha must also be flagged naive.
    Checked as a fuzz invariant across many random p-value vectors, not a
    handful of examples.
    """
    rng = np.random.default_rng(4)
    for _ in range(500):
        m = rng.integers(2, 30)
        pvals = rng.uniform(0, 1, size=m)
        alpha = 0.05
        naive = pvals < alpha
        bh = bh_reject_hand_rolled(pvals, q=alpha)
        assert np.all(bh <= naive)


def test_bh_false_win_rate_never_exceeds_naive_false_win_rate():
    result = run_metric_family_experiment(
        MetricFamilyConfig(n_metrics=20, n_trials=5000, rho=0.3, seed=5)
    )
    assert result.bh_false_win_rate <= result.naive_false_win_rate


def test_higher_correlation_does_not_increase_naive_false_win_rate():
    """More positive correlation between metrics makes them move together,
    which can only reduce (never increase) the chance that at least one of
    them crosses the threshold purely by chance, holding the marginal null
    distribution of every metric fixed.
    """
    rng_seed = 6
    low = run_metric_family_experiment(
        MetricFamilyConfig(n_metrics=20, n_trials=8000, rho=0.0, seed=rng_seed)
    )
    high = run_metric_family_experiment(
        MetricFamilyConfig(n_metrics=20, n_trials=8000, rho=0.6, seed=rng_seed)
    )
    assert high.naive_false_win_rate <= low.naive_false_win_rate + 0.02
