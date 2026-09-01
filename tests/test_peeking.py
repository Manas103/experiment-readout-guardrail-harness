import numpy as np
from scipy import stats

from erg.peeking import (
    PeekingConfig,
    look_schedule,
    run_null_peeking_experiment,
    welch_p_values,
)


def test_look_schedule_reaches_full_n_and_is_increasing():
    sizes = look_schedule(4000, 20)
    assert sizes[-1] == 4000
    assert (np.diff(sizes) > 0).all()
    assert len(sizes) == 20


def test_welch_p_values_matches_scipy_scalar_oracle():
    """Reference-oracle test: for a handful of small fixtures, the vectorized
    welch_p_values function must match scipy.stats.ttest_ind called one test
    at a time, to floating point precision."""
    rng = np.random.default_rng(42)
    a = rng.normal(size=(5, 30))
    b = rng.normal(loc=0.1, size=(5, 30))
    vectorized = welch_p_values(a, b)
    for i in range(5):
        _, p_scalar = stats.ttest_ind(a[i], b[i], equal_var=False)
        assert abs(vectorized[i] - p_scalar) < 1e-12


def test_single_look_null_fpr_close_to_nominal_alpha():
    """Sanity/invariant check: testing ONCE (no peeking) on a true null must
    control the false positive rate near the nominal alpha. This validates
    the simulation harness itself (the DGP, the null construction, and the
    Welch test) independent of the peeking behavior under test."""
    config = PeekingConfig(n_tests=2000, n_per_arm=4000, n_looks=20, seed=7)
    result = run_null_peeking_experiment(config)
    fpr = result.single_look_fpr
    # binomial sampling noise around 0.05 with n=2000: SD ~ sqrt(.05*.95/2000) ~ 0.0049
    assert 0.03 < fpr < 0.07, f"single-look FPR {fpr} not close to nominal alpha 0.05"


def test_peeking_inflates_fpr_above_single_look():
    config = PeekingConfig(n_tests=2000, n_per_arm=4000, n_looks=20, seed=7)
    result = run_null_peeking_experiment(config)
    assert result.peeking_fpr > result.single_look_fpr
    assert result.peeking_fpr > 0.10, "peeking should meaningfully inflate the false positive rate"


def test_more_looks_inflate_fpr_further():
    """Invariant: holding total sample size fixed, more looks should never
    decrease the peeking false positive rate (monotonicity is the whole
    mechanism the resume claim depends on)."""
    few = run_null_peeking_experiment(PeekingConfig(n_tests=1500, n_per_arm=4000, n_looks=3, seed=11))
    many = run_null_peeking_experiment(PeekingConfig(n_tests=1500, n_per_arm=4000, n_looks=20, seed=11))
    assert many.peeking_fpr >= few.peeking_fpr
