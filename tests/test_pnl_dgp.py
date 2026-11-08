import numpy as np
import pytest

from erg.pnl_dgp import PnLParams, simulate_sessions


def test_fair_game_default_has_zero_closed_form_mean():
    params = PnLParams()
    assert params.p_win_given_active == pytest.approx(0.5)
    assert params.mean_pnl == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("p_win", [0.5, 0.55, 0.45])
def test_closed_form_mean_matches_large_sample_average(p_win):
    """Reference-oracle check: the closed-form mean_pnl (no simulation
    involved) against a large Monte Carlo average, independent paths."""
    params = PnLParams(p_win_given_active=p_win)
    rng = np.random.default_rng(123)
    x = simulate_sessions(2_000_000, rng, params)
    assert x.mean() == pytest.approx(params.mean_pnl, abs=0.02)


def test_closed_form_mean_abs_magnitude_matches_active_sessions_only():
    params = PnLParams()
    rng = np.random.default_rng(7)
    x = simulate_sessions(1_000_000, rng, params)
    active = x[x != 0.0]
    assert np.abs(active).mean() == pytest.approx(params.mean_abs_magnitude, rel=0.02)


def test_flat_fraction_matches_p_flat():
    params = PnLParams(p_flat=0.30)
    rng = np.random.default_rng(1)
    x = simulate_sessions(500_000, rng, params)
    assert (x == 0.0).mean() == pytest.approx(0.30, abs=0.01)


def test_pnl_can_be_negative():
    """Unlike erg.dgp's revenue metric, this DGP must be able to produce a
    loss; this is the entire reason this module exists rather than reusing
    dgp.py's RevenueParams (see module docstring)."""
    rng = np.random.default_rng(2)
    x = simulate_sessions(10_000, rng, PnLParams())
    assert np.any(x < 0.0)
    assert np.any(x > 0.0)


def test_heavy_right_tail_in_magnitude():
    """The lognormal magnitude should show positive excess kurtosis (a
    heavier-than-Gaussian tail), the same property dgp.py's docstring
    argues matters for how badly peeking inflates false positives."""
    from scipy import stats

    rng = np.random.default_rng(4)
    x = simulate_sessions(500_000, rng, PnLParams())
    active = x[x != 0.0]
    assert stats.kurtosis(active) > 1.0


def test_determinism_same_seed_same_draws():
    p = PnLParams()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    assert np.array_equal(simulate_sessions(1000, rng1, p), simulate_sessions(1000, rng2, p))
