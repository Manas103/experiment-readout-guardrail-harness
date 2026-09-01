import numpy as np

from erg.cuped import (
    CupedConfig,
    cuped_adjust,
    required_sample_size,
    sample_size_reduction,
    simulate_pre_post,
    variance_reduction,
)
from erg.dgp import RevenueParams


def test_cuped_adjustment_uncorrelated_with_covariate():
    """Reference identity: after CUPED adjustment, the adjusted metric should
    be (to numerical precision) uncorrelated with the covariate it was
    adjusted on, since theta is chosen exactly to zero out that covariance."""
    rng = np.random.default_rng(1)
    x, y = simulate_pre_post(20_000, rng, RevenueParams())
    y_adj, theta = cuped_adjust(y, x)
    rho_adj = np.corrcoef(x, y_adj)[0, 1]
    assert abs(rho_adj) < 0.01


def test_cuped_never_increases_variance():
    """Invariant: CUPED variance must never exceed the raw variance, since
    theta is the OLS-optimal coefficient and can always fall back to 0."""
    rng = np.random.default_rng(2)
    x, y = simulate_pre_post(20_000, rng, RevenueParams())
    vr = variance_reduction(x, y)
    assert vr["var_y_cuped"] <= vr["var_y"]


def test_measured_reduction_matches_rho_squared_formula():
    """Reference-oracle check: the empirically measured variance reduction
    fraction must match the closed-form 1 - rho^2 prediction closely."""
    rng = np.random.default_rng(3)
    x, y = simulate_pre_post(50_000, rng, RevenueParams())
    vr = variance_reduction(x, y)
    predicted = vr["rho2"]
    measured = vr["measured_reduction_fraction"]
    assert abs(predicted - measured) < 0.01


def test_required_sample_size_scalar_oracle():
    """Hand-computed oracle for one specific input: z_0.025=1.959964,
    z_0.8=0.841621, variance=100, baseline_mean=50, relative_effect=0.02
    -> delta=1.0, n = 2*(1.959964+0.841621)^2*100/1^2 = 1569.78."""
    n = required_sample_size(variance=100.0, baseline_mean=50.0, relative_effect=0.02, alpha=0.05, power=0.80)
    assert abs(n - 1569.78) < 1.0


def test_zero_correlation_gives_zero_reduction():
    """Invariant: if pre/post are uncorrelated (no persistent spend
    multiplier AND no persistent payer type, i.e. pay probability is the
    same regardless of type), CUPED should buy ~0% sample size reduction,
    not a fabricated positive number."""
    rng = np.random.default_rng(4)
    params_uncorrelated = RevenueParams(
        sigma_m_frac=0.0, p_pay_given_payer_type=0.28, p_pay_given_nonpayer_type=0.28
    )
    x, y = simulate_pre_post(50_000, rng, params_uncorrelated)
    vr = variance_reduction(x, y)
    assert abs(vr["rho"]) < 0.05


def test_sample_size_reduction_end_to_end_runs():
    result = sample_size_reduction(CupedConfig(n_customers=20_000, seed=5))
    assert result["n_cuped"] < result["n_no_cuped"]
    assert 0 < result["reduction_pct"] < 100
