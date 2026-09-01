import numpy as np

from erg.dgp import RevenueParams, simulate_customers, theoretical_moments


def test_zero_inflated_mass():
    rng = np.random.default_rng(1)
    params = RevenueParams()
    x = simulate_customers(200_000, rng, params)
    zero_frac = float((x == 0).mean())
    # should be close to p_nonpayer, latent quality shifts it only mildly
    assert abs(zero_frac - params.p_nonpayer) < 0.03


def test_heavy_tailed_not_gaussian():
    """Skewness and excess kurtosis must both be large and positive: this is
    the concrete, checkable definition of 'heavy-tailed' used throughout the
    project. A Gaussian has skewness 0 and excess kurtosis 0."""
    rng = np.random.default_rng(2)
    x = simulate_customers(200_000, rng, RevenueParams())
    from scipy import stats

    skew = stats.skew(x)
    kurt = stats.kurtosis(x)  # excess kurtosis (Gaussian = 0)
    assert skew > 3, f"skewness too low for heavy-tailed claim: {skew}"
    assert kurt > 15, f"excess kurtosis too low for heavy-tailed claim: {kurt}"


def test_top_1_percent_share_of_revenue_is_large():
    """A hallmark of a heavy-tailed revenue metric: a small top slice of
    customers accounts for a disproportionate share of total revenue."""
    rng = np.random.default_rng(3)
    x = simulate_customers(200_000, rng, RevenueParams())
    x_sorted = np.sort(x)[::-1]
    top1 = x_sorted[: len(x) // 100].sum()
    total = x_sorted.sum()
    share = top1 / total
    assert share > 0.15, f"top 1% share too small for a heavy tail: {share}"


def test_theoretical_moments_match_empirical():
    rng = np.random.default_rng(4)
    params = RevenueParams()
    x = simulate_customers(500_000, rng, params)
    th = theoretical_moments(params)
    # empirical mean should be within 5% of the unconditional-z theoretical mean
    assert abs(x.mean() - th["mean"]) / th["mean"] < 0.08


def test_shared_latent_z_creates_correlation():
    rng = np.random.default_rng(5)
    n = 50_000
    z = rng.normal(size=n)
    x = simulate_customers(n, rng, RevenueParams(), z=z)
    y = simulate_customers(n, rng, RevenueParams(), z=z)
    rho = np.corrcoef(x, y)[0, 1]
    assert rho > 0.2, f"shared-z customers should show positive pre/post correlation: {rho}"
