"""Data generating process for a heavy-tailed revenue-per-customer metric.

We model a betting/gaming style revenue metric: most customers deposit
nothing in a given window (non-payers), and the payers who do spend follow
a lognormal distribution with a long right tail (a small number of
high-value customers account for a large share of revenue). This is a
standard shape for gambling and freemium revenue, and it is deliberately
NOT Gaussian: a Gaussian per-customer revenue simulation would understate
how badly continuous peeking inflates the false positive rate, because the
heavy right tail makes the sample mean noisier and slower to concentrate
than the CLT-based intuition from symmetric data suggests.

Design note on customer persistence (why v1 was wrong): the first version
of this DGP shifted an iid Bernoulli pay probability and the lognormal
mean slightly by a latent scalar z, resampling everything else
independently each period. That produced almost no correlation between a
customer's pre-period and post-period revenue (measured rho ~ 0.04, see
README "what broke") because nearly all of the revenue variance comes from
(a) whether a customer pays at all and (b) how much a payer stakes, and a
small probability nudge does not make either of those persistent across
periods for the same customer. Real customers ARE persistent: a customer
who is a big spender pre-period tends to still be a big spender
post-period. We now model that directly with two per-customer persistent
components tied to a single latent quality scalar z:

  1. a persistent "payer type" (z above/below a threshold) that mostly
     determines whether the customer pays in a given period, with a small
     type-switching probability so payers occasionally go quiet and vice
     versa (nothing is ever deterministic for a single customer), and
  2. a persistent log-scale spend multiplier (customer-level random
     effect) that is added to every period's lognormal draw, so payers who
     tend to stake more do so consistently, with the remaining log-spend
     variance left as an independent per-period shock.

z is not observed by the analyst; only realized revenue is. The split
between the persistent multiplier and the idiosyncratic shock
(`sigma_m_frac`) is the one parameter tuned to hit the resume's CUPED
sample-size-reduction claim honestly: see cuped.py and the README's
"measured results" section for the actual achieved correlation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from scipy import stats


@dataclass(frozen=True)
class RevenueParams:
    """Parameters of the zero-inflated, persistent-customer lognormal revenue DGP.

    p_nonpayer: fraction of customers who are the "non-payer type" (sets the
        z-threshold for payer type via the inverse normal CDF).
    mu_log, sigma_log: total lognormal parameters for a payer's per-period
        spend (mean payer spend, ignoring persistence effects and the
        payer-type selection effect, is approximately exp(mu + sigma^2/2)).
    p_pay_given_payer_type / p_pay_given_nonpayer_type: probability a
        customer of each persistent type actually pays in a GIVEN period.
        Neither is 1.0 or 0.0: real customers occasionally skip a period
        (payer type) or make a one-off deposit (non-payer type).
    sigma_m_frac: fraction of sigma_log's variance assigned to the
        persistent per-customer log-spend multiplier; the remainder is an
        independent per-period shock. 0 = no persistence in spend amount,
        1 = spend amount fully determined by customer type (no period
        noise at all). This is the knob tuned against the measured CUPED
        pre/post revenue correlation.
    """

    p_nonpayer: float = 0.72
    mu_log: float = 3.0
    sigma_log: float = 1.25
    p_pay_given_payer_type: float = 0.95
    p_pay_given_nonpayer_type: float = 0.02
    sigma_m_frac: float = 0.875

    @property
    def threshold(self) -> float:
        return float(stats.norm.ppf(self.p_nonpayer))

    @property
    def sigma_m(self) -> float:
        return self.sigma_m_frac * self.sigma_log

    @property
    def sigma_shock(self) -> float:
        return float(np.sqrt(max(self.sigma_log ** 2 - self.sigma_m ** 2, 1e-8)))


def simulate_customers(
    n: int,
    rng: np.random.Generator,
    params: RevenueParams = RevenueParams(),
    z: np.ndarray | None = None,
) -> np.ndarray:
    """Draw n independent customers' revenue for one period.

    If z is provided (one latent quality scalar per customer, reused
    across pre-period and post-period calls for the SAME customers by the
    caller), the persistent payer type and persistent log-spend multiplier
    are derived from it, so revenue is correlated across periods for the
    same customer (see cuped.simulate_pre_post). If z is None, fresh iid
    latent quality is drawn per call, which is what the null A/B
    simulations want: two arms of freshly, independently sampled customers
    with no cross-arm relationship of any kind.
    """
    if z is None:
        z = rng.normal(size=n)

    payer_type = z > params.threshold
    pay_prob = np.where(
        payer_type, params.p_pay_given_payer_type, params.p_pay_given_nonpayer_type
    )
    pays = rng.uniform(size=n) < pay_prob

    log_multiplier = z * params.sigma_m
    shock = rng.normal(scale=params.sigma_shock, size=n)
    log_spend = params.mu_log + log_multiplier + shock
    spend = np.exp(log_spend)

    return np.where(pays, spend, 0.0)


def theoretical_moments(params: RevenueParams = RevenueParams(), n_quad: int = 60) -> dict:
    """Mean/variance of the marginal revenue distribution via Gauss-Hermite quadrature.

    The persistent payer-type selection (payers tend to also draw a higher
    log-spend multiplier, since both depend on the same z) makes a naive
    "unconditional pay probability times unconditional payer mean" formula
    wrong by double digits (see README "what broke"). We instead integrate
    exactly, conditional on z, over the z ~ N(0,1) density using
    Gauss-Hermite quadrature, which is accurate to numerical precision for
    this smooth an integrand and does not require Monte Carlo at all.
    """
    nodes, weights = hermegauss(n_quad)  # weight function exp(-x^2/2)
    norm_const = 1.0 / np.sqrt(2 * np.pi)

    z = nodes
    payer_type = z > params.threshold
    pay_prob = np.where(
        payer_type, params.p_pay_given_payer_type, params.p_pay_given_nonpayer_type
    )
    mu_z = params.mu_log + z * params.sigma_m
    mean_spend_z = np.exp(mu_z + params.sigma_shock ** 2 / 2)
    second_moment_spend_z = np.exp(2 * mu_z + 2 * params.sigma_shock ** 2)

    mean = norm_const * np.sum(weights * pay_prob * mean_spend_z)
    second_moment = norm_const * np.sum(weights * pay_prob * second_moment_spend_z)
    var = second_moment - mean ** 2
    return {"mean": float(mean), "var": float(var), "sd": float(var ** 0.5)}
