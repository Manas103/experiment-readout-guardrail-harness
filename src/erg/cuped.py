"""CUPED variance reduction and the sample size it buys.

CUPED (Controlled-experiment Using Pre-Experiment Data, Deng et al. 2013)
adjusts the post-period metric Y using a pre-period covariate X that is
correlated with it but was measured before the experiment started (so it
cannot itself be affected by treatment):

    Y_cuped = Y - theta * (X - mean(X)),  theta = Cov(X, Y) / Var(X)

Var(Y_cuped) = Var(Y) * (1 - rho^2), where rho = corr(X, Y). Since the
two-sample required-sample-size formula for detecting a fixed absolute
effect at fixed alpha/power is n = 2 * (z_a2 + z_b)^2 * sigma^2 / delta^2,
and CUPED changes only sigma^2, the required sample size scales exactly by
(1 - rho^2). We measure rho empirically from simulated pre/post revenue
(sharing the latent customer-quality scalar from dgp.py, which is what
makes pre-period and post-period revenue correlated for the same customer
in the first place) rather than assuming it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from erg.dgp import RevenueParams, simulate_customers, theoretical_moments


@dataclass(frozen=True)
class CupedConfig:
    n_customers: int = 50_000
    relative_effect: float = 0.02
    alpha: float = 0.05
    power: float = 0.80
    seed: int = 20260501


def simulate_pre_post(
    n: int, rng: np.random.Generator, params: RevenueParams = RevenueParams()
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate correlated pre-period (X) and post-period (Y) revenue.

    Both periods share the same per-customer latent quality scalar z, so a
    customer who tends to spend more pre-period also tends to spend more
    post-period, exactly the structure CUPED is designed to exploit. z is
    resampled independently between periods only in its idiosyncratic
    (non-quality) component, since real customers are not perfectly
    consistent period to period.
    """
    z = rng.normal(size=n)
    x = simulate_customers(n, rng, params, z=z)
    y = simulate_customers(n, rng, params, z=z)
    return x, y


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    theta = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    y_adj = y - theta * (x - x.mean())
    return y_adj, float(theta)


def variance_reduction(x: np.ndarray, y: np.ndarray) -> dict:
    rho = float(np.corrcoef(x, y)[0, 1])
    var_y = float(np.var(y, ddof=1))
    y_adj, theta = cuped_adjust(y, x)
    var_y_cuped = float(np.var(y_adj, ddof=1))
    return {
        "rho": rho,
        "rho2": rho ** 2,
        "var_y": var_y,
        "var_y_cuped": var_y_cuped,
        "theta": theta,
        # empirical check: should match var_y * (1 - rho^2) closely
        "measured_reduction_fraction": 1 - var_y_cuped / var_y,
    }


def required_sample_size(
    variance: float, baseline_mean: float, relative_effect: float, alpha: float, power: float
) -> float:
    """Two-sample per-arm sample size for a fixed absolute effect delta.

    n = 2 * (z_{alpha/2} + z_{power})^2 * sigma^2 / delta^2
    """
    delta = relative_effect * baseline_mean
    z_a2 = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return 2 * (z_a2 + z_b) ** 2 * variance / delta ** 2


def sample_size_reduction(
    config: CupedConfig = CupedConfig(), params: RevenueParams = RevenueParams()
) -> dict:
    rng = np.random.default_rng(config.seed)
    x, y = simulate_pre_post(config.n_customers, rng, params)
    vr = variance_reduction(x, y)
    baseline_mean = float(np.mean(y))

    n_no_cuped = required_sample_size(
        vr["var_y"], baseline_mean, config.relative_effect, config.alpha, config.power
    )
    n_cuped = required_sample_size(
        vr["var_y_cuped"], baseline_mean, config.relative_effect, config.alpha, config.power
    )
    reduction_pct = 100 * (1 - n_cuped / n_no_cuped)

    return {
        "config": config,
        "rho": vr["rho"],
        "rho2": vr["rho2"],
        "baseline_mean": baseline_mean,
        "var_y": vr["var_y"],
        "var_y_cuped": vr["var_y_cuped"],
        "n_no_cuped": n_no_cuped,
        "n_cuped": n_cuped,
        "reduction_pct": reduction_pct,
    }
