"""Mixture sequential probability ratio test (mSPRT): an always-valid test.

Following Johari, Koomen, Pekelis & Walsh ("Peeking at A/B Tests", KDD
2017), we replace the fixed-horizon Welch's t-test with a test statistic
that is a martingale under the null for ANY stopping rule, including
"stop the first time it crosses threshold". Concretely we place a normal
mixing distribution N(0, tau^2) on the unknown true mean difference delta,
and at each look compute the mixture likelihood ratio

    Lambda = sqrt(V / (V + tau^2)) * exp( tau^2 * D^2 / (2 * V * (V + tau^2)) )

where D is the observed difference in arm means so far and V is the
(plug-in, sample-estimated) variance of that difference. By Ville's
inequality, P(sup_k Lambda_k >= 1/alpha | H0) <= alpha for every stopping
time, so declaring significance the first time Lambda crosses 1/alpha
controls the false positive rate under continuous peeking, unlike a fixed
alpha=0.05 t-test re-applied at every look.

We use the plug-in (estimated, not known) variance V at each look. This is
the standard practical approximation; it is asymptotically valid but can
be slightly anti-conservative at very small sample sizes, which is called
out explicitly in the README as a limitation.

tau (the mixture prior scale) must be chosen in the same units as D
(dollars of revenue per customer). It represents the analyst's prior
belief about the plausible size of a true effect; too small and the test
loses power to detect real small effects, too large and it loses power to
detect very small ones. We set tau from a fraction of the baseline mean
revenue (see PeekingConfig / MsprtConfig defaults), documented in the
README.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from erg.dgp import RevenueParams, simulate_customers
from erg.peeking import look_schedule


def mixture_lr(diff: np.ndarray, var_diff: np.ndarray, tau2: float) -> np.ndarray:
    """Vectorized mixture likelihood ratio Lambda for a normal mixture prior.

    diff: observed difference in arm means, any shape.
    var_diff: variance of that difference estimate, same shape, must be > 0.
    tau2: prior variance of the true effect (scalar).
    """
    ratio = var_diff / (var_diff + tau2)
    exponent = (tau2 * diff ** 2) / (2 * var_diff * (var_diff + tau2))
    return np.sqrt(ratio) * np.exp(exponent)


def mixture_lr_scalar(diff: float, var_diff: float, tau2: float) -> float:
    """Slow, obviously-correct scalar version, used as an oracle in tests."""
    ratio = var_diff / (var_diff + tau2)
    exponent = (tau2 * diff * diff) / (2 * var_diff * (var_diff + tau2))
    return math.sqrt(ratio) * math.exp(exponent)


@dataclass(frozen=True)
class MsprtConfig:
    n_tests: int = 2000
    n_per_arm: int = 4000
    n_looks: int = 20
    alpha: float = 0.05
    # tuned empirically (see docs/measurement_output.txt, "mSPRT tau tuning"):
    # the mixture false positive rate under this DGP and look schedule peaks
    # around tau_fraction_of_mean ~= 0.3, both smaller and larger priors are
    # MORE conservative, not less. See README "what broke".
    tau_fraction_of_mean: float = 0.3
    seed: int = 20260501


@dataclass
class MsprtResult:
    config: MsprtConfig
    lambda_trajectory: np.ndarray  # shape (n_tests, n_looks)
    flagged_any_look: np.ndarray  # shape (n_tests,), bool
    tau2: float

    @property
    def msprt_fpr(self) -> float:
        return float(self.flagged_any_look.mean())


def run_null_msprt_experiment(
    config: MsprtConfig = MsprtConfig(),
    params: RevenueParams = RevenueParams(),
    control: np.ndarray | None = None,
    treatment: np.ndarray | None = None,
) -> MsprtResult:
    """Run config.n_tests independent null A/B tests through the mSPRT.

    If control/treatment arrays (shape n_tests x n_per_arm) are supplied,
    reuse them (used by scripts/run_simulation.py so the peeking and
    mSPRT methods are compared on the identical simulated data, an
    apples-to-apples comparison). Otherwise fresh data is drawn.
    """
    rng = np.random.default_rng(config.seed)
    sizes = look_schedule(config.n_per_arm, config.n_looks)

    if control is None:
        control = simulate_customers(
            config.n_tests * config.n_per_arm, rng, params
        ).reshape(config.n_tests, config.n_per_arm)
    if treatment is None:
        treatment = simulate_customers(
            config.n_tests * config.n_per_arm, rng, params
        ).reshape(config.n_tests, config.n_per_arm)

    from erg.dgp import theoretical_moments

    baseline_mean = theoretical_moments(params)["mean"]
    tau = config.tau_fraction_of_mean * baseline_mean
    tau2 = tau ** 2

    threshold = 1.0 / config.alpha
    lambdas = np.empty((config.n_tests, config.n_looks))
    for k, m in enumerate(sizes):
        c = control[:, :m]
        t = treatment[:, :m]
        diff = t.mean(axis=1) - c.mean(axis=1)
        var_diff = t.var(axis=1, ddof=1) / m + c.var(axis=1, ddof=1) / m
        var_diff = np.maximum(var_diff, 1e-12)
        lambdas[:, k] = mixture_lr(diff, var_diff, tau2)

    flagged_any = (lambdas >= threshold).any(axis=1)

    return MsprtResult(
        config=config,
        lambda_trajectory=lambdas,
        flagged_any_look=flagged_any,
        tau2=tau2,
    )
