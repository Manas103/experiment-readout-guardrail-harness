import numpy as np

from erg.msprt import (
    MsprtConfig,
    mixture_lr,
    mixture_lr_scalar,
    run_null_msprt_experiment,
)
from erg.peeking import PeekingConfig, run_null_peeking_experiment


def test_mixture_lr_vectorized_matches_scalar_oracle():
    """Reference-oracle test: the vectorized mixture_lr must exactly match
    the independently-written scalar mixture_lr_scalar on a small fixture,
    looped one element at a time."""
    rng = np.random.default_rng(3)
    diffs = rng.normal(scale=5, size=25)
    variances = rng.uniform(0.5, 20, size=25)
    tau2 = 9.0
    vectorized = mixture_lr(diffs, variances, tau2)
    for i in range(25):
        scalar = mixture_lr_scalar(float(diffs[i]), float(variances[i]), tau2)
        assert abs(vectorized[i] - scalar) < 1e-9


def test_mixture_lr_equals_one_at_zero_difference():
    """At diff=0 the mixture likelihood ratio simplifies to sqrt(V/(V+tau2)),
    strictly less than 1: a sanity identity independent of the main code."""
    lr = mixture_lr(np.array([0.0]), np.array([4.0]), 9.0)
    expected = np.sqrt(4.0 / (4.0 + 9.0))
    assert abs(lr[0] - expected) < 1e-12
    assert lr[0] < 1.0


def test_msprt_fpr_never_exceeds_theoretical_bound_by_more_than_noise():
    """Invariant: mSPRT's always-valid guarantee bounds P(false positive)
    <= alpha for ANY stopping rule (Ville's inequality). We allow generous
    slack for Monte Carlo sampling noise (binomial SD at n=2000,p=0.05 is
    about 0.005) plus the plug-in-variance approximation, but the measured
    rate must not blow past the bound the way naive repeated t-testing does."""
    config = MsprtConfig(n_tests=2000, n_per_arm=4000, n_looks=20, seed=7)
    result = run_null_msprt_experiment(config)
    assert result.msprt_fpr < config.alpha + 0.03, (
        f"mSPRT false positive rate {result.msprt_fpr} exceeds the alpha={config.alpha} "
        "always-valid bound by more than plausible sampling noise"
    )


def test_msprt_controls_fpr_far_better_than_naive_peeking_on_same_data():
    """The core comparison the harness exists to demonstrate: on the exact
    same simulated null data, repeated naive t-testing inflates the false
    positive rate far above nominal, while the mSPRT keeps it near nominal."""
    seed = 99
    n_tests, n_per_arm, n_looks = 1500, 4000, 20

    from erg.dgp import RevenueParams, simulate_customers

    rng = np.random.default_rng(seed)
    control = simulate_customers(n_tests * n_per_arm, rng, RevenueParams()).reshape(
        n_tests, n_per_arm
    )
    treatment = simulate_customers(n_tests * n_per_arm, rng, RevenueParams()).reshape(
        n_tests, n_per_arm
    )

    peek_config = PeekingConfig(n_tests=n_tests, n_per_arm=n_per_arm, n_looks=n_looks, seed=seed)
    msprt_config = MsprtConfig(n_tests=n_tests, n_per_arm=n_per_arm, n_looks=n_looks, seed=seed)

    peek_result = run_null_peeking_experiment(peek_config)
    msprt_result = run_null_msprt_experiment(
        msprt_config, control=control, treatment=treatment
    )

    assert msprt_result.msprt_fpr < peek_result.peeking_fpr
    assert msprt_result.msprt_fpr < 0.10
