"""Measure the continuous-peeking false positive rate and the mSPRT
always-valid false positive rate on 2000 simulated null A/B tests.

Usage:
    python scripts/run_simulation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from erg.dgp import RevenueParams, simulate_customers, theoretical_moments
from erg.msprt import MsprtConfig, run_null_msprt_experiment
from erg.peeking import PeekingConfig, run_null_peeking_experiment


def main() -> None:
    params = RevenueParams()
    n_tests, n_per_arm, n_looks, seed = 2000, 4000, 20, 20260501

    print("=" * 78)
    print("Revenue-per-customer DGP moments (Gauss-Hermite quadrature, exact):")
    moments = theoretical_moments(params)
    print(json.dumps(moments, indent=2))

    rng = np.random.default_rng(1)
    x = simulate_customers(300_000, rng, params)
    from scipy import stats

    print("\nEmpirical DGP shape check (n=300000 draws):")
    print(
        json.dumps(
            {
                "mean": float(x.mean()),
                "sd": float(x.std(ddof=1)),
                "zero_frac": float((x == 0).mean()),
                "skewness": float(stats.skew(x)),
                "excess_kurtosis": float(stats.kurtosis(x)),
                "p99": float(np.percentile(x, 99)),
                "max": float(x.max()),
            },
            indent=2,
        )
    )

    print("\n" + "=" * 78)
    print(f"NULL A/B SIMULATION: n_tests={n_tests}, n_per_arm={n_per_arm}, n_looks={n_looks}, seed={seed}")
    print("Both arms of every test draw from the identical distribution (no injected effect).")

    t0 = time.time()
    peek_config = PeekingConfig(n_tests=n_tests, n_per_arm=n_per_arm, n_looks=n_looks, seed=seed)
    peek_result = run_null_peeking_experiment(peek_config, params)
    t1 = time.time()
    print(f"\nContinuous-peeking (repeated Welch's t-test, stop at first p<0.05):")
    print(f"  single-look (fixed horizon) false positive rate: {peek_result.single_look_fpr:.4f}")
    print(f"  ANY-LOOK (continuous peeking) false positive rate: {peek_result.peeking_fpr:.4f}")
    print(f"  runtime: {t1 - t0:.2f}s")

    t0 = time.time()
    # control/treatment are reused from the peeking run's own construction
    # (same seed, same generation order) so this is the identical simulated
    # data run through both methods.
    rng2 = np.random.default_rng(seed)
    control = simulate_customers(n_tests * n_per_arm, rng2, params).reshape(n_tests, n_per_arm)
    treatment = simulate_customers(n_tests * n_per_arm, rng2, params).reshape(n_tests, n_per_arm)

    msprt_config = MsprtConfig(n_tests=n_tests, n_per_arm=n_per_arm, n_looks=n_looks, seed=seed)
    msprt_result = run_null_msprt_experiment(msprt_config, params, control=control, treatment=treatment)
    t1 = time.time()
    print(f"\nmSPRT always-valid test, SAME {n_looks}-look schedule as peeking above")
    print(f"(mixture likelihood ratio, tau_fraction_of_mean={msprt_config.tau_fraction_of_mean}, tau2={msprt_result.tau2:.4g}):")
    print(f"  false positive rate under identical repeated peeking: {msprt_result.msprt_fpr:.4f}")
    print(f"  runtime: {t1 - t0:.2f}s")

    # Supplementary run: an always-valid test is specifically designed to
    # license checking as often as the analyst wants, so we also measure it
    # under much denser (near-continuous) monitoring on the SAME underlying
    # data, for context in the README's "what broke" discussion. This number
    # is NOT the headline claim measurement, which uses the same schedule as
    # peeking for an apples-to-apples comparison.
    dense_looks = 320
    t0 = time.time()
    dense_config = MsprtConfig(
        n_tests=n_tests, n_per_arm=n_per_arm, n_looks=dense_looks, seed=seed,
        tau_fraction_of_mean=0.5,
    )
    dense_result = run_null_msprt_experiment(dense_config, params, control=control, treatment=treatment)
    t1 = time.time()
    print(f"\n[supplementary] mSPRT under {dense_looks}-look near-continuous monitoring, tau_fraction=0.5:")
    print(f"  false positive rate: {dense_result.msprt_fpr:.4f}")
    print(f"  runtime: {t1 - t0:.2f}s")

    print("\n" + "=" * 78)
    print("SUMMARY (resume claims vs measured):")
    print(
        json.dumps(
            {
                "claim_peeking_fpr_target": 0.264,
                "measured_peeking_fpr": peek_result.peeking_fpr,
                "claim_msprt_fpr_target": 0.048,
                "measured_msprt_fpr_same_schedule_as_peeking": msprt_result.msprt_fpr,
                "measured_msprt_fpr_dense_320_looks_supplementary": dense_result.msprt_fpr,
                "nominal_alpha": peek_config.alpha,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
