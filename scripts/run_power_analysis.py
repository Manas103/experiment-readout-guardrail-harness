"""The "Sequential Decision Rule for Promoting a Strategy Change" extension
(Nov. 2026): re-measures the peeking-vs-mSPRT false-positive comparison on
heavy-tailed per-session trading P&L (not the base repo's per-customer
revenue metric), then measures how many per-arm sessions each method needs
to detect a true 5% edge gain at 80% power.

Usage:
    python scripts/run_power_analysis.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erg.pnl_dgp import PnLParams
from erg.power import NullPnLConfig, PowerConfig, run_null_pnl_experiment, run_power_analysis


def main() -> None:
    params = PnLParams()
    print("=" * 78)
    print("P&L DGP parameters:")
    print(json.dumps(params.__dict__, indent=2))
    print(f"mean_abs_magnitude (the 'stake' scale): {params.mean_abs_magnitude:.4f}")
    print(f"mean_pnl under the default (fair-game, p_win=0.5) params: {params.mean_pnl:.6f}")

    print("\n" + "=" * 78)
    null_config = NullPnLConfig()
    print(f"NULL SESSIONS: n_tests={null_config.n_tests}, n_per_arm={null_config.n_per_arm}, "
          f"n_looks={null_config.n_looks}, seed={null_config.seed}")
    print("Both arms draw from the identical heavy-tailed P&L distribution (no injected effect).")
    t0 = time.time()
    null_result = run_null_pnl_experiment(null_config, params)
    t1 = time.time()
    print(f"  single-look (fixed horizon) false positive rate: {null_result.single_look_fpr:.4f}")
    print(f"  ANY-LOOK (continuous peeking) false positive rate: {null_result.peeking_fpr:.4f}")
    print(f"  mSPRT always-valid false positive rate (same {null_config.n_looks}-look schedule): {null_result.msprt_fpr:.4f}")
    print(f"  nominal alpha: {null_config.alpha}")
    print(f"  runtime: {t1 - t0:.2f}s")

    print("\n" + "=" * 78)
    power_config = PowerConfig()
    print(f"POWER ANALYSIS: n_trials={power_config.n_trials}, n_per_arm_max={power_config.n_per_arm_max}, "
          f"n_looks={power_config.n_looks}, relative_lift={power_config.relative_lift}, "
          f"power_target={power_config.power_target}, tau_fraction_of_mean={power_config.tau_fraction_of_mean}, "
          f"seed={power_config.seed}")
    t0 = time.time()
    power_result = run_power_analysis(power_config, params)
    t1 = time.time()
    print(f"  lift in dollars (relative_lift * mean_abs_magnitude): {power_result.lift_dollars:.4f}")
    print(f"  per-arm sizes checked: {power_result.sizes.tolist()}")
    print(f"  fixed-horizon (single look) power by size: {[round(x, 4) for x in power_result.fixed_power_by_size.tolist()]}")
    print(f"  mSPRT (continuous monitoring) power by size: {[round(x, 4) for x in power_result.msprt_power_by_size.tolist()]}")
    print(f"  smallest N for {power_config.power_target:.0%} power, fixed-horizon: {power_result.fixed_n_for_target}")
    print(f"  smallest N for {power_config.power_target:.0%} power, mSPRT (cumulative detection by N): {power_result.msprt_n_for_target}")
    print(f"  runtime: {t1 - t0:.2f}s")

    print("\n" + "=" * 78)
    print("SUMMARY (resume claims vs measured):")
    print(
        json.dumps(
            {
                "claim_peeking_fpr_target": 0.26,
                "measured_peeking_fpr": null_result.peeking_fpr,
                "nominal_alpha_target": 0.05,
                "measured_single_look_fpr": null_result.single_look_fpr,
                "claim_msprt_held_under_nominal": True,
                "measured_msprt_fpr": null_result.msprt_fpr,
                "claim_fixed_horizon_sessions_for_80pct_power": 1840,
                "measured_fixed_horizon_sessions_for_80pct_power": power_result.fixed_n_for_target,
                "claim_msprt_sessions_for_80pct_power": 640,
                "measured_msprt_sessions_for_80pct_power": power_result.msprt_n_for_target,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
