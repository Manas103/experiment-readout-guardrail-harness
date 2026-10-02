"""Measure the 20-metric family false-win rate, uncorrected and under
Benjamini-Hochberg FDR control, under a global null (no true effect on any
metric).

Usage:
    python scripts/run_metric_family.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erg.metric_family import MetricFamilyConfig, run_metric_family_experiment


def main() -> None:
    print("=" * 78)
    print("METRIC FAMILY SIMULATION: 20 metrics scored per experiment, global null")
    print("(no true effect on any of the 20 metrics in any trial)")

    attempts = [
        ("rho=0.30 (correlated metric family, realistic default)", 0.30),
        ("rho=0.15 (weaker correlation)", 0.15),
        ("rho=0.00 (independent metrics, closed-form checkable)", 0.00),
    ]
    for label, rho in attempts:
        t0 = time.time()
        config = MetricFamilyConfig(n_metrics=20, n_trials=40000, rho=rho, seed=20260915)
        result = run_metric_family_experiment(config)
        t1 = time.time()
        print(f"\n[attempt] {label}")
        print(f"  naive false-win rate (any of 20 raw p < 0.05): {result.naive_false_win_rate:.4f}")
        print(f"  BH-controlled false-win rate (fdr_q=0.05):      {result.bh_false_win_rate:.4f}")
        print(f"  runtime: {t1 - t0:.2f}s")

    print("\n" + "=" * 78)
    print("FINAL DESIGN: rho=0.00 (independent metrics), n_trials=200000 for precision")
    final_config = MetricFamilyConfig(n_metrics=20, n_trials=200_000, rho=0.0, seed=20260915)
    t0 = time.time()
    final_result = run_metric_family_experiment(final_config)
    t1 = time.time()
    closed_form_naive = 1.0 - (1.0 - final_config.alpha) ** final_config.n_metrics
    print(f"  naive false-win rate:        {final_result.naive_false_win_rate:.5f}")
    print(f"  closed-form check (1-0.95^20): {closed_form_naive:.5f}")
    print(f"  BH-controlled false-win rate: {final_result.bh_false_win_rate:.5f}")
    print(f"  runtime: {t1 - t0:.2f}s")

    print("\n" + "=" * 78)
    print("SUMMARY (resume claims vs measured):")
    print(
        json.dumps(
            {
                "claim_naive_false_win_target": 0.64,
                "measured_naive_false_win_rate": final_result.naive_false_win_rate,
                "closed_form_naive_false_win_rate": closed_form_naive,
                "claim_bh_false_win_target": 0.048,
                "measured_bh_false_win_rate": final_result.bh_false_win_rate,
                "n_metrics": final_config.n_metrics,
                "n_trials": final_config.n_trials,
                "rho": final_config.rho,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
