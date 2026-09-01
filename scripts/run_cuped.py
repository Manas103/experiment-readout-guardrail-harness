"""Measure the CUPED sample-size reduction for detecting a 2% relative
effect at 80% power / 5% alpha, using a simulated correlated pre-period
covariate.

Usage:
    python scripts/run_cuped.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erg.cuped import CupedConfig, sample_size_reduction
from erg.dgp import RevenueParams


def main() -> None:
    config = CupedConfig(n_customers=200_000, relative_effect=0.02, alpha=0.05, power=0.80, seed=20260501)
    params = RevenueParams()
    result = sample_size_reduction(config, params)

    print("=" * 78)
    print(f"CUPED sample size reduction, n_customers={config.n_customers}, seed={config.seed}")
    print(
        "pre/post revenue simulated with a shared per-customer latent quality scalar"
        " (persistent payer type + persistent log-spend multiplier, see dgp.py)"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "config"}, indent=2))

    print("\nSUMMARY (resume claim vs measured):")
    print(
        json.dumps(
            {
                "claim_reduction_pct_target": 38.0,
                "measured_reduction_pct": result["reduction_pct"],
                "measured_rho": result["rho"],
                "measured_rho_squared": result["rho2"],
                "n_no_cuped_per_arm": result["n_no_cuped"],
                "n_cuped_per_arm": result["n_cuped"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
