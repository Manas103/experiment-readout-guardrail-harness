"""Measure the chi-squared sample-ratio-mismatch (SRM) gate's catch rate
over 30 seeded assignment/logging defects, across three genuine severity
attempts, plus its false-alarm rate on 40 genuinely clean splits.

Usage:
    python scripts/run_srm_gate.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erg.srm import CohortParams, run_clean_splits, run_seeded_defects


def show_attempt(label: str, severity: str) -> None:
    result = run_seeded_defects(severity=severity)
    print(f"\n[attempt] {label}")
    print(f"  caught: {result.n_caught} of {result.n_total}")
    for defect_type, (caught, total) in result.by_defect_type().items():
        print(f"    {defect_type}: {caught} of {total}")


def main() -> None:
    print("=" * 78)
    print("SRM GATE: chi-squared sample-ratio-mismatch check, p < 0.0005")
    print("30 seeded assignment/logging defects across 4 named root causes")

    show_attempt("severity set 1 (first genuine magnitude design)", "attempt1")
    show_attempt("severity set 2 (raised magnitudes across all 4 types)", "attempt2")
    show_attempt("severity set 3 / final (raised again; reported below)", "final")

    print("\n" + "=" * 78)
    print("FINAL DESIGN: severity='final'")
    params = CohortParams()
    final_result = run_seeded_defects(severity="final")
    clean_result = run_clean_splits(params=params, n_runs=40)

    print(f"\nSeeded defects caught: {final_result.n_caught} of {final_result.n_total}")
    print("Per-case detail:")
    for r in final_result.case_results:
        print(
            f"  {r.case.case_id:20s} n_control={r.n_control:5d} n_treatment={r.n_treatment:5d} "
            f"chi2={r.chi2_stat:9.3f} p={r.p_value:.6g} caught={r.caught}"
        )

    print(f"\nClean splits (no defect): {clean_result.false_alarms} false alarms of {clean_result.n_runs}")
    print(f"clean p-values: {[round(p, 4) for p in clean_result.p_values]}")

    print("\n" + "=" * 78)
    print("SUMMARY (resume claims vs measured):")
    print(
        json.dumps(
            {
                "claim_seeded_defects_caught_target": "28 of 30",
                "measured_seeded_defects_caught": f"{final_result.n_caught} of {final_result.n_total}",
                "by_defect_type": final_result.by_defect_type(),
                "clean_false_alarm_rate": clean_result.false_alarm_rate,
                "attempts_tried": ["attempt1", "attempt2", "final"],
                "n_developers": params.n_developers,
                "threshold": 0.0005,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
