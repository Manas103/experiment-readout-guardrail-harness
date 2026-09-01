"""Demonstrate the guardrail harness blocking a readout whose guardrail
metric regressed, even though the primary metric is a significant win.

Usage:
    python scripts/run_guardrail_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from erg.guardrail import Decision, GuardrailSpec, MetricResult, evaluate_readout


def show(title: str, primary: MetricResult, guardrails: list[MetricResult], specs: list[GuardrailSpec]) -> None:
    result = evaluate_readout(primary, guardrails, specs)
    print(f"\n--- {title} ---")
    print(f"primary: {primary}")
    for g in guardrails:
        print(f"guardrail: {g}")
    print(f"DECISION: {result.decision.value}")
    for r in result.reasons:
        print(f"  - {r}")


def main() -> None:
    rg_spec = GuardrailSpec(name="rg_flag_rate", max_regression=0.03, require_significance=True)

    show(
        "Case 1: primary wins big, guardrail regresses significantly -> BLOCK",
        MetricResult("revenue_per_customer", delta=0.08, p_value=0.0003, higher_is_better=True),
        [MetricResult("rg_flag_rate", delta=0.09, p_value=0.01, higher_is_better=False)],
        [rg_spec],
    )

    show(
        "Case 2: primary wins, guardrail clean -> SHIP",
        MetricResult("revenue_per_customer", delta=0.05, p_value=0.001, higher_is_better=True),
        [MetricResult("rg_flag_rate", delta=0.004, p_value=0.7, higher_is_better=False)],
        [rg_spec],
    )

    show(
        "Case 3: primary flat, guardrail clean -> HOLD",
        MetricResult("revenue_per_customer", delta=0.01, p_value=0.4, higher_is_better=True),
        [MetricResult("rg_flag_rate", delta=0.0, p_value=0.9, higher_is_better=False)],
        [rg_spec],
    )

    show(
        "Case 4: guardrail moved past tolerance but not significant -> SHIP with watch flag",
        MetricResult("revenue_per_customer", delta=0.05, p_value=0.001, higher_is_better=True),
        [MetricResult("rg_flag_rate", delta=0.09, p_value=0.4, higher_is_better=False)],
        [rg_spec],
    )


if __name__ == "__main__":
    main()
