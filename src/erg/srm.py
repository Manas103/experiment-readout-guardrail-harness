"""Sample Ratio Mismatch (SRM) gate: a chi-squared check of an experiment's
realized assignment split against its intended ratio, run BEFORE any effect
is read.

Microsoft's own published guidance is that no A/B test result should be
trusted until its sample ratio has been checked, at a chi-squared
p-value below 0.0005, because a skewed split is evidence that something
upstream of the metric (randomization, logging, enrollment) is broken in a
way that can bias the effect estimate itself, not just the headline count.
This module implements exactly that one check: a one-sample chi-squared
goodness-of-fit test of the two realized per-arm counts against the
intended ratio, plus a seeded-defect harness (`dev_engagement_dgp.py`) that
measures how often the gate actually catches four named, mechanistically
different root causes of a mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats

from erg.dev_engagement_dgp import (
    CohortParams,
    assign_clean,
    inject_assignment_bucketing,
    inject_log_join_loss,
    inject_uneven_ramp,
    inject_variant_triggered_dropout,
    observed_counts_clean,
)

MICROSOFT_SRM_THRESHOLD = 0.0005


def srm_chi2(n_a: int, n_b: int, expected_ratio: float = 0.5) -> tuple[float, float]:
    """One-sample chi-squared goodness-of-fit test of an observed 2-arm
    split against its EXPECTED ratio (not against each other symmetrically;
    the two-category chi-squared test is mathematically the same either
    way, but this is the framing Microsoft's own published SRM guidance
    uses: "does the realized split match what randomization promised").
    Returns (chi2 statistic, p-value), df=1.
    """
    total = n_a + n_b
    if total <= 0:
        raise ValueError("total count must be positive")
    expected_a = total * expected_ratio
    expected_b = total * (1.0 - expected_ratio)
    stat = (n_a - expected_a) ** 2 / expected_a + (n_b - expected_b) ** 2 / expected_b
    p = float(stats.chi2.sf(stat, df=1))
    return float(stat), p


def srm_gate_blocks(n_a: int, n_b: int, expected_ratio: float = 0.5, threshold: float = MICROSOFT_SRM_THRESHOLD) -> bool:
    """True if the SRM gate refuses to let an effect be read: p < threshold."""
    _, p = srm_chi2(n_a, n_b, expected_ratio)
    return p < threshold


class DefectType(str, Enum):
    ASSIGNMENT_BUCKETING = "assignment_bucketing"
    LOG_JOIN_LOSS = "log_join_loss"
    VARIANT_TRIGGERED_DROPOUT = "variant_triggered_dropout"
    UNEVEN_RAMP = "uneven_ramp"


@dataclass(frozen=True)
class DefectCase:
    """One seeded defect instance: a name, a type, and the exact keyword
    arguments passed to that type's injector in `dev_engagement_dgp.py`."""

    case_id: str
    defect_type: DefectType
    kwargs: dict


@dataclass
class DefectCaseResult:
    case: DefectCase
    n_control: int
    n_treatment: int
    chi2_stat: float
    p_value: float
    caught: bool


def _run_case(case: DefectCase, params: CohortParams, rng: np.random.Generator) -> DefectCaseResult:
    if case.defect_type is DefectType.ASSIGNMENT_BUCKETING:
        n_control, n_treatment = inject_assignment_bucketing(params, rng, **case.kwargs)
    elif case.defect_type is DefectType.LOG_JOIN_LOSS:
        n_control, n_treatment = inject_log_join_loss(params, rng, **case.kwargs)
    elif case.defect_type is DefectType.VARIANT_TRIGGERED_DROPOUT:
        n_control, n_treatment = inject_variant_triggered_dropout(params, rng, **case.kwargs)
    elif case.defect_type is DefectType.UNEVEN_RAMP:
        n_control, n_treatment = inject_uneven_ramp(params, rng, **case.kwargs)
    else:
        raise ValueError(f"unknown defect type {case.defect_type}")

    stat, p = srm_chi2(n_control, n_treatment, params.p_treatment)
    return DefectCaseResult(
        case=case,
        n_control=n_control,
        n_treatment=n_treatment,
        chi2_stat=stat,
        p_value=p,
        caught=p < MICROSOFT_SRM_THRESHOLD,
    )


def seeded_defect_cases(severity: str = "final") -> list[DefectCase]:
    """30 seeded defect cases (one `case_id` per case, deterministic given
    a fixed base seed in `run_seeded_defects`), 7 or 8 per defect type,
    spanning subtle to severe for each. `severity` selects one of three
    genuine magnitude attempts (see README "What broke"); "final" is the
    design actually reported in the headline numbers.
    """
    if severity == "attempt1":
        bucketing_bias = [0.004, 0.008, 0.012, 0.018, 0.025, 0.035, 0.05, 0.07]
        join_loss = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10]
        dropout_hazard_treat = [0.002, 0.004, 0.006, 0.008, 0.012, 0.018, 0.025]
        ramp_start = [0.35, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.60]
    elif severity == "attempt2":
        bucketing_bias = [0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10]
        join_loss = [0.015, 0.03, 0.045, 0.06, 0.09, 0.12, 0.16]
        dropout_hazard_treat = [0.004, 0.007, 0.010, 0.014, 0.020, 0.028, 0.038]
        ramp_start = [0.30, 0.25, 0.20, 0.15, 0.08, 0.02, 0.70, 0.75]
    else:  # final
        bucketing_bias = [0.015, 0.02, 0.03, 0.04, 0.055, 0.08, 0.11, 0.15]
        join_loss = [0.02, 0.035, 0.05, 0.07, 0.10, 0.14, 0.19]
        dropout_hazard_treat = [0.005, 0.008, 0.012, 0.017, 0.024, 0.033, 0.045]
        ramp_start = [0.25, 0.18, 0.12, 0.05, 0.75, 0.85, 0.90, 0.95]

    cases: list[DefectCase] = []
    for i, bias in enumerate(bucketing_bias):
        cases.append(DefectCase(f"bucketing_{i}", DefectType.ASSIGNMENT_BUCKETING, {"confound_bias": bias}))
    for i, loss in enumerate(join_loss):
        # loss hits one arm only (treatment), control stays clean, per the
        # "a client bug that only fires for one code path" framing
        cases.append(
            DefectCase(
                f"log_join_loss_{i}",
                DefectType.LOG_JOIN_LOSS,
                {"loss_rate_control": 0.0, "loss_rate_treatment": loss},
            )
        )
    for i, hazard in enumerate(dropout_hazard_treat):
        cases.append(
            DefectCase(
                f"dropout_{i}",
                DefectType.VARIANT_TRIGGERED_DROPOUT,
                {"daily_hazard_control": 0.0, "daily_hazard_treatment": hazard},
            )
        )
    for i, start in enumerate(ramp_start):
        cases.append(
            DefectCase(
                f"ramp_{i}",
                DefectType.UNEVEN_RAMP,
                {"ramp_days": 5, "p_treatment_start": start},
            )
        )
    assert len(cases) == 30, f"expected 30 seeded cases, got {len(cases)}"
    return cases


@dataclass
class SeededDefectRunResult:
    params: CohortParams
    severity: str
    case_results: list[DefectCaseResult]

    @property
    def n_caught(self) -> int:
        return sum(1 for r in self.case_results if r.caught)

    @property
    def n_total(self) -> int:
        return len(self.case_results)

    def by_defect_type(self) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for r in self.case_results:
            key = r.case.defect_type.value
            caught, total = out.get(key, [0, 0])
            out[key] = [caught + (1 if r.caught else 0), total + 1]
        return {k: (v[0], v[1]) for k, v in out.items()}


def run_seeded_defects(
    params: CohortParams = CohortParams(),
    severity: str = "final",
    base_seed: int = 20261101,
) -> SeededDefectRunResult:
    """Run all 30 seeded defect cases, each with its own, reproducible
    offset from `base_seed` so re-running this function always reproduces
    the same case-by-case verdicts (a "seeded" defect, not a freshly
    randomized one each time)."""
    cases = seeded_defect_cases(severity)
    results = []
    for i, case in enumerate(cases):
        rng = np.random.default_rng(base_seed + i)
        results.append(_run_case(case, params, rng))
    return SeededDefectRunResult(params=params, severity=severity, case_results=results)


@dataclass
class CleanSplitRunResult:
    params: CohortParams
    n_runs: int
    false_alarms: int
    p_values: list[float]

    @property
    def false_alarm_rate(self) -> float:
        return self.false_alarms / self.n_runs


def run_clean_splits(
    params: CohortParams = CohortParams(),
    n_runs: int = 40,
    base_seed: int = 20261201,
) -> CleanSplitRunResult:
    """Run the SRM gate on `n_runs` genuinely clean (fair-coin-flip,
    no defect) assignments, to measure the gate's false-alarm rate."""
    false_alarms = 0
    p_values = []
    for i in range(n_runs):
        rng = np.random.default_rng(base_seed + i)
        assignment = assign_clean(params, rng)
        n_control, n_treatment = observed_counts_clean(assignment)
        _, p = srm_chi2(n_control, n_treatment, params.p_treatment)
        p_values.append(p)
        if p < MICROSOFT_SRM_THRESHOLD:
            false_alarms += 1
    return CleanSplitRunResult(params=params, n_runs=n_runs, false_alarms=false_alarms, p_values=p_values)
