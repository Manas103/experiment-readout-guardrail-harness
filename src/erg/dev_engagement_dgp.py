"""Data generating process for a developer-engagement A/B assignment, for
the "SRM gate" extension (Nov. 2026, Microsoft Data Scientist Intern,
CoreAI req).

This is a third, parallel data-generating process alongside `dgp.py`'s
zero-inflated revenue-per-customer metric and `pnl_dgp.py`'s signed P&L
metric, not a replacement for either: the resume bullet for this req
measures a chi-squared sample-ratio-mismatch (SRM) gate, and SRM is a
property of the ASSIGNMENT AND LOGGING pipeline, not of any particular
metric's distribution. What matters here is not how much a developer uses
the product (no metric value below is ever compared between arms), it is
how many developers EACH ARM ENDS UP WITH once a specific, named failure
mode has corrupted the assignment-to-logging path. The per-developer daily
usage-event stream is simulated anyway, and the per-arm counts the SRM
gate actually tests are always derived from that stream (not drawn
directly as a biased coin flip), so that every defect is a genuine
property of a simulated logging pipeline, not a parameter handed straight
to the test.

Four named defect types are modeled, after the published SRM root-cause
taxonomy (assignment bucketing, log-join loss, variant-triggered dropout,
uneven ramp):

1. `inject_assignment_bucketing`: the randomization itself is not a fair
   coin flip. A deterministic hash of developer id is used to bucket
   developers, and that hash is correlated with a confound (here,
   seniority band), systematically over-assigning one band to treatment.
2. `inject_log_join_loss`: assignment is a fair coin flip, but the join
   between the assignment table and the event-logging pipeline drops rows
   at a different rate per arm (e.g. a client-side logging bug that only
   fires for one code path).
3. `inject_variant_triggered_dropout`: assignment is fair, logging is
   complete, but developers in one arm are more likely to STOP
   contributing logs partway through the observation window because the
   variant itself drove them away (e.g. a broken feature that gets
   disabled locally). This is simulated as a day-by-day attrition hazard
   that differs by arm, not as an instantaneous logging failure, which is
   what makes it mechanistically distinct from (2).
4. `inject_uneven_ramp`: assignment is fair and stable once the experiment
   is fully ramped, but the first several days ran at an unannounced,
   different split while a feature flag was still ramping up, and the SRM
   gate was run on the FULL window including that ramp period.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CohortParams:
    """Parameters of the clean (no-defect) developer cohort.

    n_developers: total developers enrolled in the experiment window.
    p_treatment: the INTENDED assignment ratio (0.5 = a 50/50 split).
    n_days: length of the observation window used for per-day attrition
        and ramp defects.
    daily_event_rate: mean of a Poisson daily product-usage-event count
        per active developer-day (used only to make the log table
        realistic; the SRM gate never reads event counts, only presence).
    """

    n_developers: int = 5000
    p_treatment: float = 0.5
    n_days: int = 14
    daily_event_rate: float = 3.0


def assign_clean(params: CohortParams, rng: np.random.Generator) -> np.ndarray:
    """Fair, independent coin-flip assignment: True = treatment."""
    return rng.uniform(size=params.n_developers) < params.p_treatment


def simulate_daily_events(
    assignment: np.ndarray, params: CohortParams, rng: np.random.Generator
) -> np.ndarray:
    """Per-developer, per-day usage-event counts, shape (n_developers, n_days).

    Every developer in the clean cohort is active every day (Poisson
    counts can be 0, but the developer still has a log row for that day);
    attrition and logging loss are applied on top of this by the defect
    injectors below, never baked in here.
    """
    return rng.poisson(
        lam=params.daily_event_rate, size=(params.n_developers, params.n_days)
    )


def observed_counts_clean(assignment: np.ndarray) -> tuple[int, int]:
    """Per-arm developer counts with no defect applied."""
    n_treatment = int(assignment.sum())
    n_control = int(assignment.size - n_treatment)
    return n_control, n_treatment


def inject_assignment_bucketing(
    params: CohortParams, rng: np.random.Generator, confound_bias: float
) -> tuple[int, int]:
    """Assignment-level SRM: a hash-bucketing scheme correlated with a
    confound (seniority band) instead of a fair coin flip.

    Half of developers are drawn as "senior" (a stand-in for any
    confound correlated with the bucketing key, e.g. a hash of employee
    id that happens to correlate with tenure cohorts created at the same
    time). Senior developers are assigned to treatment at
    `0.5 + confound_bias`; everyone else stays at a fair 0.5. The result
    is a systematic, reproducible skew in the realized split, not sampling
    noise.
    """
    is_senior = rng.uniform(size=params.n_developers) < 0.5
    p_treat = np.where(is_senior, 0.5 + confound_bias, 0.5)
    assignment = rng.uniform(size=params.n_developers) < p_treat
    return observed_counts_clean(assignment)


def inject_log_join_loss(
    params: CohortParams,
    rng: np.random.Generator,
    loss_rate_control: float,
    loss_rate_treatment: float,
) -> tuple[int, int]:
    """Logging-pipeline SRM: assignment is fair, but a developer's row is
    dropped from the join between the assignment table and the event log
    at a different rate per arm (e.g. a client bug that silently fails to
    emit the join key for one variant more often than the other)."""
    assignment = assign_clean(params, rng)
    loss_rate = np.where(assignment, loss_rate_treatment, loss_rate_control)
    survives_join = rng.uniform(size=params.n_developers) >= loss_rate
    surviving = assignment[survives_join]
    n_treatment = int(surviving.sum())
    n_control = int(surviving.size - n_treatment)
    return n_control, n_treatment


def inject_variant_triggered_dropout(
    params: CohortParams,
    rng: np.random.Generator,
    daily_hazard_control: float,
    daily_hazard_treatment: float,
) -> tuple[int, int]:
    """Attrition SRM: assignment and logging both start clean, but each
    day a developer has an arm-specific hazard of going permanently
    silent (e.g. the treatment variant breaks their workflow and they
    stop touching the product, so no further log rows are ever produced
    for them). Counted as "observed" at the gate: present in the log on
    the LAST day of the window, i.e. never dropped out."""
    assignment = assign_clean(params, rng)
    hazard = np.where(assignment, daily_hazard_treatment, daily_hazard_control)
    still_active = np.ones(params.n_developers, dtype=bool)
    for _ in range(params.n_days):
        dropout_today = rng.uniform(size=params.n_developers) < hazard
        still_active &= ~dropout_today
    surviving = assignment[still_active]
    n_treatment = int(surviving.sum())
    n_control = int(surviving.size - n_treatment)
    return n_control, n_treatment


def inject_uneven_ramp(
    params: CohortParams,
    rng: np.random.Generator,
    ramp_days: int,
    p_treatment_start: float,
) -> tuple[int, int]:
    """Ramp-schedule SRM: the feature flag's rollout started at
    `p_treatment_start` (an unannounced, non-50/50 split used to limit
    blast radius while confidence was built) and transitioned linearly to
    the intended `params.p_treatment` by `ramp_days`, after which it held
    steady. Developers are enrolled at a constant rate across the whole
    `n_days` window (new developers trickle into the experiment every
    day, not all on day one), and the SRM gate is run on the FULL
    enrollment window including the ramp period, so the overall split is
    pulled away from 50/50 by however many developers enrolled before the
    ramp finished."""
    per_day = params.n_developers // params.n_days
    remainder = params.n_developers - per_day * params.n_days
    n_treatment_total = 0
    n_control_total = 0
    for day in range(params.n_days):
        n_today = per_day + (remainder if day == params.n_days - 1 else 0)
        if day < ramp_days:
            frac = day / max(ramp_days - 1, 1)
            p_today = p_treatment_start + frac * (params.p_treatment - p_treatment_start)
        else:
            p_today = params.p_treatment
        assigned_treatment = int(rng.binomial(n_today, p_today))
        n_treatment_total += assigned_treatment
        n_control_total += n_today - assigned_treatment
    return n_control_total, n_treatment_total
