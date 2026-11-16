from __future__ import annotations

import numpy as np

from erg.dev_engagement_dgp import (
    CohortParams,
    assign_clean,
    inject_assignment_bucketing,
    inject_log_join_loss,
    inject_uneven_ramp,
    inject_variant_triggered_dropout,
    observed_counts_clean,
    simulate_daily_events,
)


def test_clean_assignment_close_to_intended_ratio():
    params = CohortParams(n_developers=20000, p_treatment=0.5)
    rng = np.random.default_rng(1)
    assignment = assign_clean(params, rng)
    n_control, n_treatment = observed_counts_clean(assignment)
    assert n_control + n_treatment == params.n_developers
    assert abs(n_treatment / params.n_developers - 0.5) < 0.02


def test_simulate_daily_events_shape_and_nonnegative():
    params = CohortParams(n_developers=200, n_days=10)
    rng = np.random.default_rng(2)
    assignment = assign_clean(params, rng)
    events = simulate_daily_events(assignment, params, rng)
    assert events.shape == (params.n_developers, params.n_days)
    assert (events >= 0).all()


def test_log_join_loss_zero_rates_reduces_to_clean_assignment():
    params = CohortParams(n_developers=5000, p_treatment=0.5)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    expected_control, expected_treatment = observed_counts_clean(assign_clean(params, rng_a))
    n_control, n_treatment = inject_log_join_loss(
        params, rng_b, loss_rate_control=0.0, loss_rate_treatment=0.0
    )
    assert (n_control, n_treatment) == (expected_control, expected_treatment)


def test_log_join_loss_removes_only_the_lossy_arm():
    params = CohortParams(n_developers=5000, p_treatment=0.5)
    rng_clean = np.random.default_rng(7)
    expected_control, _ = observed_counts_clean(assign_clean(params, rng_clean))
    rng_loss = np.random.default_rng(7)
    n_control, n_treatment = inject_log_join_loss(
        params, rng_loss, loss_rate_control=0.0, loss_rate_treatment=0.5
    )
    # control arm is untouched by a treatment-only logging bug
    assert n_control == expected_control
    assert n_treatment < expected_control  # roughly halved, well below the clean count


def test_variant_triggered_dropout_zero_hazard_keeps_everyone():
    params = CohortParams(n_developers=3000, n_days=10, p_treatment=0.5)
    rng_a = np.random.default_rng(99)
    rng_b = np.random.default_rng(99)
    expected_control, expected_treatment = observed_counts_clean(assign_clean(params, rng_a))
    n_control, n_treatment = inject_variant_triggered_dropout(
        params, rng_b, daily_hazard_control=0.0, daily_hazard_treatment=0.0
    )
    assert (n_control, n_treatment) == (expected_control, expected_treatment)


def test_variant_triggered_dropout_shrinks_only_hazardous_arm():
    params = CohortParams(n_developers=4000, n_days=14, p_treatment=0.5)
    rng_clean = np.random.default_rng(13)
    expected_control, _ = observed_counts_clean(assign_clean(params, rng_clean))
    rng_dropout = np.random.default_rng(13)
    n_control, n_treatment = inject_variant_triggered_dropout(
        params, rng_dropout, daily_hazard_control=0.0, daily_hazard_treatment=0.05
    )
    assert n_control == expected_control
    assert n_treatment < expected_control


def test_uneven_ramp_conserves_total_developers():
    params = CohortParams(n_developers=5000, n_days=14, p_treatment=0.5)
    rng = np.random.default_rng(5)
    n_control, n_treatment = inject_uneven_ramp(params, rng, ramp_days=5, p_treatment_start=0.1)
    assert n_control + n_treatment == params.n_developers


def test_uneven_ramp_with_no_actual_ramp_effect_stays_close_to_intended_split():
    """Starting the 'ramp' at the already-intended ratio is a no-op defect;
    the realized split should land close to 50/50, same as a clean assignment."""
    params = CohortParams(n_developers=20000, n_days=14, p_treatment=0.5)
    rng = np.random.default_rng(6)
    n_control, n_treatment = inject_uneven_ramp(params, rng, ramp_days=5, p_treatment_start=0.5)
    assert abs(n_treatment / (n_control + n_treatment) - 0.5) < 0.02


def test_assignment_bucketing_matches_closed_form_expectation_on_average():
    """Reference-oracle check: half the cohort is 'senior' (biased toward
    treatment at 0.5 + bias), half is not (fair 0.5), so the expected
    treatment fraction is exactly 0.5 + bias/2. Average many independent
    runs to beat sampling noise and compare to that closed form."""
    params = CohortParams(n_developers=4000, p_treatment=0.5)
    bias = 0.08
    fractions = []
    for seed in range(30):
        rng = np.random.default_rng(1000 + seed)
        n_control, n_treatment = inject_assignment_bucketing(params, rng, confound_bias=bias)
        fractions.append(n_treatment / (n_control + n_treatment))
    measured = float(np.mean(fractions))
    expected = 0.5 + bias / 2
    assert abs(measured - expected) < 0.01


def test_assignment_bucketing_zero_bias_matches_clean_assignment_rate():
    params = CohortParams(n_developers=20000, p_treatment=0.5)
    rng = np.random.default_rng(21)
    n_control, n_treatment = inject_assignment_bucketing(params, rng, confound_bias=0.0)
    assert abs(n_treatment / (n_control + n_treatment) - 0.5) < 0.02
