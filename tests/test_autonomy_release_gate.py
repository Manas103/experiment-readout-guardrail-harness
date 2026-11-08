import numpy as np
import pytest
from scipy import stats

from erg.autonomy_metrics import simulate_release_corpus
from erg.autonomy_release_gate import (
    RuleScorecard,
    log_rate_ratio_p_value,
    run_release_gate,
    score_rule,
)
from erg.autonomy_metrics import SliceObservation


def _make_obs(base_count, base_exposure, cand_count, cand_exposure, seeded=False):
    return SliceObservation(
        metric_name="m", family="disengagement", geofence="g",
        baseline_count=base_count, baseline_exposure_miles=base_exposure,
        candidate_count=cand_count, candidate_exposure_miles=cand_exposure,
        is_seeded_shift=seeded, true_shift_multiplier=1.0,
    )


def test_identical_rates_give_a_high_p_value_on_average():
    """Many null (no true difference) slices should mostly NOT be
    significant at alpha=0.05, since there is nothing to detect."""
    rng = np.random.default_rng(1)
    flags = []
    for _ in range(500):
        base = rng.poisson(50)
        cand = rng.poisson(50)
        _, p = log_rate_ratio_p_value(_make_obs(base, 100_000.0, cand, 100_000.0))
        flags.append(p < 0.05)
    # should be close to the 5% nominal false-positive rate, generously bounded
    assert np.mean(flags) < 0.15


def test_clearly_different_rates_are_significant():
    _, p = log_rate_ratio_p_value(_make_obs(20, 100_000.0, 80, 100_000.0))
    assert p < 0.001


def test_zero_counts_do_not_crash_due_to_continuity_correction():
    theta, p = log_rate_ratio_p_value(_make_obs(0, 100_000.0, 0, 100_000.0))
    assert np.isfinite(theta)
    assert np.isfinite(p)


def _exact_conditional_binomial_p_value(obs: SliceObservation) -> float:
    """Independent reference oracle: exact conditional test for two
    Poisson rates. Given n = x_base + x_cand, under H0 (equal rates),
    x_cand is Binomial(n, T_cand / (T_base + T_cand)). This is a
    different, exact method (no normal approximation, no continuity
    correction) used only to cross-check the approximate z-test's
    significance decisions, not to replace it.
    """
    n = obs.baseline_count + obs.candidate_count
    if n == 0:
        return 1.0
    p0 = obs.candidate_exposure_miles / (obs.baseline_exposure_miles + obs.candidate_exposure_miles)
    return float(stats.binomtest(obs.candidate_count, n, p0, alternative="two-sided").pvalue)


def test_approximate_test_agrees_with_exact_conditional_binomial_oracle_on_most_cases():
    """The delta-method z-test is an approximation; an exact conditional
    binomial test is the independent, non-obvious-correctness oracle for
    the same null hypothesis. The two should agree on the reject/accept
    decision at alpha=0.05 in the large majority of a swept grid of
    counts, with disagreement expected and disclosed only near the
    decision boundary (an asymptotic test and an exact test need not agree
    exactly there)."""
    rng = np.random.default_rng(42)
    agreements = 0
    total = 0
    for _ in range(2000):
        base_count = int(rng.integers(0, 200))
        cand_count = int(rng.integers(0, 200))
        obs = _make_obs(base_count, 100_000.0, cand_count, 100_000.0)
        _, p_approx = log_rate_ratio_p_value(obs)
        p_exact = _exact_conditional_binomial_p_value(obs)
        agreements += (p_approx < 0.05) == (p_exact < 0.05)
        total += 1
    agreement_rate = agreements / total
    assert agreement_rate > 0.90


def test_run_release_gate_bh_never_flags_more_than_naive_raw_p_rule():
    rng = np.random.default_rng(5)
    corpus = simulate_release_corpus(
        n_seeded_shifts=30, shift_multiplier=1.35,
        baseline_exposure_miles=200_000.0, candidate_exposure_miles=200_000.0, rng=rng,
    )
    result = run_release_gate(corpus)
    naive_raw_p_flagged = result.p_values < 0.05
    # BH is strictly more conservative than the uncorrected rule at the same alpha
    assert np.all(~result.bh_flagged | naive_raw_p_flagged)


def test_score_rule_arithmetic_on_a_constructed_case():
    rng = np.random.default_rng(6)
    corpus = simulate_release_corpus(
        n_seeded_shifts=4, shift_multiplier=2.0,
        baseline_exposure_miles=500_000.0, candidate_exposure_miles=500_000.0, rng=rng,
    )
    flagged = np.array([o.is_seeded_shift for o in corpus.observations])  # a rule that flags exactly the seeded slices
    card = score_rule(flagged, corpus)
    assert card.catch_rate == 1.0
    assert card.false_discovery_rate == 0.0
    assert card.false_alarm_rate == 0.0
    assert card.n_caught == 4


def test_score_rule_counts_false_alarms_correctly():
    rng = np.random.default_rng(7)
    corpus = simulate_release_corpus(
        n_seeded_shifts=2, shift_multiplier=2.0,
        baseline_exposure_miles=500_000.0, candidate_exposure_miles=500_000.0, rng=rng,
    )
    flagged = np.ones(len(corpus.observations), dtype=bool)  # flag everything
    card = score_rule(flagged, corpus)
    assert card.catch_rate == 1.0
    n_null = len(corpus.observations) - 2
    assert card.n_flagged == len(corpus.observations)
    assert card.false_alarm_rate == pytest.approx(n_null / n_null)  # every null slice flagged
