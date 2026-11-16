from __future__ import annotations

import numpy as np
from scipy import stats

from erg.dev_engagement_dgp import CohortParams
from erg.srm import (
    MICROSOFT_SRM_THRESHOLD,
    run_clean_splits,
    run_seeded_defects,
    seeded_defect_cases,
    srm_chi2,
    srm_gate_blocks,
)


def test_srm_chi2_matches_scipy_chisquare_oracle():
    """Reference-oracle diff: our hand-written chi-squared formula against
    scipy.stats.chisquare called on the same two-category table, over a
    grid of counts and expected ratios."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        total = int(rng.integers(100, 20000))
        ratio = float(rng.uniform(0.2, 0.8))
        n_a = int(rng.integers(0, total))
        n_b = total - n_a
        stat, p = srm_chi2(n_a, n_b, expected_ratio=ratio)
        expected = [total * ratio, total * (1 - ratio)]
        oracle_stat, oracle_p = stats.chisquare([n_a, n_b], f_exp=expected)
        assert abs(stat - oracle_stat) < 1e-9
        assert abs(p - oracle_p) < 1e-9


def test_srm_gate_threshold_boundary():
    # a 50/50 intended split realized as an extreme 60/40 over 5000 total
    # is a large, obvious mismatch and must block
    assert srm_gate_blocks(3000, 2000, expected_ratio=0.5) is True
    # a 50/50 intended split realized almost exactly as 50/50 must not block
    assert srm_gate_blocks(2501, 2499, expected_ratio=0.5) is False


def test_gate_blocks_strictly_more_often_as_the_split_skews_further():
    """Invariant: holding the total fixed, moving the realized count
    further from the expected count can only ever raise the chi-squared
    statistic (and so can only make the gate more, never less, willing
    to block), since the statistic is a monotonically increasing function
    of |observed - expected|."""
    total = 5000
    expected = total * 0.5
    prev_stat = -1.0
    for n_a in range(2500, 3200, 25):
        stat, _ = srm_chi2(n_a, total - n_a, expected_ratio=0.5)
        assert stat >= prev_stat
        prev_stat = stat


def test_clean_splits_almost_never_trigger_the_gate():
    result = run_clean_splits(n_runs=40)
    # by construction (true fair coin flips, no defect), a false alarm at
    # p < 0.0005 should be rare; this is a nominal-alpha sanity check, not
    # a guarantee of exactly zero
    assert result.false_alarm_rate <= 0.10


def test_seeded_defect_cases_cover_all_four_types():
    cases = seeded_defect_cases("final")
    assert len(cases) == 30
    types = {c.defect_type.value for c in cases}
    assert types == {
        "assignment_bucketing",
        "log_join_loss",
        "variant_triggered_dropout",
        "uneven_ramp",
    }


def test_seeded_defects_are_reproducible():
    """Running the seeded-defect harness twice with the same base seed
    must give byte-identical per-case verdicts; 'seeded' means
    deterministic, not merely random-looking."""
    r1 = run_seeded_defects(severity="final")
    r2 = run_seeded_defects(severity="final")
    assert [r.caught for r in r1.case_results] == [r.caught for r in r2.case_results]
    assert [r.p_value for r in r1.case_results] == [r.p_value for r in r2.case_results]


def test_seeded_defects_catch_rate_beats_the_clean_false_alarm_rate():
    """The whole point of the gate: it should catch a clearly-seeded
    defect corpus far more often than it falsely fires on clean data."""
    defects = run_seeded_defects(severity="final")
    clean = run_clean_splits(n_runs=40)
    defect_catch_rate = defects.n_caught / defects.n_total
    assert defect_catch_rate > clean.false_alarm_rate
    assert defect_catch_rate > 0.5


def test_threshold_is_the_published_microsoft_value():
    assert MICROSOFT_SRM_THRESHOLD == 0.0005
