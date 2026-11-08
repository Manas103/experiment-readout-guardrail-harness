import numpy as np

from erg.autonomy_metrics import GEOFENCES, METRICS, simulate_release_corpus


def test_exactly_24_metrics_across_3_families():
    assert len(METRICS) == 24
    families = {m.family for m in METRICS}
    assert families == {"disengagement", "safety_critical", "comfort"}
    assert sum(m.family == "disengagement" for m in METRICS) == 8
    assert sum(m.family == "safety_critical" for m in METRICS) == 8
    assert sum(m.family == "comfort" for m in METRICS) == 8


def test_base_rates_are_positive():
    for m in METRICS:
        assert m.base_rate_per_mile > 0.0


def test_corpus_has_one_observation_per_metric_geofence_pair():
    rng = np.random.default_rng(1)
    corpus = simulate_release_corpus(
        n_seeded_shifts=30,
        shift_multiplier=1.35,
        baseline_exposure_miles=200_000.0,
        candidate_exposure_miles=200_000.0,
        rng=rng,
    )
    assert len(corpus.observations) == len(METRICS) * len(GEOFENCES)


def test_exactly_n_seeded_shifts_marked():
    rng = np.random.default_rng(2)
    corpus = simulate_release_corpus(
        n_seeded_shifts=30,
        shift_multiplier=1.35,
        baseline_exposure_miles=200_000.0,
        candidate_exposure_miles=200_000.0,
        rng=rng,
    )
    n_marked = sum(o.is_seeded_shift for o in corpus.observations)
    assert n_marked == 30
    for o in corpus.observations:
        if o.is_seeded_shift:
            assert o.true_shift_multiplier == 1.35
        else:
            assert o.true_shift_multiplier == 1.0


def test_rejects_more_seeded_shifts_than_slices():
    rng = np.random.default_rng(3)
    import pytest

    with pytest.raises(ValueError):
        simulate_release_corpus(
            n_seeded_shifts=10_000,
            shift_multiplier=1.35,
            baseline_exposure_miles=1000.0,
            candidate_exposure_miles=1000.0,
            rng=rng,
        )


def test_seeded_slices_have_higher_mean_candidate_count_than_null_on_average():
    """Statistical sanity check, not a bug check on one draw: average over
    many corpora, seeded slices' candidate counts should exceed their
    baseline counts more often than null slices' do."""
    rng = np.random.default_rng(4)
    seeded_diffs = []
    null_diffs = []
    for _ in range(50):
        corpus = simulate_release_corpus(
            n_seeded_shifts=30,
            shift_multiplier=1.35,
            baseline_exposure_miles=200_000.0,
            candidate_exposure_miles=200_000.0,
            rng=rng,
        )
        for o in corpus.observations:
            diff = o.candidate_count - o.baseline_count
            if o.is_seeded_shift:
                seeded_diffs.append(diff)
            else:
                null_diffs.append(diff)
    assert np.mean(seeded_diffs) > np.mean(null_diffs)
