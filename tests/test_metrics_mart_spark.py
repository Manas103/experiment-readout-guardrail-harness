import numpy as np
import pytest

from erg.autonomy_metrics import simulate_release_corpus
from erg.metrics_mart_spark import (
    SparkUnavailableError,
    events_to_raw_rows,
    get_or_create_spark_session,
    rollup_counts_pandas_reference,
    rollup_counts_spark,
)


@pytest.fixture(scope="module")
def spark_session():
    try:
        session = get_or_create_spark_session()
    except SparkUnavailableError as e:
        pytest.skip(f"local Spark session unavailable on this machine/python: {e}")
    yield session
    session.stop()


@pytest.fixture(scope="module")
def small_corpus():
    rng = np.random.default_rng(99)
    return simulate_release_corpus(
        n_seeded_shifts=5,
        shift_multiplier=1.5,
        baseline_exposure_miles=50_000.0,
        candidate_exposure_miles=50_000.0,
        rng=rng,
    )


def test_raw_rows_total_matches_slice_counts(small_corpus):
    raw = events_to_raw_rows(small_corpus)
    expected_total = sum(o.baseline_count + o.candidate_count for o in small_corpus.observations)
    assert len(raw) == expected_total


def test_spark_rollup_matches_pandas_reference_exactly(spark_session, small_corpus):
    raw = events_to_raw_rows(small_corpus)
    spark_result = rollup_counts_spark(raw, spark_session)
    pandas_result = rollup_counts_pandas_reference(raw)
    assert spark_result.equals(pandas_result)


def test_spark_rollup_total_count_matches_raw_row_count(spark_session, small_corpus):
    raw = events_to_raw_rows(small_corpus)
    spark_result = rollup_counts_spark(raw, spark_session)
    assert spark_result["count"].sum() == len(raw)
