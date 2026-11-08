"""Builds the raw-event-to-slice-count rollup with a real local PySpark
job and cross-checks it against an independent pandas groupby, for the
same release-gate corpus the headline measurement uses. Must be run
under WSL2 on this machine (see metrics_mart_spark.py's module docstring
for why); on native Windows this prints a clean "unavailable" message
and exits 0 rather than crashing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from erg.autonomy_metrics import simulate_release_corpus
from erg.metrics_mart_spark import (
    SparkUnavailableError,
    events_to_raw_rows,
    get_or_create_spark_session,
    rollup_counts_pandas_reference,
    rollup_counts_spark,
)


def main() -> None:
    rng = np.random.default_rng(2026090103)
    corpus = simulate_release_corpus(
        n_seeded_shifts=30,
        shift_multiplier=1.60,
        baseline_exposure_miles=500_000.0,
        candidate_exposure_miles=500_000.0,
        rng=rng,
    )
    raw = events_to_raw_rows(corpus)
    print(f"Raw per-event rows expanded from {len(corpus.observations)} slices: {len(raw)} total events")

    try:
        spark = get_or_create_spark_session()
    except SparkUnavailableError as e:
        print(f"Local Spark session unavailable on this machine/python: {e}")
        print("Skipping cleanly, as documented in metrics_mart_spark.py.")
        return

    try:
        t0 = time.perf_counter()
        spark_result = rollup_counts_spark(raw, spark)
        spark_elapsed = time.perf_counter() - t0

        t0 = time.perf_counter()
        pandas_result = rollup_counts_pandas_reference(raw)
        pandas_elapsed = time.perf_counter() - t0

        matches = spark_result.equals(pandas_result)
        print(f"Spark rollup: {len(spark_result)} (metric, geofence, period) rows in {spark_elapsed:.3f}s")
        print(f"Pandas reference rollup: {len(pandas_result)} rows in {pandas_elapsed:.3f}s")
        print(f"Spark rollup exactly matches independent pandas reference: {matches}")
        print(f"Total event count agrees: {spark_result['count'].sum()} (spark) == {pandas_result['count'].sum()} (pandas) == {len(raw)} (raw rows)")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
