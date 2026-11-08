"""Builds the versioned metrics mart's per-slice event counts with a real
local PySpark job, from a raw, one-row-per-event table, rather than
simulating the counts directly with `numpy.random.poisson`.

`autonomy_metrics.simulate_release_corpus` simulates the fleet by drawing
a Poisson COUNT directly for each (metric, geofence, baseline/candidate)
slice, which is the right level of fidelity for measuring the release
gate's statistics. This module instead starts one level further back, at
individual raw event rows (one row per simulated disengagement or
comfort/safety-critical event, each tagged with its metric, geofence and
period), and rolls them up into the same per-slice counts with a Spark
DataFrame groupBy-count, the shape a real fleet telemetry mart's rollup
job would actually take. The two paths are cross-checked for exact
agreement in `tests/test_metrics_mart_spark.py` (skipped cleanly if no
local Spark session can be started on this machine; see the README's
"Honest framing" for why that guard exists).

Must be run under WSL2 on this machine: local PySpark reliably starts a
Python worker process on WSL2's Linux Python, but repeatedly crashed the
worker on native Windows Python 3.12 after two genuine attempts (pyspark
3.5.3 and 4.0.1, both with JAVA_HOME and PYSPARK_PYTHON set correctly);
see the README for the exact errors. `get_or_create_spark_session` raises
`SparkUnavailableError` if no session can be started at all, and callers
are expected to skip cleanly in that case.
"""
from __future__ import annotations

import pandas as pd


class SparkUnavailableError(RuntimeError):
    pass


def get_or_create_spark_session():
    try:
        from pyspark.sql import SparkSession
    except ImportError as e:
        raise SparkUnavailableError(f"pyspark not importable: {e}") from e
    session = (
        SparkSession.builder.master("local[2]")
        .appName("fleet-metrics-mart")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    try:
        # getOrCreate() succeeds even when the JVM cannot actually launch a
        # Python worker process (the native-Windows failure mode this repo
        # hit twice); force one real Python-worker round trip here so a
        # caller gets SparkUnavailableError up front, not a mid-test crash.
        session.sparkContext.parallelize([1]).map(lambda x: x).collect()
    except Exception as e:  # pragma: no cover - environment-dependent
        session.stop()
        raise SparkUnavailableError(f"could not start a working local Spark session: {e}") from e
    return session


def events_to_raw_rows(corpus) -> pd.DataFrame:
    """Expand every slice's baseline_count and candidate_count into one
    row per individual simulated event, the raw shape a telemetry
    ingestion pipeline would actually produce (one row per detected
    event), rather than a pre-aggregated count.
    """
    rows = []
    for o in corpus.observations:
        for _ in range(o.baseline_count):
            rows.append((o.metric_name, o.family, o.geofence, "baseline"))
        for _ in range(o.candidate_count):
            rows.append((o.metric_name, o.family, o.geofence, "candidate"))
    return pd.DataFrame(rows, columns=["metric_name", "family", "geofence", "period"])


def rollup_counts_spark(raw_events: pd.DataFrame, spark) -> pd.DataFrame:
    """Groups the raw per-event rows by (metric_name, family, geofence,
    period) and counts, using a real Spark DataFrame aggregation, and
    returns the result as a pandas DataFrame sorted for deterministic
    comparison against the pandas reference.
    """
    sdf = spark.createDataFrame(raw_events)
    rolled = (
        sdf.groupBy("metric_name", "family", "geofence", "period")
        .count()
        .toPandas()
        .sort_values(["metric_name", "geofence", "period"])
        .reset_index(drop=True)
    )
    return rolled


def rollup_counts_pandas_reference(raw_events: pd.DataFrame) -> pd.DataFrame:
    """Independent pandas groupby used only as a reference oracle against
    the Spark rollup, never as the production path."""
    return (
        raw_events.groupby(["metric_name", "family", "geofence", "period"])
        .size()
        .reset_index(name="count")
        .sort_values(["metric_name", "geofence", "period"])
        .reset_index(drop=True)
    )
