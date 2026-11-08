"""Fleet autonomy metric family definitions and the per-slice simulation
used by the release gate (`autonomy_release_gate.py`).

A fleet release decision does not score one metric; it scores a whole
family of named metrics (disengagement rate, safety-critical-event rate,
comfort), each one sliced by software version and by geofence, against a
pinned baseline. This module defines that family (24 metrics across 3
named groups) and simulates baseline/candidate event counts per slice
given each slice's exposure, including an explicit mechanism to seed a
known subset of slices with a true rate shift so the release gate's catch
rate and false-discovery rate can be measured against ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GEOFENCES: tuple[str, ...] = (
    "foster_city",
    "san_francisco",
    "las_vegas",
    "austin",
    "phoenix",
    "seattle",
)


@dataclass(frozen=True)
class MetricDef:
    name: str
    family: str  # "disengagement", "safety_critical", "comfort"
    base_rate_per_mile: float  # the pinned baseline's true rate, same across geofences


def _build_metric_family() -> tuple[MetricDef, ...]:
    """8 metrics per family x 3 families = 24, matching the resume's
    named example metrics (miles-per-disengagement, safety-critical-event
    rate, comfort) as the first metric in each family; the other 7 per
    family are named sub-causes/variants at nearby base rates, not padding
    chosen only to reach 24.
    """
    disengagement_causes = [
        "perception_handoff", "planning_handoff", "localization_handoff",
        "communication_handoff", "actuator_fault_handoff", "map_mismatch_handoff",
        "weather_handoff", "construction_zone_handoff",
    ]
    safety_critical_causes = [
        "hard_brake_avoidance", "near_miss_pedestrian", "near_miss_cyclist",
        "near_miss_vehicle", "red_light_violation_avoided", "unprotected_left_conflict",
        "lane_departure_recovery", "stopped_in_travel_lane",
    ]
    comfort_causes = [
        "harsh_brake", "harsh_accel", "harsh_steer", "jerk_event",
        "excess_dwell_pickup", "late_lane_change", "uncomfortable_following_gap",
        "abrupt_yield",
    ]
    base_rates_disengagement = np.array([3.5e-4, 2.8e-4, 2.2e-4, 5.0e-4, 1.8e-4, 4.2e-4, 6.0e-4, 3.0e-4])
    base_rates_safety_critical = np.array([2.0e-5, 1.2e-5, 1.5e-5, 1.8e-5, 1.0e-5, 2.5e-5, 3.0e-5, 1.4e-5])
    base_rates_comfort = np.array([2.2e-3, 1.8e-3, 1.5e-3, 2.6e-3, 3.0e-3, 1.2e-3, 2.0e-3, 1.7e-3])

    metrics = []
    for name, rate in zip(disengagement_causes, base_rates_disengagement):
        metrics.append(MetricDef(name, "disengagement", float(rate)))
    for name, rate in zip(safety_critical_causes, base_rates_safety_critical):
        metrics.append(MetricDef(name, "safety_critical", float(rate)))
    for name, rate in zip(comfort_causes, base_rates_comfort):
        metrics.append(MetricDef(name, "comfort", float(rate)))
    return tuple(metrics)


METRICS: tuple[MetricDef, ...] = _build_metric_family()
assert len(METRICS) == 24


@dataclass
class SliceObservation:
    metric_name: str
    family: str
    geofence: str
    baseline_count: int
    baseline_exposure_miles: float
    candidate_count: int
    candidate_exposure_miles: float
    is_seeded_shift: bool
    true_shift_multiplier: float  # 1.0 for null slices


@dataclass
class ReleaseGateCorpus:
    observations: list[SliceObservation] = field(default_factory=list)
    n_seeded: int = 0


def simulate_release_corpus(
    n_seeded_shifts: int,
    shift_multiplier: float,
    baseline_exposure_miles: float,
    candidate_exposure_miles: float,
    rng: np.random.Generator,
    metrics: tuple[MetricDef, ...] = METRICS,
    geofences: tuple[str, ...] = GEOFENCES,
) -> ReleaseGateCorpus:
    """One release-gate evaluation: every (metric, geofence) slice gets a
    baseline-period count and a candidate-period count, both Poisson draws
    from the pinned base rate times exposure, except for `n_seeded_shifts`
    randomly chosen slices where the candidate period's true rate is
    multiplied by `shift_multiplier` (a true, known regression) before
    drawing the candidate count. Every slice is otherwise null: same true
    rate in both periods, so any flag on a null slice is a false alarm by
    construction, and any flag on a seeded slice is a correct catch.
    """
    all_slices = [(m, g) for m in metrics for g in geofences]
    n_slices = len(all_slices)
    if n_seeded_shifts > n_slices:
        raise ValueError("cannot seed more shifts than there are slices")
    seeded_idx = set(rng.choice(n_slices, size=n_seeded_shifts, replace=False).tolist())

    observations = []
    for i, (metric, geofence) in enumerate(all_slices):
        is_seeded = i in seeded_idx
        true_multiplier = shift_multiplier if is_seeded else 1.0
        baseline_lambda = metric.base_rate_per_mile * baseline_exposure_miles
        candidate_lambda = metric.base_rate_per_mile * true_multiplier * candidate_exposure_miles
        baseline_count = int(rng.poisson(baseline_lambda))
        candidate_count = int(rng.poisson(candidate_lambda))
        observations.append(
            SliceObservation(
                metric_name=metric.name,
                family=metric.family,
                geofence=geofence,
                baseline_count=baseline_count,
                baseline_exposure_miles=baseline_exposure_miles,
                candidate_count=candidate_count,
                candidate_exposure_miles=candidate_exposure_miles,
                is_seeded_shift=is_seeded,
                true_shift_multiplier=true_multiplier,
            )
        )
    return ReleaseGateCorpus(observations=observations, n_seeded=n_seeded_shifts)
