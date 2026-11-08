"""The release gate itself: a per-slice rate-ratio significance test, two
competing decision rules (Benjamini-Hochberg across the whole metric
family, and a naive fixed-threshold rule applied slice by slice), and the
measurement of each rule's catch rate against the seeded shifts and its
false-alarm rate against the null slices.

The statistical core (Benjamini-Hochberg step-up) is reused unchanged from
`metric_family.bh_reject_hand_rolled`, verified in that module's own test
suite against `statsmodels`; this module only supplies the rate-ratio
p-values the autonomy metric family actually produces.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from erg.autonomy_metrics import ReleaseGateCorpus, SliceObservation
from erg.metric_family import bh_reject_hand_rolled

HALDANE_ANSCOMBE_CORRECTION = 0.5  # added to counts before taking a log, so a zero count is defined


def log_rate_ratio_p_value(obs: SliceObservation) -> tuple[float, float]:
    """Two-sided p-value for H0: candidate rate == baseline rate, via the
    delta-method log-rate-ratio z-test standard in incidence-rate-ratio
    epidemiology: theta = log(rate_candidate / rate_baseline),
    Var(theta) ~= 1/x_baseline + 1/x_candidate (both counts continuity-
    corrected by 0.5 so a zero count does not make log or the variance
    undefined). Returns (log_rate_ratio, p_value).
    """
    x_base = obs.baseline_count + HALDANE_ANSCOMBE_CORRECTION
    x_cand = obs.candidate_count + HALDANE_ANSCOMBE_CORRECTION
    rate_base = x_base / obs.baseline_exposure_miles
    rate_cand = x_cand / obs.candidate_exposure_miles
    theta = np.log(rate_cand / rate_base)
    var_theta = 1.0 / x_base + 1.0 / x_cand
    z = theta / np.sqrt(var_theta)
    p = float(2.0 * stats.norm.sf(np.abs(z)))
    return float(theta), p


@dataclass
class GateResult:
    bh_flagged: np.ndarray  # bool, one per slice, in corpus.observations order
    fixed_threshold_flagged: np.ndarray
    log_rate_ratios: np.ndarray
    p_values: np.ndarray


def run_release_gate(
    corpus: ReleaseGateCorpus, fdr_q: float = 0.05, fixed_relative_threshold: float = 0.15
) -> GateResult:
    """Evaluate both decision rules over every slice in one release-gate
    family at once (a single family-wise BH correction across all
    slices, matching the claim "Benjamini-Hochberg control across the
    metric family").

    fixed_relative_threshold: the naive baseline rule flags any slice
    whose candidate rate exceeds its own baseline rate by more than this
    fraction, regardless of statistical significance. This is the rule a
    team would reach for without any multiple-comparisons machinery at
    all: "did the number go up by more than X%?"
    """
    thetas = np.empty(len(corpus.observations))
    pvals = np.empty(len(corpus.observations))
    fixed_flagged = np.empty(len(corpus.observations), dtype=bool)

    for i, obs in enumerate(corpus.observations):
        theta, p = log_rate_ratio_p_value(obs)
        thetas[i] = theta
        pvals[i] = p
        rate_base = obs.baseline_count / obs.baseline_exposure_miles if obs.baseline_count > 0 else 0.0
        rate_cand = obs.candidate_count / obs.candidate_exposure_miles if obs.candidate_count > 0 else 0.0
        if rate_base == 0.0:
            fixed_flagged[i] = rate_cand > 0.0
        else:
            fixed_flagged[i] = (rate_cand - rate_base) / rate_base > fixed_relative_threshold

    bh_flagged = bh_reject_hand_rolled(pvals, fdr_q)
    return GateResult(
        bh_flagged=bh_flagged,
        fixed_threshold_flagged=fixed_flagged,
        log_rate_ratios=thetas,
        p_values=pvals,
    )


@dataclass
class RuleScorecard:
    catch_rate: float  # fraction of seeded shifts flagged
    n_caught: int
    n_seeded: int
    false_discovery_rate: float  # fraction of ALL flags that are not seeded shifts
    false_alarm_rate: float  # fraction of null slices that are (falsely) flagged
    n_flagged: int


def score_rule(flagged: np.ndarray, corpus: ReleaseGateCorpus) -> RuleScorecard:
    is_seeded = np.array([o.is_seeded_shift for o in corpus.observations])
    n_seeded = int(is_seeded.sum())
    n_null = len(corpus.observations) - n_seeded

    n_caught = int((flagged & is_seeded).sum())
    n_flagged = int(flagged.sum())
    n_false_flags = int((flagged & ~is_seeded).sum())

    catch_rate = n_caught / n_seeded if n_seeded else float("nan")
    fdr = n_false_flags / n_flagged if n_flagged else 0.0
    false_alarm_rate = n_false_flags / n_null if n_null else 0.0

    return RuleScorecard(
        catch_rate=catch_rate,
        n_caught=n_caught,
        n_seeded=n_seeded,
        false_discovery_rate=fdr,
        false_alarm_rate=false_alarm_rate,
        n_flagged=n_flagged,
    )
