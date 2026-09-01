"""Readout decision harness with guardrail checks.

A "readout" is the output of an experiment analysis: a primary metric
result plus zero or more guardrail metric results (protective metrics that
must not regress, e.g. a complaint rate or a responsible-gaming signal).
The harness's whole reason to exist is that a significant, positive
primary metric is NOT sufficient to ship: if a guardrail metric regressed
beyond its tolerance and that regression is itself statistically
significant, the harness blocks the readout, full stop, regardless of how
good the primary metric looks. This is the guardrail check the resume
bullet claims, and it is the one invariant the test suite enforces most
aggressively (see tests/test_guardrail.py::test_never_ships_with_blocked_guardrail).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Decision(str, Enum):
    SHIP = "ship"
    HOLD = "hold"
    BLOCK = "block"


@dataclass(frozen=True)
class MetricResult:
    """One metric's measured result for a single experiment.

    name: metric identifier, e.g. "revenue_per_customer" or "complaint_rate".
    delta: relative change of treatment vs control, e.g. 0.03 means +3%.
    p_value: two-sided p-value for the delta being different from zero.
    higher_is_better: True if a larger delta is the desired direction
        (revenue). False if a larger delta is bad (complaint rate,
        problem-gambling flag rate): for those metrics a positive delta is
        a regression, not an improvement.
    """

    name: str
    delta: float
    p_value: float
    higher_is_better: bool = True


@dataclass(frozen=True)
class GuardrailSpec:
    """Tolerance for one guardrail metric.

    max_regression: the largest tolerated move in the bad direction,
        expressed as a positive fraction, e.g. 0.03 means the harness
        tolerates up to a 3% regression before it is willing to block.
    require_significance: if True (the default, and the honest choice),
        the harness only blocks on a regression that is itself
        statistically significant at `alpha`; an regression that could
        plausibly be noise is surfaced as a hold-level warning, not a hard
        block, since blocking on noisy guardrail readings would make the
        harness useless in practice (every launch would eventually trip
        some guardrail by chance). This is a documented design tradeoff,
        not a loophole: it never allows a *significant* regression past
        threshold to ship.
    """

    name: str
    max_regression: float
    require_significance: bool = True


@dataclass
class ReadoutDecision:
    decision: Decision
    reasons: list[str] = field(default_factory=list)
    blocked_guardrails: list[str] = field(default_factory=list)
    watch_guardrails: list[str] = field(default_factory=list)


def _is_regression(result: MetricResult, spec: GuardrailSpec) -> bool:
    if result.name != spec.name:
        raise ValueError(f"guardrail spec/result name mismatch: {spec.name} vs {result.name}")
    if result.higher_is_better:
        # a drop is bad
        return -result.delta > spec.max_regression
    # a rise is bad
    return result.delta > spec.max_regression


def evaluate_readout(
    primary: MetricResult,
    guardrails: list[MetricResult],
    guardrail_specs: list[GuardrailSpec],
    alpha: float = 0.05,
) -> ReadoutDecision:
    """Decide ship / hold / block for one experiment readout.

    Blocking guardrail regressions are checked FIRST and unconditionally:
    no combination of primary-metric significance or effect size can
    override a significant guardrail regression beyond its threshold.
    """
    specs_by_name = {s.name: s for s in guardrail_specs}
    blocked: list[str] = []
    watch: list[str] = []
    reasons: list[str] = []

    for g in guardrails:
        spec = specs_by_name.get(g.name)
        if spec is None:
            continue
        if _is_regression(g, spec):
            significant = g.p_value < alpha
            if significant and spec.require_significance:
                blocked.append(g.name)
                reasons.append(
                    f"guardrail '{g.name}' regressed {g.delta:+.3%} "
                    f"(tolerance {spec.max_regression:.3%}, p={g.p_value:.4g}, significant)"
                )
            elif not spec.require_significance:
                blocked.append(g.name)
                reasons.append(
                    f"guardrail '{g.name}' regressed {g.delta:+.3%} beyond tolerance "
                    f"{spec.max_regression:.3%} (significance not required by spec)"
                )
            else:
                watch.append(g.name)
                reasons.append(
                    f"guardrail '{g.name}' regressed {g.delta:+.3%} but not significant "
                    f"(p={g.p_value:.4g} >= alpha {alpha}), flagged for watch, not blocked"
                )

    if blocked:
        return ReadoutDecision(
            decision=Decision.BLOCK,
            reasons=reasons,
            blocked_guardrails=blocked,
            watch_guardrails=watch,
        )

    primary_significant = primary.p_value < alpha
    primary_good_direction = primary.delta > 0 if primary.higher_is_better else primary.delta < 0

    if primary_significant and primary_good_direction:
        reasons.append(
            f"primary '{primary.name}' moved {primary.delta:+.3%}, p={primary.p_value:.4g}, significant"
        )
        return ReadoutDecision(
            decision=Decision.SHIP, reasons=reasons, watch_guardrails=watch
        )

    reasons.append(
        f"primary '{primary.name}' not a significant improvement "
        f"(delta={primary.delta:+.3%}, p={primary.p_value:.4g})"
    )
    return ReadoutDecision(decision=Decision.HOLD, reasons=reasons, watch_guardrails=watch)
