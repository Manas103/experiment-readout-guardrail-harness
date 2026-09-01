import random

import pytest

from erg.guardrail import Decision, GuardrailSpec, MetricResult, evaluate_readout


def test_blocks_when_guardrail_regressed_despite_great_primary():
    """The headline demonstration: primary metric is a huge, highly
    significant win, but the responsible-gaming guardrail regressed
    significantly beyond tolerance. The harness must block, not ship."""
    primary = MetricResult(name="revenue_per_customer", delta=0.08, p_value=0.0003, higher_is_better=True)
    guardrail = MetricResult(
        name="rg_flag_rate", delta=0.09, p_value=0.01, higher_is_better=False
    )
    spec = GuardrailSpec(name="rg_flag_rate", max_regression=0.03, require_significance=True)

    result = evaluate_readout(primary, [guardrail], [spec], alpha=0.05)

    assert result.decision == Decision.BLOCK
    assert "rg_flag_rate" in result.blocked_guardrails


def test_ships_when_primary_significant_and_guardrails_clean():
    primary = MetricResult(name="revenue_per_customer", delta=0.05, p_value=0.001, higher_is_better=True)
    guardrail = MetricResult(name="rg_flag_rate", delta=0.005, p_value=0.6, higher_is_better=False)
    spec = GuardrailSpec(name="rg_flag_rate", max_regression=0.03)

    result = evaluate_readout(primary, [guardrail], [spec], alpha=0.05)
    assert result.decision == Decision.SHIP


def test_holds_when_primary_not_significant_and_guardrails_clean():
    primary = MetricResult(name="revenue_per_customer", delta=0.01, p_value=0.4, higher_is_better=True)
    guardrail = MetricResult(name="rg_flag_rate", delta=0.0, p_value=0.9, higher_is_better=False)
    spec = GuardrailSpec(name="rg_flag_rate", max_regression=0.03)

    result = evaluate_readout(primary, [guardrail], [spec], alpha=0.05)
    assert result.decision == Decision.HOLD


def test_watch_not_block_for_non_significant_regression():
    """A guardrail that moved past threshold but is not statistically
    significant is surfaced as a watch item, not a hard block: this is a
    documented design tradeoff (see guardrail.py docstring), distinct from
    the always-block case above where the regression WAS significant."""
    primary = MetricResult(name="revenue_per_customer", delta=0.05, p_value=0.001, higher_is_better=True)
    guardrail = MetricResult(name="rg_flag_rate", delta=0.09, p_value=0.4, higher_is_better=False)
    spec = GuardrailSpec(name="rg_flag_rate", max_regression=0.03, require_significance=True)

    result = evaluate_readout(primary, [guardrail], [spec], alpha=0.05)
    assert result.decision == Decision.SHIP
    assert "rg_flag_rate" in result.watch_guardrails


def test_guardrail_improving_never_blocks():
    primary = MetricResult(name="revenue_per_customer", delta=0.05, p_value=0.001, higher_is_better=True)
    guardrail = MetricResult(name="rg_flag_rate", delta=-0.10, p_value=0.001, higher_is_better=False)
    spec = GuardrailSpec(name="rg_flag_rate", max_regression=0.03)

    result = evaluate_readout(primary, [guardrail], [spec], alpha=0.05)
    assert result.decision == Decision.SHIP


def test_higher_is_better_guardrail_blocks_on_a_drop():
    """A guardrail where higher is better (e.g. a trust/NPS-style score)
    should block on a significant DROP, the mirror image of the
    lower-is-better case."""
    primary = MetricResult(name="revenue_per_customer", delta=0.05, p_value=0.001, higher_is_better=True)
    guardrail = MetricResult(name="trust_score", delta=-0.06, p_value=0.02, higher_is_better=True)
    spec = GuardrailSpec(name="trust_score", max_regression=0.03)

    result = evaluate_readout(primary, [guardrail], [spec], alpha=0.05)
    assert result.decision == Decision.BLOCK


def test_never_ships_with_blocked_guardrail():
    """Fuzz invariant: across many randomized readouts, whenever at least one
    guardrail has a significant regression beyond its threshold, the
    decision must never be SHIP. This is the property the whole harness
    exists to guarantee."""
    rng = random.Random(12345)
    spec = GuardrailSpec(name="rg_flag_rate", max_regression=0.03, require_significance=True)

    violations = 0
    for _ in range(5000):
        primary = MetricResult(
            name="revenue_per_customer",
            delta=rng.uniform(-0.1, 0.2),
            p_value=rng.uniform(0.0, 1.0),
            higher_is_better=True,
        )
        guardrail = MetricResult(
            name="rg_flag_rate",
            delta=rng.uniform(-0.1, 0.2),
            p_value=rng.uniform(0.0, 1.0),
            higher_is_better=False,
        )
        result = evaluate_readout(primary, [guardrail], [spec], alpha=0.05)

        guardrail_significantly_regressed = guardrail.delta > spec.max_regression and guardrail.p_value < 0.05
        if guardrail_significantly_regressed and result.decision == Decision.SHIP:
            violations += 1

    assert violations == 0


def test_unknown_guardrail_name_raises():
    spec = GuardrailSpec(name="rg_flag_rate", max_regression=0.03)
    bad_result = MetricResult(name="other_metric", delta=0.1, p_value=0.01, higher_is_better=False)
    with pytest.raises(ValueError):
        from erg.guardrail import _is_regression

        _is_regression(bad_result, spec)
