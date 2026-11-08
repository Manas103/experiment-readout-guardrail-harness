import numpy as np
import pytest

from erg.pnl_dgp import PnLParams
from erg.power import NullPnLConfig, PowerConfig, run_null_pnl_experiment, run_power_analysis


def test_null_pnl_peeking_fpr_near_resume_claim_and_far_above_nominal():
    config = NullPnLConfig(n_tests=1500, n_per_arm=4000, n_looks=20, seed=11)
    result = run_null_pnl_experiment(config, PnLParams())
    assert result.peeking_fpr > 3 * config.alpha
    assert result.peeking_fpr == pytest.approx(0.26, abs=0.08)


def test_null_pnl_single_look_fpr_near_nominal():
    config = NullPnLConfig(n_tests=1500, n_per_arm=4000, n_looks=20, seed=11)
    result = run_null_pnl_experiment(config, PnLParams())
    assert result.single_look_fpr == pytest.approx(config.alpha, abs=0.03)


def test_null_pnl_msprt_controls_fpr_far_better_than_peeking_on_same_data():
    """The core comparison, re-measured on heavy-tailed P&L instead of
    revenue-per-customer: on the same simulated null sessions, mSPRT's
    false positive rate is far below naive continuous peeking's."""
    config = NullPnLConfig(n_tests=1500, n_per_arm=4000, n_looks=20, seed=13)
    result = run_null_pnl_experiment(config, PnLParams())
    assert result.msprt_fpr < result.peeking_fpr
    assert result.msprt_fpr < config.alpha + 0.03


def test_lift_dollars_is_exactly_relative_lift_times_mean_abs_magnitude():
    params = PnLParams()
    config = PowerConfig(n_trials=50, n_per_arm_max=1000, n_looks=5, relative_lift=0.05)
    result = run_power_analysis(config, params)
    assert result.lift_dollars == pytest.approx(0.05 * params.mean_abs_magnitude)


def test_zero_lift_gives_power_at_roughly_the_null_false_positive_level():
    """Sanity/invariant check cross-referencing the null measurement: with
    no true effect injected, 'power' (the rate of flagging significance)
    should sit near the method's own false-positive rate, not near 80%."""
    params = PnLParams()
    config = PowerConfig(n_trials=1000, n_per_arm_max=4000, n_looks=20, relative_lift=0.0, seed=21)
    result = run_power_analysis(config, params)
    assert result.fixed_power_by_size[-1] < 0.15
    assert result.msprt_power_by_size[-1] < 0.15


def test_msprt_cumulative_power_is_nondecreasing_in_sample_size():
    """By construction (np.maximum.accumulate over 'detected by this
    look'), the mSPRT's cumulative detection rate can never decrease as
    more sessions are added; this is an invariant, not a measured result."""
    config = PowerConfig(n_trials=300, n_per_arm_max=6000, n_looks=15, seed=5)
    result = run_power_analysis(config, PnLParams())
    assert np.all(np.diff(result.msprt_power_by_size) >= 0.0)


def test_power_rises_with_a_real_effect_vs_no_effect():
    params = PnLParams()
    config_null = PowerConfig(n_trials=500, n_per_arm_max=8000, n_looks=20, relative_lift=0.0, seed=8)
    config_lift = PowerConfig(n_trials=500, n_per_arm_max=8000, n_looks=20, relative_lift=0.05, seed=8)
    null_result = run_power_analysis(config_null, params)
    lift_result = run_power_analysis(config_lift, params)
    assert lift_result.fixed_power_by_size[-1] > null_result.fixed_power_by_size[-1]
    assert lift_result.msprt_power_by_size[-1] > null_result.msprt_power_by_size[-1]


def test_fixed_horizon_and_msprt_sample_sizes_are_found_within_the_horizon():
    config = PowerConfig(n_trials=1000, n_per_arm_max=20000, n_looks=50, seed=20261108)
    result = run_power_analysis(config, PnLParams())
    assert result.fixed_n_for_target is not None
    assert result.msprt_n_for_target is not None
    assert result.fixed_n_for_target > 0
    assert result.msprt_n_for_target > 0


def test_n_for_target_returns_none_when_never_reached():
    from erg.power import _n_for_target

    sizes = np.array([100, 200, 300])
    power = np.array([0.1, 0.2, 0.3])
    assert _n_for_target(sizes, power, 0.80) is None
    assert _n_for_target(sizes, power, 0.25) == 300
