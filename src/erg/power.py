"""Two measurements on the heavy-tailed P&L DGP (`pnl_dgp.py`), reusing the
generic, already-validated primitives from `peeking.py` (look_schedule,
welch_p_values) and `msprt.py` (mixture_lr), not reimplementing either
test:

1. The same null comparison `scripts/run_simulation.py` runs on the
   revenue DGP (continuous-peeking false positive rate vs. the mSPRT
   always-valid boundary), re-measured on the P&L DGP so this extension's
   own headline numbers are not inherited unchanged from a different
   metric's distribution.
2. How many per-arm sessions each method needs to detect a true
   `relative_lift` edge gain at `power_target` power: the smallest sample
   size at which a single-look fixed-horizon test (evaluated as if THAT
   size were the pre-registered horizon, no peeking) and, separately, the
   mSPRT under continuous monitoring, each first reach that power.

Both measurements share one pair of control/treatment arrays per run
(common-random-number pairing): the fixed-horizon sweep and the mSPRT
sweep in the power analysis read prefixes of the exact same simulated
sessions, so the comparison between them is not also absorbing simulation
noise from drawing separate data for each method.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from erg.msprt import mixture_lr
from erg.peeking import look_schedule, welch_p_values
from erg.pnl_dgp import PnLParams, simulate_sessions


@dataclass(frozen=True)
class NullPnLConfig:
    n_tests: int = 2000
    n_per_arm: int = 4000
    n_looks: int = 20
    alpha: float = 0.05
    tau_fraction_of_mean: float = 0.3
    seed: int = 20261101


@dataclass
class NullPnLResult:
    config: NullPnLConfig
    peeking_fpr: float
    single_look_fpr: float
    msprt_fpr: float
    tau2: float


def run_null_pnl_experiment(
    config: NullPnLConfig = NullPnLConfig(), params: PnLParams = PnLParams()
) -> NullPnLResult:
    """The peeking-vs-mSPRT false-positive comparison, on heavy-tailed P&L
    sessions with NO true effect in either arm (both arms use `params`
    unmodified)."""
    rng = np.random.default_rng(config.seed)
    sizes = look_schedule(config.n_per_arm, config.n_looks)

    control = simulate_sessions(config.n_tests * config.n_per_arm, rng, params).reshape(
        config.n_tests, config.n_per_arm
    )
    treatment = simulate_sessions(config.n_tests * config.n_per_arm, rng, params).reshape(
        config.n_tests, config.n_per_arm
    )

    pvals = np.empty((config.n_tests, config.n_looks))
    for k, m in enumerate(sizes):
        pvals[:, k] = welch_p_values(control[:, :m], treatment[:, :m])
    peeking_fpr = float((pvals < config.alpha).any(axis=1).mean())
    single_look_fpr = float((pvals[:, -1] < config.alpha).mean())

    tau = config.tau_fraction_of_mean * params.mean_abs_magnitude
    tau2 = tau ** 2
    threshold = 1.0 / config.alpha
    lambdas = np.empty((config.n_tests, config.n_looks))
    for k, m in enumerate(sizes):
        c, t = control[:, :m], treatment[:, :m]
        diff = t.mean(axis=1) - c.mean(axis=1)
        var_diff = t.var(axis=1, ddof=1) / m + c.var(axis=1, ddof=1) / m
        var_diff = np.maximum(var_diff, 1e-12)
        lambdas[:, k] = mixture_lr(diff, var_diff, tau2)
    msprt_fpr = float((lambdas >= threshold).any(axis=1).mean())

    return NullPnLResult(
        config=config, peeking_fpr=peeking_fpr, single_look_fpr=single_look_fpr,
        msprt_fpr=msprt_fpr, tau2=tau2,
    )


@dataclass(frozen=True)
class PowerConfig:
    n_trials: int = 2000
    n_per_arm_max: int = 20000
    n_looks: int = 50
    alpha: float = 0.05
    power_target: float = 0.80
    relative_lift: float = 0.05
    # Matched to relative_lift (not the null FPR measurement's 0.3): an
    # mSPRT's mixture prior is most powerful against the specific
    # alternative it is sized for (see Findings for the attempt that used
    # 0.3 and the attempt that used a denser look schedule, neither of
    # which changed the qualitative result).
    tau_fraction_of_mean: float = 0.05
    seed: int = 20261108


@dataclass
class PowerResult:
    config: PowerConfig
    sizes: np.ndarray
    fixed_power_by_size: np.ndarray
    msprt_power_by_size: np.ndarray
    fixed_n_for_target: int | None
    msprt_n_for_target: int | None
    baseline_mean: float
    lift_dollars: float


def _n_for_target(sizes: np.ndarray, power: np.ndarray, target: float) -> int | None:
    hits = np.where(power >= target)[0]
    if hits.size == 0:
        return None
    return int(sizes[hits[0]])


def run_power_analysis(
    config: PowerConfig = PowerConfig(), params: PnLParams = PnLParams()
) -> PowerResult:
    """Smallest per-arm sample size at which (a) a single-look fixed-horizon
    Welch test and (b) the mSPRT under continuous monitoring each first
    reach `power_target` power against a true `relative_lift` shift in
    mean P&L, with CRN: control and the lift-free half of treatment share
    one simulated draw, truncated to every candidate size, for both
    methods."""
    rng = np.random.default_rng(config.seed)
    sizes = look_schedule(config.n_per_arm_max, config.n_looks)

    control = simulate_sessions(
        config.n_trials * config.n_per_arm_max, rng, params
    ).reshape(config.n_trials, config.n_per_arm_max)
    treatment_raw = simulate_sessions(
        config.n_trials * config.n_per_arm_max, rng, params
    ).reshape(config.n_trials, config.n_per_arm_max)

    baseline_mean = params.mean_pnl
    lift_dollars = config.relative_lift * params.mean_abs_magnitude
    treatment = treatment_raw + lift_dollars

    fixed_power = np.empty(config.n_looks)
    for k, m in enumerate(sizes):
        p = welch_p_values(control[:, :m], treatment[:, :m])
        fixed_power[k] = (p < config.alpha).mean()

    tau = config.tau_fraction_of_mean * params.mean_abs_magnitude
    tau2 = tau ** 2
    threshold = 1.0 / config.alpha
    detected_by_size = np.zeros((config.n_trials, config.n_looks), dtype=bool)
    for k, m in enumerate(sizes):
        c, t = control[:, :m], treatment[:, :m]
        diff = t.mean(axis=1) - c.mean(axis=1)
        var_diff = t.var(axis=1, ddof=1) / m + c.var(axis=1, ddof=1) / m
        var_diff = np.maximum(var_diff, 1e-12)
        lam = mixture_lr(diff, var_diff, tau2)
        detected_by_size[:, k] = lam >= threshold
    cumulative_detected = np.maximum.accumulate(detected_by_size, axis=1)
    msprt_power = cumulative_detected.mean(axis=0)

    return PowerResult(
        config=config,
        sizes=sizes,
        fixed_power_by_size=fixed_power,
        msprt_power_by_size=msprt_power,
        fixed_n_for_target=_n_for_target(sizes, fixed_power, config.power_target),
        msprt_n_for_target=_n_for_target(sizes, msprt_power, config.power_target),
        baseline_mean=baseline_mean,
        lift_dollars=lift_dollars,
    )
