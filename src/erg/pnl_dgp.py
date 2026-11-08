"""Heavy-tailed per-session trading P&L, for the "Sequential Decision Rule
for Promoting a Strategy Change" extension (Nov. 2026).

This is a second, parallel data-generating process to erg.dgp's zero-
inflated revenue-per-customer metric, not a replacement for it: `dgp.py`'s
metric cannot go negative (a customer deposits zero or more), while the
resume bullet this extension answers is about per-session trading P&L,
which must be able to lose money. The shape that matters for the FPR/power
claims below -- heavy right tail, so the sample mean is noisier and slower
to concentrate than a Gaussian intuition suggests -- is preserved by giving
each active session's magnitude a lognormal draw and attaching a sign.

A session is: flat (P&L = 0, no trade taken) with probability `p_flat`,
otherwise a win (positive P&L) with probability `p_win_given_active` and a
loss (negative P&L) otherwise, with |P&L| drawn lognormal(mu_log,
sigma_log) either way. The default `p_win_given_active=0.5` is a literal
fair game: the baseline strategy's mean P&L is exactly 0, which is what
"a simulated null session" means here, matching the base DGP's "both arms
draw from the identical distribution" convention. "A 5% edge gain" is
then unambiguous only relative to some dollar scale, and the natural one
in a trading context is the size of a trade, not the (zero) baseline
mean: `power.py` adds a constant dollar shift equal to exactly
`relative_lift * params.mean_abs_magnitude` to every treatment-arm
session, i.e. a 5% edge means a per-session expected P&L improvement
worth 5% of the average trade's stake, not a change to win probability,
flatness or magnitude that would shift the mean indirectly and by an
unstated amount.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PnLParams:
    p_flat: float = 0.30
    p_win_given_active: float = 0.50
    mu_log: float = 2.0
    sigma_log: float = 0.80

    @property
    def mean_abs_magnitude(self) -> float:
        """E[lognormal(mu_log, sigma_log)], closed form."""
        return float(np.exp(self.mu_log + 0.5 * self.sigma_log ** 2))

    @property
    def mean_pnl(self) -> float:
        """Exact mean of one session's P&L under this DGP, closed form:
        P(active) * E[|magnitude|] * (P(win|active) - P(loss|active))."""
        active_frac = 1.0 - self.p_flat
        skew = 2.0 * self.p_win_given_active - 1.0
        return active_frac * self.mean_abs_magnitude * skew


def simulate_sessions(n: int, rng: np.random.Generator, params: PnLParams = PnLParams()) -> np.ndarray:
    """Draw n independent sessions' P&L under `params`, mean exactly
    `params.mean_pnl` in expectation. No persistence across sessions is
    modeled (unlike `dgp.py`'s customers): each session is an independent
    draw, consistent with the resume's "simulated null sessions", not
    "simulated customers"."""
    active = rng.uniform(size=n) >= params.p_flat
    win = rng.uniform(size=n) < params.p_win_given_active
    sign = np.where(win, 1.0, -1.0)
    magnitude = rng.lognormal(mean=params.mu_log, sigma=params.sigma_log, size=n)
    return np.where(active, sign * magnitude, 0.0)
