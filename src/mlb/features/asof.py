"""Shared as-of-date exponential-decay utilities.

The one function every leakage test cares about: for a time-ordered series of
(date, value) pairs belonging to one entity (a pitcher, a team, ...), produce
for each row the decayed cumulative sum of all STRICTLY EARLIER rows. Row 0
always gets 0 — there is no history before the first observation.

Rows sharing the same calendar day are treated as simultaneous: none of them
sees any other same-day row's value (this matters for doubleheaders — a
team's or pitcher's second game of the day must not see the first game's
result baked into its "as-of" projection).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decay(delta_days: np.ndarray, halflife_days: float) -> np.ndarray:
    return np.power(0.5, delta_days / halflife_days)


def asof_ewm_cumsum(dates: pd.Series, values: pd.Series, halflife_days: float) -> np.ndarray:
    d = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    v = pd.Series(values).reset_index(drop=True).to_numpy(dtype=float)
    n = len(v)
    out = np.zeros(n)

    order = pd.DataFrame({"d": d, "v": v, "orig_pos": np.arange(n)})
    running = 0.0
    last_day = None
    for day, group in order.groupby("d", sort=True):
        if last_day is not None:
            delta = (day - last_day) / np.timedelta64(1, "D")
            running = running * decay(np.array([delta]), halflife_days)[0]
        # Every row sharing this calendar day sees the SAME as-of value —
        # the state accumulated through the end of the previous distinct day.
        out[group["orig_pos"].to_numpy()] = running
        running = running + group["v"].sum()
        last_day = day
    return out


def asof_expanding_mean(dates: pd.Series, num: pd.Series, den: pd.Series) -> np.ndarray:
    """A non-decayed, strictly-prior-days expanding ratio — used only as the
    fallback prior when the exponential league prior itself has zero mass
    (the very first day(s) of an entire dataset). Implemented as EWM cumsum
    with an effectively-infinite halflife, so it inherits the same
    leak-free, same-day-safe guarantees as `asof_ewm_cumsum`.
    """
    huge_halflife = 1e9
    num_cum = asof_ewm_cumsum(dates, num, huge_halflife)
    den_cum = asof_ewm_cumsum(dates, den, huge_halflife)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den_cum > 0, num_cum / den_cum, np.nan)


def shrink(own_num: np.ndarray, own_den: np.ndarray, prior_mean: np.ndarray, k: float) -> np.ndarray:
    """Empirical-Bayes / credibility shrinkage toward `prior_mean`."""
    return (prior_mean * k + own_num) / (k + own_den)
