"""The core anti-leakage guarantee: `asof_ewm_cumsum` (used by every
projection module — pitcher, bullpen, team offense, home-field baseline)
must never let a row's own value, or any later row's value, influence its
own output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlb.features.asof import asof_ewm_cumsum


def test_first_row_has_no_history():
    dates = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"])
    values = pd.Series([10.0, 20.0, 30.0])
    out = asof_ewm_cumsum(dates, values, halflife_days=30)
    assert out[0] == 0.0


def test_changing_a_future_value_does_not_change_past_outputs():
    dates = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-15"])
    values = pd.Series([10.0, 20.0, 30.0, 40.0])
    out_original = asof_ewm_cumsum(dates, values, halflife_days=30)

    # Mutate only the LAST value — every prior row's as-of output must be identical.
    values_mutated = values.copy()
    values_mutated.iloc[-1] = 99999.0
    out_mutated = asof_ewm_cumsum(dates, values_mutated, halflife_days=30)

    np.testing.assert_array_equal(out_original[:-1], out_mutated[:-1])


def test_changing_a_past_value_only_affects_strictly_later_rows():
    dates = pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-15"])
    values = pd.Series([10.0, 20.0, 30.0, 40.0])
    out_original = asof_ewm_cumsum(dates, values, halflife_days=30)

    values_mutated = values.copy()
    values_mutated.iloc[0] = 99999.0
    out_mutated = asof_ewm_cumsum(dates, values_mutated, halflife_days=30)

    # Row 0's own as-of output is always 0 regardless of its own value.
    assert out_original[0] == out_mutated[0] == 0.0
    # Every row after it must differ, since row 0 fed their history.
    assert not np.allclose(out_original[1:], out_mutated[1:])


def test_same_day_rows_do_not_leak_into_each_other():
    """Two starts on the same calendar day (a doubleheader-adjacent edge
    case) must not see each other — only strictly earlier CALENDAR DAYS
    count as history."""
    dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-05"])
    values = pd.Series([10.0, 20.0, 30.0])
    out = asof_ewm_cumsum(dates, values, halflife_days=30)
    assert out[0] == 0.0
    assert out[1] == 0.0  # same day as row 0 — must not see row 0's value


def test_decay_reduces_weight_of_older_observations():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-06-01"])
    values = pd.Series([100.0, 100.0, 0.0])
    out = asof_ewm_cumsum(dates, values, halflife_days=10)
    # By the third row (~150 days later, many halflives), the accumulated
    # history should have decayed to nearly nothing.
    assert out[2] < 0.01


@pytest.mark.parametrize("halflife", [1.0, 10.0, 45.0, 365.0])
def test_output_never_negative(halflife):
    rng = np.random.default_rng(0)
    dates = pd.to_datetime(pd.date_range("2024-01-01", periods=50, freq="D"))
    values = pd.Series(rng.uniform(0, 10, size=50))
    out = asof_ewm_cumsum(dates, values, halflife_days=halflife)
    assert np.all(out >= 0)
