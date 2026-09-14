"""Unit tests for real-odds parsing/de-vig math: verifies the arithmetic
and game-matching logic against known inputs, since a bug here would
silently corrupt the CLV comparison rather than crash loudly.
"""
from __future__ import annotations

import datetime as dt

from mlb.data.odds import _devig_two_way, extract_game_closing_odds


def test_devig_removes_vig_and_sums_to_one():
    # Two even-money-ish prices with obvious vig baked in (both < true 50/50 fair odds of 2.0)
    p_home, p_away = _devig_two_way(1.91, 1.91)
    assert abs((p_home + p_away) - 1.0) < 1e-9
    assert abs(p_home - 0.5) < 1e-9  # symmetric prices -> symmetric de-vigged probs


def test_devig_favors_the_shorter_price():
    p_home, p_away = _devig_two_way(1.50, 2.80)  # home is the favorite (shorter price)
    assert p_home > p_away
    assert abs((p_home + p_away) - 1.0) < 1e-9


def _fake_snapshot(commence_time_str, home, away, h2h_prices=(1.90, 1.95)):
    return {
        "timestamp": "2024-06-01T23:55:00Z",
        "data": [{
            "commence_time": commence_time_str,
            "home_team": home,
            "away_team": away,
            "bookmakers": [{
                "key": "fanduel",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": home, "price": h2h_prices[0]},
                        {"name": away, "price": h2h_prices[1]},
                    ],
                }],
            }],
        }],
    }


def test_extract_matches_game_within_tolerance():
    snap = _fake_snapshot("2024-06-01T23:15:00Z", "Seattle Mariners", "Los Angeles Angels")
    commence = dt.datetime(2024, 6, 1, 23, 20, tzinfo=dt.timezone.utc)  # 5 min off, within 20-min tolerance
    result = extract_game_closing_odds(snap, "Seattle Mariners", "Los Angeles Angels", commence)
    assert result is not None
    assert result["n_bookmakers"] == 1
    assert 0.0 < result["ml_home_implied_prob"] < 1.0


def test_extract_rejects_wrong_team_names():
    snap = _fake_snapshot("2024-06-01T23:15:00Z", "Seattle Mariners", "Los Angeles Angels")
    commence = dt.datetime(2024, 6, 1, 23, 15, tzinfo=dt.timezone.utc)
    result = extract_game_closing_odds(snap, "Boston Red Sox", "New York Yankees", commence)
    assert result is None  # never fabricate a match that isn't there


def test_extract_rejects_games_outside_time_tolerance():
    snap = _fake_snapshot("2024-06-01T23:15:00Z", "Seattle Mariners", "Los Angeles Angels")
    commence = dt.datetime(2024, 6, 2, 3, 0, tzinfo=dt.timezone.utc)  # hours off
    result = extract_game_closing_odds(snap, "Seattle Mariners", "Los Angeles Angels", commence, tolerance_minutes=20)
    assert result is None
