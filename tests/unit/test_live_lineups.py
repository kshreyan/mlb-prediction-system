"""Unit tests for live confirmed-lineup ingestion: verifies the parsing
logic distinguishes "not posted yet" from "posted" correctly, since that
distinction is the entire point of this module (never fabricate a lineup
that hasn't been announced).
"""
from __future__ import annotations

from mlb.lineups.live import fetch_confirmed_lineup, fetch_probable_pitchers


def _fake_game_data(home_batting_order, away_batting_order, home_players=None, away_players=None):
    return {
        "gameData": {
            "probablePitchers": {
                "home": {"id": 111, "fullName": "Home Starter"},
                "away": {"id": 222, "fullName": "Away Starter"},
            },
        },
        "liveData": {
            "boxscore": {
                "teams": {
                    "home": {
                        "battingOrder": home_batting_order,
                        "players": home_players or {},
                        "pitchers": [111] if home_batting_order else [],
                    },
                    "away": {
                        "battingOrder": away_batting_order,
                        "players": away_players or {},
                        "pitchers": [222] if away_batting_order else [],
                    },
                },
            },
        },
    }


def test_probable_pitchers_available_before_lineups_posted(monkeypatch):
    import mlb.lineups.live as live_mod
    monkeypatch.setattr(live_mod.statsapi, "get", lambda endpoint, params: _fake_game_data([], []))
    probables = fetch_probable_pitchers(12345)
    assert probables["home"] == {"id": 111, "name": "Home Starter"}
    assert probables["away"] == {"id": 222, "name": "Away Starter"}


def test_unposted_lineup_returns_none_not_a_guess(monkeypatch):
    import mlb.lineups.live as live_mod
    monkeypatch.setattr(live_mod.statsapi, "get", lambda endpoint, params: _fake_game_data([], []))
    lineups = fetch_confirmed_lineup(12345)
    assert lineups["home"] is None
    assert lineups["away"] is None


def test_posted_lineup_parses_batting_order_correctly(monkeypatch):
    home_players = {
        f"ID{pid}": {"person": {"fullName": f"Player {pid}", "batSide": {"code": "R"}}}
        for pid in [901, 902, 903]
    }
    import mlb.lineups.live as live_mod
    monkeypatch.setattr(
        live_mod.statsapi, "get",
        lambda endpoint, params: _fake_game_data([901, 902, 903], [], home_players=home_players),
    )
    lineups = fetch_confirmed_lineup(12345)
    assert lineups["away"] is None  # away lineup still not posted
    assert lineups["home"]["starting_pitcher_id"] == 111
    order = lineups["home"]["batting_order"]
    assert [b["slot"] for b in order] == [1, 2, 3]
    assert [b["batter_id"] for b in order] == [901, 902, 903]
    assert order[0]["name"] == "Player 901"
    assert order[0]["stand"] == "R"
