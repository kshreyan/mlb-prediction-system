"""Regression test for a real MLB Stats API artifact discovered while
building the 2023/2024 datasets: a postponed-and-resumed game is listed
under both its original date (status "Postponed") and its actual played
date (status "Final") with the SAME game_pk. Left un-deduplicated, this
silently fans out into duplicate rows through every downstream join.
"""
from __future__ import annotations

import pandas as pd

from mlb.data.schedule import fetch_schedule


def test_postponed_duplicate_resolves_to_single_final_row(monkeypatch):
    fake_games = [
        {
            "game_id": 555, "game_date": "2024-06-10", "game_datetime": None, "status": "Postponed",
            "home_name": "Team A", "away_name": "Team B", "home_score": "", "away_score": "",
            "home_probable_pitcher": None, "away_probable_pitcher": None, "venue_id": 1,
            "venue_name": "Park A", "doubleheader": "N", "game_num": 1, "game_type": "R",
        },
        {
            "game_id": 555, "game_date": "2024-06-11", "game_datetime": None, "status": "Final",
            "home_name": "Team A", "away_name": "Team B", "home_score": "4", "away_score": "2",
            "home_probable_pitcher": None, "away_probable_pitcher": None, "venue_id": 1,
            "venue_name": "Park A", "doubleheader": "N", "game_num": 1, "game_type": "R",
        },
    ]

    import mlb.data.schedule as sched_mod
    monkeypatch.setattr(sched_mod.statsapi, "schedule", lambda **kwargs: fake_games)
    monkeypatch.setattr(
        sched_mod, "season_regular_season_bounds",
        lambda season: (pd.Timestamp("2024-06-01").date(), pd.Timestamp("2024-06-30").date()),
    )

    df = fetch_schedule(2024)
    assert len(df) == 1
    assert df.iloc[0]["status"] == "Final"
    assert df.iloc[0]["home_score"] == 4
    assert df.iloc[0]["game_pk"] == 555
