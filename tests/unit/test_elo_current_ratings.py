"""current_ratings must return each team's POST-game rating from its most
recent appearance, not the pre-game rating (using pre-game would understate
a team's current strength by exactly one game's worth of update)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd

from mlb.models.moneyline.elo import compute_elo_ratings, current_ratings


def test_current_ratings_uses_post_game_value():
    games = pd.DataFrame({
        "game_pk": [1, 2, 3],
        "season": [2024, 2024, 2024],
        "game_date": pd.to_datetime(["2024-04-01", "2024-04-02", "2024-04-03"]),
        "home_team": ["A", "B", "A"],
        "away_team": ["B", "C", "C"],
        "actual_home_win": [1, 0, 1],
    })
    elo_df = compute_elo_ratings(games, k=20.0, home_advantage=24.0)
    ratings = current_ratings(elo_df)

    # Team A's last appearance is game 3 (home, won) -> its post-game rating there.
    last_row = elo_df[elo_df["game_pk"] == 3].iloc[0]
    assert ratings["A"] == last_row["elo_home_post"]
    assert ratings["A"] != last_row["elo_home_pre"]

    # Team C's last appearance is also game 3 (away, lost).
    assert ratings["C"] == last_row["elo_away_post"]


def test_current_ratings_missing_team_absent():
    games = pd.DataFrame({
        "game_pk": [1], "season": [2024], "game_date": pd.to_datetime(["2024-04-01"]),
        "home_team": ["A"], "away_team": ["B"], "actual_home_win": [1],
    })
    elo_df = compute_elo_ratings(games)
    ratings = current_ratings(elo_df)
    assert "Z" not in ratings
