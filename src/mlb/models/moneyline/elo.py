"""Classic Elo (team-only) and a pitcher-adjusted variant, both walk-forward
SAFE BY CONSTRUCTION: ratings are only ever updated using a game's result
AFTER that game's pre-game rating has already been used to predict it, and
games must be processed in strict chronological order (across concatenated
seasons, with a regression-to-mean at each season boundary — teams change
in the offseason, so full rating persistence season-to-season overstates
confidence).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def elo_win_prob(rating_diff: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.power(10.0, -rating_diff / 400.0))


def compute_elo_ratings(
    games: pd.DataFrame,
    k: float = 20.0,
    home_advantage: float = 24.0,
    season_regress_frac: float = 0.33,
    start_rating: float = 1500.0,
) -> pd.DataFrame:
    """`games` must span potentially multiple seasons, sorted by game_date,
    with columns game_pk, season, game_date, home_team, away_team,
    actual_home_win (or home_win). Returns the same frame with
    `elo_home_pre`, `elo_away_pre`, `elo_pred_home_win_prob` added — the
    PRE-game ratings and prediction, i.e. exactly what would have been known
    before that game was played.
    """
    df = games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    current_season = None

    elo_home_pre = np.zeros(len(df))
    elo_away_pre = np.zeros(len(df))

    for i, row in df.iterrows():
        season = row["season"]
        if current_season is not None and season != current_season:
            # Regress every known team's rating toward the mean between seasons.
            for team in list(ratings.keys()):
                ratings[team] = start_rating + (1 - season_regress_frac) * (ratings[team] - start_rating)
        current_season = season

        home, away = row["home_team"], row["away_team"]
        r_home = ratings.get(home, start_rating)
        r_away = ratings.get(away, start_rating)
        elo_home_pre[i] = r_home
        elo_away_pre[i] = r_away

        expected_home = elo_win_prob(np.array([r_home + home_advantage - r_away]))[0]
        actual_home = float(row["actual_home_win"]) if "actual_home_win" in row else float(row["home_win"])

        ratings[home] = r_home + k * (actual_home - expected_home)
        ratings[away] = r_away + k * ((1 - actual_home) - (1 - expected_home))

    df["elo_home_pre"] = elo_home_pre
    df["elo_away_pre"] = elo_away_pre
    df["elo_pred_home_win_prob"] = elo_win_prob(elo_home_pre + home_advantage - elo_away_pre)
    return df
