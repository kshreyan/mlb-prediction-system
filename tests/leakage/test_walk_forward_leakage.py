"""Structural leakage check on the walk-forward backtest loop itself: the
training pool used to predict any game must contain only games strictly
before that game's own date, and the reported `n_train_games` for a given
prediction must never exceed the count of games actually before it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlb.backtest.walk_forward import BacktestConfig, walk_forward_backtest


def _synthetic_season(n_games: int = 40, n_teams: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-04-01", periods=n_games, freq="D")
    teams = [f"Team {i}" for i in range(n_teams)]
    rows = []
    for i, d in enumerate(dates):
        home, away = rng.choice(teams, size=2, replace=False)
        home_score = int(rng.poisson(4.3))
        away_score = int(rng.poisson(4.3))
        rows.append({
            "game_pk": 1000 + i,
            "game_date": d,
            "home_team": home,
            "away_team": away,
            "home_score": home_score,
            "away_score": away_score,
            "home_starter_proj_xwoba_against": 0.30 + rng.normal(0, 0.02),
            "away_starter_proj_xwoba_against": 0.30 + rng.normal(0, 0.02),
            "home_off_proj_runs_scored_per_game": 4.3 + rng.normal(0, 0.3),
            "away_off_proj_runs_scored_per_game": 4.3 + rng.normal(0, 0.3),
            "home_bullpen_proj_bullpen_xwoba_against": 0.31 + rng.normal(0, 0.02),
            "away_bullpen_proj_bullpen_xwoba_against": 0.31 + rng.normal(0, 0.02),
            "park_factor": 100.0,
        })
    return pd.DataFrame(rows)


def test_training_pool_never_includes_current_or_future_games():
    season = _synthetic_season()
    cfg = BacktestConfig(retrain_freq_days=5, n_sims=500, seed=1, min_training_games=10)
    preds = walk_forward_backtest(season, prior_history=None, cfg=cfg)

    # The model retrains only at each window boundary (not daily), so a
    # game mid-window is legitimately trained on slightly fewer games than
    # everything strictly before ITS OWN date — that's conservative, not
    # leaky. The real safety invariant: n_train_games must NEVER exceed the
    # count of season games strictly before that game's date (never >=,
    # which would mean same-day or future games leaked into training).
    season_sorted = season.sort_values("game_date").reset_index(drop=True)
    for _, row in preds.iterrows():
        max_allowed_n_train = int((season_sorted["game_date"] < row["game_date"]).sum())
        assert row["n_train_games"] <= max_allowed_n_train, (
            f"game on {row['game_date']} trained on {row['n_train_games']} games, "
            f"but only {max_allowed_n_train} games existed strictly before it — LEAKAGE"
        )
