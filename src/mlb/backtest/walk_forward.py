"""Day/week-by-day walk-forward backtest for the moneyline simulation model.

The model is retrained periodically (default: weekly) on an EXPANDING window
of every game completed strictly before the retrain date — prior full
seasons plus whatever of the current season has already been played. Each
retrain then generates predictions for every game in the following window
before it sees any of those games' actual results. This mirrors how the
system would actually have been run in production: no result is ever used
to predict itself or an earlier game.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlb.simulation.engine import simulate_game
from mlb.simulation.run_environment import build_long_training_frame, fit_run_environment

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    retrain_freq_days: int = 7
    n_sims: int = 20000
    seed: int = 42
    min_training_games: int = 200
    feature_set: str = "team_offense"  # "team_offense" | "lineup" | "both"


def _predict_row(model, row: pd.Series, n_sims: int, seed: int):
    X_home = pd.DataFrame([{
        "own_offense_proj": row.get("home_off_proj_runs_scored_per_game"),
        "own_lineup_xwoba": row.get("home_lineup_proj_xwoba"),
        "opp_starter_xwoba": row["away_starter_proj_xwoba_against"],
        "opp_bullpen_xwoba": row["away_bullpen_proj_bullpen_xwoba_against"],
        "park_factor": row["park_factor"],
        "is_home": 1.0,
    }])
    X_away = pd.DataFrame([{
        "own_offense_proj": row.get("away_off_proj_runs_scored_per_game"),
        "own_lineup_xwoba": row.get("away_lineup_proj_xwoba"),
        "opp_starter_xwoba": row["home_starter_proj_xwoba_against"],
        "opp_bullpen_xwoba": row["home_bullpen_proj_bullpen_xwoba_against"],
        "park_factor": row["park_factor"],
        "is_home": 0.0,
    }])
    mu_h = model.predict_mu(X_home)[0]
    mu_a = model.predict_mu(X_away)[0]
    sim = simulate_game(mu_h, mu_a, model.dispersion_k, n_sims, seed)
    return mu_h, mu_a, sim


def walk_forward_backtest(
    game_features: pd.DataFrame,
    prior_history: pd.DataFrame | None,
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """`game_features` is the season under test (must have actual scores —
    they're used only to grade predictions AFTER they're made, and to grow
    the training window for later weeks). `prior_history` is fully-completed
    prior-season game_features used as the initial training pool.
    """
    df = game_features.dropna(subset=[
        "home_starter_proj_xwoba_against", "away_starter_proj_xwoba_against",
    ]).sort_values("game_date").reset_index(drop=True)

    history_frames = [prior_history] if prior_history is not None and not prior_history.empty else []

    start_date = df["game_date"].min()
    end_date = df["game_date"].max()

    results = []
    window_start = start_date
    rng_seed = cfg.seed
    while window_start <= end_date:
        window_end = window_start + pd.Timedelta(days=cfg.retrain_freq_days - 1)

        train_current = df[df["game_date"] < window_start]
        train_pool = pd.concat(history_frames + [train_current], ignore_index=True) if history_frames else train_current
        test_slice = df[(df["game_date"] >= window_start) & (df["game_date"] <= window_end)]

        if test_slice.empty:
            window_start = window_end + pd.Timedelta(days=1)
            continue

        if len(train_pool) < cfg.min_training_games:
            logger.info("skipping window %s..%s: only %d training games so far", window_start, window_end, len(train_pool))
            window_start = window_end + pd.Timedelta(days=1)
            continue

        long_train = build_long_training_frame(train_pool, feature_set=cfg.feature_set)
        model = fit_run_environment(long_train, feature_set=cfg.feature_set)

        for _, row in test_slice.iterrows():
            rng_seed += 1
            mu_h, mu_a, sim = _predict_row(model, row, cfg.n_sims, rng_seed)
            results.append({
                "game_pk": row["game_pk"],
                "game_date": row["game_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "n_train_games": len(train_pool),
                "mu_home": mu_h,
                "mu_away": mu_a,
                "dispersion_k": model.dispersion_k,
                "pred_home_win_prob": sim.home_win_prob,
                "pred_home_minus_1_5_cover_prob": sim.home_minus_1_5_cover_prob,
                "pred_away_plus_1_5_cover_prob": sim.away_plus_1_5_cover_prob,
                "pred_prob_one_run_game": sim.prob_one_run_game,
                "pred_mean_total": sim.mean_total,
                "pred_median_total": sim.median_total,
                "actual_home_score": row["home_score"],
                "actual_away_score": row["away_score"],
                "actual_home_win": int(row["home_score"] > row["away_score"]),
                "actual_margin": int(row["home_score"] - row["away_score"]),
                "actual_total": int(row["home_score"] + row["away_score"]),
            })

        logger.info("window %s..%s: predicted %d games (train pool=%d)", window_start, window_end, len(test_slice), len(train_pool))
        window_start = window_end + pd.Timedelta(days=1)

    return pd.DataFrame(results)
