"""Leakage check for `walk_forward_logistic_baseline` — the function both
the pitcher-adjusted-Elo baseline AND the stacked ensemble's meta-model
depend on — and for `walk_forward_gbm_baseline`, the 4th ensemble
component tried in this build. Same invariant as the main simulation
backtest: a game's prediction must never be influenced by that game's own
outcome or any later game's outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlb.models.moneyline.baselines import walk_forward_logistic_baseline
from mlb.models.moneyline.gbm import walk_forward_gbm_baseline


def _synthetic_games(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-04-01", periods=n, freq="D")
    x = rng.normal(0, 1, size=n)
    # actual outcome correlated with x, but with real noise
    p_true = 1 / (1 + np.exp(-x))
    y = (rng.random(n) < p_true).astype(int)
    return pd.DataFrame({"game_pk": np.arange(n), "game_date": dates, "feat": x, "actual_home_win": y})


def test_prediction_for_a_game_does_not_change_when_its_own_or_later_outcomes_change():
    df = _synthetic_games()
    result_original = walk_forward_logistic_baseline(df, ["feat"], retrain_freq_days=7, min_training_games=10)

    # Flip the outcome of a game roughly 3/4 of the way through the season —
    # every prediction dated STRICTLY BEFORE that game must be identical.
    flip_idx = int(len(df) * 0.75)
    flip_date = df.iloc[flip_idx]["game_date"]
    altered = df.copy()
    altered.loc[flip_idx, "actual_home_win"] = 1 - altered.loc[flip_idx, "actual_home_win"]
    result_altered = walk_forward_logistic_baseline(altered, ["feat"], retrain_freq_days=7, min_training_games=10)

    before = result_original[result_original["game_date"] < flip_date].dropna(subset=["pred_prob"])
    before_altered = result_altered[result_altered["game_date"] < flip_date].dropna(subset=["pred_prob"])

    pd.testing.assert_series_equal(
        before["pred_prob"].reset_index(drop=True),
        before_altered["pred_prob"].reset_index(drop=True),
        check_names=False,
    )


def test_predictions_only_exist_after_min_training_games():
    df = _synthetic_games()
    result = walk_forward_logistic_baseline(df, ["feat"], retrain_freq_days=7, min_training_games=20)
    has_pred = result.dropna(subset=["pred_prob"])
    # No prediction should be generated before the model has seen at least
    # min_training_games' worth of strictly-prior games.
    for _, row in has_pred.iterrows():
        n_prior = int((df["game_date"] < row["game_date"]).sum())
        assert n_prior >= 20


def _synthetic_games_multi_feature(n: int = 80, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-04-01", periods=n, freq="D")
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    p_true = 1 / (1 + np.exp(-(x1 + 0.5 * x2)))
    y = (rng.random(n) < p_true).astype(int)
    return pd.DataFrame({
        "game_pk": np.arange(n), "game_date": dates, "diff_starter_xwoba": x1,
        "diff_bullpen_xwoba": x2, "actual_home_win": y,
    })


def test_gbm_prediction_for_a_game_does_not_change_when_later_outcomes_change():
    df = _synthetic_games_multi_feature()
    feature_cols = ["diff_starter_xwoba", "diff_bullpen_xwoba"]
    result_original = walk_forward_gbm_baseline(df, feature_cols, retrain_freq_days=7, min_training_games=20)

    flip_idx = int(len(df) * 0.75)
    flip_date = df.iloc[flip_idx]["game_date"]
    altered = df.copy()
    altered.loc[flip_idx, "actual_home_win"] = 1 - altered.loc[flip_idx, "actual_home_win"]
    result_altered = walk_forward_gbm_baseline(altered, feature_cols, retrain_freq_days=7, min_training_games=20)

    before = result_original[result_original["game_date"] < flip_date].dropna(subset=["pred_prob"])
    before_altered = result_altered[result_altered["game_date"] < flip_date].dropna(subset=["pred_prob"])
    pd.testing.assert_series_equal(
        before["pred_prob"].reset_index(drop=True),
        before_altered["pred_prob"].reset_index(drop=True),
        check_names=False,
    )
