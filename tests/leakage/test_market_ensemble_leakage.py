"""Leakage checks for the totals (Ridge regression) and run-line (logistic
regression) direct models — the second component in each market's stacked
ensemble. Same invariant as every other walk-forward function in this
build: a game's prediction must never be influenced by that game's own
outcome or any later game's outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlb.models.total.direct import walk_forward_total_regression
from mlb.models.runline.direct import walk_forward_runline_classifier


def _synthetic_games(n: int = 80, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-04-01", periods=n, freq="D")
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    total = 8.5 + x1 + 0.5 * x2 + rng.normal(0, 2, size=n)
    margin = rng.normal(x1, 3, size=n)
    return pd.DataFrame({
        "game_pk": np.arange(n), "game_date": dates, "diff_starter_xwoba": x1,
        "diff_bullpen_xwoba": x2, "actual_total": total,
        "actual_home_minus_1_5_cover": (margin >= 2).astype(int),
    })


def test_total_regression_unaffected_by_later_outcomes():
    df = _synthetic_games()
    feature_cols = ["diff_starter_xwoba", "diff_bullpen_xwoba"]
    result_original = walk_forward_total_regression(df, feature_cols, retrain_freq_days=7, min_training_games=20)

    flip_idx = int(len(df) * 0.75)
    flip_date = df.iloc[flip_idx]["game_date"]
    altered = df.copy()
    altered.loc[flip_idx, "actual_total"] = 99.0  # an absurd outlier
    result_altered = walk_forward_total_regression(altered, feature_cols, retrain_freq_days=7, min_training_games=20)

    before = result_original[result_original["game_date"] < flip_date].dropna(subset=["pred_total"])
    before_altered = result_altered[result_altered["game_date"] < flip_date].dropna(subset=["pred_total"])
    pd.testing.assert_series_equal(
        before["pred_total"].reset_index(drop=True),
        before_altered["pred_total"].reset_index(drop=True),
        check_names=False,
    )


def test_runline_classifier_unaffected_by_later_outcomes():
    df = _synthetic_games()
    feature_cols = ["diff_starter_xwoba", "diff_bullpen_xwoba"]
    result_original = walk_forward_runline_classifier(df, feature_cols, retrain_freq_days=7, min_training_games=20)

    flip_idx = int(len(df) * 0.75)
    flip_date = df.iloc[flip_idx]["game_date"]
    altered = df.copy()
    altered.loc[flip_idx, "actual_home_minus_1_5_cover"] = 1 - altered.loc[flip_idx, "actual_home_minus_1_5_cover"]
    result_altered = walk_forward_runline_classifier(altered, feature_cols, retrain_freq_days=7, min_training_games=20)

    before = result_original[result_original["game_date"] < flip_date].dropna(subset=["pred_home_minus_1_5_prob"])
    before_altered = result_altered[result_altered["game_date"] < flip_date].dropna(subset=["pred_home_minus_1_5_prob"])
    pd.testing.assert_series_equal(
        before["pred_home_minus_1_5_prob"].reset_index(drop=True),
        before_altered["pred_home_minus_1_5_prob"].reset_index(drop=True),
        check_names=False,
    )


def test_linear_stacking_unaffected_by_later_outcomes():
    from mlb.ensemble.stacking import walk_forward_linear_stacking

    df = _synthetic_games()
    df["p_a"] = df["actual_total"] + np.random.default_rng(0).normal(0, 0.5, len(df))
    df["p_b"] = df["actual_total"] + np.random.default_rng(1).normal(0, 0.5, len(df))
    result_original = walk_forward_linear_stacking(df, ["p_a", "p_b"], "actual_total", retrain_freq_days=7, min_training_games=20)

    flip_idx = int(len(df) * 0.75)
    flip_date = df.iloc[flip_idx]["game_date"]
    altered = df.copy()
    altered.loc[flip_idx, "actual_total"] = 99.0
    result_altered = walk_forward_linear_stacking(altered, ["p_a", "p_b"], "actual_total", retrain_freq_days=7, min_training_games=20)

    before = result_original[result_original["game_date"] < flip_date].dropna(subset=["pred_ensemble"])
    before_altered = result_altered[result_altered["game_date"] < flip_date].dropna(subset=["pred_ensemble"])
    pd.testing.assert_series_equal(
        before["pred_ensemble"].reset_index(drop=True),
        before_altered["pred_ensemble"].reset_index(drop=True),
        check_names=False,
    )
