"""Direct classification baseline for the run line (home team -1.5,
i.e. P(home wins by 2+)), on the matchup features — a second signal to
blend with the simulation's own run-line probability (which comes from
the same simulated score distribution as moneyline/totals). A logistic
regression, same family as the pitcher-adjusted-Elo moneyline baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def walk_forward_runline_classifier(
    games_with_features: pd.DataFrame,
    feature_cols: list[str],
    retrain_freq_days: int,
    min_training_games: int = 400,
) -> pd.DataFrame:
    """Target: `actual_home_minus_1_5_cover` (1 if home wins by 2+ runs)."""
    df = games_with_features.dropna(subset=feature_cols + ["actual_home_minus_1_5_cover"]).sort_values("game_date").reset_index(drop=True)
    start_date, end_date = df["game_date"].min(), df["game_date"].max()

    preds = np.full(len(df), np.nan)
    window_start = start_date
    while window_start <= end_date:
        window_end = window_start + pd.Timedelta(days=retrain_freq_days - 1)
        train_mask = df["game_date"] < window_start
        test_mask = (df["game_date"] >= window_start) & (df["game_date"] <= window_end)
        if test_mask.sum() == 0:
            window_start = window_end + pd.Timedelta(days=1)
            continue
        if train_mask.sum() < min_training_games:
            window_start = window_end + pd.Timedelta(days=1)
            continue

        clf = LogisticRegression(max_iter=1000)
        clf.fit(df.loc[train_mask, feature_cols], df.loc[train_mask, "actual_home_minus_1_5_cover"])
        preds[test_mask.to_numpy().nonzero()[0]] = clf.predict_proba(df.loc[test_mask, feature_cols])[:, 1]
        window_start = window_end + pd.Timedelta(days=1)

    df["pred_home_minus_1_5_prob"] = preds
    return df
