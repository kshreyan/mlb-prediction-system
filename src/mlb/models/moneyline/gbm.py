"""Direct gradient-boosted-trees baseline on the matchup features (the
"direct ML baselines" the original spec calls for, alongside Elo and the
simulation). Walk-forward: retrained periodically on an expanding window of
strictly-prior games, same discipline as every other model in this build.

Deliberately conservative hyperparameters (shallow trees, high min-leaf
sample count, few boosting rounds) given the modest per-window training set
size (a few thousand games at most) — an aggressive GBM would overfit
long before it found real signal beyond what the other components already
capture.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def walk_forward_gbm_baseline(
    games_with_features: pd.DataFrame,
    feature_cols: list[str],
    retrain_freq_days: int,
    min_training_games: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    df = games_with_features.dropna(subset=feature_cols + ["actual_home_win"]).sort_values("game_date").reset_index(drop=True)
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

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, "actual_home_win"]
        clf = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=30, l2_regularization=1.0, random_state=seed,
        )
        clf.fit(X_train, y_train)
        X_test = df.loc[test_mask, feature_cols]
        preds[test_mask.to_numpy().nonzero()[0]] = clf.predict_proba(X_test)[:, 1]
        window_start = window_end + pd.Timedelta(days=1)

    df["pred_prob"] = preds
    return df
