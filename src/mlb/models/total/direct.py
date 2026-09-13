"""Direct regression baseline for total runs, on the matchup features —
a second, differently-shaped signal (linear, not derived from a Poisson
run-environment simulation) to blend with the simulation's own total-runs
prediction. Ridge (not GBM): the moneyline GBM experiment underperformed on
this project's modest per-window training-set size, and a heavily
regularized linear model is a more conservative first thing to try for a
continuous target before reaching for trees again.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def walk_forward_total_regression(
    games_with_features: pd.DataFrame,
    feature_cols: list[str],
    retrain_freq_days: int,
    min_training_games: int = 400,
    alpha: float = 5.0,
) -> pd.DataFrame:
    df = games_with_features.dropna(subset=feature_cols + ["actual_total"]).sort_values("game_date").reset_index(drop=True)
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

        pipeline = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
        pipeline.fit(df.loc[train_mask, feature_cols], df.loc[train_mask, "actual_total"])
        preds[test_mask.to_numpy().nonzero()[0]] = pipeline.predict(df.loc[test_mask, feature_cols])
        window_start = window_end + pd.Timedelta(days=1)

    df["pred_total"] = preds
    return df
