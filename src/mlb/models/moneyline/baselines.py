"""Baselines the simulation model must beat out-of-sample (or the README
says honestly that it doesn't): home-field-always, elo-only, and
pitcher-adjusted elo (elo + starter/bullpen quality diff, refit walk-forward
on the same expanding-window cadence as the main model — never fit on a
game before predicting it).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlb.features.asof import asof_ewm_cumsum


def home_field_baseline(games: pd.DataFrame, halflife_days: float = 365.0) -> np.ndarray:
    """As-of-date league-wide home win rate (not a fixed constant) — still a
    legitimate walk-forward baseline since it only uses strictly prior games."""
    df = games.sort_values("game_date").reset_index(drop=True)
    day_totals = df.groupby("game_date")["actual_home_win"].agg(["sum", "count"]).reset_index()
    num = asof_ewm_cumsum(day_totals["game_date"], day_totals["sum"], halflife_days)
    den = asof_ewm_cumsum(day_totals["game_date"], day_totals["count"], halflife_days)
    global_mean = df["actual_home_win"].mean()
    rate = np.where(den > 0, num / den, global_mean)
    rate_df = pd.DataFrame({"game_date": day_totals["game_date"], "home_rate": rate})
    merged = df.merge(rate_df, on="game_date", how="left")
    return merged["home_rate"].to_numpy()


def better_record_baseline(games: pd.DataFrame, halflife_days: float = 60.0) -> np.ndarray:
    """"Better record always wins" baseline via Log5 (Bill James): combines
    each team's as-of-date win pct into a matchup probability
    P(home) = (pH - pH*pA) / (pH + pA - 2*pH*pA), the standard sabermetric
    method for turning two teams' win rates into a head-to-head probability
    (not an invented formula)."""
    df = games.sort_values("game_date").reset_index(drop=True)

    long_home = pd.DataFrame({"game_date": df["game_date"], "team": df["home_team"], "win": df["actual_home_win"].astype(float), "one": 1.0})
    long_away = pd.DataFrame({"game_date": df["game_date"], "team": df["away_team"], "win": 1.0 - df["actual_home_win"].astype(float), "one": 1.0})
    long_df = pd.concat([long_home, long_away], ignore_index=True).sort_values(["game_date", "team"]).reset_index(drop=True)

    win_num = np.zeros(len(long_df))
    win_den = np.zeros(len(long_df))
    for _, idx in long_df.groupby("team").groups.items():
        sub = long_df.loc[idx]
        win_num[idx] = asof_ewm_cumsum(sub["game_date"], sub["win"], halflife_days)
        win_den[idx] = asof_ewm_cumsum(sub["game_date"], sub["one"], halflife_days)
    long_df["win_pct"] = np.where(win_den > 0, win_num / win_den, 0.5)

    # One row per (game_date, team) isn't unique across a team's double
    # games, so key on (game_date, team, cumulative occurrence) instead.
    long_df["_occ"] = long_df.groupby(["game_date", "team"]).cumcount()
    wp_lookup = long_df.set_index(["game_date", "team", "_occ"])["win_pct"]

    df = df.copy()
    df["_occ_home"] = df.groupby(["game_date", "home_team"]).cumcount()
    df["_occ_away"] = df.groupby(["game_date", "away_team"]).cumcount()
    p_home = df.apply(lambda r: wp_lookup.get((r["game_date"], r["home_team"], r["_occ_home"]), 0.5), axis=1).to_numpy()
    p_away = df.apply(lambda r: wp_lookup.get((r["game_date"], r["away_team"], r["_occ_away"]), 0.5), axis=1).to_numpy()

    denom = p_home + p_away - 2 * p_home * p_away
    prob_home = np.where(np.abs(denom) > 1e-9, (p_home - p_home * p_away) / denom, 0.5)
    return np.clip(prob_home, 0.01, 0.99)


def walk_forward_logistic_baseline(
    games_with_features: pd.DataFrame,
    feature_cols: list[str],
    retrain_freq_days: int,
    min_training_games: int = 200,
    label_col: str = "actual_home_win",
) -> pd.DataFrame:
    """Generic walk-forward logistic-regression baseline over arbitrary
    pre-computed feature columns (e.g. [elo_diff] for elo-only, or
    [elo_diff, diff_starter_xwoba, diff_bullpen_xwoba] for pitcher-adjusted
    elo; also reused for run-line stacking with a different `label_col`).
    Same expanding-window discipline as the main model."""
    df = games_with_features.dropna(subset=feature_cols + [label_col]).sort_values("game_date").reset_index(drop=True)
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
        y_train = df.loc[train_mask, label_col]
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        X_test = df.loc[test_mask, feature_cols]
        preds[test_mask.to_numpy().nonzero()[0]] = clf.predict_proba(X_test)[:, 1]
        window_start = window_end + pd.Timedelta(days=1)

    df["pred_prob"] = preds
    return df
