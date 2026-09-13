"""Stacked ensembles for all three markets:
  - Moneyline / run line (binary): blend on the LOG-ODDS scale via
    walk-forward logistic regression (`mlb.models.moneyline.baselines.
    walk_forward_logistic_baseline`) — per the original spec's "calibrated
    stacked ensemble on log-odds" requirement.
  - Totals (continuous): blend on the RAW SCALE via walk-forward linear
    regression (there's no natural logit transform for a run total).
Both share the same walk-forward discipline as everything else in this
build: weights are refit periodically on an expanding window, never using
a game's own or a later game's outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def logit(p: pd.Series | np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def build_ensemble_frame(
    base: pd.DataFrame,
    components: dict[str, tuple[pd.DataFrame, str]],
    label_col: str = "actual_home_win",
    binary: bool = True,
) -> pd.DataFrame:
    """`base` provides game_pk, game_date, and `label_col` (typically the
    simulation model's own predictions frame). `components` maps a short
    name (e.g. "elo") to (that model's predictions dataframe, its
    probability/point-prediction column name) — each merged in on game_pk
    (inner join, so a game missing from any component is dropped from the
    ensemble entirely). For `binary=True` markets (moneyline, run line),
    adds `logit_<name>`; for `binary=False` (totals), keeps the raw
    `p_<name>` column instead (no logit transform for a continuous target).
    """
    merged = base[["game_pk", "game_date", label_col]].copy()
    for name, (df, prob_col) in components.items():
        merged = merged.merge(
            df[["game_pk", prob_col]].rename(columns={prob_col: f"p_{name}"}),
            on="game_pk", how="inner",
        )
        if binary:
            merged[f"logit_{name}"] = logit(merged[f"p_{name}"])
    merged["game_date"] = pd.to_datetime(merged["game_date"])
    return merged


def walk_forward_linear_stacking(
    ensemble_frame: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    retrain_freq_days: int,
    min_training_games: int = 200,
    alpha: float = 10.0,
) -> pd.DataFrame:
    """Same expanding-window walk-forward discipline as
    `walk_forward_logistic_baseline`, but Ridge regression on the raw scale
    — for blending continuous predictions (e.g. total runs) rather than
    probabilities. Ridge, not plain OLS: with only 2-3 highly-correlated
    component predictions and a weekly refit on a modest window, an
    unregularized regression's coefficients are noisy enough to hurt
    rather than help (this was tried and confirmed empirically — see
    docs/limitations.md).
    """
    df = ensemble_frame.dropna(subset=feature_cols + [label_col]).sort_values("game_date").reset_index(drop=True)
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

        reg = Ridge(alpha=alpha)
        reg.fit(df.loc[train_mask, feature_cols], df.loc[train_mask, label_col])
        preds[test_mask.to_numpy().nonzero()[0]] = reg.predict(df.loc[test_mask, feature_cols])
        window_start = window_end + pd.Timedelta(days=1)

    df["pred_ensemble"] = preds
    return df
