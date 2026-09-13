"""Stacked ensemble: blend an arbitrary set of moneyline win-probability
models on the LOG-ODDS scale, with weights learned via walk-forward
logistic regression (never fit on a game before predicting it) — per the
original spec's "calibrated stacked ensemble on log-odds" requirement.
Reuses `mlb.models.moneyline.baselines.walk_forward_logistic_baseline`,
since stacking on logits is just logistic regression whose inputs happen
to be other models' logit-transformed probabilities instead of raw
features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def logit(p: pd.Series | np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def build_ensemble_frame(
    base: pd.DataFrame,
    components: dict[str, tuple[pd.DataFrame, str]],
) -> pd.DataFrame:
    """`base` provides game_pk, game_date, actual_home_win (typically the
    simulation model's own predictions frame). `components` maps a short
    name (e.g. "elo") to (that model's predictions dataframe, its
    probability column name) — each merged in on game_pk (inner join, so a
    game missing from any component is dropped from the ensemble entirely).
    Returns one row per game with `logit_<name>` for every component.
    """
    merged = base[["game_pk", "game_date", "actual_home_win"]].copy()
    for name, (df, prob_col) in components.items():
        merged = merged.merge(
            df[["game_pk", prob_col]].rename(columns={prob_col: f"p_{name}"}),
            on="game_pk", how="inner",
        )
        merged[f"logit_{name}"] = logit(merged[f"p_{name}"])
    merged["game_date"] = pd.to_datetime(merged["game_date"])
    return merged
