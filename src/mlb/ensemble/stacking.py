"""Stacked ensemble: blend the simulation model, Elo-only, and
pitcher-adjusted-Elo win probabilities on the LOG-ODDS scale, with weights
learned via walk-forward logistic regression (never fit on a game before
predicting it) — per the original spec's "calibrated stacked ensemble on
log-odds" requirement. Reuses `mlb.models.moneyline.baselines.
walk_forward_logistic_baseline`, since stacking on logits is just logistic
regression whose inputs happen to be other models' logit-transformed
probabilities instead of raw features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def logit(p: pd.Series | np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def build_ensemble_frame(
    sim_preds: pd.DataFrame,
    elo_preds: pd.DataFrame,
    pae_preds: pd.DataFrame,
) -> pd.DataFrame:
    """Each input has (at least) game_pk, game_date, actual_home_win, and
    its own probability column. Returns one row per game with
    logit_sim/logit_elo/logit_pae features ready for
    `walk_forward_logistic_baseline`.
    """
    base = sim_preds[["game_pk", "game_date", "actual_home_win", "pred_home_win_prob"]].rename(
        columns={"pred_home_win_prob": "p_sim"}
    )
    merged = base.merge(
        elo_preds[["game_pk", "elo_pred_home_win_prob"]].rename(columns={"elo_pred_home_win_prob": "p_elo"}),
        on="game_pk", how="inner",
    )
    merged = merged.merge(
        pae_preds[["game_pk", "pred_prob"]].rename(columns={"pred_prob": "p_pae"}),
        on="game_pk", how="inner",
    )
    merged["logit_sim"] = logit(merged["p_sim"])
    merged["logit_elo"] = logit(merged["p_elo"])
    merged["logit_pae"] = logit(merged["p_pae"])
    merged["game_date"] = pd.to_datetime(merged["game_date"])
    return merged
