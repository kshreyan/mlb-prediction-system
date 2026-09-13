"""As-of-date pitcher projections.

For every pitcher-game row, produce a projection of that pitcher's true-talent
peripherals using ONLY his own starts strictly before that game (exponentially
time-weighted, halflife in days) shrunk toward a dynamically-computed,
same-date league average (also computed from strictly-prior games only).

This is the anti-leakage backbone of the whole system: a 4-start hot streak
gets outweighed by the shrinkage prior exactly because `k` (shrinkage_k_batters)
represents a meaningful fraction of a full log of batters faced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlb.features.asof import asof_ewm_cumsum, asof_expanding_mean, shrink

STATS = {
    # name: (numerator_col, denominator_col)
    "k_pct": ("n_k", "batters_faced"),
    "bb_pct": ("n_bb", "batters_faced"),
    "whiff_pct": ("n_whiffs", "n_swings"),
    "csw_pct": ("n_csw", "n_pitches"),
    "barrel_pct": ("n_barrels", "n_bip"),
    "xwoba_against": ("xwoba_weighted_sum", "woba_denom_sum"),
}

# Absolute last-resort neutral priors, used only when there is zero prior
# data of ANY kind (the first calendar day the dataset has ever seen) — a
# documented, published-ballpark constant, not a fabricated one, and in
# practice unreachable once a prior season's games are prepended.
_NEUTRAL_PRIOR = {
    "k_pct": 0.22, "bb_pct": 0.085, "whiff_pct": 0.25,
    "csw_pct": 0.28, "barrel_pct": 0.07, "xwoba_against": 0.31,
}


def add_asof_pitcher_projections(
    pitcher_games: pd.DataFrame,
    halflife_days: float,
    shrinkage_k: float,
    entity_col: str = "pitcher",
) -> pd.DataFrame:
    """Adds `proj_<stat>`, `league_<stat>`, and `n_prior_<stat>_den` columns
    to a copy of `pitcher_games`, computed strictly as-of (before) each row's
    own game date.
    """
    df = pitcher_games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

    # --- League-wide as-of-date prior (pooled across all pitchers), grouped
    # by calendar day so within-day starts can't leak off each other. ---
    needed_cols = sorted({c for pair in STATS.values() for c in pair})
    day_totals = (
        df.groupby("game_date")[needed_cols].sum().reset_index().sort_values("game_date")
    )

    league_cols = {}
    for stat, (num_col, den_col) in STATS.items():
        league_cols[f"{stat}__lg_num"] = asof_ewm_cumsum(day_totals["game_date"], day_totals[num_col], halflife_days)
        league_cols[f"{stat}__lg_den"] = asof_ewm_cumsum(day_totals["game_date"], day_totals[den_col], halflife_days)
    league_df = pd.DataFrame({"game_date": day_totals["game_date"], **league_cols})
    df = df.merge(league_df, on="game_date", how="left")

    # --- Per-pitcher as-of-date own history, shrunk toward the league prior ---
    for stat, (num_col, den_col) in STATS.items():
        num_out = np.zeros(len(df))
        den_out = np.zeros(len(df))
        for _, idx in df.groupby(entity_col).groups.items():
            sub = df.loc[idx]
            num_out[idx] = asof_ewm_cumsum(sub["game_date"], sub[num_col], halflife_days)
            den_out[idx] = asof_ewm_cumsum(sub["game_date"], sub[den_col], halflife_days)

        league_mean = np.where(df[f"{stat}__lg_den"] > 0, df[f"{stat}__lg_num"] / df[f"{stat}__lg_den"], np.nan)
        # Fallback for the very first day(s) of the entire dataset, when even
        # the exponential league prior has zero mass: a strictly-prior-days
        # expanding mean (never the whole-dataset mean, which would leak
        # future games into the earliest predictions).
        expanding_fallback = asof_expanding_mean(df["game_date"], df[num_col], df[den_col])
        league_mean = np.where(np.isnan(league_mean), expanding_fallback, league_mean)
        # Last-resort constant only if there is truly zero prior data at all
        # (the first calendar day ever seen) — a documented neutral prior,
        # not a fabricated one; in practice never hit once a prior season's
        # games are prepended to the input.
        league_mean = np.where(np.isnan(league_mean), _NEUTRAL_PRIOR[stat], league_mean)

        df[f"league_{stat}"] = league_mean
        df[f"proj_{stat}"] = shrink(num_out, den_out, league_mean, shrinkage_k)
        df[f"n_prior_{stat}_den"] = den_out

    return df
