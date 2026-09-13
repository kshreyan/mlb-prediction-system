"""As-of-date BATTER projections, split by the handedness of the pitcher
faced (the platoon split) — the whole point of doing this at batter level
instead of team level. Same anti-leakage machinery as pitcher projections:
exponentially time-weighted own history, shrunk toward a same-date,
same-split league average, both computed from strictly-prior games only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlb.features.asof import asof_ewm_cumsum, asof_expanding_mean, shrink

_NEUTRAL_XWOBA_PRIOR = 0.31
HALFLIFE_MULTIPLIER_FOR_SPARSE_SPLITS = 1.0  # kept as a hook, not tuned


def add_asof_batter_projections(batter_games: pd.DataFrame, halflife_days: float, shrinkage_k: float) -> pd.DataFrame:
    """Adds `proj_xwoba` computed separately within each `p_throws` split.
    Returns one row per (game_pk, batter, p_throws) — same grain as input.
    """
    df = batter_games.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

    out_frames = []
    for split, split_df in df.groupby("p_throws"):
        split_df = split_df.sort_values(["game_date", "game_pk"]).reset_index(drop=True)

        day_totals = split_df.groupby("game_date")[["xwoba_weighted_sum", "woba_denom_sum"]].sum().reset_index()
        lg_num = asof_ewm_cumsum(day_totals["game_date"], day_totals["xwoba_weighted_sum"], halflife_days)
        lg_den = asof_ewm_cumsum(day_totals["game_date"], day_totals["woba_denom_sum"], halflife_days)
        league_df = pd.DataFrame({"game_date": day_totals["game_date"], "lg_num": lg_num, "lg_den": lg_den})
        split_df = split_df.merge(league_df, on="game_date", how="left")

        expanding_mean = asof_expanding_mean(split_df["game_date"], split_df["xwoba_weighted_sum"], split_df["woba_denom_sum"])
        league_mean = np.where(split_df["lg_den"] > 0, split_df["lg_num"] / split_df["lg_den"], expanding_mean)
        league_mean = np.where(np.isnan(league_mean), _NEUTRAL_XWOBA_PRIOR, league_mean)

        num_out = np.zeros(len(split_df))
        den_out = np.zeros(len(split_df))
        for _, idx in split_df.groupby("batter").groups.items():
            sub = split_df.loc[idx]
            num_out[idx] = asof_ewm_cumsum(sub["game_date"], sub["xwoba_weighted_sum"], halflife_days)
            den_out[idx] = asof_ewm_cumsum(sub["game_date"], sub["woba_denom_sum"], halflife_days)

        split_df["proj_xwoba"] = shrink(num_out, den_out, league_mean, shrinkage_k)
        split_df["n_prior_pa"] = den_out
        out_frames.append(split_df)

    result = pd.concat(out_frames, ignore_index=True)
    return result[["game_pk", "game_date", "batter", "batting_team", "p_throws", "proj_xwoba", "n_prior_pa"]]
