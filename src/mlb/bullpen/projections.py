"""As-of-date team bullpen quality + fatigue.

Bullpen quality is modeled at the team level (aggregate relief xwOBA-against,
exponentially time-weighted, shrunk to the league bullpen average) rather
than tracking individual relievers' rest days — a real, if simplified,
proxy for "is this bullpen good and fresh right now." Fatigue is a
non-decayed rolling workload sum: total relief pitches thrown by the team
in the trailing `fatigue_lookback_days` calendar days, which is a
predictable, explainable edge (a bullpen that threw 60 relief pitches
yesterday is more gassed today) independent of season-long quality.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlb.features.asof import asof_ewm_cumsum, asof_expanding_mean, shrink

_NEUTRAL_XWOBA_PRIOR = 0.31


def _all_team_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """Every (game_pk, team) the team was scheduled in, home or away —
    the reindex target so a complete-game start (zero relief appearances)
    still produces a row with 0 contribution that day, rather than being
    silently absent and leaving a hole for the feature-join to fill with NaN."""
    completed = schedule[schedule["is_final"]]
    home = completed[["game_pk", "game_date", "home_team"]].rename(columns={"home_team": "team"})
    away = completed[["game_pk", "game_date", "away_team"]].rename(columns={"away_team": "team"})
    return pd.concat([home, away], ignore_index=True)


def _team_bullpen_games(pitcher_games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    relief = pitcher_games[~pitcher_games["is_starter"]].copy()
    relief["game_date"] = pd.to_datetime(relief["game_date"])
    agg = relief.groupby(["game_pk", "game_date", "pitching_team"]).agg(
        n_pitches=("n_pitches", "sum"),
        batters_faced=("batters_faced", "sum"),
        woba_denom_sum=("woba_denom_sum", "sum"),
        xwoba_weighted_sum=("xwoba_weighted_sum", "sum"),
    ).reset_index().rename(columns={"pitching_team": "team"})

    all_games = _all_team_games(schedule)
    all_games["game_date"] = pd.to_datetime(all_games["game_date"])
    merged = all_games.merge(agg, on=["game_pk", "game_date", "team"], how="left")
    value_cols = ["n_pitches", "batters_faced", "woba_denom_sum", "xwoba_weighted_sum"]
    merged[value_cols] = merged[value_cols].fillna(0.0)
    return merged.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def _rolling_window_sum(dates: pd.Series, values: pd.Series, window_days: int) -> np.ndarray:
    """Non-decayed sum of `values` over the trailing `window_days` calendar
    days, STRICTLY EXCLUDING the current row's own date (as-of before)."""
    d = pd.to_datetime(dates).to_numpy()
    v = values.to_numpy(dtype=float)
    out = np.zeros(len(v))
    for i in range(len(v)):
        cutoff_low = d[i] - np.timedelta64(window_days, "D")
        mask = (d < d[i]) & (d >= cutoff_low)
        out[i] = v[mask].sum()
    return out


def add_asof_bullpen_projections(
    pitcher_games: pd.DataFrame,
    schedule: pd.DataFrame,
    halflife_days: float,
    shrinkage_k_batters: float,
    fatigue_lookback_days: int,
) -> pd.DataFrame:
    """Returns one row per (game_pk, team) — for EVERY scheduled game, even
    ones the bullpen didn't pitch in — with `proj_bullpen_xwoba_against` and
    `bullpen_recent_pitches` (trailing workload), both as-of strictly before
    that game.
    """
    df = _team_bullpen_games(pitcher_games, schedule)

    day_totals = df.groupby("game_date")[["woba_denom_sum", "xwoba_weighted_sum"]].sum().reset_index()
    lg_den = asof_ewm_cumsum(day_totals["game_date"], day_totals["woba_denom_sum"], halflife_days)
    lg_num = asof_ewm_cumsum(day_totals["game_date"], day_totals["xwoba_weighted_sum"], halflife_days)
    league_df = pd.DataFrame({"game_date": day_totals["game_date"], "lg_num": lg_num, "lg_den": lg_den})
    df = df.merge(league_df, on="game_date", how="left")

    expanding_mean = asof_expanding_mean(df["game_date"], df["xwoba_weighted_sum"], df["woba_denom_sum"])
    league_mean = np.where(df["lg_den"] > 0, df["lg_num"] / df["lg_den"], expanding_mean)
    league_mean = np.where(np.isnan(league_mean), _NEUTRAL_XWOBA_PRIOR, league_mean)

    num_out = np.zeros(len(df))
    den_out = np.zeros(len(df))
    fatigue_out = np.zeros(len(df))
    for _, idx in df.groupby("team").groups.items():
        sub = df.loc[idx]
        num_out[idx] = asof_ewm_cumsum(sub["game_date"], sub["xwoba_weighted_sum"], halflife_days)
        den_out[idx] = asof_ewm_cumsum(sub["game_date"], sub["woba_denom_sum"], halflife_days)
        fatigue_out[idx] = _rolling_window_sum(sub["game_date"], sub["n_pitches"], fatigue_lookback_days)

    df["proj_bullpen_xwoba_against"] = shrink(num_out, den_out, league_mean, shrinkage_k_batters)
    df["bullpen_recent_pitches"] = fatigue_out
    df["n_prior_bullpen_den"] = den_out

    return df[[
        "game_pk", "game_date", "team", "proj_bullpen_xwoba_against",
        "bullpen_recent_pitches", "n_prior_bullpen_den",
    ]]
