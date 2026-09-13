"""As-of-date team offense proxy: rolling runs-scored/game, shrunk toward the
same-date league average. This is a TEAM-LEVEL SHRINKAGE PRIOR, not the
primary offensive signal — see docs/limitations.md. It stands in for a real
confirmed-lineup wOBA-vs-handedness model, which the `mlb.lineups` package is
scaffolded for but not yet wired into the backtest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mlb.features.asof import asof_ewm_cumsum, asof_expanding_mean, shrink

# Neutral last-resort prior (runs/game per team, MLB long-run ballpark
# figure) — used only if there is zero prior data of any kind; see the
# identical guard in mlb.pitchers.projections for why this can't be a
# whole-dataset mean.
_NEUTRAL_RUNS_PRIOR = 4.4


def _long_format(schedule: pd.DataFrame) -> pd.DataFrame:
    completed = schedule[schedule["is_final"] & schedule["home_score"].notna() & schedule["away_score"].notna()]
    home = completed.rename(columns={
        "home_team": "team", "away_team": "opponent", "home_score": "runs_scored", "away_score": "runs_allowed",
    })[["game_pk", "game_date", "team", "opponent", "runs_scored", "runs_allowed"]].assign(is_home=True)
    away = completed.rename(columns={
        "away_team": "team", "home_team": "opponent", "away_score": "runs_scored", "home_score": "runs_allowed",
    })[["game_pk", "game_date", "team", "opponent", "runs_scored", "runs_allowed"]].assign(is_home=False)
    return pd.concat([home, away], ignore_index=True)


def add_asof_team_offense(schedule: pd.DataFrame, halflife_days: float, shrinkage_k_games: float) -> pd.DataFrame:
    """Returns one row per (game_pk, team) with `proj_runs_scored_per_game`
    and `proj_runs_allowed_per_game`, each computed strictly as-of (before)
    that game's date from that team's own prior games only.
    """
    df = _long_format(schedule)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk"]).reset_index(drop=True)
    df["one"] = 1.0

    day_totals = df.groupby("game_date")[["runs_scored", "runs_allowed", "one"]].sum().reset_index()
    lg_num_scored = asof_ewm_cumsum(day_totals["game_date"], day_totals["runs_scored"], halflife_days)
    lg_num_allowed = asof_ewm_cumsum(day_totals["game_date"], day_totals["runs_allowed"], halflife_days)
    lg_den = asof_ewm_cumsum(day_totals["game_date"], day_totals["one"], halflife_days)
    league_df = pd.DataFrame({
        "game_date": day_totals["game_date"],
        "lg_scored_num": lg_num_scored,
        "lg_allowed_num": lg_num_allowed,
        "lg_den": lg_den,
    })
    df = df.merge(league_df, on="game_date", how="left")

    expanding_scored = asof_expanding_mean(df["game_date"], df["runs_scored"], df["one"])
    expanding_allowed = asof_expanding_mean(df["game_date"], df["runs_allowed"], df["one"])
    lg_scored_mean = np.where(df["lg_den"] > 0, df["lg_scored_num"] / df["lg_den"], expanding_scored)
    lg_scored_mean = np.where(np.isnan(lg_scored_mean), _NEUTRAL_RUNS_PRIOR, lg_scored_mean)
    lg_allowed_mean = np.where(df["lg_den"] > 0, df["lg_allowed_num"] / df["lg_den"], expanding_allowed)
    lg_allowed_mean = np.where(np.isnan(lg_allowed_mean), _NEUTRAL_RUNS_PRIOR, lg_allowed_mean)

    for col, prior in [("runs_scored", lg_scored_mean), ("runs_allowed", lg_allowed_mean)]:
        num_out = np.zeros(len(df))
        den_out = np.zeros(len(df))
        for _, idx in df.groupby("team").groups.items():
            sub = df.loc[idx]
            num_out[idx] = asof_ewm_cumsum(sub["game_date"], sub[col], halflife_days)
            den_out[idx] = asof_ewm_cumsum(sub["game_date"], sub["one"], halflife_days)
        df[f"proj_{col}_per_game"] = shrink(num_out, den_out, prior, shrinkage_k_games)
        df[f"n_prior_{col}_games"] = den_out

    return df[[
        "game_pk", "game_date", "team", "opponent", "is_home",
        "proj_runs_scored_per_game", "proj_runs_allowed_per_game",
        "n_prior_runs_scored_games",
    ]]
