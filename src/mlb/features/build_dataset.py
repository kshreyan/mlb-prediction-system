"""Assemble one row per game: starter projections (home + away), team offense
proxy, bullpen quality/fatigue, and park factor — all as-of strictly before
that game. This is the feature matrix every model (simulation, GBM baseline,
Elo) consumes.
"""
from __future__ import annotations

import pandas as pd


def build_game_features(
    schedule: pd.DataFrame,
    starter_proj: pd.DataFrame,
    team_offense: pd.DataFrame,
    bullpen: pd.DataFrame,
    park_factors: pd.DataFrame | None = None,
    lineup_offense: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    sched = schedule[schedule["is_final"] & schedule["home_score"].notna()].copy()
    sched["game_date"] = pd.to_datetime(sched["game_date"])

    starters = starter_proj[[
        "game_pk", "pitching_team", "player_name", "proj_k_pct", "proj_bb_pct",
        "proj_whiff_pct", "proj_csw_pct", "proj_barrel_pct", "proj_xwoba_against",
        "n_prior_xwoba_against_den",
    ]].copy()

    home_starters = starters.rename(columns={c: f"home_starter_{c}" for c in starters.columns if c not in ("game_pk", "pitching_team")})
    away_starters = starters.rename(columns={c: f"away_starter_{c}" for c in starters.columns if c not in ("game_pk", "pitching_team")})

    df = sched.merge(home_starters, left_on=["game_pk", "home_team"], right_on=["game_pk", "pitching_team"], how="left")
    df = df.drop(columns=["pitching_team"])
    df = df.merge(away_starters, left_on=["game_pk", "away_team"], right_on=["game_pk", "pitching_team"], how="left")
    df = df.drop(columns=["pitching_team"])

    off = team_offense[["game_pk", "team", "proj_runs_scored_per_game", "proj_runs_allowed_per_game", "n_prior_runs_scored_games"]]
    home_off = off.rename(columns={c: f"home_off_{c}" for c in off.columns if c not in ("game_pk", "team")})
    away_off = off.rename(columns={c: f"away_off_{c}" for c in off.columns if c not in ("game_pk", "team")})
    df = df.merge(home_off, left_on=["game_pk", "home_team"], right_on=["game_pk", "team"], how="left").drop(columns=["team"])
    df = df.merge(away_off, left_on=["game_pk", "away_team"], right_on=["game_pk", "team"], how="left").drop(columns=["team"])

    bp = bullpen[["game_pk", "team", "proj_bullpen_xwoba_against", "bullpen_recent_pitches", "n_prior_bullpen_den"]]
    home_bp = bp.rename(columns={c: f"home_bullpen_{c}" for c in bp.columns if c not in ("game_pk", "team")})
    away_bp = bp.rename(columns={c: f"away_bullpen_{c}" for c in bp.columns if c not in ("game_pk", "team")})
    df = df.merge(home_bp, left_on=["game_pk", "home_team"], right_on=["game_pk", "team"], how="left").drop(columns=["team"])
    df = df.merge(away_bp, left_on=["game_pk", "away_team"], right_on=["game_pk", "team"], how="left").drop(columns=["team"])

    if lineup_offense is not None and not lineup_offense.empty:
        lo = lineup_offense[["game_pk", "team", "lineup_proj_xwoba", "lineup_n_batters_matched"]]
        home_lo = lo.rename(columns={c: f"home_{c}" for c in lo.columns if c not in ("game_pk", "team")})
        away_lo = lo.rename(columns={c: f"away_{c}" for c in lo.columns if c not in ("game_pk", "team")})
        df = df.merge(home_lo, left_on=["game_pk", "home_team"], right_on=["game_pk", "team"], how="left").drop(columns=["team"])
        df = df.merge(away_lo, left_on=["game_pk", "away_team"], right_on=["game_pk", "team"], how="left").drop(columns=["team"])
        df["diff_lineup_xwoba"] = df["home_lineup_proj_xwoba"] - df["away_lineup_proj_xwoba"]

    if weather is not None and not weather.empty:
        w = weather[["game_pk", "wind_effect", "temp_f_filled", "is_indoor_or_roof_closed"]]
        df = df.merge(w, on="game_pk", how="left")
        df["wind_effect"] = df["wind_effect"].fillna(0.0)
        df["temp_f_filled"] = df["temp_f_filled"].fillna(72.0)
        df["is_indoor_or_roof_closed"] = df["is_indoor_or_roof_closed"].fillna(False)
    else:
        df["wind_effect"] = 0.0
        df["temp_f_filled"] = 72.0
        df["is_indoor_or_roof_closed"] = False

    if park_factors is not None and not park_factors.empty:
        pf = park_factors[["venue_name", "home_team", "park_factor"]]
        df = df.merge(pf, on=["venue_name", "home_team"], how="left")
        df["park_factor"] = df["park_factor"].fillna(100.0)
    else:
        df["park_factor"] = 100.0

    # Matchup diffs: positive = favors the home team.
    df["diff_starter_xwoba"] = df["away_starter_proj_xwoba_against"] - df["home_starter_proj_xwoba_against"]
    df["diff_bullpen_xwoba"] = df["away_bullpen_proj_bullpen_xwoba_against"] - df["home_bullpen_proj_bullpen_xwoba_against"]
    df["diff_offense"] = df["home_off_proj_runs_scored_per_game"] - df["away_off_proj_runs_scored_per_game"]
    df["diff_defense"] = df["away_off_proj_runs_allowed_per_game"] - df["home_off_proj_runs_allowed_per_game"]

    return df


FEATURE_COLUMNS = [
    "diff_starter_xwoba", "diff_bullpen_xwoba", "diff_offense", "diff_defense",
    "diff_lineup_xwoba",
    "home_starter_proj_xwoba_against", "away_starter_proj_xwoba_against",
    "home_bullpen_proj_bullpen_xwoba_against", "away_bullpen_proj_bullpen_xwoba_against",
    "home_off_proj_runs_scored_per_game", "away_off_proj_runs_scored_per_game",
    "home_lineup_proj_xwoba", "away_lineup_proj_xwoba",
    "home_bullpen_recent_pitches", "away_bullpen_recent_pitches",
    "park_factor",
]
