"""Pull Statcast pitch-level data and aggregate to one row per
(pitcher, game) with the peripheral stats that are more predictively stable
than ERA/W-L: xwOBA against, K%, BB%, whiff%, CSW%, barrel%.

Raw pitch-level data is not persisted long-term (it's large and cheaply
re-fetchable via pybaseball's own on-disk cache); only the aggregated
per-pitcher-game rows are written to data/raw, one file per season.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pybaseball as pb

logger = logging.getLogger(__name__)

SOURCE = "statcast_via_pybaseball"

pb.cache.enable()


def _fetch_range(start: dt.date, end: dt.date) -> pd.DataFrame:
    df = pb.statcast(start_dt=start.isoformat(), end_dt=end.isoformat(), verbose=False)
    return df


def aggregate_pitcher_games(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw pitches to one row per (game_pk, pitcher)."""
    if pitch_df.empty:
        return pd.DataFrame()

    df = pitch_df.copy()
    df["is_swing"] = df["description"].isin(
        ["swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
         "hit_into_play", "foul_bunt", "missed_bunt"]
    )
    df["is_whiff"] = df["description"].isin(["swinging_strike", "swinging_strike_blocked", "foul_tip"])
    df["is_called_strike"] = df["description"] == "called_strike"
    df["is_csw"] = df["is_whiff"] | df["is_called_strike"]
    df["is_bip"] = df["type"] == "X"
    df["is_barrel"] = df["launch_speed_angle"] == 6  # Statcast's own barrel flag

    # xwOBA-against value per plate-appearance-ending pitch: use the
    # contact-quality estimate for balls in play, and the actual (already
    # deterministic) wOBA value for non-batted-ball PA endings (K/BB/HBP).
    df["xwoba_value"] = np.where(df["is_bip"], df["estimated_woba_using_speedangle"], df["woba_value"])

    # The pitcher's own team: he pitches while the OTHER side bats, so
    # inning_topbot=='Top' (visitors batting) means the home team is pitching.
    df["pitching_team"] = np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])

    group_keys = ["game_pk", "game_date", "pitcher", "player_name", "home_team", "away_team", "pitching_team"]
    agg = df.groupby(group_keys).agg(
        n_pitches=("pitch_type", "size"),
        n_swings=("is_swing", "sum"),
        n_whiffs=("is_whiff", "sum"),
        n_csw=("is_csw", "sum"),
        n_bip=("is_bip", "sum"),
        n_barrels=("is_barrel", "sum"),
        woba_denom_sum=("woba_denom", "sum"),
    ).reset_index()

    # xwOBA-against weighted sum, computed explicitly (not inside the agg
    # above) so the weighting by woba_denom is unambiguous.
    weighted = df.assign(w=df["woba_denom"].fillna(0) * df["xwoba_value"].fillna(0))
    wsum = weighted.groupby(["game_pk", "pitcher"])["w"].sum().rename("xwoba_weighted_sum")
    agg = agg.merge(wsum, on=["game_pk", "pitcher"], how="left")

    # Batters faced and true outcomes come from plate-appearance-ending rows only.
    pa_end = df[df["events"].notna()].copy()
    pa_end["is_k"] = pa_end["events"].isin(["strikeout", "strikeout_double_play"])
    pa_end["is_bb"] = pa_end["events"] == "walk"
    pa_end["is_hbp"] = pa_end["events"] == "hit_by_pitch"

    pa_agg = pa_end.groupby(["game_pk", "pitcher"]).agg(
        batters_faced=("events", "size"),
        n_k=("is_k", "sum"),
        n_bb=("is_bb", "sum"),
        n_hbp=("is_hbp", "sum"),
    ).reset_index()

    # Starter = the pitcher who faced the first plate appearance for his team
    # in the game (min at_bat_number within that game+pitching_team group).
    df["_first_pa_of_team"] = df.groupby(["game_pk", "pitching_team"])["at_bat_number"].transform("min")
    starters = (
        df[df["at_bat_number"] == df["_first_pa_of_team"]]
        .groupby(["game_pk", "pitching_team"])["pitcher"].first()
        .reset_index().rename(columns={"pitcher": "starter_pitcher"})
    )
    agg = agg.merge(starters, on=["game_pk", "pitching_team"], how="left")
    agg["is_starter"] = agg["pitcher"] == agg["starter_pitcher"]
    agg = agg.drop(columns=["starter_pitcher"])

    out = agg.merge(pa_agg, on=["game_pk", "pitcher"], how="left")

    out["k_pct"] = out["n_k"] / out["batters_faced"]
    out["bb_pct"] = out["n_bb"] / out["batters_faced"]
    out["whiff_pct"] = out["n_whiffs"] / out["n_swings"].replace(0, np.nan)
    out["csw_pct"] = out["n_csw"] / out["n_pitches"]
    out["barrel_pct"] = out["n_barrels"] / out["n_bip"].replace(0, np.nan)
    out["xwoba_against"] = out["xwoba_weighted_sum"] / out["woba_denom_sum"].replace(0, np.nan)

    out["source"] = SOURCE
    out["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()

    keep = [
        "game_pk", "game_date", "pitcher", "player_name", "home_team", "away_team",
        "pitching_team", "is_starter", "batters_faced", "n_pitches", "n_swings", "n_whiffs",
        "n_csw", "n_bip", "n_barrels", "woba_denom_sum", "xwoba_weighted_sum",
        "n_k", "n_bb", "n_hbp",
        "k_pct", "bb_pct", "whiff_pct", "csw_pct", "barrel_pct", "xwoba_against",
        "source", "fetched_at",
    ]
    return out[keep].sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def load_or_fetch_pitcher_games(season: int, raw_dir: Path, start: dt.date, end: dt.date, force: bool = False) -> pd.DataFrame:
    out_path = raw_dir / "pitcher_games" / f"{season}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    monthly = []
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + dt.timedelta(days=13))
        logger.info("fetching statcast %s..%s", cur, chunk_end)
        raw = _fetch_range(cur, chunk_end)
        agg = aggregate_pitcher_games(raw)
        monthly.append(agg)
        cur = chunk_end + dt.timedelta(days=1)

    df = pd.concat(monthly, ignore_index=True) if monthly else pd.DataFrame()
    df.to_parquet(out_path, index=False)
    logger.info("saved pitcher-game aggregates season=%s rows=%d -> %s", season, len(df), out_path)
    return df
