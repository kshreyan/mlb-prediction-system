"""Pull Statcast pitch-level data and derive, per game:
  1. Each batter's plate-appearance outcomes, split by the throwing hand of
     the pitcher he faced (the platoon split is the whole point — a batter's
     true talent vs LHP and vs RHP are different quantities).
  2. The actual starting lineup used (9 batters in batting order), inferred
     directly from plate-appearance sequence — no extra API calls needed.

Reuses pybaseball's on-disk Statcast cache, so re-running this after
`mlb.pitchers.statcast_pull` has already pulled a season costs no extra
network time.
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
    return pb.statcast(start_dt=start.isoformat(), end_dt=end.isoformat(), verbose=False)


def aggregate_batter_games(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, batter, opposing pitcher hand faced) — almost
    always one row per batter per game, occasionally two if a mid-game
    pitching change crossed handedness."""
    if pitch_df.empty:
        return pd.DataFrame()

    df = pitch_df.copy()
    df["is_bip"] = df["type"] == "X"
    df["xwoba_value"] = np.where(df["is_bip"], df["estimated_woba_using_speedangle"], df["woba_value"])
    df["batting_team"] = np.where(df["inning_topbot"] == "Top", df["away_team"], df["home_team"])

    pa_end = df[df["events"].notna()].copy()
    weighted = pa_end.assign(w=pa_end["woba_denom"].fillna(0) * pa_end["xwoba_value"].fillna(0))

    agg = weighted.groupby(["game_pk", "game_date", "batter", "batting_team", "p_throws"]).agg(
        pa=("events", "size"),
        woba_denom_sum=("woba_denom", "sum"),
        xwoba_weighted_sum=("w", "sum"),
    ).reset_index()
    agg["source"] = SOURCE
    agg["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    return agg.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def extract_starting_lineups(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_pk, team, batting_order_slot 1-9, batter, stand) —
    the actual lineup used, inferred as the first 9 distinct batters (by
    at_bat_number) for that team in that game. This is the REAL lineup that
    played (not a pre-game "confirmed" listing), which is the right ground
    truth for backtesting a lineup-aware model even though a live/future
    prediction pipeline would need the pre-game confirmed lineup instead
    (see docs/limitations.md).
    """
    if pitch_df.empty:
        return pd.DataFrame()

    df = pitch_df.copy()
    df["batting_team"] = np.where(df["inning_topbot"] == "Top", df["away_team"], df["home_team"])

    ordered = df.sort_values(["game_pk", "batting_team", "at_bat_number"])
    first_appearance = ordered.drop_duplicates(subset=["game_pk", "batting_team", "batter"], keep="first")

    lineups = []
    for (game_pk, team), group in first_appearance.groupby(["game_pk", "batting_team"]):
        top9 = group.sort_values("at_bat_number").head(9).reset_index(drop=True)
        for slot, row in top9.iterrows():
            lineups.append({
                "game_pk": game_pk, "game_date": row["game_date"], "team": team,
                "batting_order_slot": slot + 1, "batter": row["batter"], "stand": row["stand"],
            })
    return pd.DataFrame(lineups)


def load_or_fetch_batter_data(season: int, raw_dir: Path, start: dt.date, end: dt.date, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    batter_path = raw_dir / "batter_games" / f"{season}.parquet"
    lineup_path = raw_dir / "lineups" / f"{season}.parquet"
    batter_path.parent.mkdir(parents=True, exist_ok=True)
    lineup_path.parent.mkdir(parents=True, exist_ok=True)

    if batter_path.exists() and lineup_path.exists() and not force:
        return pd.read_parquet(batter_path), pd.read_parquet(lineup_path)

    batter_frames, lineup_frames = [], []
    cur = start
    while cur <= end:
        chunk_end = min(end, cur + dt.timedelta(days=13))
        logger.info("fetching statcast (batters/lineups) %s..%s", cur, chunk_end)
        raw = _fetch_range(cur, chunk_end)
        batter_frames.append(aggregate_batter_games(raw))
        lineup_frames.append(extract_starting_lineups(raw))
        cur = chunk_end + dt.timedelta(days=1)

    batter_df = pd.concat(batter_frames, ignore_index=True) if batter_frames else pd.DataFrame()
    lineup_df = pd.concat(lineup_frames, ignore_index=True) if lineup_frames else pd.DataFrame()
    batter_df.to_parquet(batter_path, index=False)
    lineup_df.to_parquet(lineup_path, index=False)
    logger.info("saved batter_games rows=%d, lineups rows=%d", len(batter_df), len(lineup_df))
    return batter_df, lineup_df
