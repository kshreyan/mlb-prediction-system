"""Schedule + final-score ingestion via the MLB Stats API.

Every row carries provenance (`source`, `fetched_at`) so downstream code can
tell real, dated data from anything else. Nothing here is ever fabricated:
a field MLB Stats API doesn't return comes back as null, not a guess.
"""
from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path

import pandas as pd
import statsapi

logger = logging.getLogger(__name__)

SOURCE = "mlb_stats_api"


def season_regular_season_bounds(season: int) -> tuple[dt.date, dt.date]:
    """Look up the actual regular-season start/end dates for `season` instead
    of assuming a fixed calendar window (openers/closers shift year to year)."""
    info = statsapi.get(
        "season", {"seasonId": season, "sportId": 1}
    )["seasons"][0]
    start = dt.date.fromisoformat(info["regularSeasonStartDate"])
    end = dt.date.fromisoformat(info["regularSeasonEndDate"])
    return start, end


def fetch_schedule(season: int, start: dt.date | None = None, end: dt.date | None = None) -> pd.DataFrame:
    """Fetch every regular-season game for `season` (optionally clipped to
    [start, end]) with final scores where the game has completed."""
    s_start, s_end = season_regular_season_bounds(season)
    start = max(start, s_start) if start else s_start
    end = min(end, s_end) if end else s_end

    # The Stats API's schedule endpoint times out on a full-season single
    # request; chunk month-by-month with retries instead.
    games: list[dict] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(end, chunk_start + dt.timedelta(days=29))
        for attempt in range(4):
            try:
                games.extend(
                    statsapi.schedule(
                        start_date=chunk_start.isoformat(),
                        end_date=chunk_end.isoformat(),
                        sportId=1,
                    )
                )
                break
            except Exception as exc:  # noqa: BLE001 - retry transient 503s
                if attempt == 3:
                    raise
                logger.warning("schedule chunk %s..%s failed (%s), retrying", chunk_start, chunk_end, exc)
                time.sleep(2 * (attempt + 1))
        chunk_start = chunk_end + dt.timedelta(days=1)
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat()

    rows = []
    for g in games:
        if g.get("game_type") != "R":  # regular season only
            continue
        rows.append(
            {
                "game_pk": g["game_id"],
                "season": season,
                "game_date": g["game_date"],
                "game_datetime": g.get("game_datetime"),
                "status": g.get("status"),
                "home_team": g["home_name"],
                "away_team": g["away_name"],
                "home_score": g.get("home_score"),
                "away_score": g.get("away_score"),
                "home_probable_pitcher": g.get("home_probable_pitcher") or None,
                "away_probable_pitcher": g.get("away_probable_pitcher") or None,
                "venue_id": g.get("venue_id"),
                "venue_name": g.get("venue_name"),
                "doubleheader": g.get("doubleheader"),
                "game_num": g.get("game_num"),
                "source": SOURCE,
                "fetched_at": fetched_at,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["is_final"] = df["status"].isin(["Final", "Completed Early", "Game Over"])
    df["home_win"] = None
    completed = df["is_final"] & df["home_score"].notna() & df["away_score"].notna()
    df.loc[completed, "home_win"] = (df.loc[completed, "home_score"] > df.loc[completed, "away_score"]).astype(int)

    # The Stats API lists a postponed-then-resumed/rescheduled game under
    # BOTH its original date (status "Postponed", 0-0) and its actual played
    # date (status "Final") — same game_pk twice. Keep one canonical row per
    # game_pk: prefer a completed status, and among completed duplicates
    # prefer the later date (the actual completion date).
    df = df.sort_values(["game_pk", "is_final", "game_date"], ascending=[True, False, False])
    df = df.drop_duplicates(subset="game_pk", keep="first").sort_values("game_date").reset_index(drop=True)
    return df


def load_or_fetch_schedule(season: int, raw_dir: Path, force: bool = False) -> pd.DataFrame:
    out_path = raw_dir / "schedule" / f"{season}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)
    df = fetch_schedule(season)
    df.to_parquet(out_path, index=False)
    logger.info("fetched schedule season=%s games=%d -> %s", season, len(df), out_path)
    return df
