"""Empirical park factors derived from real game results (schedule data),
not a fabricated/hardcoded table.

Standard sabermetric method: for each home venue, compare the run-scoring
environment (runs scored + allowed per game) in games played there against
the same teams' run environment in games played elsewhere. A 3-season
trailing window is used (park factors are stable but not perfectly so —
e.g. humidor changes, fence changes), and — critically for leakage — the
factor applied to any given season is built only from *prior* completed
seasons, never the season being predicted.
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compute_park_factors(schedule: pd.DataFrame, min_games: int = 20) -> pd.DataFrame:
    """Compute one park factor per (venue_name, home_team) pair from a
    schedule dataframe (as returned by `mlb.data.schedule.fetch_schedule`,
    concatenated across the seasons that should inform the factor).

    Returns columns: venue_name, home_team, home_rpg, road_rpg, park_factor,
    n_home_games. `park_factor` is indexed to 100 = neutral.
    """
    df = schedule[schedule["is_final"] & schedule["home_score"].notna() & schedule["away_score"].notna()].copy()
    df["total_runs"] = df["home_score"] + df["away_score"]

    home_side = df.groupby(["venue_name", "home_team"]).agg(
        home_runs_per_game=("total_runs", "mean"),
        n_home_games=("total_runs", "size"),
    ).reset_index()

    # Road runs-per-game for that same team, across all their away games.
    road = df.groupby("away_team").agg(
        road_runs_per_game=("total_runs", "mean"),
        n_road_games=("total_runs", "size"),
    ).reset_index().rename(columns={"away_team": "home_team"})

    merged = home_side.merge(road, on="home_team", how="left")
    merged = merged[merged["n_home_games"] >= min_games]
    merged["park_factor"] = 100 * (merged["home_runs_per_game"] / merged["road_runs_per_game"])
    return merged.sort_values("park_factor", ascending=False).reset_index(drop=True)


def load_or_compute_park_factors(
    seasons_prior: list[int],
    raw_dir: Path,
    schedule_loader,
    force: bool = False,
) -> pd.DataFrame:
    """`schedule_loader(season, raw_dir)` should return that season's schedule
    dataframe (injected to avoid a circular import on mlb.data.schedule)."""
    tag = "-".join(str(s) for s in sorted(seasons_prior))
    out_path = raw_dir / "park_factors" / f"{tag}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    frames = [schedule_loader(s, raw_dir) for s in seasons_prior]
    combined = pd.concat(frames, ignore_index=True)
    pf = compute_park_factors(combined)
    pf["seasons_used"] = tag
    pf["computed_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    pf.to_parquet(out_path, index=False)
    logger.info("computed park factors from seasons=%s -> %d venues", tag, len(pf))
    return pf
