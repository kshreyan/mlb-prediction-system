"""Build the full game-feature dataset for one season: schedule + as-of-date
starter/bullpen/offense projections + leak-free (prior-seasons-only) park
factors. Run as: python scripts/build_features.py 2024
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mlb.config import load_backtest_config
from mlb.data.schedule import load_or_fetch_schedule
from mlb.data.team_ids import normalize_team_column
from mlb.pitchers.projections import add_asof_pitcher_projections
from mlb.bullpen.projections import add_asof_bullpen_projections
from mlb.features.team_offense import add_asof_team_offense
from mlb.features.build_dataset import build_game_features
from mlb.park_weather.park_factors import load_or_compute_park_factors


def build_season(season: int, cfg, park_factor_seasons: list[int]) -> pd.DataFrame:
    raw_dir = cfg.raw_dir
    sched = load_or_fetch_schedule(season, raw_dir)

    pg_path = raw_dir / "pitcher_games" / f"{season}.parquet"
    pg = pd.read_parquet(pg_path)
    for col in ["pitching_team", "home_team", "away_team"]:
        pg = normalize_team_column(pg, col, season, raw_dir)

    starters = pg[pg["is_starter"]].copy()
    sproj = add_asof_pitcher_projections(
        starters,
        halflife_days=cfg.pitcher_projection.halflife_days,
        shrinkage_k=cfg.pitcher_projection.shrinkage_k_batters,
    )
    bp = add_asof_bullpen_projections(
        pg,
        sched,
        halflife_days=cfg.bullpen.halflife_days,
        shrinkage_k_batters=cfg.pitcher_projection.shrinkage_k_batters,
        fatigue_lookback_days=cfg.bullpen.fatigue_lookback_days,
    )
    off = add_asof_team_offense(
        sched,
        halflife_days=cfg.team_offense.halflife_days,
        shrinkage_k_games=cfg.team_offense.shrinkage_k_games,
    )
    pf = load_or_compute_park_factors(park_factor_seasons, raw_dir, load_or_fetch_schedule)

    gf = build_game_features(sched, sproj, off, bp, pf)
    gf["season"] = season
    return gf


if __name__ == "__main__":
    season = int(sys.argv[1])
    cfg = load_backtest_config()
    pf_seasons = [season - 2, season - 1] if season - 2 >= 2022 else [season - 1]
    gf = build_season(season, cfg, pf_seasons)
    out_path = cfg.processed_dir / f"game_features_{season}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gf.to_parquet(out_path, index=False)
    print(f"season={season} shape={gf.shape} missing_starter_proj="
          f"{gf['home_starter_proj_xwoba_against'].isna().sum()} -> {out_path}")
