"""A targeted, lighter-weight follow-up to scripts/tune_hyperparams.py: adds
2019 as a 4th validation season (2020 skipped deliberately — a 60-game
pandemic-shortened season is a poor validation signal) and checks
specifically whether the THREE halflife parameters that trended upward in
every prior tuning pass (pitcher 45->75->110, batter 60->100->140,
team-offense 30->50->70) continue that trend, plateau, or reverse — rather
than re-running the full 8-parameter x 3-candidate grid, which would mostly
just re-confirm parameters that have already been stable across 3
consecutive independent validation seasons. This is a deliberate,
documented scope choice given diminishing returns, not a shortcut taken
silently.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from mlb.config import load_backtest_config
from mlb.data.schedule import load_or_fetch_schedule, season_regular_season_bounds
from mlb.data.team_ids import normalize_team_column
from mlb.pitchers.projections import add_asof_pitcher_projections
from mlb.bullpen.projections import add_asof_bullpen_projections
from mlb.features.team_offense import add_asof_team_offense
from mlb.features.build_dataset import build_game_features
from mlb.park_weather.park_factors import load_or_compute_park_factors
from mlb.park_weather.weather import load_or_fetch_weather, add_derived_weather_features
from mlb.lineups.statcast_pull import load_or_fetch_batter_data
from mlb.lineups.projections import add_asof_batter_projections
from mlb.lineups.lineup_offense import build_lineup_offense_features, derive_starter_hand_by_team_game
from mlb.backtest.walk_forward import BacktestConfig, walk_forward_backtest
from mlb.evaluation.calibration import binary_metrics
from sklearn.metrics import log_loss, brier_score_loss

TUNING_SEASONS = [2019, 2021, 2022, 2023]
CUTOFFS = {
    2019: pd.Timestamp("2019-07-13"),
    2021: pd.Timestamp("2021-07-21"),
    2022: pd.Timestamp("2022-07-24"),
    2023: pd.Timestamp("2023-07-19"),
}
PARK_FACTOR_PRIOR = {2019: [2018], 2021: [2020], 2022: [2021], 2023: [2022]}

cfg = load_backtest_config()
raw_dir = cfg.raw_dir

print("Loading raw data for 4 tuning seasons (cached, no network)...")
RAW = {}
for season in TUNING_SEASONS:
    sched = load_or_fetch_schedule(season, raw_dir)
    pg = pd.read_parquet(raw_dir / "pitcher_games" / f"{season}.parquet")
    for col in ["pitching_team", "home_team", "away_team"]:
        pg = normalize_team_column(pg, col, season, raw_dir)
    starters_raw = pg[pg["is_starter"]].copy()

    start, end = season_regular_season_bounds(season)
    batters_raw, lineups_raw = load_or_fetch_batter_data(season, raw_dir, start, end)
    batters_raw = normalize_team_column(batters_raw, "batting_team", season, raw_dir)
    lineups_raw = normalize_team_column(lineups_raw, "team", season, raw_dir)

    pf = load_or_compute_park_factors(PARK_FACTOR_PRIOR[season], raw_dir, load_or_fetch_schedule)
    game_pks = [int(x) for x in sched[sched["is_final"]]["game_pk"].tolist()]
    weather = add_derived_weather_features(load_or_fetch_weather(season, game_pks, raw_dir))

    pa_weight_frames = []
    for prior_s in PARK_FACTOR_PRIOR[season]:
        lp, bp_path = raw_dir / "lineups" / f"{prior_s}.parquet", raw_dir / "batter_games" / f"{prior_s}.parquet"
        if lp.exists() and bp_path.exists():
            pa_weight_frames.append(pd.read_parquet(lp).merge(
                pd.read_parquet(bp_path)[["game_pk", "batter", "pa"]], on=["game_pk", "batter"], how="left"
            ))
    pa_weights = pd.concat(pa_weight_frames, ignore_index=True).groupby("batting_order_slot")["pa"].mean().to_dict() if pa_weight_frames else None

    RAW[season] = dict(sched=sched, pg=pg, starters_raw=starters_raw, batters_raw=batters_raw,
                       lineups_raw=lineups_raw, pf=pf, weather=weather, pa_weights=pa_weights)
print("Loaded:", {s: RAW[s]["sched"].shape for s in TUNING_SEASONS})


def build_gf(season: int, hp: dict) -> pd.DataFrame:
    r = RAW[season]
    sproj = add_asof_pitcher_projections(r["starters_raw"], halflife_days=hp["pitcher_halflife"], shrinkage_k=hp["pitcher_k"])
    bp = add_asof_bullpen_projections(r["pg"], r["sched"], halflife_days=hp["bullpen_halflife"], shrinkage_k_batters=hp["bullpen_k"], fatigue_lookback_days=3)
    off = add_asof_team_offense(r["sched"], halflife_days=hp["team_offense_halflife"], shrinkage_k_games=hp["team_offense_k"])
    bproj = add_asof_batter_projections(r["batters_raw"], halflife_days=hp["batter_halflife"], shrinkage_k=hp["batter_k"])
    starter_hand = derive_starter_hand_by_team_game(r["starters_raw"])
    lineup_off = build_lineup_offense_features(r["lineups_raw"], bproj, starter_hand, pa_weights_by_slot=r["pa_weights"])
    gf = build_game_features(r["sched"], sproj, off, bp, r["pf"], lineup_offense=lineup_off, weather=r["weather"])
    gf["game_date"] = pd.to_datetime(gf["game_date"])
    return gf


def score(hp: dict) -> dict:
    all_actual, all_pred = [], []
    per_season = {}
    for season in TUNING_SEASONS:
        gf = build_gf(season, hp)
        cutoff = CUTOFFS[season]
        prior_pool = gf[gf["game_date"] < cutoff]
        validation = gf[gf["game_date"] >= cutoff]
        bt_cfg = BacktestConfig(retrain_freq_days=7, n_sims=cfg.simulation.n_sims, seed=cfg.random_seed, feature_set="both")
        preds = walk_forward_backtest(validation, prior_pool, bt_cfg)
        all_actual.extend(preds["actual_home_win"].tolist())
        all_pred.extend(preds["pred_home_win_prob"].tolist())
        per_season[season] = binary_metrics(preds["actual_home_win"], preds["pred_home_win_prob"])["log_loss"]

    all_actual = np.array(all_actual)
    all_pred = np.clip(np.array(all_pred), 1e-6, 1 - 1e-6)
    return {
        "pooled_log_loss": float(log_loss(all_actual, all_pred, labels=[0, 1])),
        "pooled_brier": float(brier_score_loss(all_actual, all_pred)),
        "n": len(all_actual),
        "per_season_log_loss": per_season,
    }


# The 3-season (2021+2022+2023) winner — the config currently shipped.
CURRENT = {
    "pitcher_halflife": 110.0, "pitcher_k": 250.0,
    "batter_halflife": 140.0, "batter_k": 100.0,
    "bullpen_halflife": 35.0, "bullpen_k": 400.0,
    "team_offense_halflife": 70.0, "team_offense_k": 20.0,
}

# Only the 3 parameters that have trended upward every round get new,
# further-extended candidates tested here; everything else is held fixed
# at CURRENT since it was already confirmed stable across 3 seasons.
TREND_CANDIDATES = {
    "pitcher_halflife": [110.0, 150.0],
    "batter_halflife": [140.0, 180.0],
    "team_offense_halflife": [70.0, 100.0],
}

if __name__ == "__main__":
    baseline = score(CURRENT)
    print(f"CURRENT config (3-season winner) on 4 seasons: {baseline}\n")

    best = dict(CURRENT)
    best_score = baseline["pooled_log_loss"]
    for param, candidates in TREND_CANDIDATES.items():
        print(f"Checking {param} (holding all else at current)...")
        for candidate in candidates:
            if candidate == best[param]:
                continue
            trial = dict(best)
            trial[param] = candidate
            s = score(trial)
            print(f"  {param}={candidate}: pooled_log_loss={s['pooled_log_loss']:.4f} per_season={s['per_season_log_loss']}")
            if s["pooled_log_loss"] < best_score:
                best_score = s["pooled_log_loss"]
                best[param] = candidate
        print(f"  -> best {param} = {best[param]} (pooled_log_loss={best_score:.4f})\n")

    print("FINAL (4-season-checked) HYPERPARAMETERS:")
    print(best)
    final = score(best)
    print(f"Final pooled validation score (4 seasons): {final}")

    import json
    (cfg.processed_dir / "tuned_hyperparams_v4_fourseason_check.json").write_text(json.dumps(best, indent=2))
