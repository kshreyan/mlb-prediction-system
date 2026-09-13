"""Hyperparameter search for the as-of-date halflife/shrinkage values, using
ONLY 2023 data — split chronologically into a warm-start pool (games before
a cutoff) and a validation slice (games from the cutoff on), scored by
walk-forward log loss on the validation slice. This keeps 2024 completely
untouched as the final honest holdout (tuning on it would be a leakage risk
of its own kind — overfitting the hyperparameters to the reporting season).

Coordinate descent: tune one module's (halflife, shrinkage_k) at a time,
holding the others at their current-best value, over a small grid. Not a
full joint grid search (too expensive for the time budget) or the nested
CV the original spec calls for, but a real, leakage-safe, held-out search —
see docs/limitations.md for the honest caveat.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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

SEASON = 2023
CUTOFF = pd.Timestamp("2023-07-20")

cfg = load_backtest_config()
raw_dir = cfg.raw_dir

print("Loading raw 2023 data once (cached, no network)...")
sched = load_or_fetch_schedule(SEASON, raw_dir)
pg = pd.read_parquet(raw_dir / "pitcher_games" / f"{SEASON}.parquet")
for col in ["pitching_team", "home_team", "away_team"]:
    pg = normalize_team_column(pg, col, SEASON, raw_dir)
starters_raw = pg[pg["is_starter"]].copy()

start, end = season_regular_season_bounds(SEASON)
batters_raw, lineups_raw = load_or_fetch_batter_data(SEASON, raw_dir, start, end)
batters_raw = normalize_team_column(batters_raw, "batting_team", SEASON, raw_dir)
lineups_raw = normalize_team_column(lineups_raw, "team", SEASON, raw_dir)

pf = load_or_compute_park_factors([SEASON - 1], raw_dir, load_or_fetch_schedule)
game_pks = [int(x) for x in sched[sched["is_final"]]["game_pk"].tolist()]
weather = add_derived_weather_features(load_or_fetch_weather(SEASON, game_pks, raw_dir))


def build_gf(hp: dict) -> pd.DataFrame:
    sproj = add_asof_pitcher_projections(starters_raw, halflife_days=hp["pitcher_halflife"], shrinkage_k=hp["pitcher_k"])
    bp = add_asof_bullpen_projections(pg, sched, halflife_days=hp["bullpen_halflife"], shrinkage_k_batters=hp["bullpen_k"], fatigue_lookback_days=3)
    off = add_asof_team_offense(sched, halflife_days=hp["team_offense_halflife"], shrinkage_k_games=hp["team_offense_k"])
    bproj = add_asof_batter_projections(batters_raw, halflife_days=hp["batter_halflife"], shrinkage_k=hp["batter_k"])
    starter_hand = derive_starter_hand_by_team_game(starters_raw)
    lineup_off = build_lineup_offense_features(lineups_raw, bproj, starter_hand)
    gf = build_game_features(sched, sproj, off, bp, pf, lineup_offense=lineup_off, weather=weather)
    gf["game_date"] = pd.to_datetime(gf["game_date"])
    return gf


def score(hp: dict) -> dict:
    gf = build_gf(hp)
    prior_pool = gf[gf["game_date"] < CUTOFF]
    validation = gf[gf["game_date"] >= CUTOFF]
    bt_cfg = BacktestConfig(retrain_freq_days=7, n_sims=cfg.simulation.n_sims, seed=cfg.random_seed, feature_set="both")
    preds = walk_forward_backtest(validation, prior_pool, bt_cfg)
    m = binary_metrics(preds["actual_home_win"], preds["pred_home_win_prob"])
    return {"log_loss": m["log_loss"], "brier": m["brier"], "accuracy": m["accuracy"], "n": m["n"]}


DEFAULT = {
    "pitcher_halflife": 45.0, "pitcher_k": 250.0,
    "batter_halflife": 60.0, "batter_k": 200.0,
    "bullpen_halflife": 20.0, "bullpen_k": 250.0,
    "team_offense_halflife": 30.0, "team_offense_k": 12.0,
}

GRID = {
    "pitcher_halflife": [25.0, 45.0, 75.0],
    "pitcher_k": [125.0, 250.0, 400.0],
    "batter_halflife": [35.0, 60.0, 100.0],
    "batter_k": [100.0, 200.0, 350.0],
    "bullpen_halflife": [10.0, 20.0, 35.0],
    "bullpen_k": [125.0, 250.0, 400.0],
    "team_offense_halflife": [15.0, 30.0, 50.0],
    "team_offense_k": [6.0, 12.0, 20.0],
}

if __name__ == "__main__":
    best = dict(DEFAULT)
    print(f"Baseline (defaults): {score(best)}\n")

    results_log = []
    for param in ["pitcher_halflife", "pitcher_k", "batter_halflife", "batter_k",
                  "bullpen_halflife", "bullpen_k", "team_offense_halflife", "team_offense_k"]:
        best_val = best[param]
        best_score = score(best)["log_loss"]
        print(f"Tuning {param} (holding others at current best)...")
        for candidate in GRID[param]:
            if candidate == best[param]:
                continue
            trial = dict(best)
            trial[param] = candidate
            s = score(trial)
            print(f"  {param}={candidate}: log_loss={s['log_loss']:.4f} brier={s['brier']:.4f} acc={s['accuracy']:.4f}")
            results_log.append({"param": param, "value": candidate, **s})
            if s["log_loss"] < best_score:
                best_score = s["log_loss"]
                best_val = candidate
        best[param] = best_val
        print(f"  -> best {param} = {best_val} (log_loss={best_score:.4f})\n")

    print("FINAL TUNED HYPERPARAMETERS:")
    print(best)
    final_score = score(best)
    print(f"Final validation score: {final_score}")

    pd.DataFrame(results_log).to_csv(cfg.processed_dir / "tuning_log.csv", index=False)
    import json
    (cfg.processed_dir / "tuned_hyperparams.json").write_text(json.dumps(best, indent=2))
