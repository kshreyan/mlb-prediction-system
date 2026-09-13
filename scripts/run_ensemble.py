"""Build and evaluate the stacked ensemble (simulation + Elo-only +
pitcher-adjusted-Elo, blended on log-odds via walk-forward logistic
regression). Requires simulation walk-forward predictions to already exist
for both 2023 (predictions_2023_both.parquet, using 2022 as prior) and 2024
(predictions_2024_both.parquet, using 2023 as prior) — see
scripts/run_backtest.py.

Usage: python scripts/run_ensemble.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mlb.config import load_backtest_config
from mlb.models.moneyline.elo import compute_elo_ratings
from mlb.models.moneyline.baselines import walk_forward_logistic_baseline
from mlb.ensemble.stacking import build_ensemble_frame
from mlb.evaluation.calibration import binary_metrics, expected_calibration_error

cfg = load_backtest_config()
processed = cfg.processed_dir

# --- Load game_features for all 3 seasons (needed for Elo + pitcher-adjusted-Elo inputs) ---
frames = []
for season in [2022, 2023, 2024]:
    gf = pd.read_parquet(processed / f"game_features_{season}.parquet")
    gf["game_date"] = pd.to_datetime(gf["game_date"])
    gf["actual_home_win"] = (gf["home_score"] > gf["away_score"]).astype(int)
    frames.append(gf)
all_seasons = pd.concat(frames, ignore_index=True).sort_values("game_date").reset_index(drop=True)

# --- Elo across all 3 seasons (walk-forward safe by construction) ---
elo_df = compute_elo_ratings(all_seasons, k=20.0, home_advantage=24.0, season_regress_frac=0.33)
elo_df["elo_diff"] = elo_df["elo_home_pre"] + 24.0 - elo_df["elo_away_pre"]

# --- Pitcher-adjusted Elo, walk-forward, for 2023 (using 2022 as its own
# warm-up via the same expanding-window discipline) and 2024 (using 2023). ---
merged = all_seasons.merge(elo_df[["game_pk", "elo_diff"]], on="game_pk", how="left")
pae_2023 = walk_forward_logistic_baseline(
    merged[merged["season"].isin([2022, 2023])].copy(),
    ["elo_diff", "diff_starter_xwoba", "diff_bullpen_xwoba"], retrain_freq_days=7,
)
pae_2023 = pae_2023[pae_2023["season"] == 2023].dropna(subset=["pred_prob"])

pae_2024 = walk_forward_logistic_baseline(
    merged[merged["season"].isin([2023, 2024])].copy(),
    ["elo_diff", "diff_starter_xwoba", "diff_bullpen_xwoba"], retrain_freq_days=7,
)
pae_2024 = pae_2024[pae_2024["season"] == 2024].dropna(subset=["pred_prob"])

pae_all = pd.concat([pae_2023, pae_2024], ignore_index=True)

# --- Simulation model walk-forward predictions (already generated) ---
sim_2023 = pd.read_parquet(processed / "predictions_2023_both.parquet")
sim_2024 = pd.read_parquet(processed / "predictions_2024_both.parquet")
sim_all = pd.concat([sim_2023, sim_2024], ignore_index=True)

# --- Build the ensemble frame and run the walk-forward stacking regression ---
ens_frame = build_ensemble_frame(sim_all, elo_df, pae_all)
print(f"Ensemble frame: {ens_frame.shape}, spanning {ens_frame['game_date'].min()} to {ens_frame['game_date'].max()}")

ens_result = walk_forward_logistic_baseline(
    ens_frame, ["logit_sim", "logit_elo", "logit_pae"], retrain_freq_days=7, min_training_games=200,
)
ens_result = ens_result.dropna(subset=["pred_prob"])

# Restrict final reporting to 2024 games only (the true holdout), matching
# every other result in this project.
gf24_pks = set(frames[2]["game_pk"])
ens_2024 = ens_result[ens_result["game_pk"].isin(gf24_pks)]

print(f"\nEnsemble predictions on 2024 holdout: n={len(ens_2024)}")
m = binary_metrics(ens_2024["actual_home_win"], ens_2024["pred_prob"])
ece = expected_calibration_error(ens_2024["actual_home_win"], ens_2024["pred_prob"])
print(f"ENSEMBLE: acc={m['accuracy']:.4f} brier={m['brier']:.4f} logloss={m['log_loss']:.4f} ece={ece:.4f}")

# Compare against each component on the SAME game subset for fairness.
common_pks = set(ens_2024["game_pk"])
for name, df, col in [
    ("SIM (component)", sim_2024, "pred_home_win_prob"),
    ("ELO (component)", elo_df[elo_df["game_pk"].isin(gf24_pks)], "elo_pred_home_win_prob"),
    ("PAE (component)", pae_2024, "pred_prob"),
]:
    sub = df[df["game_pk"].isin(common_pks)]
    label_col = "actual_home_win" if "actual_home_win" in sub.columns else None
    if label_col is None:
        sub = sub.merge(all_seasons[["game_pk", "actual_home_win"]], on="game_pk", how="left")
    mm = binary_metrics(sub["actual_home_win"], sub[col])
    ee = expected_calibration_error(sub["actual_home_win"], sub[col])
    print(f"{name:20} acc={mm['accuracy']:.4f} brier={mm['brier']:.4f} logloss={mm['log_loss']:.4f} ece={ee:.4f} n={mm['n']}")

ens_result.to_parquet(processed / "predictions_2024_ensemble.parquet", index=False)
print(f"\nSaved -> {processed / 'predictions_2024_ensemble.parquet'}")
