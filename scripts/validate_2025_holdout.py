"""Out-of-sample validation on the 2025 season — a genuinely untouched
holdout, never used in any tuning decision (unlike 2019/2021/2022/2023,
which were used across the 4 tuning attempts, or 2024, the primary
validation season this build's headline numbers are reported against).
This answers directly: does the shipped model (final hyperparameters,
final 3-way stacked ensemble) generalize to a season it has never
influenced in any way, or were the headline 2024 numbers partly luck?

Requires game_features_2025.parquet (scripts/build_features.py 2025) and
predictions_2025_both.parquet (scripts/run_backtest.py 2025 --prior 2024
--feature-set both --out-suffix _both) to already exist.

Usage: python scripts/validate_2025_holdout.py
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

frames = []
for season in [2024, 2025]:
    gf = pd.read_parquet(processed / f"game_features_{season}.parquet")
    gf["game_date"] = pd.to_datetime(gf["game_date"])
    gf["actual_home_win"] = (gf["home_score"] > gf["away_score"]).astype(int)
    frames.append(gf)
all_seasons = pd.concat(frames, ignore_index=True).sort_values("game_date").reset_index(drop=True)
gf25_pks = set(frames[1]["game_pk"])

elo_df = compute_elo_ratings(all_seasons, k=20.0, home_advantage=24.0, season_regress_frac=0.33)
elo_df["elo_diff"] = elo_df["elo_home_pre"] + 24.0 - elo_df["elo_away_pre"]

merged = all_seasons.merge(elo_df[["game_pk", "elo_diff"]], on="game_pk", how="left")
pae_2025 = walk_forward_logistic_baseline(
    merged, ["elo_diff", "diff_starter_xwoba", "diff_bullpen_xwoba"], retrain_freq_days=7,
)
pae_2025 = pae_2025[pae_2025["season"] == 2025].dropna(subset=["pred_prob"])

sim_2025 = pd.read_parquet(processed / "predictions_2025_both.parquet")

components = {"sim": (sim_2025, "pred_home_win_prob"), "elo": (elo_df, "elo_pred_home_win_prob"), "pae": (pae_2025, "pred_prob")}
ens_frame = build_ensemble_frame(sim_2025, components)
ens_result = walk_forward_logistic_baseline(ens_frame, [f"logit_{c}" for c in components], retrain_freq_days=7, min_training_games=200)
ens_result = ens_result.dropna(subset=["pred_prob"])
ens_2025 = ens_result[ens_result["game_pk"].isin(gf25_pks)]

print("=== Moneyline, 2025 (genuinely untouched holdout) vs. reported 2024 headline numbers ===")
print(f"{'model':28} {'n':>5} {'acc':>7} {'brier':>8} {'logloss':>8} {'ece':>7}   (2024 reported)")

def row(name, y_true, y_pred, reported_2024=""):
    m = binary_metrics(y_true, y_pred)
    ece = expected_calibration_error(y_true, y_pred)
    print(f"{name:28} {m['n']:5} {m['accuracy']:7.4f} {m['brier']:8.4f} {m['log_loss']:8.4f} {ece:7.4f}   {reported_2024}")

tmp_pks = set(ens_2025["game_pk"])
sim_sub = sim_2025[sim_2025["game_pk"].isin(tmp_pks)]
row("simulation (component)", sim_sub["actual_home_win"], sim_sub["pred_home_win_prob"], "acc=0.554 brier=0.2470")
elo_sub = elo_df[elo_df["game_pk"].isin(tmp_pks)]
row("elo-only (component)", elo_sub["actual_home_win"], elo_sub["elo_pred_home_win_prob"], "acc=0.550 brier=0.2483")
pae_sub = pae_2025[pae_2025["game_pk"].isin(tmp_pks)]
row("pitcher-adjusted elo (component)", pae_sub["actual_home_win"], pae_sub["pred_prob"], "acc=0.549 brier=0.2455")
row("STACKED ENSEMBLE (shipped)", ens_2025["actual_home_win"], ens_2025["pred_prob"], "acc=0.563 brier=0.2447 ece=0.0095")

ens_2025.to_parquet(processed / "predictions_2025_ensemble_final.parquet", index=False)
print(f"\nSaved -> {processed / 'predictions_2025_ensemble_final.parquet'}")
