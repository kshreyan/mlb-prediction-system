"""Build and evaluate the stacked ensemble (simulation + Elo-only +
pitcher-adjusted-Elo [+ GBM], blended on log-odds via walk-forward logistic
regression). Requires simulation walk-forward predictions to already exist
for both 2023 (predictions_2023_both.parquet, using 2022 as prior) and 2024
(predictions_2024_both.parquet, using 2023 as prior) — see
scripts/run_backtest.py — and, if including GBM, gbm_2023.parquet /
gbm_2024.parquet — see scripts/run_gbm_baseline.py.

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
gf24_pks = set(frames[2]["game_pk"])

# --- Elo across all 3 seasons (walk-forward safe by construction) ---
elo_df = compute_elo_ratings(all_seasons, k=20.0, home_advantage=24.0, season_regress_frac=0.33)
elo_df["elo_diff"] = elo_df["elo_home_pre"] + 24.0 - elo_df["elo_away_pre"]

# --- Pitcher-adjusted Elo, walk-forward, for 2023 (using 2022 as warm-up) and 2024 (using 2023). ---
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

# --- Simulation model + GBM walk-forward predictions (already generated) ---
sim_all = pd.concat([
    pd.read_parquet(processed / "predictions_2023_both.parquet"),
    pd.read_parquet(processed / "predictions_2024_both.parquet"),
], ignore_index=True)
gbm_all = pd.concat([
    pd.read_parquet(processed / "gbm_2023.parquet"),
    pd.read_parquet(processed / "gbm_2024.parquet"),
], ignore_index=True)


def evaluate_ensemble(name: str, components: dict) -> pd.DataFrame:
    ens_frame = build_ensemble_frame(sim_all, components)
    feature_cols = [f"logit_{c}" for c in components]
    ens_result = walk_forward_logistic_baseline(ens_frame, feature_cols, retrain_freq_days=7, min_training_games=200)
    ens_result = ens_result.dropna(subset=["pred_prob"])
    ens_2024 = ens_result[ens_result["game_pk"].isin(gf24_pks)]
    m = binary_metrics(ens_2024["actual_home_win"], ens_2024["pred_prob"])
    ece = expected_calibration_error(ens_2024["actual_home_win"], ens_2024["pred_prob"])
    print(f"{name:28} n={m['n']:5} acc={m['accuracy']:.4f} brier={m['brier']:.4f} logloss={m['log_loss']:.4f} ece={ece:.4f}")
    return ens_2024


print("=== Component models alone (on 2024, matched to 3-way ensemble's game set) ===")
three_way_components = {"sim": (sim_all, "pred_home_win_prob"), "elo": (elo_df, "elo_pred_home_win_prob"), "pae": (pae_all, "pred_prob")}
tmp_frame = build_ensemble_frame(sim_all, three_way_components)
tmp_2024_pks = set(tmp_frame[tmp_frame["game_pk"].isin(gf24_pks)]["game_pk"])
for name, (df, col) in three_way_components.items():
    sub = df[df["game_pk"].isin(tmp_2024_pks)]
    if "actual_home_win" not in sub.columns:
        sub = sub.merge(all_seasons[["game_pk", "actual_home_win"]], on="game_pk", how="left")
    m = binary_metrics(sub["actual_home_win"], sub[col])
    ece = expected_calibration_error(sub["actual_home_win"], sub[col])
    print(f"{name:28} n={m['n']:5} acc={m['accuracy']:.4f} brier={m['brier']:.4f} logloss={m['log_loss']:.4f} ece={ece:.4f}")
sub = gbm_all[gbm_all["game_pk"].isin(tmp_2024_pks)]
m = binary_metrics(sub["actual_home_win"], sub["pred_prob"])
ece = expected_calibration_error(sub["actual_home_win"], sub["pred_prob"])
print(f"{'gbm':28} n={m['n']:5} acc={m['accuracy']:.4f} brier={m['brier']:.4f} logloss={m['log_loss']:.4f} ece={ece:.4f}")

print("\n=== Ensembles ===")
ens3 = evaluate_ensemble(
    "3-way (sim+elo+pae)",
    {"sim": (sim_all, "pred_home_win_prob"), "elo": (elo_df, "elo_pred_home_win_prob"), "pae": (pae_all, "pred_prob")},
)
ens4 = evaluate_ensemble(
    "4-way (sim+elo+pae+gbm)",
    {"sim": (sim_all, "pred_home_win_prob"), "elo": (elo_df, "elo_pred_home_win_prob"),
     "pae": (pae_all, "pred_prob"), "gbm": (gbm_all, "pred_prob")},
)

ens3.to_parquet(processed / "predictions_2024_ensemble.parquet", index=False)
ens4.to_parquet(processed / "predictions_2024_ensemble4.parquet", index=False)
print(f"\nSaved 3-way (shipped) ensemble -> {processed / 'predictions_2024_ensemble.parquet'}")
print(f"Saved 4-way (experimental, GBM) ensemble -> {processed / 'predictions_2024_ensemble4.parquet'}")
