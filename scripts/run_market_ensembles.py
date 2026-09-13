"""Extend the stacked-ensemble approach to totals and run line, each
blending the simulation model with a second, differently-shaped direct
model (Ridge regression for totals, logistic regression for run line) on
the matchup features. Requires predictions_2023_both.parquet /
predictions_2024_both.parquet (scripts/run_backtest.py) to already exist.

Usage: python scripts/run_market_ensembles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from mlb.config import load_backtest_config
from mlb.features.build_dataset import FEATURE_COLUMNS
from mlb.models.total.direct import walk_forward_total_regression
from mlb.models.runline.direct import walk_forward_runline_classifier
from mlb.ensemble.stacking import build_ensemble_frame, walk_forward_linear_stacking
from mlb.models.moneyline.baselines import walk_forward_logistic_baseline
from mlb.evaluation.calibration import total_runs_metrics, binary_metrics, expected_calibration_error

cfg = load_backtest_config()
processed = cfg.processed_dir

frames = {}
for season in [2022, 2023, 2024]:
    gf = pd.read_parquet(processed / f"game_features_{season}.parquet")
    gf["game_date"] = pd.to_datetime(gf["game_date"])
    gf["actual_total"] = gf["home_score"] + gf["away_score"]
    gf["actual_margin"] = gf["home_score"] - gf["away_score"]
    gf["actual_home_minus_1_5_cover"] = (gf["actual_margin"] >= 2).astype(int)
    frames[season] = gf
gf24_pks = set(frames[2024]["game_pk"])

sim_2023 = pd.read_parquet(processed / "predictions_2023_both.parquet")
sim_2024 = pd.read_parquet(processed / "predictions_2024_both.parquet")
sim_all = pd.concat([sim_2023, sim_2024], ignore_index=True)
sim_all["actual_home_minus_1_5_cover"] = (sim_all["actual_margin"] >= 2).astype(int)

# ============================= TOTALS =============================
print("=== TOTALS ===")
pool_2023 = pd.concat([frames[2022], frames[2023]], ignore_index=True)
tot_2023 = walk_forward_total_regression(pool_2023, FEATURE_COLUMNS, retrain_freq_days=7, min_training_games=400)
tot_2023 = tot_2023[tot_2023["season"] == 2023].dropna(subset=["pred_total"])

pool_2024 = pd.concat([frames[2023], frames[2024]], ignore_index=True)
tot_2024 = walk_forward_total_regression(pool_2024, FEATURE_COLUMNS, retrain_freq_days=7, min_training_games=400)
tot_2024 = tot_2024[tot_2024["season"] == 2024].dropna(subset=["pred_total"])
tot_all = pd.concat([tot_2023, tot_2024], ignore_index=True)

ens_tot_frame = build_ensemble_frame(
    sim_all, {"sim": (sim_all, "pred_mean_total"), "direct": (tot_all, "pred_total")},
    label_col="actual_total", binary=False,
)
ens_tot = walk_forward_linear_stacking(ens_tot_frame, ["p_sim", "p_direct"], "actual_total", retrain_freq_days=7, min_training_games=200)
ens_tot_2024 = ens_tot[ens_tot["game_pk"].isin(gf24_pks)].dropna(subset=["pred_ensemble"])

common_pks_tot = set(ens_tot_2024["game_pk"])
sim_sub = sim_all[sim_all["game_pk"].isin(common_pks_tot)]
direct_sub = tot_all[tot_all["game_pk"].isin(common_pks_tot)]
naive_pred = sim_sub["actual_total"].mean()  # crude reference only; real naive uses as-of rolling avg (see README)

print("SIM (component):    ", total_runs_metrics(sim_sub["actual_total"], sim_sub["pred_mean_total"]))
print("DIRECT (component):  ", total_runs_metrics(direct_sub["actual_total"], direct_sub["pred_total"]))
print("ENSEMBLE:            ", total_runs_metrics(ens_tot_2024["actual_total"], ens_tot_2024["pred_ensemble"]))

ens_tot_2024.to_parquet(processed / "predictions_2024_ensemble_totals.parquet", index=False)

# ============================= RUN LINE =============================
print("\n=== RUN LINE (home -1.5) ===")
rl_2023 = walk_forward_runline_classifier(pool_2023, FEATURE_COLUMNS, retrain_freq_days=7, min_training_games=400)
rl_2023 = rl_2023[rl_2023["season"] == 2023].dropna(subset=["pred_home_minus_1_5_prob"])
rl_2024 = walk_forward_runline_classifier(pool_2024, FEATURE_COLUMNS, retrain_freq_days=7, min_training_games=400)
rl_2024 = rl_2024[rl_2024["season"] == 2024].dropna(subset=["pred_home_minus_1_5_prob"])
rl_all = pd.concat([rl_2023, rl_2024], ignore_index=True)

ens_rl_frame = build_ensemble_frame(
    sim_all, {"sim": (sim_all, "pred_home_minus_1_5_cover_prob"), "direct": (rl_all, "pred_home_minus_1_5_prob")},
    label_col="actual_home_minus_1_5_cover", binary=True,
)
ens_rl = walk_forward_logistic_baseline(
    ens_rl_frame, ["logit_sim", "logit_direct"], retrain_freq_days=7, min_training_games=200,
    label_col="actual_home_minus_1_5_cover",
)
ens_rl = ens_rl.rename(columns={"pred_prob": "pred_home_minus_1_5_ensemble"})
ens_rl_2024 = ens_rl[ens_rl["game_pk"].isin(gf24_pks)].dropna(subset=["pred_home_minus_1_5_ensemble"])

common_pks_rl = set(ens_rl_2024["game_pk"])
sim_sub_rl = sim_all[sim_all["game_pk"].isin(common_pks_rl)]
direct_sub_rl = rl_all[rl_all["game_pk"].isin(common_pks_rl)]

def rl_metrics(y, p):
    m = binary_metrics(y, p)
    ece = expected_calibration_error(y, p)
    return {**m, "ece": ece}

print("SIM (component):    ", rl_metrics(sim_sub_rl["actual_home_minus_1_5_cover"], sim_sub_rl["pred_home_minus_1_5_cover_prob"]))
print("DIRECT (component):  ", rl_metrics(direct_sub_rl["actual_home_minus_1_5_cover"], direct_sub_rl["pred_home_minus_1_5_prob"]))
print("ENSEMBLE:            ", rl_metrics(ens_rl_2024["actual_home_minus_1_5_cover"], ens_rl_2024["pred_home_minus_1_5_ensemble"]))

ens_rl_2024.to_parquet(processed / "predictions_2024_ensemble_runline.parquet", index=False)
print(f"\nSaved -> predictions_2024_ensemble_totals.parquet, predictions_2024_ensemble_runline.parquet")
