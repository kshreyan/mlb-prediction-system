"""Generate walk-forward GBM baseline predictions for 2023 (using 2022 as
warm-up) and 2024 (using 2023 as warm-up), matching the pattern used for
pitcher-adjusted Elo — so the ensemble has an honest out-of-sample GBM
component for both seasons.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mlb.config import load_backtest_config
from mlb.features.build_dataset import FEATURE_COLUMNS
from mlb.models.moneyline.gbm import walk_forward_gbm_baseline
from mlb.evaluation.calibration import binary_metrics, expected_calibration_error

cfg = load_backtest_config()
processed = cfg.processed_dir

frames = {}
for season in [2022, 2023, 2024]:
    gf = pd.read_parquet(processed / f"game_features_{season}.parquet")
    gf["game_date"] = pd.to_datetime(gf["game_date"])
    gf["actual_home_win"] = (gf["home_score"] > gf["away_score"]).astype(int)
    frames[season] = gf

pool_2023 = pd.concat([frames[2022], frames[2023]], ignore_index=True)
gbm_2023 = walk_forward_gbm_baseline(pool_2023, FEATURE_COLUMNS, retrain_freq_days=7, min_training_games=400)
gbm_2023 = gbm_2023[gbm_2023["season"] == 2023].dropna(subset=["pred_prob"])

pool_2024 = pd.concat([frames[2023], frames[2024]], ignore_index=True)
gbm_2024 = walk_forward_gbm_baseline(pool_2024, FEATURE_COLUMNS, retrain_freq_days=7, min_training_games=400)
gbm_2024 = gbm_2024[gbm_2024["season"] == 2024].dropna(subset=["pred_prob"])

print(f"GBM 2023 predictions: {len(gbm_2023)}, 2024 predictions: {len(gbm_2024)}")
m23 = binary_metrics(gbm_2023["actual_home_win"], gbm_2023["pred_prob"])
m24 = binary_metrics(gbm_2024["actual_home_win"], gbm_2024["pred_prob"])
print("GBM 2023 (val):", m23)
print("GBM 2024 (holdout):", m24, "ECE:", expected_calibration_error(gbm_2024["actual_home_win"], gbm_2024["pred_prob"]))

gbm_2023[["game_pk", "game_date", "pred_prob", "actual_home_win"]].to_parquet(processed / "gbm_2023.parquet", index=False)
gbm_2024[["game_pk", "game_date", "pred_prob", "actual_home_win"]].to_parquet(processed / "gbm_2024.parquet", index=False)
print(f"Saved -> {processed / 'gbm_2023.parquet'}, {processed / 'gbm_2024.parquet'}")
