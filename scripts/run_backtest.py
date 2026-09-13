"""Run the walk-forward moneyline backtest for `season`, using `prior_seasons`
(fully completed) as the initial training pool. Saves immutable predictions
(with actual results attached for grading) to data/processed/.

Usage: python scripts/run_backtest.py 2024 --prior 2023
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mlb.backtest.walk_forward import BacktestConfig, walk_forward_backtest
from mlb.config import load_backtest_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=int)
    parser.add_argument("--prior", type=int, nargs="*", default=[])
    parser.add_argument("--retrain-freq-days", type=int, default=7)
    args = parser.parse_args()

    cfg = load_backtest_config()
    processed = cfg.processed_dir

    gf_season = pd.read_parquet(processed / f"game_features_{args.season}.parquet")
    gf_season["game_date"] = pd.to_datetime(gf_season["game_date"])

    prior_frames = []
    for p in args.prior:
        prior_df = pd.read_parquet(processed / f"game_features_{p}.parquet")
        prior_df["game_date"] = pd.to_datetime(prior_df["game_date"])
        prior_frames.append(prior_df)
    prior_history = pd.concat(prior_frames, ignore_index=True) if prior_frames else None

    bt_cfg = BacktestConfig(
        retrain_freq_days=args.retrain_freq_days,
        n_sims=cfg.simulation.n_sims,
        seed=cfg.random_seed,
    )
    preds = walk_forward_backtest(gf_season, prior_history, bt_cfg)
    out_path = processed / f"predictions_{args.season}.parquet"
    preds.to_parquet(out_path, index=False)
    print(f"season={args.season} n_predictions={len(preds)} -> {out_path}")
