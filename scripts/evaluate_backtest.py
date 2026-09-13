"""Produce the honest evaluation report for one season's walk-forward
backtest: moneyline calibration (Brier/log loss/ECE/reliability), run-line
calibration, total-runs error, and baseline comparisons (Elo-only,
pitcher-adjusted Elo, home-field-always, better-record/Log5) — all
restricted to the same set of games so comparisons are apples-to-apples.

Usage: python scripts/evaluate_backtest.py 2024 --prior 2023
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from mlb.config import load_backtest_config
from mlb.evaluation.calibration import binary_metrics, expected_calibration_error, reliability_table, total_runs_metrics
from mlb.models.moneyline.elo import compute_elo_ratings
from mlb.models.moneyline.baselines import home_field_baseline, better_record_baseline, walk_forward_logistic_baseline


def main(season: int, prior_seasons: list[int]) -> dict:
    cfg = load_backtest_config()
    processed = cfg.processed_dir

    preds = pd.read_parquet(processed / f"predictions_{season}.parquet")
    preds["game_date"] = pd.to_datetime(preds["game_date"])

    frames = []
    for s in prior_seasons + [season]:
        gf = pd.read_parquet(processed / f"game_features_{s}.parquet")
        gf["game_date"] = pd.to_datetime(gf["game_date"])
        gf["actual_home_win"] = (gf["home_score"] > gf["away_score"]).astype(int)
        frames.append(gf)
    all_seasons = pd.concat(frames, ignore_index=True).sort_values("game_date").reset_index(drop=True)

    elo_df = compute_elo_ratings(all_seasons, k=20.0, home_advantage=24.0, season_regress_frac=0.33)
    elo_df["elo_diff"] = elo_df["elo_home_pre"] + 24.0 - elo_df["elo_away_pre"]
    merged = all_seasons.merge(elo_df[["game_pk", "elo_diff", "elo_pred_home_win_prob"]], on="game_pk", how="left")
    merged["pred_hf"] = home_field_baseline(all_seasons)
    merged["pred_br"] = better_record_baseline(all_seasons)

    season_merged = merged[merged["season"] == season].copy()
    wf = walk_forward_logistic_baseline(
        season_merged, ["elo_diff", "diff_starter_xwoba", "diff_bullpen_xwoba"], retrain_freq_days=7,
    )
    valid_pks = set(wf.dropna(subset=["pred_prob"])["game_pk"])

    def subset(df, pk_col="game_pk"):
        return df[df[pk_col].isin(valid_pks)]

    report = {"season": season, "n_common_games": len(valid_pks)}

    sim_sub = subset(preds)
    report["simulation_model"] = {
        **binary_metrics(sim_sub["actual_home_win"], sim_sub["pred_home_win_prob"]),
        "ece": expected_calibration_error(sim_sub["actual_home_win"], sim_sub["pred_home_win_prob"]),
    }

    elo_sub = subset(merged[merged["season"] == season])
    report["elo_only_baseline"] = {
        **binary_metrics(elo_sub["actual_home_win"], elo_sub["elo_pred_home_win_prob"]),
        "ece": expected_calibration_error(elo_sub["actual_home_win"], elo_sub["elo_pred_home_win_prob"]),
    }
    report["home_field_always_baseline"] = {
        **binary_metrics(elo_sub["actual_home_win"], elo_sub["pred_hf"]),
        "ece": expected_calibration_error(elo_sub["actual_home_win"], elo_sub["pred_hf"]),
    }
    report["better_record_log5_baseline"] = {
        **binary_metrics(elo_sub["actual_home_win"], elo_sub["pred_br"]),
        "ece": expected_calibration_error(elo_sub["actual_home_win"], elo_sub["pred_br"]),
    }
    pae_sub = wf.dropna(subset=["pred_prob"])
    report["pitcher_adjusted_elo_baseline"] = {
        **binary_metrics(pae_sub["actual_home_win"], pae_sub["pred_prob"]),
        "ece": expected_calibration_error(pae_sub["actual_home_win"], pae_sub["pred_prob"]),
    }

    report["totals"] = total_runs_metrics(preds["actual_total"], preds["pred_mean_total"])
    report["run_line"] = {
        "actual_home_minus_1_5_cover_rate": float((preds["actual_margin"] >= 2).mean()),
        "mean_predicted_home_minus_1_5_cover_prob": float(preds["pred_home_minus_1_5_cover_prob"].mean()),
        "actual_prob_one_run_game": float((preds["actual_margin"].abs() == 1).mean()),
        "mean_predicted_prob_one_run_game": float(preds["pred_prob_one_run_game"].mean()),
    }

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("season", type=int)
    parser.add_argument("--prior", type=int, nargs="*", default=[])
    args = parser.parse_args()

    report = main(args.season, args.prior)
    print(json.dumps(report, indent=2))

    cfg = load_backtest_config()
    out_path = cfg.processed_dir / f"evaluation_report_{args.season}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {out_path}")
