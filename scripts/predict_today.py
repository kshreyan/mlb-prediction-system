"""Predict every game on today's real MLB slate.

Fits FINAL production versions of every model (run-environment simulation,
Elo, pitcher-adjusted-Elo, the moneyline stacking ensemble) on ALL available
historical data through yesterday, pulls today's actual schedule/probable
pitchers/lineups from the MLB Stats API, and writes one immutable,
timestamped prediction file per run — matching the project's "predictions
are immutable" rule (a later run for the same day, e.g. after lineups post,
writes a NEW file rather than overwriting the earlier one).

Usage: python scripts/predict_today.py [YYYY-MM-DD]
"""
from __future__ import annotations

import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb.config import load_backtest_config
from mlb.daily.predict import build_todays_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

SEASONS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]


def main() -> None:
    today = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date.today()
    cfg = load_backtest_config()

    out = build_todays_predictions(cfg, SEASONS, today=today)
    if out.empty:
        print(f"No predictions generated for {today} (no games, or no games with enough info yet).")
        return

    out_dir = cfg.processed_dir / "daily_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%H%M%SZ")
    out_path = out_dir / f"{today.isoformat()}_{stamp}.parquet"
    out.to_parquet(out_path, index=False)

    cols = ["away_team", "home_team", "away_probable_pitcher", "home_probable_pitcher",
            "home_lineup_confirmed", "away_lineup_confirmed", "ensemble_home_win_prob",
            "pred_mean_total", "sim_home_minus_1_5_cover_prob"]
    print(out[cols].to_string(index=False))
    print(f"\n{len(out)} game(s) -> {out_path}")


if __name__ == "__main__":
    main()
