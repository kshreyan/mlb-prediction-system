"""Predict every game on today's real MLB slate.

Fits FINAL production versions of every model (run-environment simulation,
Elo, pitcher-adjusted-Elo, the moneyline stacking ensemble) on ALL available
historical data through yesterday, pulls today's actual schedule/probable
pitchers/lineups from the MLB Stats API, and writes one immutable,
timestamped prediction file per run — matching the project's "predictions
are immutable" rule (a later run for the same day, e.g. after lineups post,
writes a NEW file rather than overwriting the earlier one).

Also refreshes docs/data/daily_predictions.json — the small, human-readable
snapshot the GitHub Pages dashboard reads to show today's slate. Unlike the
immutable parquet history, this file IS overwritten each run (it's a "latest
state" view, not a record), and is the one output of this script meant to be
committed to git.

Usage: python scripts/predict_today.py [YYYY-MM-DD]
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mlb.config import load_backtest_config
from mlb.daily.predict import build_todays_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

SEASONS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
DOCS_DATA_PATH = Path(__file__).resolve().parents[1] / "docs" / "data" / "daily_predictions.json"


def _write_site_json(out, today: dt.date, generated_at: str) -> None:
    games = []
    for _, r in out.iterrows():
        games.append({
            "away_team": r["away_team"], "home_team": r["home_team"],
            "away_probable_pitcher": r["away_probable_pitcher"],
            "home_probable_pitcher": r["home_probable_pitcher"],
            "home_lineup_confirmed": bool(r["home_lineup_confirmed"]),
            "away_lineup_confirmed": bool(r["away_lineup_confirmed"]),
            "weather_available": bool(r["weather_available"]),
            "ensemble_home_win_prob": round(float(r["ensemble_home_win_prob"]), 4),
            "pred_mean_total": round(float(r["pred_mean_total"]), 2),
            "sim_home_minus_1_5_cover_prob": round(float(r["sim_home_minus_1_5_cover_prob"]), 4),
        })
    payload = {"date": today.isoformat(), "generated_at": generated_at, "games": games}
    DOCS_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCS_DATA_PATH.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    today = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date.today()
    cfg = load_backtest_config()

    out = build_todays_predictions(cfg, SEASONS, today=today)
    if out.empty:
        print(f"No predictions generated for {today} (no games, or no games with enough info yet).")
        return

    out_dir = cfg.processed_dir / "daily_predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = dt.datetime.now(dt.timezone.utc).isoformat()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%H%M%SZ")
    out_path = out_dir / f"{today.isoformat()}_{stamp}.parquet"
    out.to_parquet(out_path, index=False)
    _write_site_json(out, today, generated_at)

    cols = ["away_team", "home_team", "away_probable_pitcher", "home_probable_pitcher",
            "home_lineup_confirmed", "away_lineup_confirmed", "ensemble_home_win_prob",
            "pred_mean_total", "sim_home_minus_1_5_cover_prob"]
    print(out[cols].to_string(index=False))
    print(f"\n{len(out)} game(s) -> {out_path}")
    print(f"site data -> {DOCS_DATA_PATH}")


if __name__ == "__main__":
    main()
