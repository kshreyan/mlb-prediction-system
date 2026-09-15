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

import pandas as pd

from mlb.config import load_backtest_config
from mlb.daily.predict import build_todays_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

SEASONS = [2019, 2021, 2022, 2023, 2024, 2025, 2026]
DOCS_DATA_PATH = Path(__file__).resolve().parents[1] / "docs" / "data" / "daily_predictions.json"


def _r(v, ndigits=4):
    """Round a possibly-missing (NaN/None) value, keeping it JSON-null rather
    than fabricating a number when the underlying market data wasn't found."""
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if fv != fv:  # NaN
        return None
    return round(fv, ndigits)


def _fmt_point(p: float) -> str:
    return f"{p:+g}"


def _top_edges(out, max_n: int = 8) -> list[dict]:
    """Ranks every available market/side across the whole slate by how much
    the model's probability diverges from the real de-vigged market
    probability, and returns the top `max_n`. This is a MODEL-VS-MARKET
    DIVERGENCE ranking, not a claim of positive expected value: this
    project's own CLV backtest (see README/limitations) found the real
    market beats this model on moneyline and run line in aggregate, and
    ties it on totals — a bigger divergence from a historically sharper
    market is not obviously more trustworthy than a smaller one. Every
    entry here is a real, computed number from a real live market quote;
    nothing here is fabricated, but nothing here is betting advice either.
    """
    candidates = []
    for _, r in out.iterrows():
        game = f"{r['away_team']} @ {r['home_team']}"

        if pd.notna(r.get("ml_edge_home")):
            home_favored = r["ml_edge_home"] >= r["ml_edge_away"]
            candidates.append({
                "game": game, "market": "Moneyline",
                "pick": f"{r['home_team']} ML" if home_favored else f"{r['away_team']} ML",
                "model_prob": r["ensemble_home_win_prob"] if home_favored else 1 - r["ensemble_home_win_prob"],
                "market_prob": r["ml_market_home_prob"] if home_favored else r["ml_market_away_prob"],
                "edge_pp": abs(r["ml_edge_home"]), "n_bookmakers": r.get("n_bookmakers"),
            })

        if pd.notna(r.get("run_line_edge_home")):
            point = r["run_line_market_point"]
            home_side = r["run_line_edge_home"] >= 0
            candidates.append({
                "game": game, "market": "Run line",
                "pick": f"{r['home_team']} {_fmt_point(point)}" if home_side else f"{r['away_team']} {_fmt_point(-point)}",
                "model_prob": r["run_line_model_home_prob"] if home_side else 1 - r["run_line_model_home_prob"],
                "market_prob": r["run_line_market_home_prob"] if home_side else 1 - r["run_line_market_home_prob"],
                "edge_pp": abs(r["run_line_edge_home"]), "n_bookmakers": r.get("n_bookmakers"),
            })

        if pd.notna(r.get("total_edge_over")):
            line = r["total_market_point"]
            over_side = r["total_edge_over"] >= 0
            candidates.append({
                "game": game, "market": "Total",
                "pick": f"Over {line:g}" if over_side else f"Under {line:g}",
                "model_prob": r["total_model_over_prob"] if over_side else 1 - r["total_model_over_prob"],
                "market_prob": r["total_market_over_prob"] if over_side else 1 - r["total_market_over_prob"],
                "edge_pp": abs(r["total_edge_over"]), "n_bookmakers": r.get("n_bookmakers"),
            })

    candidates.sort(key=lambda c: c["edge_pp"], reverse=True)
    return [{
        "game": c["game"], "market": c["market"], "pick": c["pick"],
        "model_prob": _r(c["model_prob"]), "market_prob": _r(c["market_prob"]),
        "edge_pp": _r(c["edge_pp"]), "n_bookmakers": c["n_bookmakers"],
    } for c in candidates[:max_n]]


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
            "n_bookmakers": r.get("n_bookmakers"),
            "moneyline": {
                "model_home_prob": _r(r["ensemble_home_win_prob"]),
                "market_home_prob": _r(r.get("ml_market_home_prob")),
                "market_away_prob": _r(r.get("ml_market_away_prob")),
            },
            "run_line": {
                "market_point": _r(r.get("run_line_market_point"), 1),
                "model_home_prob": _r(r.get("run_line_model_home_prob")),
                "market_home_prob": _r(r.get("run_line_market_home_prob")),
            },
            "total": {
                "model_mean": _r(r["pred_mean_total"], 2),
                "market_point": _r(r.get("total_market_point"), 1),
                "model_over_prob": _r(r.get("total_model_over_prob")),
                "market_over_prob": _r(r.get("total_market_over_prob")),
            },
        })
    payload = {
        "date": today.isoformat(), "generated_at": generated_at,
        "games": games, "top_edges": _top_edges(out),
    }
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

    edges = _top_edges(out)
    if edges:
        print("\ntop model-vs-market edges (not betting advice — see docs/limitations.md):")
        for e in edges:
            print(f"  {e['edge_pp']*100:5.1f}pp  {e['market']:<10} {e['pick']:<28} model={e['model_prob']*100:.1f}% mkt={e['market_prob']*100:.1f}%  ({e['game']}, {e['n_bookmakers']} books)")
    else:
        print("\nno live market data available today — edges not computed.")


if __name__ == "__main__":
    main()
