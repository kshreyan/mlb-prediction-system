"""Pull real historical MLB closing-line odds, incrementally.

Two modes:
  - One-shot systematic sample: `python scripts/pull_historical_odds.py 500`
    picks an evenly-spread sample of 500 games across the season.
  - Daily incremental batch (for a scheduled routine):
    `python scripts/pull_historical_odds.py --daily-batch 15`
    walks the full season IN DATE ORDER, skips games already pulled, and
    fetches up to 15 new games this run — safe to run on a cron/launchd
    schedule day after day, gradually covering more of the season without
    ever exceeding the API key's remaining credit budget (stops with a
    safety margin, never spends the account down to zero unexpectedly).

Usage: source .env && python scripts/pull_historical_odds.py [n_games | --daily-batch N]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mlb.config import load_backtest_config
from mlb.data.odds import fetch_historical_snapshot, extract_game_closing_odds, remaining_credits

cfg = load_backtest_config()
raw_dir = cfg.raw_dir

SEASON = 2024
CREDITS_PER_GAME_ESTIMATE = 38  # observed ~33-38; use the conservative end
SAFETY_MARGIN_CREDITS = 100     # stop this many credits before empty, never run a key to exactly zero

parser = argparse.ArgumentParser()
parser.add_argument("n_games", nargs="?", type=int, default=None, help="One-shot: evenly-spread sample size")
parser.add_argument("--daily-batch", type=int, default=None, help="Incremental: new games to add this run, in date order")
args = parser.parse_args()

sched = pd.read_parquet(raw_dir / "schedule" / f"{SEASON}.parquet")
sched = sched[sched["is_final"] & sched["game_datetime"].notna()].copy()
sched["game_datetime"] = pd.to_datetime(sched["game_datetime"])
sched = sched.sort_values("game_datetime").reset_index(drop=True)

out_path = raw_dir / "odds" / f"{SEASON}_sample.parquet"
out_path.parent.mkdir(parents=True, exist_ok=True)
existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["game_pk"])
already_have = set(existing["game_pk"].tolist())

if args.daily_batch is not None:
    # Full season, in date order, skipping what we already have — a
    # predictable, resumable walk across the whole season over many runs.
    todo = sched[~sched["game_pk"].isin(already_have)].copy()
    sample = todo.head(args.daily_batch)
    print(f"Daily batch mode: {len(already_have)}/{len(sched)} games already covered "
          f"({len(already_have)/len(sched)*100:.1f}%). Attempting {len(sample)} more.")
    if sample.empty:
        print("Full season already covered — nothing to do.")
        sys.exit(0)
else:
    n_games = args.n_games or 550
    step = max(1, len(sched) // n_games)
    sample = sched.iloc[::step].head(n_games).copy()
    print(f"One-shot sample mode: season has {len(sched)} games; sampling every {step}th -> {len(sample)} games")

rows = []
n_found, n_missing, n_skipped = 0, 0, 0
for i, (_, g) in enumerate(sample.iterrows()):
    if g["game_pk"] in already_have:
        n_skipped += 1
        continue

    credits_left = remaining_credits()
    if credits_left is not None and credits_left < SAFETY_MARGIN_CREDITS:
        print(f"STOPPING: only {credits_left} credits left (safety margin {SAFETY_MARGIN_CREDITS}) — "
              "saving progress and exiting cleanly rather than risking an exhausted/errored key.")
        break

    commence = g["game_datetime"].to_pydatetime().replace(tzinfo=dt.timezone.utc)
    snapshot_time = commence - dt.timedelta(minutes=8)
    try:
        snapshot = fetch_historical_snapshot(snapshot_time)
    except RuntimeError as exc:
        print(f"STOPPING: {exc}")
        break
    result = extract_game_closing_odds(snapshot, g["home_team"], g["away_team"], commence)
    if result is None:
        n_missing += 1
    else:
        result["game_pk"] = g["game_pk"]
        rows.append(result)
        n_found += 1
    if (i + 1) % 50 == 0:
        print(f"  progress: {i+1}/{len(sample)} attempted, {n_found} found, {n_missing} missing "
              f"(credits remaining: {remaining_credits()})")
    time.sleep(0.4)

new_df = pd.DataFrame(rows)
combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(subset="game_pk", keep="last") if len(new_df) else existing
combined.to_parquet(out_path, index=False)
print(f"\nDone: found={n_found} missing={n_missing} already_had={n_skipped} "
      f"credits_remaining={remaining_credits()}")
print(f"Total coverage: {len(combined)}/{len(sched)} games ({len(combined)/len(sched)*100:.1f}%) -> {out_path}")
