"""Pull real historical MLB closing-line odds for a systematic SAMPLE of
the 2024 season (budget-limited — see mlb.data.odds module docstring for
why this isn't full-season coverage) and save them for CLV evaluation.

Usage: source .env && python scripts/pull_historical_odds.py [n_games]
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from mlb.config import load_backtest_config
from mlb.data.odds import fetch_historical_snapshot, extract_game_closing_odds

cfg = load_backtest_config()
raw_dir = cfg.raw_dir

SEASON = 2024
N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 550

sched = pd.read_parquet(raw_dir / "schedule" / f"{SEASON}.parquet")
sched = sched[sched["is_final"] & sched["game_datetime"].notna()].copy()
sched["game_datetime"] = pd.to_datetime(sched["game_datetime"])
sched = sched.sort_values("game_datetime").reset_index(drop=True)

# Systematic sample spread evenly across the season (not clustered in one
# stretch), for representative temporal coverage within budget.
step = max(1, len(sched) // N_GAMES)
sample = sched.iloc[::step].head(N_GAMES).copy()
print(f"Season has {len(sched)} games; sampling every {step}th -> {len(sample)} games")

out_path = raw_dir / "odds" / f"{SEASON}_sample.parquet"
out_path.parent.mkdir(parents=True, exist_ok=True)
existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["game_pk"])
already_have = set(existing["game_pk"].tolist())

rows = []
n_found, n_missing, n_skipped = 0, 0, 0
for i, (_, g) in enumerate(sample.iterrows()):
    if g["game_pk"] in already_have:
        n_skipped += 1
        continue
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
        print(f"  progress: {i+1}/{len(sample)} sampled, {n_found} found, {n_missing} missing")
    time.sleep(0.4)

new_df = pd.DataFrame(rows)
combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(subset="game_pk", keep="last") if len(new_df) else existing
combined.to_parquet(out_path, index=False)
print(f"\nDone: found={n_found} missing={n_missing} already_had={n_skipped}")
print(f"Saved {len(combined)} total odds rows -> {out_path}")
