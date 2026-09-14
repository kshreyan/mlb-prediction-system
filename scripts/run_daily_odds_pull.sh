#!/bin/bash
# Daily incremental CLV-odds pull, meant to run on a schedule (launchd/cron).
# Sources .env locally (never leaves this machine), pulls up to 15 new
# games' real closing-line odds, and appends to data/raw/odds/2024_sample.parquet.
# Stops gracefully with a safety margin before the API key's credits run out.
set -euo pipefail

REPO_DIR="/Users/kshreyan12/Documents/mlb"
LOG_DIR="$REPO_DIR/data/raw/odds"
LOG_FILE="$LOG_DIR/daily_pull.log"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) starting daily odds pull ==="
  source .env
  export ODDS_API_KEY
  source .venv/bin/activate
  python3 scripts/pull_historical_odds.py --daily-batch 15
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) finished ==="
} >> "$LOG_FILE" 2>&1
