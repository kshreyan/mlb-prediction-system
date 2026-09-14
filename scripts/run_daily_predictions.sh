#!/bin/bash
# Daily live-slate prediction run, meant to run on a schedule (launchd/cron).
# Predicts every game on today's real MLB schedule and writes an immutable,
# timestamped parquet file under data/processed/daily_predictions/.
# Needs no API key/secret — only the public MLB Stats API.
set -euo pipefail

REPO_DIR="/Users/kshreyan12/Documents/mlb"
LOG_DIR="$REPO_DIR/data/processed/daily_predictions"
LOG_FILE="$LOG_DIR/run.log"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) starting daily prediction run ==="
  source .venv/bin/activate
  python3 scripts/predict_today.py
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) finished ==="
} >> "$LOG_FILE" 2>&1
