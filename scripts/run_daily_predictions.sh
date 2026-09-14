#!/bin/bash
# Daily live-slate prediction run, meant to run on a schedule (launchd/cron).
# Predicts every game on today's real MLB schedule, writes an immutable,
# timestamped parquet file under data/processed/daily_predictions/, and
# publishes the refreshed docs/data/daily_predictions.json to GitHub so the
# live Pages dashboard's "Today's Live Slate" section updates itself —
# ONLY that one file is ever staged/committed/pushed here, never a broad
# `git add`. Needs no API key/secret — only the public MLB Stats API.
set -euo pipefail

REPO_DIR="/Users/kshreyan12/Documents/mlb"
LOG_DIR="$REPO_DIR/data/processed/daily_predictions"
LOG_FILE="$LOG_DIR/run.log"
SITE_JSON="docs/data/daily_predictions.json"

mkdir -p "$LOG_DIR"
cd "$REPO_DIR"

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) starting daily prediction run ==="
  source .venv/bin/activate
  python3 scripts/predict_today.py

  if [ -n "$(git status --porcelain -- "$SITE_JSON")" ]; then
    git add "$SITE_JSON"
    git commit -m "Update live slate predictions ($(date -u +%Y-%m-%d))"
    git push origin main
    echo "published $SITE_JSON to GitHub"
  else
    echo "$SITE_JSON unchanged, nothing to publish"
  fi
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) finished ==="
} >> "$LOG_FILE" 2>&1
