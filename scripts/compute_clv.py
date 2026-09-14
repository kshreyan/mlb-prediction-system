"""Compute honest CLV (model probability vs. real closing-line market
probability) for moneyline, run line, and totals, on the sample of 2024
games with real pulled odds (scripts/pull_historical_odds.py).

Moneyline: our final ensemble's win probability vs. the de-vigged market
consensus.
Run line: our run-line ensemble vs. the market — ONLY on games where the
market posted home at -1.5 (home favored on the run line). Games where the
market posted home at +1.5 (away favored) are excluded from this specific
comparison: our saved run-line output is defined as P(home covers -1.5) /
P(away covers +1.5), which isn't the complement needed for the
away-favored framing without re-deriving from the full margin
distribution — excluded and reported as such rather than silently
mismatched.
Totals: our simulation is RE-RUN at the market's own posted total line
(using the saved mu_home/mu_away/dispersion_k from the walk-forward
backtest — the same simulation, just queried at a different line) to get
a genuinely comparable P(over) at the identical number the market used.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from mlb.config import load_backtest_config
from mlb.simulation.engine import simulate_game, over_under_prob

cfg = load_backtest_config()
processed = cfg.processed_dir
raw_dir = cfg.raw_dir

odds = pd.read_parquet(raw_dir / "odds" / "2024_sample.parquet")
sim = pd.read_parquet(processed / "predictions_2024_both.parquet")
ml_ens = pd.read_parquet(processed / "predictions_2024_ensemble.parquet")[["game_pk", "pred_prob"]].rename(columns={"pred_prob": "ml_ensemble_prob"})
rl_ens = pd.read_parquet(processed / "predictions_2024_ensemble_runline.parquet")[["game_pk", "pred_home_minus_1_5_ensemble"]]

merged = odds.merge(sim, on="game_pk", how="inner").merge(ml_ens, on="game_pk", how="left").merge(rl_ens, on="game_pk", how="left")
print(f"Matched {len(merged)}/{len(odds)} odds rows to backtest predictions (some odds-sample games may be outside the walk-forward-scored range)")


def report(name, y_true, model_p, market_p):
    model_p = np.clip(model_p, 1e-6, 1 - 1e-6)
    market_p = np.clip(market_p, 1e-6, 1 - 1e-6)
    n = len(y_true)
    print(f"\n{name} (n={n}):")
    print(f"  MODEL : brier={brier_score_loss(y_true, model_p):.4f} logloss={log_loss(y_true, model_p, labels=[0,1]):.4f}")
    print(f"  MARKET: brier={brier_score_loss(y_true, market_p):.4f} logloss={log_loss(y_true, market_p, labels=[0,1]):.4f}")
    print(f"  mean(model_p - market_p) = {np.mean(model_p - market_p):+.4f}")


# --- MONEYLINE ---
ml = merged.dropna(subset=["ml_ensemble_prob", "ml_home_implied_prob"])
report("MONEYLINE (ensemble vs. de-vigged market)", ml["actual_home_win"], ml["ml_ensemble_prob"], ml["ml_home_implied_prob"])

# --- RUN LINE (home -1.5 only) ---
rl = merged.dropna(subset=["pred_home_minus_1_5_ensemble", "run_line_point", "run_line_home_cover_prob"])
rl_home_favored = rl[rl["run_line_point"] == -1.5]
print(f"\nRun-line sample: {len(rl)} total, {len(rl_home_favored)} with home favored (-1.5) — "
      f"{len(rl) - len(rl_home_favored)} excluded (home +1.5, different event definition, not silently mismatched)")
if len(rl_home_favored) > 5:
    actual_home_cover = (rl_home_favored["actual_margin"] >= 2).astype(int)
    report("RUN LINE, home -1.5 subset (ensemble vs. market)", actual_home_cover,
           rl_home_favored["pred_home_minus_1_5_ensemble"], rl_home_favored["run_line_home_cover_prob"])

# --- TOTALS (re-simulate at the market's own line) ---
tot = merged.dropna(subset=["total_point", "total_over_prob"])
our_over_probs = []
for _, row in tot.iterrows():
    sim_result = simulate_game(row["mu_home"], row["mu_away"], row["dispersion_k"], n_sims=20000, seed=int(row["game_pk"]))
    over_p, _ = over_under_prob(sim_result.total_distribution, row["total_point"])
    our_over_probs.append(over_p)
tot = tot.copy()
tot["our_over_prob"] = our_over_probs
actual_over = (tot["actual_total"] > tot["total_point"]).astype(int)
report("TOTALS, re-simulated at market's own line (model vs. market)", actual_over, tot["our_over_prob"], tot["total_over_prob"])

print("\n--- Sample scope note ---")
print(f"This covers a systematic sample of {len(odds)} games (~{len(odds)/2429*100:.0f}% of the 2024 season), "
      "not full-season coverage - see docs/limitations.md for the budget reasoning. "
      "Treat these as real but sample-sized estimates, not a full-season CLV claim.")
