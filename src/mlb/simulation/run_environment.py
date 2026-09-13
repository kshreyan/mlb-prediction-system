"""Fit each team-side's expected-runs mean from the matchup features, and
estimate the negative-binomial overdispersion of actual team runs-per-game
around that mean — both fit ONLY on data passed in (the caller is
responsible for passing only strictly-prior games during walk-forward
backtesting; see mlb.backtest.walk_forward).

Runs are treated as overdispersed count data (NB), not plain Poisson,
because real team runs-per-game variance exceeds its mean.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MEAN_MODEL_FEATURES = [
    "own_offense_proj", "opp_starter_xwoba", "opp_bullpen_xwoba", "park_factor", "is_home",
]


def build_long_training_frame(game_features: pd.DataFrame) -> pd.DataFrame:
    """One row per team-per-game-side, target = that side's actual runs."""
    home = pd.DataFrame({
        "game_pk": game_features["game_pk"],
        "team": game_features["home_team"],
        "is_home": 1.0,
        "actual_runs": game_features["home_score"],
        "own_offense_proj": game_features["home_off_proj_runs_scored_per_game"],
        "opp_starter_xwoba": game_features["away_starter_proj_xwoba_against"],
        "opp_bullpen_xwoba": game_features["away_bullpen_proj_bullpen_xwoba_against"],
        "park_factor": game_features["park_factor"],
    })
    away = pd.DataFrame({
        "game_pk": game_features["game_pk"],
        "team": game_features["away_team"],
        "is_home": 0.0,
        "actual_runs": game_features["away_score"],
        "own_offense_proj": game_features["away_off_proj_runs_scored_per_game"],
        "opp_starter_xwoba": game_features["home_starter_proj_xwoba_against"],
        "opp_bullpen_xwoba": game_features["home_bullpen_proj_bullpen_xwoba_against"],
        "park_factor": game_features["park_factor"],
    })
    long_df = pd.concat([home, away], ignore_index=True)
    return long_df.dropna(subset=MEAN_MODEL_FEATURES + ["actual_runs"])


@dataclass
class RunEnvironmentModel:
    pipeline: Pipeline
    dispersion_k: float  # negative-binomial size parameter (higher = closer to Poisson)

    def predict_mu(self, X: pd.DataFrame) -> np.ndarray:
        mu = self.pipeline.predict(X[MEAN_MODEL_FEATURES])
        return np.clip(mu, 0.2, None)  # avoid degenerate near-zero means


def fit_run_environment(long_df: pd.DataFrame) -> RunEnvironmentModel:
    X = long_df[MEAN_MODEL_FEATURES]
    y = long_df["actual_runs"].to_numpy(dtype=float)

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=1.0, max_iter=500)),
    ])
    pipeline.fit(X, y)
    mu = np.clip(pipeline.predict(X), 0.2, None)

    # Pearson method-of-moments estimator for NB dispersion (Var = mu + mu^2/k):
    # 1/k = mean[(y - mu)^2 - mu] / mean[mu^2]
    resid_term = (y - mu) ** 2 - mu
    inv_k = np.mean(resid_term) / np.mean(mu ** 2)
    k = 1.0 / inv_k if inv_k > 1e-6 else 1e6  # near-Poisson fallback if no overdispersion detected
    k = float(np.clip(k, 1.0, 1e6))

    return RunEnvironmentModel(pipeline=pipeline, dispersion_k=k)
