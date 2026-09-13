"""Fit each team-side's expected-runs mean from the matchup features, and
estimate the negative-binomial overdispersion of actual team runs-per-game
around that mean — both fit ONLY on data passed in (the caller is
responsible for passing only strictly-prior games during walk-forward
backtesting; see mlb.backtest.walk_forward).

Runs are treated as overdispersed count data (NB), not plain Poisson,
because real team runs-per-game variance exceeds its mean.

`feature_set` selects which offense signal feeds the mean model, so the
team-level proxy and the lineup-level (confirmed-lineup, platoon-aware)
signal can be honestly ablated against each other rather than just swapped:
  - "team_offense": the original team-level rolling-runs proxy only.
  - "lineup": the lineup-level projected xwOBA vs. the opposing starter's
    hand, in place of the team proxy.
  - "both": both signals included; lets the regression itself weigh them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = ["opp_starter_xwoba", "opp_bullpen_xwoba", "park_factor", "is_home"]

FEATURE_SETS = {
    "team_offense": ["own_offense_proj"] + BASE_FEATURES,
    "lineup": ["own_lineup_xwoba"] + BASE_FEATURES,
    "both": ["own_offense_proj", "own_lineup_xwoba"] + BASE_FEATURES,
}


def build_long_training_frame(game_features: pd.DataFrame, feature_set: str = "team_offense") -> pd.DataFrame:
    """One row per team-per-game-side, target = that side's actual runs."""
    features = FEATURE_SETS[feature_set]

    home = pd.DataFrame({
        "game_pk": game_features["game_pk"],
        "team": game_features["home_team"],
        "is_home": 1.0,
        "actual_runs": game_features["home_score"],
        "own_offense_proj": game_features.get("home_off_proj_runs_scored_per_game"),
        "own_lineup_xwoba": game_features.get("home_lineup_proj_xwoba"),
        "opp_starter_xwoba": game_features["away_starter_proj_xwoba_against"],
        "opp_bullpen_xwoba": game_features["away_bullpen_proj_bullpen_xwoba_against"],
        "park_factor": game_features["park_factor"],
    })
    away = pd.DataFrame({
        "game_pk": game_features["game_pk"],
        "team": game_features["away_team"],
        "is_home": 0.0,
        "actual_runs": game_features["away_score"],
        "own_offense_proj": game_features.get("away_off_proj_runs_scored_per_game"),
        "own_lineup_xwoba": game_features.get("away_lineup_proj_xwoba"),
        "opp_starter_xwoba": game_features["home_starter_proj_xwoba_against"],
        "opp_bullpen_xwoba": game_features["home_bullpen_proj_bullpen_xwoba_against"],
        "park_factor": game_features["park_factor"],
    })
    long_df = pd.concat([home, away], ignore_index=True)
    long_df.attrs["feature_set"] = feature_set
    return long_df.dropna(subset=features + ["actual_runs"])


@dataclass
class RunEnvironmentModel:
    pipeline: Pipeline
    dispersion_k: float  # negative-binomial size parameter (higher = closer to Poisson)
    feature_set: str

    def predict_mu(self, X: pd.DataFrame) -> np.ndarray:
        mu = self.pipeline.predict(X[FEATURE_SETS[self.feature_set]])
        return np.clip(mu, 0.2, None)  # avoid degenerate near-zero means


def fit_run_environment(long_df: pd.DataFrame, feature_set: str = "team_offense") -> RunEnvironmentModel:
    features = FEATURE_SETS[feature_set]
    X = long_df[features]
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

    return RunEnvironmentModel(pipeline=pipeline, dispersion_k=k, feature_set=feature_set)
