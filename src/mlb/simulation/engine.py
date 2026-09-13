"""Monte Carlo game simulator: samples each team's runs from a
Gamma-Poisson mixture (== negative binomial with real-valued dispersion),
jointly producing moneyline win probability, run-line cover probability,
and the total-runs distribution from ONE consistent simulation — rather
than three separately-fit models that could disagree with each other.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimResult:
    home_win_prob: float
    home_minus_1_5_cover_prob: float  # home favored by 1.5
    away_plus_1_5_cover_prob: float
    prob_one_run_game: float
    mean_total: float
    median_total: float
    total_distribution: np.ndarray  # for over/under at arbitrary lines
    margin_distribution: np.ndarray


def simulate_game(mu_home: float, mu_away: float, dispersion_k: float, n_sims: int, seed: int) -> SimResult:
    rng = np.random.default_rng(seed)

    # NB(mu, k) via Gamma-Poisson mixture: lambda ~ Gamma(shape=k, scale=mu/k), Y ~ Poisson(lambda)
    lam_home = rng.gamma(shape=dispersion_k, scale=mu_home / dispersion_k, size=n_sims)
    lam_away = rng.gamma(shape=dispersion_k, scale=mu_away / dispersion_k, size=n_sims)
    home_runs = rng.poisson(lam_home)
    away_runs = rng.poisson(lam_away)

    margin = home_runs - away_runs  # positive = home wins by this many
    total = home_runs + away_runs

    home_wins = margin > 0
    ties = margin == 0
    # Extra-innings resolution for simulated ties: coin flip (no run-total
    # info to break the tie on; real MLB has a small extra-inning home edge
    # but modeling it precisely is out of scope, so this is deliberately
    # the conservative/neutral choice — documented in docs/limitations.md).
    tie_breaks = rng.random(n_sims) < 0.5
    home_win_prob = float(np.mean(home_wins | (ties & tie_breaks)))

    home_minus_1_5_cover_prob = float(np.mean(margin >= 2))
    away_plus_1_5_cover_prob = float(np.mean(margin <= 1))
    prob_one_run_game = float(np.mean(np.abs(margin) == 1))

    return SimResult(
        home_win_prob=home_win_prob,
        home_minus_1_5_cover_prob=home_minus_1_5_cover_prob,
        away_plus_1_5_cover_prob=away_plus_1_5_cover_prob,
        prob_one_run_game=prob_one_run_game,
        mean_total=float(np.mean(total)),
        median_total=float(np.median(total)),
        total_distribution=total,
        margin_distribution=margin,
    )


def over_under_prob(total_distribution: np.ndarray, line: float) -> tuple[float, float]:
    """Returns (P(over), P(under)) for a given total-runs betting line."""
    over = float(np.mean(total_distribution > line))
    under = float(np.mean(total_distribution < line))
    return over, under
