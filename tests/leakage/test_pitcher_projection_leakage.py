"""End-to-end leakage check on the actual pitcher-projection module: a
pitcher's projection for start N must be identical whether or not we know
what happens in start N+1 (and beyond).
"""
from __future__ import annotations

import pandas as pd

from mlb.pitchers.projections import add_asof_pitcher_projections


def _synthetic_pitcher_games(n_games: int = 6) -> pd.DataFrame:
    dates = pd.date_range("2024-04-01", periods=n_games, freq="5D")
    rows = []
    for i, d in enumerate(dates):
        rows.append({
            "game_pk": 1000 + i,
            "game_date": d,
            "pitcher": 111,
            "player_name": "Test, Pitcher",
            "home_team": "Team A",
            "away_team": "Team B",
            "pitching_team": "Team A",
            "is_starter": True,
            "batters_faced": 25,
            "n_pitches": 90,
            "n_swings": 40,
            "n_whiffs": 10,
            "n_csw": 35,
            "n_bip": 15,
            "n_barrels": 2,
            "woba_denom_sum": 25,
            "xwoba_weighted_sum": 25 * 0.30,
            "n_k": 6,
            "n_bb": 2,
            "n_hbp": 0,
        })
    return pd.DataFrame(rows)


def test_projection_for_early_start_unaffected_by_later_results():
    df_full = _synthetic_pitcher_games(6)
    proj_full = add_asof_pitcher_projections(df_full, halflife_days=45, shrinkage_k=250)

    # A wildly different final start (e.g. pitcher gets shelled) must not
    # change the projection computed for any EARLIER start.
    df_altered = df_full.copy()
    df_altered.loc[df_altered.index[-1], ["n_k", "n_bb", "xwoba_weighted_sum"]] = [0, 20, 25 * 0.99]
    proj_altered = add_asof_pitcher_projections(df_altered, halflife_days=45, shrinkage_k=250)

    for col in ["proj_k_pct", "proj_bb_pct", "proj_xwoba_against"]:
        pd.testing.assert_series_equal(
            proj_full[col].iloc[:-1].reset_index(drop=True),
            proj_altered[col].iloc[:-1].reset_index(drop=True),
            check_names=False,
        )


def test_truncating_history_matches_incremental_projection():
    """The projection for start k, computed from the full season log, must
    equal the projection you'd get computing as-of that same date from ONLY
    the games up to and including start k. This is the strongest possible
    leakage check: no future data can be present, structurally."""
    df_full = _synthetic_pitcher_games(6)
    proj_full = add_asof_pitcher_projections(df_full, halflife_days=45, shrinkage_k=250)

    truncated = df_full.iloc[:4].copy()  # only first 4 starts known
    proj_truncated = add_asof_pitcher_projections(truncated, halflife_days=45, shrinkage_k=250)

    # The 4th (last, index 3) row's projection should match between the two runs.
    for col in ["proj_k_pct", "proj_xwoba_against"]:
        assert abs(proj_full[col].iloc[3] - proj_truncated[col].iloc[3]) < 1e-9
