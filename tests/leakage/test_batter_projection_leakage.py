"""Leakage checks for the batter/platoon-split projection module: a batter's
projection for game N must be unaffected by games after N, AND his vs-LHP
history must never leak into his vs-RHP projection or vice versa (the whole
point of splitting by handedness).
"""
from __future__ import annotations

import pandas as pd

from mlb.lineups.projections import add_asof_batter_projections


def _synthetic_batter_games(n_each_split: int = 4) -> pd.DataFrame:
    rows = []
    date = pd.Timestamp("2024-04-01")
    # Alternate facing RHP and LHP starters across the season.
    for i in range(n_each_split * 2):
        split = "R" if i % 2 == 0 else "L"
        rows.append({
            "game_pk": 1000 + i,
            "game_date": date + pd.Timedelta(days=3 * i),
            "batter": 555,
            "batting_team": "Team A",
            "p_throws": split,
            "pa": 4,
            "woba_denom_sum": 4,
            "xwoba_weighted_sum": 4 * (0.30 if split == "R" else 0.30),
        })
    return pd.DataFrame(rows)


def test_vs_hand_projection_unaffected_by_later_games():
    df = _synthetic_batter_games()
    proj_full = add_asof_batter_projections(df, halflife_days=60, shrinkage_k=200)

    altered = df.copy()
    last_idx = altered.index[-1]
    altered.loc[last_idx, "xwoba_weighted_sum"] = 4 * 0.99  # a huge final game
    proj_altered = add_asof_batter_projections(altered, halflife_days=60, shrinkage_k=200)

    proj_full_sorted = proj_full.sort_values(["batter", "p_throws", "game_date"]).reset_index(drop=True)
    proj_altered_sorted = proj_altered.sort_values(["batter", "p_throws", "game_date"]).reset_index(drop=True)

    # Drop the single altered row (last game overall) and compare everything else.
    altered_game_pk = df.loc[last_idx, "game_pk"]
    a = proj_full_sorted[proj_full_sorted["game_pk"] != altered_game_pk]["proj_xwoba"].reset_index(drop=True)
    b = proj_altered_sorted[proj_altered_sorted["game_pk"] != altered_game_pk]["proj_xwoba"].reset_index(drop=True)
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_vs_lhp_and_vs_rhp_histories_are_independent():
    """Changing a batter's vs-RHP results must not move his vs-LHP
    projection at all — they are separate as-of-date histories."""
    df = _synthetic_batter_games()
    proj_original = add_asof_batter_projections(df, halflife_days=60, shrinkage_k=200)

    altered = df.copy()
    rhp_rows = altered[altered["p_throws"] == "R"].index
    altered.loc[rhp_rows, "xwoba_weighted_sum"] = 4 * 0.05  # tank all vs-RHP games
    proj_altered = add_asof_batter_projections(altered, halflife_days=60, shrinkage_k=200)

    lhp_original = proj_original[proj_original["p_throws"] == "L"].sort_values("game_date")["proj_xwoba"].reset_index(drop=True)
    lhp_altered = proj_altered[proj_altered["p_throws"] == "L"].sort_values("game_date")["proj_xwoba"].reset_index(drop=True)
    pd.testing.assert_series_equal(lhp_original, lhp_altered, check_names=False)
