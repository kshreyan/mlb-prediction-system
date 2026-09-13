"""Combine starting lineups + as-of-date batter projections + the opposing
starter's throwing hand into one team-game feature: the lineup's projected
average wOBA against that specific hand. This replaces the team-level
rolling-runs proxy with an actual lineup-level, platoon-aware signal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_NEUTRAL_XWOBA_PRIOR = 0.31


def compute_pa_weights_by_slot(lineups: pd.DataFrame, batter_games: pd.DataFrame) -> dict[int, float]:
    """Empirical average plate-appearances-per-game by batting-order slot
    (1-9), computed from real data — NOT invented. Real baseball structure:
    a leadoff hitter genuinely bats more often per game than a #9 hitter,
    since the order cycles and #1 comes up first most often. Used as fixed
    weights for a PA-weighted lineup average instead of an unweighted mean
    (previously every batter counted equally regardless of how often he'd
    actually bat). Should be computed from seasons STRICTLY BEFORE the one
    being predicted, matching every other prior in this pipeline (see
    docs/limitations.md — this replaces the earlier equal-weighting
    simplification).
    """
    merged = lineups.merge(batter_games[["game_pk", "batter", "pa"]], on=["game_pk", "batter"], how="left")
    avg_pa = merged.groupby("batting_order_slot")["pa"].mean()
    return avg_pa.to_dict()


def derive_starter_hand_by_team_game(starters: pd.DataFrame) -> pd.DataFrame:
    """`starters` = pitcher_games rows where is_starter, with columns
    game_pk, pitching_team, home_team, away_team, p_throws. Returns
    game_pk, team, opp_starter_p_throws for BOTH teams in each game (the
    hand that TEAM's own lineup faces, i.e. the OTHER team's starter)."""
    s = starters[["game_pk", "home_team", "away_team", "pitching_team", "p_throws"]].drop_duplicates()
    home_starter = s[s["pitching_team"] == s["home_team"]][["game_pk", "p_throws"]].rename(columns={"p_throws": "home_starter_hand"})
    away_starter = s[s["pitching_team"] == s["away_team"]][["game_pk", "p_throws"]].rename(columns={"p_throws": "away_starter_hand"})
    games = s[["game_pk", "home_team", "away_team"]].drop_duplicates()
    games = games.merge(home_starter, on="game_pk", how="left").merge(away_starter, on="game_pk", how="left")

    home_rows = games[["game_pk", "home_team", "away_starter_hand"]].rename(
        columns={"home_team": "team", "away_starter_hand": "opp_starter_p_throws"}
    )
    away_rows = games[["game_pk", "away_team", "home_starter_hand"]].rename(
        columns={"away_team": "team", "home_starter_hand": "opp_starter_p_throws"}
    )
    return pd.concat([home_rows, away_rows], ignore_index=True)


def build_lineup_offense_features(
    lineups: pd.DataFrame,
    batter_proj: pd.DataFrame,
    starter_hand_by_team_game: pd.DataFrame,
    pa_weights_by_slot: dict[int, float] | None = None,
) -> pd.DataFrame:
    """
    lineups: game_pk, team, batting_order_slot, batter, stand
    batter_proj: game_pk, batter, batting_team, p_throws, proj_xwoba, n_prior_pa
    starter_hand_by_team_game: game_pk, team, opp_starter_p_throws — the hand
      of the pitcher THIS team's lineup faces in this game (i.e. keyed by the
      batting team, valued with the opposing starter's hand).
    pa_weights_by_slot: optional {1: weight, ..., 9: weight} from
      `compute_pa_weights_by_slot`, computed on PRIOR seasons only. If
      omitted, falls back to an equal-weighted average (the original,
      simpler behavior).

    Returns: game_pk, team, lineup_proj_xwoba, lineup_n_batters_matched
    """
    merged = lineups.merge(
        starter_hand_by_team_game, on=["game_pk", "team"], how="left",
    )
    merged = merged.merge(
        batter_proj.rename(columns={"batting_team": "team"}),
        left_on=["game_pk", "team", "batter", "opp_starter_p_throws"],
        right_on=["game_pk", "team", "batter", "p_throws"],
        how="left",
    )

    # A batter who didn't face that exact hand in that game (rare — e.g. a
    # pinch-hitter who only saw a mid-game reliever of the other hand)
    # falls back to the neutral league prior rather than being dropped,
    # so a lineup's average isn't skewed by missing bench bats.
    merged["proj_xwoba_filled"] = merged["proj_xwoba"].fillna(_NEUTRAL_XWOBA_PRIOR)

    if pa_weights_by_slot:
        merged["_pa_weight"] = merged["batting_order_slot"].map(pa_weights_by_slot).fillna(
            np.mean(list(pa_weights_by_slot.values()))
        )
    else:
        merged["_pa_weight"] = 1.0

    def _weighted_mean(g: pd.DataFrame) -> float:
        return float(np.average(g["proj_xwoba_filled"], weights=g["_pa_weight"]))

    out = merged.groupby(["game_pk", "team"]).apply(_weighted_mean, include_groups=False).rename("lineup_proj_xwoba").reset_index()
    counts = merged.groupby(["game_pk", "team"]).agg(
        lineup_n_batters_matched=("proj_xwoba", lambda s: s.notna().sum()),
        lineup_size=("batter", "size"),
    ).reset_index()
    return out.merge(counts, on=["game_pk", "team"], how="left")
