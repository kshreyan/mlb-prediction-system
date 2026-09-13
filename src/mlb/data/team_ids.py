"""Team abbreviation <-> full-name mapping, sourced from the MLB Stats API
(authoritative, not guessed) — bridges Statcast's abbreviation convention
(e.g. "NYY") with the Stats API schedule's full names (e.g. "New York
Yankees").
"""
from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd
import statsapi

logger = logging.getLogger(__name__)


def fetch_team_id_map(season: int) -> pd.DataFrame:
    teams = statsapi.get("teams", {"sportId": 1, "season": season})["teams"]
    rows = [{"abbreviation": t["abbreviation"], "name": t["name"], "team_id": t["id"], "season": season} for t in teams]
    return pd.DataFrame(rows)


def load_or_fetch_team_id_map(season: int, raw_dir: Path, force: bool = False) -> pd.DataFrame:
    out_path = raw_dir / "team_ids" / f"{season}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)
    df = fetch_team_id_map(season)
    df.to_parquet(out_path, index=False)
    return df


# A team's abbreviation is stable within a season but franchises do
# occasionally rebrand (e.g. Cleveland Indians -> Guardians in 2022) — always
# resolve per-season rather than caching one global map across years.
# Statcast sometimes tags games with an abbreviation the Stats API's
# per-season `teams` endpoint doesn't return for that season — e.g. Statcast
# tags some of the Athletics' 2024 games "ATH" even though the Stats API
# still calls them "Oakland Athletics"/"OAK" that season (the franchise's
# later Sacramento-relocation abbreviation). Map the alias onto whatever
# canonical name the Stats API map already uses, so both sides of any join
# agree — never introduce a name the schedule side doesn't already have.
_KNOWN_ALIASES = {"ATH": "OAK"}


def abbrev_to_name_map(season: int, raw_dir: Path) -> dict[str, str]:
    df = load_or_fetch_team_id_map(season, raw_dir)
    mapping = dict(zip(df["abbreviation"], df["name"]))
    for alias, canonical_abbrev in _KNOWN_ALIASES.items():
        if canonical_abbrev in mapping:
            mapping.setdefault(alias, mapping[canonical_abbrev])
    return mapping


def normalize_team_column(df: pd.DataFrame, col: str, season: int, raw_dir: Path) -> pd.DataFrame:
    """Rewrite an abbreviation column (Statcast convention) to full team
    names (Stats API schedule convention) in place, returning the frame."""
    mapping = abbrev_to_name_map(season, raw_dir)
    out = df.copy()
    unmapped = set(out[col].dropna().unique()) - set(mapping.keys())
    if unmapped:
        logger.warning("unmapped team abbreviations in %s: %s", col, unmapped)
    out[col] = out[col].map(mapping).fillna(out[col])
    return out
