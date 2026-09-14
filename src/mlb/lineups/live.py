"""Live confirmed-lineup ingestion for TODAY's slate — distinct from the
backtest's `mlb.lineups.statcast_pull`, which derives the ACTUAL lineup a
game already played from historical pitch-by-pitch data. Here, the game
hasn't happened yet: MLB Stats API's `game` endpoint reports probable
pitchers well in advance, but `liveData.boxscore.teams.<side>.battingOrder`
stays empty until the lineup card is actually posted — typically ~1-3
hours before first pitch, sometimes later. This module reports that
distinction honestly (`lineup_confirmed: False`) rather than guessing.
"""
from __future__ import annotations

import datetime as dt

import statsapi


def fetch_probable_pitchers(game_pk: int) -> dict:
    """Real, officially-announced probable starters — available far in
    advance of first pitch, unlike the confirmed lineup."""
    data = statsapi.get("game", {"gamePk": game_pk})
    probables = data.get("gameData", {}).get("probablePitchers", {}) or {}
    out = {}
    for side in ("home", "away"):
        p = probables.get(side)
        out[side] = {"id": p["id"], "name": p.get("fullName")} if p else None
    return out


def fetch_confirmed_lineup(game_pk: int) -> dict:
    """Returns {'home': {...}, 'away': {...}} where each side is either
    None (lineup not posted yet) or a dict with 'batting_order' (a list of
    9 dicts: {slot, batter_id, name}) and 'starting_pitcher'. Never
    fabricates a lineup that hasn't been posted — that's the whole point.
    """
    data = statsapi.get("game", {"gamePk": game_pk})
    box = data.get("liveData", {}).get("boxscore", {})
    teams = box.get("teams", {})
    out = {}
    for side in ("home", "away"):
        t = teams.get(side, {})
        order_ids = t.get("battingOrder", [])
        if not order_ids:
            out[side] = None
            continue
        players = t.get("players", {})
        batting_order = []
        for slot_idx, pid in enumerate(order_ids[:9], start=1):
            p = players.get(f"ID{pid}", {})
            batting_order.append({
                "slot": slot_idx,
                "batter_id": pid,
                "name": p.get("person", {}).get("fullName"),
                "stand": p.get("person", {}).get("batSide", {}).get("code"),
            })
        pitchers = t.get("pitchers", [])
        starter_id = pitchers[0] if pitchers else None
        out[side] = {"batting_order": batting_order, "starting_pitcher_id": starter_id}
    return out


def fetch_slate_status(target_date: dt.date | None = None) -> list[dict]:
    """One row per scheduled game on `target_date` (default: today), with
    probable pitchers always populated and confirmed lineups populated
    only once posted — the honest, real-time state of the slate."""
    target_date = target_date or dt.date.today()
    games = statsapi.schedule(start_date=target_date.isoformat(), end_date=target_date.isoformat(), sportId=1)

    rows = []
    for g in games:
        if g.get("game_type") != "R":
            continue
        game_pk = g["game_id"]
        probables = fetch_probable_pitchers(game_pk)
        lineups = fetch_confirmed_lineup(game_pk)
        rows.append({
            "game_pk": game_pk,
            "game_date": g["game_date"],
            "home_team": g["home_name"],
            "away_team": g["away_name"],
            "status": g.get("status"),
            "home_probable_pitcher": probables["home"],
            "away_probable_pitcher": probables["away"],
            "home_lineup": lineups["home"],
            "away_lineup": lineups["away"],
            "lineups_confirmed": lineups["home"] is not None and lineups["away"] is not None,
        })
    return rows
