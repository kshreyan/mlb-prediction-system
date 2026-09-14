"""Real historical closing-line odds from The Odds API
(https://the-odds-api.com), for CLV evaluation. Every row carries
provenance (`source`, `fetched_at`, `is_closing`, `is_real_data`) per this
project's data-honesty policy — never fabricated, never approximated from
something else.

Budget reality, stated plainly: a full-season pull at 3 markets (h2h,
spreads, totals) x games with 5-minute-granularity snapshots would cost far
more credits than a typical paid plan grants (see docs/limitations.md).
This module fetches ONE targeted snapshot per requested game, timed just
before that specific game's own commence_time, which is the only way to
get a genuine closing line (as opposed to a price hours or days out) —
so it is used for a representative SAMPLE of games, not exhaustive
coverage, and that scope is recorded in the output, not hidden.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"
SOURCE = "the_odds_api"


def _api_key() -> str:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise RuntimeError(
            "ODDS_API_KEY not set. Put it in a .env file (never commit it) "
            "and `source .env` before running odds ingestion."
        )
    return key


_last_remaining_credits: int | None = None


def remaining_credits() -> int | None:
    """Credits remaining as of the most recent call, per the API's own
    `x-requests-remaining` response header — None until a call has been made."""
    return _last_remaining_credits


def fetch_historical_snapshot(timestamp: dt.datetime, markets: str = "h2h,spreads,totals", regions: str = "us") -> dict:
    """One real API call: the full odds board as it stood at `timestamp`
    (UTC), across the requested markets/regions. Costs credits — see the
    module docstring. Retries transient failures; raises on real errors
    (e.g. a deactivated or exhausted key) rather than silently degrading.
    """
    global _last_remaining_credits
    params = {
        "apiKey": _api_key(),
        "regions": regions,
        "markets": markets,
        "date": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for attempt in range(3):
        resp = requests.get(f"{BASE_URL}/historical/sports/{SPORT_KEY}/odds", params=params, timeout=30)
        if "x-requests-remaining" in resp.headers:
            _last_remaining_credits = int(resp.headers["x-requests-remaining"])
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (401, 402, 403):
            raise RuntimeError(f"Odds API auth/quota error {resp.status_code}: {resp.text}")
        logger.warning("odds API transient error %s, attempt %d: %s", resp.status_code, attempt + 1, resp.text[:200])
        time.sleep(2 * (attempt + 1))
    resp.raise_for_status()


def _devig_two_way(price_a: float, price_b: float) -> tuple[float, float]:
    """Decimal odds -> de-vigged (no-hold) implied probabilities."""
    p_a, p_b = 1.0 / price_a, 1.0 / price_b
    total = p_a + p_b
    return p_a / total, p_b / total


def extract_game_closing_odds(snapshot: dict, home_team: str, away_team: str, commence_time: dt.datetime, tolerance_minutes: int = 20) -> dict | None:
    """Finds the target game in a snapshot response and returns a
    structured, de-vigged consensus closing line across whatever US
    bookmakers were present. Returns None if the game isn't in this
    snapshot (wrong timing, or the book hadn't posted yet) — never
    fabricates a line that wasn't actually quoted.
    """
    for g in snapshot.get("data", []):
        g_commence = dt.datetime.strptime(g["commence_time"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        if g["home_team"] != home_team or g["away_team"] != away_team:
            continue
        if abs((g_commence - commence_time).total_seconds()) > tolerance_minutes * 60:
            continue

        h2h_home, h2h_away, spread_home, spread_away, spread_point, total_over, total_under, total_point = ([] for _ in range(8))
        books_seen = []
        for bm in g.get("bookmakers", []):
            books_seen.append(bm["key"])
            for m in bm.get("markets", []):
                outcomes = {o["name"]: o for o in m["outcomes"]}
                if m["key"] == "h2h" and home_team in outcomes and away_team in outcomes:
                    h2h_home.append(outcomes[home_team]["price"])
                    h2h_away.append(outcomes[away_team]["price"])
                elif m["key"] == "spreads" and home_team in outcomes and away_team in outcomes:
                    spread_home.append(outcomes[home_team]["price"])
                    spread_away.append(outcomes[away_team]["price"])
                    spread_point.append(outcomes[home_team].get("point"))
                elif m["key"] == "totals" and "Over" in outcomes and "Under" in outcomes:
                    total_over.append(outcomes["Over"]["price"])
                    total_under.append(outcomes["Under"]["price"])
                    total_point.append(outcomes["Over"].get("point"))

        if not h2h_home:
            return None  # game found but no moneyline quoted yet — real absence, not an error

        import statistics
        ml_home_prob, ml_away_prob = _devig_two_way(statistics.median(h2h_home), statistics.median(h2h_away))
        result = {
            "game_pk": None,  # filled in by the caller, which knows the mapping
            "home_team": home_team, "away_team": away_team,
            "commence_time": commence_time.isoformat(),
            "snapshot_timestamp": snapshot.get("timestamp"),
            "n_bookmakers": len(books_seen), "bookmakers": ",".join(sorted(set(books_seen))),
            "ml_home_implied_prob": ml_home_prob, "ml_away_implied_prob": ml_away_prob,
            "source": SOURCE, "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "is_closing": True, "is_real_data": True,
        }
        if spread_home:
            rl_home_prob, rl_away_prob = _devig_two_way(statistics.median(spread_home), statistics.median(spread_away))
            result["run_line_point"] = statistics.median([p for p in spread_point if p is not None]) if any(p is not None for p in spread_point) else None
            result["run_line_home_cover_prob"] = rl_home_prob
            result["run_line_away_cover_prob"] = rl_away_prob
        if total_over:
            over_prob, under_prob = _devig_two_way(statistics.median(total_over), statistics.median(total_under))
            result["total_point"] = statistics.median([p for p in total_point if p is not None]) if any(p is not None for p in total_point) else None
            result["total_over_prob"] = over_prob
            result["total_under_prob"] = under_prob
        return result
    return None
