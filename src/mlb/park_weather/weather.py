"""Real per-game weather from the MLB Stats API `game` endpoint
(gameData.weather: condition, temp in F, wind speed+direction relative to
the park, e.g. "Out To CF" / "In From LF" / "L To R" / "Varies" / "None" /
"Roof Closed"). One API call per game — resumable and rate-limited-friendly
via a modest thread pool, with results cached to parquet so a re-run only
fetches games not already on disk.
"""
from __future__ import annotations

import concurrent.futures as cf
import datetime as dt
import logging
import re
import time
from pathlib import Path

import pandas as pd
import statsapi

logger = logging.getLogger(__name__)

SOURCE = "mlb_stats_api_game_weather"

# Directional wind strings are relative to the park's own layout, as MLB
# reports them — "Out" tends to help hitters (ball carries), "In" tends to
# suppress offense. Categorized, not scored numerically, since the actual
# run impact should be LEARNED by the model, not assumed.
_OUT_PATTERN = re.compile(r"^Out To", re.IGNORECASE)
_IN_PATTERN = re.compile(r"^In From", re.IGNORECASE)
_CROSS_PATTERN = re.compile(r"To (L|R)$|^(L|R) To", re.IGNORECASE)


def _parse_wind(wind_str: str | None) -> tuple[float | None, str | None]:
    if not wind_str:
        return None, None
    m = re.match(r"(\d+)\s*mph,\s*(.+)", wind_str)
    if not m:
        return None, None
    speed = float(m.group(1))
    direction_raw = m.group(2).strip()
    if direction_raw.lower() in ("none", "calm"):
        category = "none"
    elif _OUT_PATTERN.match(direction_raw):
        category = "out"
    elif _IN_PATTERN.match(direction_raw):
        category = "in"
    elif _CROSS_PATTERN.search(direction_raw) or "varies" in direction_raw.lower():
        category = "cross_or_varies"
    else:
        category = "other"
    return speed, category


def _fetch_one(game_pk: int) -> dict:
    try:
        data = statsapi.get("game", {"gamePk": game_pk})
        weather = data.get("gameData", {}).get("weather", {}) or {}
        condition = weather.get("condition")
        temp_raw = weather.get("temp")
        temp = float(temp_raw) if temp_raw not in (None, "") else None
        is_roof_closed = condition in ("Roof Closed", "Dome")
        # The Stats API sometimes reports temp=0 for indoor games as a
        # placeholder (there's no outdoor reading to give) — treat as
        # missing rather than a real 0F, which would be a data artifact,
        # not an honest observation.
        if is_roof_closed and temp == 0.0:
            temp = None
        wind_speed, wind_category = _parse_wind(weather.get("wind"))
        return {
            "game_pk": game_pk, "condition": condition, "temp_f": temp,
            "wind_speed_mph": wind_speed, "wind_category": wind_category,
            "is_indoor_or_roof_closed": is_roof_closed,
            "source": SOURCE, "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    except Exception as exc:  # noqa: BLE001 - one bad game_pk shouldn't kill the whole pull
        logger.warning("weather fetch failed for game_pk=%s: %s", game_pk, exc)
        return {"game_pk": game_pk, "condition": None, "temp_f": None,
                "wind_speed_mph": None, "wind_category": None,
                "is_indoor_or_roof_closed": None, "source": SOURCE,
                "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat()}


def load_or_fetch_weather(season: int, game_pks: list[int], raw_dir: Path, max_workers: int = 8, force: bool = False) -> pd.DataFrame:
    out_path = raw_dir / "weather" / f"{season}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.read_parquet(out_path) if (out_path.exists() and not force) else pd.DataFrame(columns=["game_pk"])
    already_have = set(existing["game_pk"].tolist())
    to_fetch = [pk for pk in game_pks if pk not in already_have]

    if not to_fetch:
        return existing

    logger.info("fetching weather for %d/%d games (season=%s)", len(to_fetch), len(game_pks), season)
    rows = []
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, row in enumerate(ex.map(_fetch_one, to_fetch)):
            rows.append(row)
            if (i + 1) % 500 == 0:
                logger.info("  weather progress: %d/%d (%.0fs elapsed)", i + 1, len(to_fetch), time.time() - t0)

    new_df = pd.DataFrame(rows)
    combined = pd.concat([existing, new_df], ignore_index=True).drop_duplicates(subset="game_pk", keep="last")
    combined.to_parquet(out_path, index=False)
    logger.info("saved weather season=%s rows=%d -> %s (%.0fs)", season, len(combined), out_path, time.time() - t0)
    return combined


_NEUTRAL_INDOOR_TEMP = 72.0


def add_derived_weather_features(weather: pd.DataFrame) -> pd.DataFrame:
    """Adds `wind_effect` (signed speed: +mph blowing out, -mph blowing in,
    0 for crosswind/none/indoor — the model learns the coefficient, we only
    encode direction) and `temp_f_filled` (indoor games get a neutral 72F
    rather than a missing value, since indoor climate control genuinely
    makes outdoor temperature irrelevant to that game)."""
    df = weather.copy()
    sign = df["wind_category"].map({"out": 1.0, "in": -1.0}).fillna(0.0)
    df["wind_effect"] = sign * df["wind_speed_mph"].fillna(0.0)
    df["temp_f_filled"] = df["temp_f"].fillna(_NEUTRAL_INDOOR_TEMP)
    return df
