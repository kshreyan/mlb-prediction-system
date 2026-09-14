"""Live, same-day predictions for today's actual MLB slate.

The core trick, used consistently for every projection module (pitcher,
batter, bullpen, team offense): append ONE synthetic row per entity dated
TODAY with every numerator/denominator stat column set to zero, run the
EXACT SAME as-of-date functions already leak-tested in the backtest
(`mlb.pitchers.projections`, `mlb.lineups.projections`,
`mlb.bullpen.projections`, `mlb.features.team_offense`), and read off that
synthetic row's own `proj_*` output. Because those functions compute each
row's projection from STRICTLY EARLIER rows only (verified by
`tests/leakage/`), a zero-valued row dated today can never leak into any
earlier row's projection, and today's own output is exactly "the same
projection the model would compute as of right now" — no new, separately-
tested logic required, just leak-free reuse of what's already proven.

Weather: the MLB Stats API returns no forecast for a game that hasn't
happened yet (`gameData.weather` is `{}` until near game time) — there is
no honest way to know today's wind/temp in advance, so this module uses
the same neutral defaults as an indoor game and flags `weather_available:
False` rather than guessing.

Lineups: uses the CONFIRMED lineup when posted (`mlb.lineups.live`); when
not yet posted, falls back to that team's most recent ACTUAL lineup from
real historical data as an honest "expected lineup" proxy, with
`lineup_confirmed: False` so a re-run after lineups post gets a genuinely
different, more accurate prediction.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd
import statsapi

from mlb.config import BacktestConfig
from mlb.data.schedule import load_or_fetch_schedule
from mlb.data.team_ids import normalize_team_column
from mlb.pitchers.projections import add_asof_pitcher_projections
from mlb.bullpen.projections import add_asof_bullpen_projections
from mlb.features.team_offense import add_asof_team_offense
from mlb.features.build_dataset import build_game_features
from mlb.lineups.projections import add_asof_batter_projections
from mlb.lineups.lineup_offense import build_lineup_offense_features, compute_pa_weights_by_slot
from mlb.lineups.live import fetch_slate_status
from mlb.park_weather.park_factors import load_or_compute_park_factors
from mlb.simulation.run_environment import build_long_training_frame, fit_run_environment
from mlb.simulation.engine import simulate_game
from mlb.models.moneyline.elo import compute_elo_ratings, current_ratings
from mlb.ensemble.stacking import logit

logger = logging.getLogger(__name__)


def _person_throwing_hand(person_id: int) -> str | None:
    try:
        data = statsapi.get("person", {"personId": person_id})
        return data["people"][0].get("pitchHand", {}).get("code")
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not look up throwing hand for person_id=%s: %s", person_id, exc)
        return None


def _recent_actual_lineup(lineups_hist: pd.DataFrame, team: str, before_date: pd.Timestamp) -> pd.DataFrame | None:
    """Fallback when today's lineup isn't posted yet: that team's most
    recent real starting lineup from historical data — an honest proxy,
    not a guess, and clearly flagged as unconfirmed by the caller."""
    team_games = lineups_hist[(lineups_hist["team"] == team) & (lineups_hist["game_date"] < before_date)]
    if team_games.empty:
        return None
    last_game_pk = team_games.sort_values("game_date")["game_pk"].iloc[-1]
    return team_games[team_games["game_pk"] == last_game_pk][["batting_order_slot", "batter", "stand"]]


def build_todays_predictions(cfg: BacktestConfig, seasons: list[int], today: dt.date | None = None) -> pd.DataFrame:
    today = today or dt.date.today()
    today_ts = pd.Timestamp(today)
    raw_dir = cfg.raw_dir

    logger.info("Loading all historical data across seasons %s ...", seasons)
    pg_frames, batters_frames, lineups_frames, sched_frames = [], [], [], []
    for season in seasons:
        pg = pd.read_parquet(raw_dir / "pitcher_games" / f"{season}.parquet")
        for col in ["pitching_team", "home_team", "away_team"]:
            pg = normalize_team_column(pg, col, season, raw_dir)
        pg_frames.append(pg)

        b = pd.read_parquet(raw_dir / "batter_games" / f"{season}.parquet")
        b = normalize_team_column(b, "batting_team", season, raw_dir)
        batters_frames.append(b)

        l = pd.read_parquet(raw_dir / "lineups" / f"{season}.parquet")
        l = normalize_team_column(l, "team", season, raw_dir)
        l["game_date"] = pd.to_datetime(l["game_date"])
        lineups_frames.append(l)

        s = pd.read_parquet(raw_dir / "schedule" / f"{season}.parquet")
        sched_frames.append(s)

    pg_all = pd.concat(pg_frames, ignore_index=True)
    batters_all = pd.concat(batters_frames, ignore_index=True)
    lineups_all = pd.concat(lineups_frames, ignore_index=True)
    sched_all = pd.concat(sched_frames, ignore_index=True)
    starters_hist = pg_all[pg_all["is_starter"]].copy()

    # --- Today's real slate: probable pitchers + confirmed-or-fallback lineups ---
    logger.info("Fetching today's real slate (%s)...", today)
    slate = fetch_slate_status(today)
    if not slate:
        logger.warning("No games scheduled today.")
        return pd.DataFrame()

    current_season = today.year

    # Real venue per home team, for a correct park-factor lookup below (not
    # a guess — each team's actual home ballpark from its most recent game).
    venue_by_team = (
        sched_all.sort_values("game_date").drop_duplicates("home_team", keep="last")
        .set_index("home_team")["venue_name"].to_dict()
    )

    live_sched_rows, pitcher_synth_rows, batter_synth_rows, lineup_rows, starter_hand_rows = [], [], [], [], []
    game_meta = []

    for g in slate:
        game_pk = g["game_pk"]
        home_probable = g["home_probable_pitcher"]
        away_probable = g["away_probable_pitcher"]
        if not home_probable or not away_probable:
            logger.warning("Skipping %s @ %s — probable pitcher(s) not yet announced", g["away_team"], g["home_team"])
            continue

        home_hand = _person_throwing_hand(home_probable["id"])
        away_hand = _person_throwing_hand(away_probable["id"])
        if home_hand is None or away_hand is None:
            logger.warning("Skipping %s @ %s — could not determine a probable starter's throwing hand", g["away_team"], g["home_team"])
            continue

        live_sched_rows.append({
            "game_pk": game_pk, "game_date": today.isoformat(), "home_team": g["home_team"], "away_team": g["away_team"],
            "home_score": 0, "away_score": 0, "is_final": True, "venue_name": venue_by_team.get(g["home_team"]),
        })
        for team, opp_team, pitcher, hand in [
            (g["home_team"], g["away_team"], home_probable, home_hand),
            (g["away_team"], g["home_team"], away_probable, away_hand),
        ]:
            pitcher_synth_rows.append({
                "game_pk": game_pk, "game_date": today_ts, "pitcher": pitcher["id"], "player_name": pitcher["name"],
                "home_team": g["home_team"], "away_team": g["away_team"], "pitching_team": team,
                "is_starter": True, "p_throws": hand,
                "batters_faced": 0, "n_pitches": 0, "n_swings": 0, "n_whiffs": 0, "n_csw": 0, "n_bip": 0,
                "n_barrels": 0, "woba_denom_sum": 0, "xwoba_weighted_sum": 0, "n_k": 0, "n_bb": 0, "n_hbp": 0,
            })
            starter_hand_rows.append({"game_pk": game_pk, "team": opp_team, "opp_starter_p_throws": hand})

        for side, team, opp_hand in [("home", g["home_team"], away_hand), ("away", g["away_team"], home_hand)]:
            confirmed = g[f"{side}_lineup"]
            if confirmed:
                order = confirmed["batting_order"]
            else:
                fallback = _recent_actual_lineup(lineups_all, team, today_ts)
                if fallback is None:
                    logger.warning("No confirmed or recent-fallback lineup for %s — skipping this side's batter detail", team)
                    order = []
                else:
                    order = [{"slot": r.batting_order_slot, "batter_id": r.batter, "stand": r.stand} for r in fallback.itertuples()]
            for b in order:
                lineup_rows.append({"game_pk": game_pk, "team": team, "batting_order_slot": b["slot"], "batter": b["batter_id"], "stand": b.get("stand")})
                batter_synth_rows.append({
                    "game_pk": game_pk, "game_date": today_ts, "batter": b["batter_id"], "batting_team": team,
                    "p_throws": opp_hand, "pa": 0, "woba_denom_sum": 0, "xwoba_weighted_sum": 0,
                })

        game_meta.append({
            "game_pk": game_pk, "home_team": g["home_team"], "away_team": g["away_team"],
            "home_probable_pitcher": home_probable["name"], "away_probable_pitcher": away_probable["name"],
            "home_lineup_confirmed": g["home_lineup"] is not None, "away_lineup_confirmed": g["away_lineup"] is not None,
        })

    if not game_meta:
        logger.warning("No games with enough information (probable pitchers + hand) to predict today.")
        return pd.DataFrame()

    live_sched = pd.DataFrame(live_sched_rows)
    live_sched["game_date"] = pd.to_datetime(live_sched["game_date"])

    # --- Pitcher projections: reuse the exact leak-tested function ---
    combined_starters = pd.concat([starters_hist, pd.DataFrame(pitcher_synth_rows)], ignore_index=True)
    sproj_all = add_asof_pitcher_projections(combined_starters, halflife_days=cfg.pitcher_projection.halflife_days, shrinkage_k=cfg.pitcher_projection.shrinkage_k_batters)
    sproj_today = sproj_all[sproj_all["game_date"] == today_ts]

    # --- Bullpen: today's games included in the reindex target (is_final=True trick, see module docstring) ---
    combined_sched_for_bullpen = pd.concat([sched_all[sched_all["is_final"]], live_sched], ignore_index=True)
    bp_all = add_asof_bullpen_projections(pg_all, combined_sched_for_bullpen, halflife_days=cfg.bullpen.halflife_days,
                                           shrinkage_k_batters=cfg.bullpen.shrinkage_k_batters, fatigue_lookback_days=cfg.bullpen.fatigue_lookback_days)
    bp_today = bp_all[bp_all["game_date"] == today_ts]

    # --- Team offense: same trick, on the schedule ---
    off_all = add_asof_team_offense(combined_sched_for_bullpen, halflife_days=cfg.team_offense.halflife_days, shrinkage_k_games=cfg.team_offense.shrinkage_k_games)
    off_today = off_all[off_all["game_date"] == today_ts]

    # --- Batter/lineup projections ---
    combined_batters = pd.concat([batters_all, pd.DataFrame(batter_synth_rows)], ignore_index=True)
    bproj_all = add_asof_batter_projections(combined_batters, halflife_days=cfg.batter_projection.halflife_days, shrinkage_k=cfg.batter_projection.shrinkage_k_pa)
    bproj_today = bproj_all[bproj_all["game_date"] == today_ts]

    lineups_today_df = pd.DataFrame(lineup_rows)
    starter_hand_today = pd.DataFrame(starter_hand_rows).drop_duplicates()
    pa_weights = compute_pa_weights_by_slot(lineups_all[lineups_all["game_date"] >= today_ts - pd.Timedelta(days=730)], batters_all)
    lineup_off_today = build_lineup_offense_features(lineups_today_df, bproj_today, starter_hand_today, pa_weights_by_slot=pa_weights)

    # --- Park factors (real, prior-seasons-only — same cached computation
    # used everywhere else in the pipeline, no live component needed) ---
    prior_seasons_sorted = sorted(s for s in seasons if s < current_season)
    prior_seasons_for_pf = prior_seasons_sorted[-2:] or [min(seasons)]
    pf = load_or_compute_park_factors(prior_seasons_for_pf, raw_dir, load_or_fetch_schedule)

    weather_today = pd.DataFrame({
        "game_pk": [m["game_pk"] for m in game_meta],
        "wind_effect": 0.0, "temp_f_filled": 72.0, "is_indoor_or_roof_closed": False,
    })

    gf_today = build_game_features(live_sched, sproj_today, off_today, bp_today, pf, lineup_offense=lineup_off_today, weather=weather_today)
    gf_today["season"] = current_season

    # --- Fit FINAL models on ALL historical data through yesterday, apply to today ---
    logger.info("Fitting final models on all historical data...")
    hist_frames = []
    for season in seasons:
        p = cfg.processed_dir / f"game_features_{season}.parquet"
        if p.exists():
            hf = pd.read_parquet(p)
            hf["game_date"] = pd.to_datetime(hf["game_date"])
            hist_frames.append(hf)
    hist_all = pd.concat(hist_frames, ignore_index=True) if hist_frames else pd.DataFrame()
    hist_all["actual_home_win"] = (hist_all["home_score"] > hist_all["away_score"]).astype(int)

    long_train = build_long_training_frame(hist_all, feature_set="both")
    run_env_model = fit_run_environment(long_train, feature_set="both")

    results = []
    for _, row in gf_today.iterrows():
        X_home = pd.DataFrame([{
            "own_offense_proj": row.get("home_off_proj_runs_scored_per_game"),
            "own_lineup_xwoba": row.get("home_lineup_proj_xwoba"),
            "opp_starter_xwoba": row["away_starter_proj_xwoba_against"],
            "opp_bullpen_xwoba": row["away_bullpen_proj_bullpen_xwoba_against"],
            "park_factor": row["park_factor"], "is_home": 1.0,
            "wind_effect": row.get("wind_effect", 0.0), "temp_f_filled": row.get("temp_f_filled", 72.0),
        }])
        X_away = pd.DataFrame([{
            "own_offense_proj": row.get("away_off_proj_runs_scored_per_game"),
            "own_lineup_xwoba": row.get("away_lineup_proj_xwoba"),
            "opp_starter_xwoba": row["home_starter_proj_xwoba_against"],
            "opp_bullpen_xwoba": row["home_bullpen_proj_bullpen_xwoba_against"],
            "park_factor": row["park_factor"], "is_home": 0.0,
            "wind_effect": row.get("wind_effect", 0.0), "temp_f_filled": row.get("temp_f_filled", 72.0),
        }])
        mu_h = run_env_model.predict_mu(X_home)[0]
        mu_a = run_env_model.predict_mu(X_away)[0]
        sim = simulate_game(mu_h, mu_a, run_env_model.dispersion_k, n_sims=20000, seed=int(row["game_pk"]) % (2**31))
        meta = next(m for m in game_meta if m["game_pk"] == row["game_pk"])
        results.append({
            **meta, "mu_home": mu_h, "mu_away": mu_a,
            "sim_home_win_prob": sim.home_win_prob,
            "sim_home_minus_1_5_cover_prob": sim.home_minus_1_5_cover_prob,
            "sim_prob_one_run_game": sim.prob_one_run_game,
            "pred_mean_total": sim.mean_total, "pred_median_total": sim.median_total,
            "weather_available": False,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        })

    out = pd.DataFrame(results)

    # --- Ensemble: fit Elo + pitcher-adjusted-Elo on ALL history, blend with today's simulation ---
    elo_df = compute_elo_ratings(hist_all, k=20.0, home_advantage=24.0, season_regress_frac=0.33)
    elo_df["elo_diff"] = elo_df["elo_home_pre"] + 24.0 - elo_df["elo_away_pre"]

    # Today's Elo: each team's rating as of right now (post its most recent
    # completed game).
    final_rating = current_ratings(elo_df)

    for r in results:
        r["elo_home_pred"] = final_rating.get(r["home_team"], 1500.0)
        r["elo_away_pred"] = final_rating.get(r["away_team"], 1500.0)
    out = pd.DataFrame(results)
    out["elo_prob"] = 1.0 / (1.0 + np.power(10.0, -(out["elo_home_pred"] + 24.0 - out["elo_away_pred"]) / 400.0))

    # Pitcher-adjusted Elo (PAE): a FINAL fit on ALL history (not walk-forward
    # windowed — same "final production" treatment as the simulation and Elo
    # models above; see mlb.models.moneyline.baselines.walk_forward_logistic_baseline
    # for the walk-forward-validated version this reuses the exact feature set of).
    from sklearn.linear_model import LogisticRegression
    merged_hist = hist_all.merge(elo_df[["game_pk", "elo_diff"]], on="game_pk", how="left")
    pae_clf_frame = merged_hist.dropna(subset=["elo_diff", "diff_starter_xwoba", "diff_bullpen_xwoba"])
    pae_clf = LogisticRegression(max_iter=1000)
    pae_clf.fit(pae_clf_frame[["elo_diff", "diff_starter_xwoba", "diff_bullpen_xwoba"]], pae_clf_frame["actual_home_win"])
    today_elo_diff = out["elo_home_pred"] + 24.0 - out["elo_away_pred"]
    pae_X = pd.DataFrame({"elo_diff": today_elo_diff, "diff_starter_xwoba": gf_today["diff_starter_xwoba"].values, "diff_bullpen_xwoba": gf_today["diff_bullpen_xwoba"].values})
    out["pitcher_adjusted_elo_prob"] = pae_clf.predict_proba(pae_X)[:, 1]

    # Ensemble meta-model: fit on the already walk-forward-OUT-OF-SAMPLE
    # 2024 ensemble frame (predictions_2024_ensemble_final.parquet) — the
    # same honestly-validated logit_sim/logit_elo/logit_pae history the
    # backtest itself used to pick this ensemble, rather than reconstructing
    # component logits in-sample here (which would mix genuinely-OOS sim
    # logits with an in-sample-fit PAE component — an inconsistent, overly
    # optimistic training signal for the meta-model).
    ens_hist_path = cfg.processed_dir / "predictions_2024_ensemble_final.parquet"
    if ens_hist_path.exists():
        ens_hist = pd.read_parquet(ens_hist_path).dropna(subset=["logit_sim", "logit_elo", "logit_pae", "actual_home_win"])
        ens_clf = LogisticRegression(max_iter=1000)
        ens_clf.fit(ens_hist[["logit_sim", "logit_elo", "logit_pae"]], ens_hist["actual_home_win"])
        today_logits = pd.DataFrame({
            "logit_sim": logit(out["sim_home_win_prob"].values),
            "logit_elo": logit(out["elo_prob"].values),
            "logit_pae": logit(out["pitcher_adjusted_elo_prob"].values),
        })
        out["ensemble_home_win_prob"] = ens_clf.predict_proba(today_logits)[:, 1]
    else:
        out["ensemble_home_win_prob"] = out["sim_home_win_prob"]
        logger.warning("No historical ensemble frame found (%s); using simulation-only probability.", ens_hist_path)

    return out
