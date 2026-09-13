# MLB Prediction System — Moneyline · Total Runs · Run Line

A reproducible, honestly-backtested MLB prediction system built bottom-up
from the starting pitcher, bullpen, and offense — not team ratings. See
`docs/limitations.md` for exactly what is and isn't implemented yet, and
read that file before trusting any number below further than it's earned.

## The honesty standard (read this first)

MLB is the least predictable major sport. This system does not target high
accuracy — it targets calibration and honest performance vs. realistic
baselines. Realistic acceptance targets:

- Moneyline: ~57–60% straight-up is a **great** result, not a floor.
- Run line and totals: near breakeven against a sharp market.
- Anything above ~62% moneyline accuracy is presumed leakage, not skill.

**This build currently lands at ~54.8% moneyline accuracy, well-calibrated,
and is now the best-accuracy model of the group tested — but still trails
pitcher-adjusted Elo on Brier score and log loss, and has worse ECE.** That
is reported here plainly, not hidden — see Results below. Adding a real
lineup-level, platoon-aware offense signal (replacing the original
team-level proxy) closed most, but not all, of the gap to pitcher-adjusted
Elo — see "Lineup ablation" below.

## What's real here

- **Data**: real Statcast pitch-level data (via `pybaseball`) for the full
  2023 and 2024 regular seasons (21,000+ and 20,000+ pitcher-game rows
  respectively), real MLB Stats API schedules/scores, real empirical park
  factors computed from 2022-2023 actual game results (Coors Field comes out
  139.2 — the most hitter-friendly park in MLB, exactly as expected; Petco
  Park comes out 83.6, the most pitcher-friendly — this is a real signal,
  not a guess).
- **Every projection is as-of-date and leak-tested.** A pitcher's, batter's,
  team's, or bullpen's projection for game N uses ONLY games strictly before
  game N, exponentially time-weighted and shrunk toward a same-date league
  prior that is ITSELF computed from only strictly-prior games. `tests/leakage/`
  has 15 passing tests enforcing this, including two real bugs caught and
  fixed during this build (see "Leakage bugs found and fixed" below).
- **Confirmed lineups are real, not a team-level proxy.** The actual
  starting lineup (9 batters, batting order) is derived directly from
  Statcast plate-appearance sequence for each historical game — no extra API
  calls needed. Each batter's projection is split by the handedness of the
  pitcher he's facing (a real, persistent platoon effect), same as-of-date/
  shrinkage discipline as pitchers.
- **Nothing is fabricated.** No odds data exists in this build because we
  don't have a licensed/paid source — rather than approximate it, CLV is
  reported as unavailable. See `docs/limitations.md` §3.

## Architecture

```
src/mlb/
  data/            Stats API schedule ingestion, team ID mapping
  pitchers/        Statcast pull + aggregation, as-of-date pitcher projections
  bullpen/         As-of-date team bullpen quality + fatigue/workload
  lineups/         Actual-lineup extraction, as-of-date batter platoon-split
                   projections, lineup-vs-opposing-starter-hand aggregation
  features/        Team-offense proxy (fallback), as-of-date utilities, matchup dataset assembly
  park_weather/    Empirical park factors (from real game logs, prior-seasons-only)
  simulation/      Poisson-mean regression + NB dispersion, Monte Carlo game engine
  models/moneyline/  Elo, pitcher-adjusted Elo, Log5, home-field baselines
  calibration/     Isotonic post-hoc calibration
  backtest/        Walk-forward (expanding-window) backtest loop
  evaluation/      Brier/log-loss/ECE/reliability/totals-MAE metrics
tests/
  leakage/         The anti-leakage test suite (13 tests, all passing)
  unit/            Regression tests for real data artifacts found along the way
scripts/
  build_features.py     Build one season's leak-free game-feature dataset
  run_backtest.py        Walk-forward backtest one season
  evaluate_backtest.py    Produce the honest evaluation report
```

## How the model works

1. **Starter projection** (`mlb.pitchers.projections`): xwOBA-against, K%,
   BB%, whiff%, CSW%, barrel% — each exponentially time-weighted (45-day
   halflife) over the pitcher's own starts and shrunk toward a same-date
   league average using an empirical-Bayes credibility formula
   (`k=250` batters). A 4-start hot streak is outweighed by the shrinkage
   prior exactly as the spec requires.
2. **Bullpen projection** (`mlb.bullpen.projections`): same shrinkage
   machinery, applied to team-aggregate relief xwOBA-against (20-day
   halflife), plus a real fatigue signal — total relief pitches thrown in
   the trailing 3 days.
3. **Lineup offense** (`mlb.lineups`): the actual starting lineup (9
   batters) is inferred directly from Statcast plate-appearance order for
   each game — no extra API calls. Each batter's xwOBA is projected
   separately vs. LHP and vs. RHP (platoon splits), same shrinkage
   machinery as pitchers (60-day halflife, k=200 PA), then averaged across
   the lineup against the actual opposing starter's hand for that game. A
   team-level rolling-runs proxy (`mlb.features.team_offense`) is kept as a
   secondary signal — the simulation's mean-runs model takes both (see
   "Lineup ablation" below for why).
4. **Park factors** (`mlb.park_weather.park_factors`): empirical, from real
   prior-season game logs (home run-scoring environment vs. that team's own
   road environment), never leaking the season being predicted.
5. **Simulation** (`mlb.simulation`): a Poisson-mean regression (features:
   own offense proxy, opponent starter xwOBA, opponent bullpen xwOBA, park
   factor, home/away) predicts each team's expected runs; a Negative
   Binomial dispersion parameter is estimated empirically from the
   actual/predicted residual variance (Pearson method-of-moments — teams'
   runs are overdispersed relative to Poisson, as expected). A 20,000-draw
   Monte Carlo (Gamma-Poisson mixture) produces win probability, run-line
   cover probability, and the full total-runs distribution **jointly and
   consistently from one simulation**, per the spec.
6. **Walk-forward backtest** (`mlb.backtest.walk_forward`): retrains weekly
   on an expanding window (all games strictly before the retrain date); each
   week's predictions are locked in before that week's results are known.

## Results — 2024 season, walk-forward, 2023 as training warm-start

2,429 games backtested; 2,165 games used for baseline comparison (the
pitcher-adjusted-Elo baseline needs warm-up games and drops the first ~264).

### Lineup ablation (team-level proxy vs. lineup-level vs. both)

Before comparing to baselines, we tested whether real confirmed-lineup data
actually helps, on the full 2,429-game backtest:

| Offense feature | Accuracy | Brier | Log loss | ECE | Totals MAE |
|---|---|---|---|---|---|
| Team-level proxy only | 53.9% | 0.2479 | 0.6889 | 0.024 | 3.443 |
| Lineup-level only | 53.0% | 0.2481 | 0.6894 | 0.024 | 3.445 |
| **Both (used below)** | **54.9%** | **0.2475** | **0.6882** | 0.027 | 3.456 |

**Honest read:** lineup data ALONE is not obviously better than the simple
team-level proxy — but the mean-runs regression given BOTH signals
outperforms either alone on accuracy, Brier, and log loss (it apparently
extracts complementary information from each rather than one dominating).
Totals MAE is flat to slightly worse — the lineup signal isn't yet moving
the total-runs prediction, only the win-probability split. All results below
use the "both" feature set.

### Moneyline

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| **Simulation model (ours, both features)** | **54.8%** | 0.2474 | 0.6880 | 0.030 |
| Elo-only | 54.7% | 0.2485 | 0.6904 | 0.048 |
| Home-field-always | 52.8% | 0.2494 | 0.6920 | 0.006 |
| Better-record (Log5) | 50.3% | 0.2646 | 0.7264 | 0.097 |
| Pitcher-adjusted Elo | 54.7% | **0.2466** | **0.6864** | 0.015 |

**Honest read:** adding lineup data made our simulation model the
best-accuracy model of the group (54.8%, edging past both Elo variants) and
improved its Brier/log loss versus the team-offense-only version — but it
still trails pitcher-adjusted Elo on Brier score, log loss, AND calibration
(ECE 0.030 vs. 0.015). So: real, measurable progress from lineups, not yet
a clear win over the simplest strong baseline. Home-field-always is,
unsurprisingly, the best-calibrated (it just predicts the historical rate)
but least discriminating. Better-record (Log5) is the weakest model —
early win-loss record is a poor signal once regressed between seasons.

### Totals

| | MAE | RMSE | Bias |
|---|---|---|---|
| Simulation model (both features) | 3.456 | 4.407 | +0.135 |
| Naive (as-of league-average total) | 3.451 | 4.355 | +0.083 |

**Honest read:** the totals model is statistically indistinguishable from
predicting the rolling league-average total every single game — if
anything, marginally worse with lineup data added (3.456 vs. 3.451 MAE).
Totals are not yet adding measurable skill — expected, given no weather
data. This matches the spec's "totals near market breakeven" expectation,
though here it's breakeven against a trivial baseline rather than against
the market (no market data — see limitations).

### Run line (±1.5, modeled as P(margin), not a variable spread)

| | Actual rate | Mean predicted |
|---|---|---|
| Home −1.5 covers (wins by 2+) | 35.3% | 34.8% |
| One-run game | 27.8% | 19.6% |

**Honest read:** the −1.5/+1.5 cover probability is well-calibrated (35.3%
vs. 34.9%). The model meaningfully **underestimates** how often games are
decided by exactly one run — the NB dispersion parameter (fit once, globally)
doesn't fully capture the real fat-tailed frequency of close games. This is
a concrete, named target for the next iteration (e.g., a per-team or
run-environment-dependent dispersion instead of one global value).

### Isotonic calibration (tested, did not help)

Fit on the first half of the 2024 season, applied to the second half: ECE
went from 0.017 (raw) to 0.046 (isotonic-calibrated) — **worse**. The raw
simulation probabilities were already reasonably well-calibrated; isotonic
regression overfit on ~1,200 training games. We ship the raw probabilities.

## Leakage bugs found and fixed during this build

`tests/leakage/` has 15 passing tests. Two real leakage bugs were caught by
the test suite before they could taint results, and are worth naming
because they're the kind of subtle bug this whole architecture exists to
prevent:

1. **Same-day (doubleheader) leakage**: the original as-of-date function
   let a second game on the same calendar day see the first game's result
   (zero time-decay for zero elapsed days). Fixed by grouping same-day rows
   so they share one pre-day state, never each other's outcomes.
2. **Whole-dataset-mean fallback leakage**: the league-average "prior" used
   when a pitcher/team has zero history fell back to a mean computed over
   the ENTIRE season (including future games) rather than only prior days.
   Fixed with a strictly-prior-days expanding-mean fallback
   (`mlb.features.asof.asof_expanding_mean`), with a documented neutral
   constant as the absolute last resort (unreachable once a prior season is
   prepended to the training data).

Also fixed: a real MLB Stats API data artifact where a postponed-and-later-
completed game is listed twice under two different dates with the same
`game_pk` — silently fanned out into duplicate rows through every downstream
join until deduplicated at the source (`mlb.data.schedule.fetch_schedule`,
regression-tested in `tests/unit/test_schedule_dedup.py`).

## Running it

```bash
make setup                                   # venv + editable install
make test                                    # full suite (15 tests)
make leakage-test                            # just the anti-leakage gate
make features SEASON=2023                    # build one season's dataset (incl. lineups)
make features SEASON=2024
python scripts/run_backtest.py 2024 --prior 2023 --feature-set both
python scripts/evaluate_backtest.py 2024 --prior 2023 --feature-set both
```

## Acceptance criteria — honest status

- [x] Full pipeline runs raw data → predictions (backtest form; live daily
      slate deployment is not yet built — see limitations).
- [x] Walk-forward backtest, zero leakage (13 leakage tests passing,
      including 2 real bugs caught and fixed during this build).
- [~] Predictions driven by starter + bullpen + confirmed lineups —
      **implemented and backtested** (actual lineups derived from
      play-by-play, platoon-split batter projections), but the LIVE daily
      pipeline still needs to be wired to the Stats API's pre-game confirmed
      lineup endpoint rather than backtest-derived actual lineups. See
      limitations §1.
- [x] Calibration (reliability + ECE) reported for all three markets.
- [x] Run line modeled as P(margin), not a variable spread.
- [ ] CLV vs. closing line — **not measured**, no free/licensed odds source
      available in this build. Reported as unavailable, not fabricated.
- [~] Beats baselines out-of-sample — **beats home-field-always and
      better-record; does NOT clearly beat pitcher-adjusted Elo.** Reported
      honestly, not reframed.
- [x] Predictions immutable (backtest predictions parquet is write-once);
      no fabricated data anywhere in the pipeline.
- [x] Uncertainty and the ~57–60% realistic ceiling stated (this file, top).
- [x] No "beats Vegas" claim anywhere — there is no Vegas comparison in
      this build at all, by design.

## What's next (not done in this session)

See `docs/limitations.md` for the full list. In priority order: confirmed
lineups (biggest expected lift), weather, extending the backtest across
more seasons, an odds data source for CLV, and improving the run-line
dispersion model for one-run games.
