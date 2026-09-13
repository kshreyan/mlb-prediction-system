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

**This build currently lands at ~53–55% moneyline accuracy, well-calibrated,
and does not yet clearly beat a simple pitcher-adjusted Elo baseline.** That
is reported here plainly, not hidden — see Results below. The most likely
reason (confirmed lineups aren't wired in yet — team offense is a rolling
proxy) is documented in `docs/limitations.md`.

## What's real here

- **Data**: real Statcast pitch-level data (via `pybaseball`) for the full
  2023 and 2024 regular seasons (21,000+ and 20,000+ pitcher-game rows
  respectively), real MLB Stats API schedules/scores, real empirical park
  factors computed from 2022-2023 actual game results (Coors Field comes out
  139.2 — the most hitter-friendly park in MLB, exactly as expected; Petco
  Park comes out 83.6, the most pitcher-friendly — this is a real signal,
  not a guess).
- **Every projection is as-of-date and leak-tested.** A pitcher's, team's,
  or bullpen's projection for game N uses ONLY games strictly before game N,
  exponentially time-weighted and shrunk toward a same-date league prior
  that is ITSELF computed from only strictly-prior games. `tests/leakage/`
  has 13 passing tests enforcing this, including a real bug caught and
  fixed during this build (see "Leakage bugs found and fixed" below).
- **Nothing is fabricated.** No odds data exists in this build because we
  don't have a licensed/paid source — rather than approximate it, CLV is
  reported as unavailable. See `docs/limitations.md` §3.

## Architecture

```
src/mlb/
  data/            Stats API schedule ingestion, team ID mapping
  pitchers/        Statcast pull + aggregation, as-of-date pitcher projections
  bullpen/         As-of-date team bullpen quality + fatigue/workload
  features/        Team-offense proxy, as-of-date utilities, matchup dataset assembly
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
3. **Offense proxy** (`mlb.features.team_offense`): team rolling
   runs-scored/allowed per game, 30-day halflife, shrunk to the league mean.
   **This is the weakest link — see limitations §1.**
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

### Moneyline

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| **Simulation model (ours)** | 53.8% | 0.2477 | 0.6886 | 0.030 |
| Elo-only | 54.7% | 0.2485 | 0.6904 | 0.048 |
| Home-field-always | 52.8% | 0.2494 | 0.6920 | 0.006 |
| Better-record (Log5) | 50.3% | 0.2646 | 0.7264 | 0.097 |
| **Pitcher-adjusted Elo** | 54.7% | **0.2466** | **0.6864** | 0.015 |

**Honest read:** our simulation model is competitive and well-calibrated but
does **not** clearly beat pitcher-adjusted Elo — that baseline has the best
Brier and log loss of the group. Home-field-always is, unsurprisingly, the
best-calibrated (it just predicts the historical rate) but least
discriminating. Better-record (Log5) is the weakest model — early
win-loss record is a poor signal once regressed between seasons.

### Totals

| | MAE | RMSE | Bias |
|---|---|---|---|
| Simulation model | 3.443 | 4.358 | +0.147 |
| Naive (as-of league-average total) | 3.451 | 4.355 | +0.083 |

**Honest read:** the totals model is statistically indistinguishable from
predicting the rolling league-average total every single game. It is not
yet adding measurable skill — expected, given no weather and a team-level
(not lineup-level) offense signal. This matches the spec's "totals near
market breakeven" expectation, though here it's breakeven against a trivial
baseline rather than against the market (no market data — see limitations).

### Run line (±1.5, modeled as P(margin), not a variable spread)

| | Actual rate | Mean predicted |
|---|---|---|
| Home −1.5 covers (wins by 2+) | 35.3% | 34.9% |
| One-run game | 27.8% | 19.6% |

**Honest read:** the −1.5/+1.5 cover probability is well-calibrated (35.3%
vs. 34.9%). The model meaningfully **underestimates** how often games are
decided by exactly one run — the NB dispersion parameter (fit once, globally)
doesn't fully capture the real fat-tailed frequency of close games. This is
a concrete, named target for the next iteration (e.g., a per-team or
run-environment-dependent dispersion instead of one global value).

### Isotonic calibration (tested, did not help)

Fit on the first half of the 2024 season, applied to the second half: ECE
went from 0.011 (raw) to 0.062 (isotonic-calibrated) — **worse**. The raw
simulation probabilities were already well-calibrated; isotonic regression
overfit on ~1,200 training games. We ship the raw probabilities.

## Leakage bugs found and fixed during this build

Two real leakage bugs were caught by the test suite before they could taint
results, and are worth naming because they're the kind of subtle bug this
whole architecture exists to prevent:

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
make test                                    # full suite (13 tests)
make leakage-test                            # just the anti-leakage gate
make features SEASON=2023                    # build one season's dataset
make features SEASON=2024
make backtest SEASON=2024 PRIOR=2023         # walk-forward backtest
make report SEASON=2024                       # honest evaluation report (needs PRIOR too — see scripts/evaluate_backtest.py)
```

## Acceptance criteria — honest status

- [x] Full pipeline runs raw data → predictions (backtest form; live daily
      slate deployment is not yet built — see limitations).
- [x] Walk-forward backtest, zero leakage (13 leakage tests passing,
      including 2 real bugs caught and fixed during this build).
- [~] Predictions driven by starter + bullpen + offense — **not yet
      confirmed lineups** (team-level proxy currently). See limitations §1.
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
