# Known limitations and scope of this build

This document exists so nobody — including future us — mistakes the current
state of the system for more than it is. Read this alongside the results in
`README.md`.

## 1. Offense is a team-level proxy, not confirmed lineups (yet)

The spec's core insight — "the pitcher is the unit, not the team," with
lineups as the second major driver — is only half-built. Starter and
bullpen projections are real, as-of-date, Statcast-peripheral-driven models.
Offense is currently a **team-level rolling runs-per-game proxy**
(`mlb.features.team_offense`), not a confirmed-lineup wOBA-vs-handedness
model. This means:

- A star hitter resting or a weak bench lineup will NOT move today's
  prediction the way the spec calls for.
- The `mlb.lineups` package is scaffolded (directory + `__init__.py`) but has
  no ingestion code yet. The real fix: pull confirmed lineups from the MLB
  Stats API boxscore/lineup endpoints (available live, ~10-15 games/day —
  cheap for daily predictions; expensive to backfill historically at
  ~2,400 games/season if done one boxscore call at a time) and replace the
  team-offense proxy with per-batter wOBA vs. the opposing starter's
  handedness, park-adjusted.

## 2. No weather data

Wind direction/speed and temperature are named in the spec as real,
predictable total-runs movers (especially at parks like Wrigley Field). None
is wired in. Park factors ARE implemented and are real (computed from actual
game-log runs, not fabricated — see `mlb.park_weather.park_factors`).

## 3. No market odds — CLV is not measured

CLV vs. the closing line is described in the spec as "the decisive and
humbling benchmark." We do not have a free, reliable historical odds source
integrated. Building `mlb.data` odds ingestion honestly requires either a
paid odds API key or a licensed historical dataset. Rather than fabricate
odds or approximate them from something else, **this system reports no CLV
number at all** — the honest-data policy in the spec ("missing → unavailable,
never invented") is intentionally more important than filling in this box.
If you have an odds API key, wire it into `mlb.data` and this becomes
straightforward to add.

## 4. Backtest coverage: 2023 (training pool) + 2024 (evaluated), not the full 2015+ history

The walk-forward backtest in this build covers the full 2024 season
(2,429 games), trained on an expanding window seeded with the full 2023
season. Extending to 2015+ is mechanical (`make features SEASON=<year>` for
each year, then include earlier seasons in `--prior`), but was out of scope
for the time budget of this build. Statcast data (the pitcher-peripheral
foundation) only exists from 2015 onward regardless.

## 5. Umpire tendencies, catcher framing splits, travel/rest beyond bullpen fatigue

Named in the spec as real signals; not implemented. Catcher framing is
available via `pybaseball.statcast_catcher_framing` and would be a
straightforward addition to the feature set.

## 6. Run-line and totals are read off the SAME simulation as moneyline

This is intentional (the spec calls for a jointly-consistent simulation
rather than three separately-fit models), not a limitation — but it does
mean an error in the run-environment mean model propagates to all three
markets simultaneously rather than being independently correctable.

## 7. Isotonic recalibration did not help (an honest negative result)

We tested post-hoc isotonic calibration (fit on the first half of the 2024
season, applied to the second half). It made calibration slightly WORSE
(ECE 0.011 → 0.062) rather than better — the raw simulation probabilities
were already well-calibrated, and isotonic regression overfit on a training
split of ~1,200 games. We are reporting the raw simulation probabilities as
the shipped output rather than the isotonic-recalibrated ones. See
`README.md` for the numbers.

## 8. The simulation model does not clearly beat a pitcher-adjusted Elo baseline

On the 2024 season (2,165 games where both models have enough warm-up data
to produce a prediction), the full simulation model and a much simpler
pitcher-adjusted-Elo logistic regression are statistically close, with
pitcher-adjusted Elo slightly AHEAD on log loss and Brier score. See
`README.md` §Results for the full table. This is reported plainly per the
project's honesty standard rather than reframed as a win. The most likely
fix is #1 above (confirmed lineups) — team-level offense is currently the
weakest link in the feature set.
