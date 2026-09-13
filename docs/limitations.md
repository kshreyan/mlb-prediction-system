# Known limitations and scope of this build

This document exists so nobody — including future us — mistakes the current
state of the system for more than it is. Read this alongside the results in
`README.md`.

## 1. Lineups: implemented for backtesting, not yet for live predictions

UPDATE: this was originally the single biggest gap; it's now partially
closed. `mlb.lineups` derives the ACTUAL starting lineup (9 batters, batting
order) directly from Statcast plate-appearance sequence for every historical
game — no extra API calls, no backfill cost. Each batter's xwOBA is
projected separately vs. LHP and vs. RHP (platoon splits), same as-of-date/
shrinkage discipline as pitchers, then averaged across the lineup against
the actual opposing starter's hand. Backtested: adding this signal alongside
the team-level proxy improved accuracy from 53.9% to 54.9% and improved
Brier/log loss (see README §Results, "Lineup ablation").

What's still missing:

- **Live/future predictions still need real pre-game confirmed lineups.**
  The backtest uses the ACTUAL lineup that played (the real ground truth,
  legitimate for backtesting since a team's real batting order is knowable
  from the lineup card ~1-3 hours before first pitch in the vast majority of
  cases). But for TODAY's slate, that data doesn't exist yet in this build —
  you'd need to pull the Stats API's pre-game confirmed-lineup endpoint,
  which updates once lineups are posted. Not yet wired up.
- **Batters are averaged equally across the lineup**, not weighted by
  expected plate appearances (leadoff hitters bat more often than #9). A
  PA-weighted average is a natural, cheap next improvement.
- **A batter not found for the exact opposing-hand split that game falls
  back to a flat neutral prior (0.31)** rather than a smarter estimate (e.g.
  from his overall, hand-agnostic history). Rare in practice (only ~0.4% of
  lineup slots hit this in the 2024 backtest) but worth tightening.
- Lineup-level offense did NOT move the totals-runs prediction (MAE was
  flat to slightly worse) — the win-probability split benefited more than
  the total-runs mean did. Worth investigating further.

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

## 8. The simulation model still doesn't clearly beat a pitcher-adjusted Elo baseline

UPDATE after adding lineups: the simulation model (with both team-offense
and lineup features) now has the BEST accuracy of every model tested
(54.8%, vs. 54.7% for both Elo variants), and improved Brier/log loss versus
the team-offense-only version. But it still trails pitcher-adjusted Elo on
Brier score (0.2474 vs 0.2466), log loss (0.6880 vs 0.6864), and ECE (0.030
vs. 0.015) on the same 2,165-game comparison. See `README.md` §Results for
the full table. Progress, not a win — reported plainly per the project's
honesty standard. Remaining likely fixes: PA-weighted lineup averaging (see
§1), tuning halflife/shrinkage hyperparameters (currently reasonable
defaults, never searched — see #9 below), and a proper stacked ensemble
instead of picking one model.

## 9. No hyperparameter tuning

All halflife/shrinkage-k values (pitcher: 45 days / k=250; batter: 60 days /
k=200; bullpen: 20 days / k=250; team offense: 30 days / k=12 games) are
reasonable defaults chosen by domain judgment, never tuned via the nested
time-series cross-validation the original spec calls for. This is a
plausible source of the remaining gap to pitcher-adjusted Elo.
