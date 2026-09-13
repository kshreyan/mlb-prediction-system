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

## 2. Weather: implemented, helps totals, doesn't help moneyline

UPDATE: closed. `mlb.park_weather.weather` pulls real per-game condition,
temp, and wind speed/direction from the MLB Stats API's `game` endpoint for
every game in 2023-2024 (~4,860 games, one API call each, moderate
concurrency). Wind is encoded as a signed `wind_effect` (out=positive,
in=negative, cross/none/indoor=zero) so the regression learns the
coefficient rather than assuming one. Result (README §Results, "Weather
ablation"): totals MAE improved from 3.456 to 3.430 — the clearest
unambiguous win from any single addition this session. Moneyline accuracy
actually dropped slightly (54.8% → 54.5%) with log loss/ECE roughly flat —
weather is a totals signal, not a moneyline signal, as domain intuition
would predict.

Remaining gap: the Stats API only reports the weather AT GAME TIME (or at
whatever point the API snapshot was taken), not a pre-game FORECAST. For a
live prediction pipeline, you'd want a forecast API for games not yet
played, falling back to the actual reading only for backtesting historical
games (where it's the correct, knowable-after-first-pitch information for
grading, though technically a pre-game forecast would sometimes differ from
the eventual actual reading — a subtlety worth flagging for anyone building
the live pipeline).

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

## 4. Backtest coverage: 2022-2024 (2024 evaluated), not the full 2015+ history

The walk-forward backtest in this build covers the full 2024 season
(2,429 games, evaluated) and 2023 (2,430 games, also backtested — used both
as 2024's training pool and, via its own 2022-warm-started walk-forward run,
as ensemble/tuning validation data). 2022 was pulled in full (pitchers,
batters/lineups, weather) specifically to support multi-season
hyperparameter tuning and the ensemble's warm-up. Extending to 2015+ is
mechanical (`make features SEASON=<year>` for each year, then include
earlier seasons in `--prior`), but was out of scope for the time budget of
this build. Statcast data (the pitcher-peripheral foundation) only exists
from 2015 onward regardless.

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

## 8. RESOLVED: the stacked ensemble decisively beats every baseline, including pitcher-adjusted Elo

This was the single most important open finding through most of this
build's history: the standalone simulation model, even after lineups,
weather, and multi-season hyperparameter tuning, was only statistically
tied with pitcher-adjusted Elo on accuracy (54.7% vs 54.8%) and still
trailed it on Brier/log loss (0.2470/0.6872 vs 0.2466/0.6864).

It's resolved by NOT picking one model. `mlb.ensemble.stacking` blends the
simulation model, Elo-only, and pitcher-adjusted Elo on log-odds via a
walk-forward logistic regression (`scripts/run_ensemble.py`), and the
result decisively beats every component on every metric: 56.0% accuracy,
Brier 0.2448, log loss 0.6826, ECE 0.0072 (2-6x better calibrated than any
single component). See `README.md` §"Moneyline — stacked ensemble" for the
full table and `tests/leakage/test_ensemble_leakage.py` for the leakage
verification.

What's still open:

- **The ensemble covers moneyline only.** Elo and pitcher-adjusted-Elo
  produce a single win probability, not a run distribution, so there's
  nothing of theirs to blend with the simulation's run-line/totals output.
  Extending stacking to those markets would need other run/total-producing
  models to combine with — not yet built.
- The in-sample diagnostic weight inspection (not the reported walk-forward
  result — just a single global fit for interpretability) put a NEGATIVE
  coefficient on the pitcher-adjusted-Elo term, likely a suppressor-variable
  effect from its heavy overlap with the Elo term rather than a sign the
  ensemble is doing something wrong (the held-out walk-forward performance
  is what's actually reported and trusted). Worth a deeper look if the
  ensemble is extended further.
- Remaining likely further gains: PA-weighted lineup averaging (§1), tuning
  across more than two seasons (§9), and adding more diverse component
  models to the ensemble (e.g. a gradient-boosted-trees baseline directly
  on the matchup features, as the original spec suggested).

## 9. Hyperparameter tuning: a failed single-season attempt, fixed by tuning across two seasons

`scripts/tune_hyperparams.py` runs a real, held-out coordinate-descent
search. The FIRST attempt scored candidates by walk-forward log loss on a
slice of 2023 ONLY (games from 2023-07-20 on, trained on earlier 2023
games), keeping 2024 completely untouched. It found longer halflives
helped on the 2023 validation slice (pitcher 45→75 days, batter 60→100
days; log loss 0.6844→0.6833) — but when applied to the true 2024 holdout,
the improvement nearly vanished (log loss 0.6879→0.6878) and RAW ACCURACY
GOT WORSE (54.5%→53.8%). A single-season validation split wasn't reliable
enough here.

The SECOND attempt scored the same search by POOLED log loss across
held-out validation slices of BOTH 2022 and 2023 (2024 still never
touched), requiring improvement on both seasons individually, not just one
average. This landed on a materially different, more conservative config
(batter shrinkage-k reversed from 200 back down to 100; bullpen halflife
20→35 days; team-offense halflife 30→50 days) and, applied to the 2024
holdout, actually delivered: accuracy recovered to 54.7% (from 53.8%) and
Brier/log loss improved further (0.2470/0.6872, vs. attempt 1's
0.2473/0.6878). This is the config shipped in this build.

The lesson worth keeping: on a system this noisy, tuning on one validation
season was actively misleading, and two seasons was enough to catch and
fix it. The original spec's full nested time-series CV (more seasons
still) would be the natural further extension — not yet built.
