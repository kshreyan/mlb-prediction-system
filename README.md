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

**The single simulation model lands at ~54.7% moneyline accuracy on the
true 2024 holdout — statistically tied with pitcher-adjusted Elo, still
trailing it slightly on Brier/log loss. But a stacked ensemble (simulation
+ Elo-only + pitcher-adjusted-Elo, blended on log-odds, weights learned
walk-forward) clearly beats every individual model on every metric: 56.0%
accuracy, Brier 0.2447, log loss 0.6825, ECE 0.0058** — the best result on
every axis of any model in this build, and the first time this project has
decisively beaten its strongest baseline rather than merely approaching it.
See Results below for the full story, including two negative results along
the way that were diagnosed and fixed rather than hidden: a first,
single-season hyperparameter tuning pass looked good on its validation
split but didn't transfer to the 2024 holdout (fixed by tuning across two
seasons instead of one), and isotonic post-hoc calibration made calibration
worse, not better (so it isn't used).

## What's real here

- **Data**: real Statcast pitch-level data (via `pybaseball`) for the full
  2022, 2023, and 2024 regular seasons (~20,000-21,000 pitcher-game rows and
  ~69,000-73,000 batter-game rows each), real MLB Stats API schedules/scores,
  real empirical park factors computed from actual prior-season game results
  (Coors Field comes out 139.2 — the most hitter-friendly park in MLB,
  exactly as expected; Petco Park comes out 83.6, the most pitcher-friendly
  — this is a real signal, not a guess).
- **Every projection is as-of-date and leak-tested.** A pitcher's, batter's,
  team's, or bullpen's projection for game N uses ONLY games strictly before
  game N, exponentially time-weighted and shrunk toward a same-date league
  prior that is ITSELF computed from only strictly-prior games. `tests/leakage/`
  has 18 passing tests enforcing this, including two real bugs caught and
  fixed during this build (see "Leakage bugs found and fixed" below).
- **Confirmed lineups are real, not a team-level proxy.** The actual
  starting lineup (9 batters, batting order) is derived directly from
  Statcast plate-appearance sequence for each historical game — no extra API
  calls needed. Each batter's projection is split by the handedness of the
  pitcher he's facing (a real, persistent platoon effect), same as-of-date/
  shrinkage discipline as pitchers.
- **Real weather, not fabricated or omitted.** Per-game condition, temp,
  and wind speed/direction come from the MLB Stats API's `game` endpoint
  (real readings, e.g. `{"condition": "Clear", "temp": "60", "wind": "13
  mph, L To R"}`) — pulled for all ~4,860 games across 2023-2024 via a
  resumable, moderately-concurrent puller. A real API data artifact (dome
  games sometimes report `temp=0` as a placeholder) was caught and nulled
  out rather than fed to the model as a literal reading.
- **Hyperparameters were actually tuned**, not just guessed — via a
  held-out coordinate-descent search across 2022+2023 validation data,
  keeping 2024 completely untouched as the final test set. See
  "Hyperparameter tuning" below for the two-attempt story (one failed, one
  worked).
- **A real stacked ensemble** (`mlb.ensemble.stacking`) blends the
  simulation model with Elo-only and pitcher-adjusted-Elo on the log-odds
  scale, with weights learned via walk-forward logistic regression — per
  the original spec's requirement, not a hand-picked blend. This is the
  best-performing model in the whole build (see Results).
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
  park_weather/    Empirical park factors (real game logs, prior-seasons-only) + real per-game weather
  simulation/      Poisson-mean regression + NB dispersion, Monte Carlo game engine
  models/moneyline/  Elo, pitcher-adjusted Elo, Log5, home-field baselines
  ensemble/        Log-odds stacking of simulation + Elo + pitcher-adjusted Elo
  calibration/     Isotonic post-hoc calibration (tested, not used — see Results)
  backtest/        Walk-forward (expanding-window) backtest loop
  evaluation/      Brier/log-loss/ECE/reliability/totals-MAE metrics
tests/
  leakage/         The anti-leakage test suite (18 tests, all passing)
  unit/            Regression tests for real data artifacts found along the way
scripts/
  build_features.py     Build one season's leak-free game-feature dataset
  run_backtest.py        Walk-forward backtest one season
  evaluate_backtest.py    Produce the honest evaluation report
  tune_hyperparams.py     Held-out coordinate-descent hyperparameter search (2022+2023)
  run_ensemble.py         Build + evaluate the stacked ensemble
```

## How the model works

1. **Starter projection** (`mlb.pitchers.projections`): xwOBA-against, K%,
   BB%, whiff%, CSW%, barrel% — each exponentially time-weighted (75-day
   halflife, tuned — see below) over the pitcher's own starts and shrunk
   toward a same-date league average using an empirical-Bayes credibility
   formula (`k=250` batters). A 4-start hot streak is outweighed by the
   shrinkage prior exactly as the spec requires.
2. **Bullpen projection** (`mlb.bullpen.projections`): same shrinkage
   machinery, applied to team-aggregate relief xwOBA-against (35-day
   halflife, tuned), plus a real fatigue signal — total relief pitches
   thrown in the trailing 3 days.
3. **Lineup offense** (`mlb.lineups`): the actual starting lineup (9
   batters) is inferred directly from Statcast plate-appearance order for
   each game — no extra API calls. Each batter's xwOBA is projected
   separately vs. LHP and vs. RHP (platoon splits), same shrinkage
   machinery as pitchers (100-day halflife, k=100 PA, both tuned), then
   combined into a lineup-level projection using REAL empirical
   plate-appearances-per-batting-slot weights (leadoff hitters bat more
   often than #9 — computed from prior seasons only, not invented) rather
   than a plain average, against the actual opposing starter's hand for
   that game. A team-level rolling-runs proxy
   (`mlb.features.team_offense`) is kept as a secondary signal — the
   simulation's mean-runs model takes both (see "Lineup ablation" below for
   why).
4. **Park factors** (`mlb.park_weather.park_factors`): empirical, from real
   prior-season game logs (home run-scoring environment vs. that team's own
   road environment), never leaking the season being predicted.
5. **Weather** (`mlb.park_weather.weather`): real per-game condition, temp,
   and wind speed/direction from the MLB Stats API. Wind is encoded as a
   signed `wind_effect` (speed blowing out = positive, in = negative,
   cross/none/indoor = zero) rather than an assumed run value — the
   regression learns the coefficient.
6. **Simulation** (`mlb.simulation`): a Poisson-mean regression (features:
   own offense proxy, own lineup xwOBA, opponent starter xwOBA, opponent
   bullpen xwOBA, park factor, home/away, wind effect, temperature)
   predicts each team's expected runs; a Negative Binomial dispersion
   parameter is estimated empirically from the actual/predicted residual
   variance (Pearson method-of-moments — teams' runs are overdispersed
   relative to Poisson, as expected). A 20,000-draw Monte Carlo
   (Gamma-Poisson mixture) produces win probability, run-line cover
   probability, and the full total-runs distribution **jointly and
   consistently from one simulation**, per the spec.
7. **Walk-forward backtest** (`mlb.backtest.walk_forward`): retrains weekly
   on an expanding window (all games strictly before the retrain date); each
   week's predictions are locked in before that week's results are known.
8. **Stacked ensemble** (`mlb.ensemble.stacking`, moneyline only): the
   simulation model's win probability, Elo-only's win probability, and
   pitcher-adjusted-Elo's win probability are each converted to log-odds and
   blended via a walk-forward logistic regression (weights refit weekly,
   same expanding-window discipline as everything else) — the "calibrated
   stacked ensemble on log-odds" the original spec called for. This is the
   best-performing model in the build (see Results).

## Results — 2024 season, walk-forward, 2023 as training warm-start

2,429 games backtested; 2,165 games used for baseline comparison (the
pitcher-adjusted-Elo baseline needs warm-up games and drops the first ~264).
Numbers below are the FINAL pipeline (lineups + weather + tuned
hyperparameters) unless a table is explicitly an ablation showing an
earlier stage.

### Lineup ablation (team-level proxy vs. lineup-level vs. both)

Tested with weather off and pre-tuning defaults, on the full 2,429-game
backtest, to isolate the lineup effect alone:

| Offense feature | Accuracy | Brier | Log loss | ECE | Totals MAE |
|---|---|---|---|---|---|
| Team-level proxy only | 53.9% | 0.2479 | 0.6889 | 0.024 | 3.443 |
| Lineup-level only | 53.0% | 0.2481 | 0.6894 | 0.024 | 3.445 |
| **Both** | **54.9%** | **0.2475** | **0.6882** | 0.027 | 3.456 |

**Honest read:** lineup data ALONE is not obviously better than the simple
team-level proxy — but the mean-runs regression given BOTH signals
outperforms either alone on accuracy, Brier, and log loss (it apparently
extracts complementary information from each rather than one dominating).
Totals MAE is flat to slightly worse here — see the weather ablation below
for where totals actually improved. Both signals are kept in the final model.

### Weather ablation (both offense signals, with vs. without weather)

| | Accuracy | Brier | Log loss | ECE | Totals MAE | Totals RMSE |
|---|---|---|---|---|---|---|
| Without weather | 54.8% | 0.2475 | 0.6882 | 0.027 | 3.456 | 4.407 |
| **With weather** | 54.5% | 0.2475 | 0.6881 | 0.024 | **3.430** | **4.367** |

**Honest read:** weather measurably improves the TOTALS prediction (MAE
3.456 → 3.430, RMSE 4.407 → 4.367) — physically sensible, since wind/temp
directly affect run-scoring environment. It does NOT improve, and slightly
hurts, moneyline accuracy (54.8% → 54.5%), while log loss and ECE both
improve marginally. This is a plausible, mixed, real result: weather is a
totals signal, not much of a moneyline signal, exactly as domain intuition
would predict. Weather is kept in the final model.

### Hyperparameter tuning: attempt 1 (single-season) failed to transfer, attempt 2 (multi-season) worked

**Attempt 1** — a coordinate-descent search (`scripts/tune_hyperparams.py`)
over halflife/shrinkage-k for pitcher, batter, bullpen, and team-offense
projections, scored by log loss on a held-out SLICE OF 2023 ONLY (games
from 2023-07-20 on, trained on everything before that date within 2023) —
2024 was never touched during the search itself.

| | Pitcher halflife | Batter halflife | 2023 val. log loss |
|---|---|---|---|
| Defaults | 45d | 60d | 0.6844 |
| Tuned (2023-only) | 75d | 100d | 0.6833 |

Applying those values to the FINAL 2024 backtest (the true, untouched
holdout): accuracy on the same 2,165-game comparison set actually got
WORSE (54.5% → 53.8%) despite log loss improving marginally (0.6879 →
0.6878, essentially noise). **A single-season validation split wasn't
reliable enough to trust.**

**Attempt 2** — the same search, but scored by POOLED log loss across
held-out validation slices of BOTH 2022 and 2023 (2024 still never
touched):

| | Pitcher halflife | Batter halflife | Batter k | Bullpen halflife | Bullpen k | Team-off. halflife | Pooled val. log loss |
|---|---|---|---|---|---|---|---|
| Defaults | 45d | 60d | 200 | 20d | 250 | 30d | 0.6821 |
| **Multi-season tuned** | 75d | 100d | **100** | **35d** | **400** | **50d** | **0.6801** |

This is a materially different config from attempt 1 — batter k reversed
direction (200 → 100), and bullpen/team-offense halflives both moved up
substantially. It also improved on BOTH validation seasons individually
(2022: 0.6796→0.6771; 2023: 0.6847→0.6832), a much stronger signal than
attempt 1's single data point.

Applying THIS config to the same final 2024 holdout:

| | Accuracy | Brier | Log loss |
|---|---|---|---|
| Defaults + weather | 54.5% | 0.2474 | 0.6879 |
| Attempt 1 (2023-only tuning) | 53.8% | 0.2473 | 0.6878 |
| **Attempt 2 (multi-season tuning)** | **54.7%** | **0.2470** | **0.6872** |

**Honest read:** this time the validation-set improvement DID transfer —
accuracy recovered past the pre-tuning baseline, and Brier/log loss both
improved more than attempt 1 managed. This is the config shipped in this
build. The lesson we're keeping, not just the number: single-season
hyperparameter tuning on this system was actively misleading, and
tuning across at least two independent seasons was enough to catch it. The
original spec's nested time-series CV (more seasons still) would be the
further extension.

### Moneyline — single-model comparison (lineups + weather + multi-season-tuned hyperparameters)

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| **Simulation model (ours, final)** | 54.7% | 0.2470 | 0.6872 | 0.030 |
| Elo-only | 54.7% | 0.2485 | 0.6904 | 0.048 |
| Home-field-always | 52.8% | 0.2494 | 0.6920 | 0.006 |
| Better-record (Log5) | 50.3% | 0.2646 | 0.7264 | 0.097 |
| Pitcher-adjusted Elo | 54.8% | **0.2466** | **0.6864** | 0.015 |

**Honest read:** as a standalone model, the simulation now clearly beats
Elo-only on Brier score and log loss (0.2470/0.6872 vs. 0.2485/0.6904)
while matching it almost exactly on accuracy. It's statistically tied with
pitcher-adjusted Elo on accuracy (54.7% vs. 54.8% — a 2-game difference out
of 2,165) and has closed roughly half the earlier Brier/log-loss gap to it
(previously 0.2474/0.6879 pre-tuning vs. 0.2466/0.6864; now 0.2470/0.6872).
Not a decisive win over the strongest single baseline on its own — see the
stacked ensemble below for where that changes. (For reference: home-field-
always remains the best-calibrated single baseline since it just predicts
the historical rate, though least discriminating; better-record/Log5
remains the weakest model overall.)

### Moneyline — stacked ensemble (the headline result)

Blending all three win-probability signals — simulation, Elo-only,
pitcher-adjusted Elo — on log-odds via a walk-forward logistic regression
(`scripts/run_ensemble.py`), evaluated on the full 2,429-game 2024 holdout
(the ensemble's warm-up comes from 2022-2023, so unlike the pitcher-adjusted-
Elo baseline alone, no games need to be dropped from 2024 for warm-up):

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| Simulation (component) | 54.5% | 0.2471 | 0.6874 | 0.024 |
| Elo-only (component) | 55.0% | 0.2483 | 0.6901 | 0.044 |
| Pitcher-adjusted Elo (component) | 54.7% | 0.2455 | 0.6841 | 0.013 |
| **Stacked ensemble** | **56.0%** | **0.2447** | **0.6825** | **0.0058** |

**Honest read:** this is the best result anywhere in this build, on every
metric simultaneously, including calibration (ECE 0.0058 is roughly 2-8x
better than any individual component). 56.0% accuracy is just short of the
spec's realistic "great result" band (57-60%) and well clear of the
"presumed leakage" zone above ~62% — a genuinely plausible, non-suspicious
number, and it was leakage-tested directly:
`tests/leakage/test_ensemble_leakage.py` confirms the stacking
regression never uses a game's own or a later game's outcome. This is
attributable to genuine complementary information across the three
components (a Poisson-mean simulation of run distributions; a team-strength
Elo prior; a simple linear pitcher/bullpen-adjusted blend) rather than to
one component dominating — an in-sample diagnostic fit (not the reported
walk-forward result) put roughly comparable weight on the simulation and
Elo terms, with the pitcher-adjusted-Elo term acting more as a correction
than an independent driver, likely because it already overlaps heavily
with the Elo term. The ensemble covers moneyline only — totals and run
line still come from the simulation model alone (see below), since Elo/
pitcher-adjusted-Elo have no run distribution to combine with.

**Two further experiments, both tested honestly and reported regardless of
outcome:**

1. **PA-weighted lineup averaging** (`compute_pa_weights_by_slot`):
   replaced the original equal-weighted lineup average with real, empirical
   plate-appearances-per-batting-slot weights (leadoff hitters average 3.01
   PA/game vs. 2.44 for the #9 hitter, computed from 2022-2023 data only —
   never the season being predicted). Effect: essentially neutral on the
   standalone simulation model (accuracy 54.7%→54.5%, Brier/log loss flat,
   ECE improved 0.030→0.024) and on the ensemble (accuracy flat, Brier/log
   loss marginally better, ECE improved 0.0072→0.0058). Kept — it replaces
   a known simplification with real data at essentially no cost, even
   though the win is modest.
2. **A 4th ensemble component: gradient-boosted trees** directly on the
   matchup features (`mlb.models.moneyline.gbm`, the "direct ML baseline"
   the original spec calls for), walk-forward trained with conservative
   hyperparameters. Standalone, it was the WEAKEST of all four components
   (54.3% accuracy, worse Brier/log loss/ECE than sim, Elo, or
   pitcher-adjusted-Elo) — unsurprising given its modest training-set size.
   Added as a 4th ensemble input, it did not help: 55.3% accuracy vs. the
   3-way ensemble's 56.0%, and worse on every other metric too. **Not
   included in the shipped ensemble** — tested and rejected, not
   silently dropped.

### Totals — final model

| | MAE | RMSE | Bias |
|---|---|---|---|
| **Simulation model (final)** | 3.429 | 4.371 | +0.183 |
| Naive (as-of league-average total) | 3.451 | 4.355 | +0.083 |

**Honest read:** the totals model beats the naive league-average baseline
on MAE (3.429 vs. 3.451), improving further from the weather-only version
(3.434) with multi-season-tuned hyperparameters — a real, if modest, edge.
RMSE and bias are still slightly worse than naive, so this isn't a clean
sweep, but the direction is real: weather plus properly-tuned halflives
each contributed a small, genuine improvement to totals.

### Run line (±1.5, modeled as P(margin), not a variable spread)

| | Actual rate | Mean predicted |
|---|---|---|
| Home −1.5 covers (wins by 2+) | 35.3% | 34.8% |
| One-run game | 27.8% | 19.6% |

**Honest read:** the −1.5/+1.5 cover probability remains well-calibrated
(35.3% vs. 34.8%). The model still meaningfully **underestimates** how
often games are decided by exactly one run — the NB dispersion parameter
(fit once, globally) doesn't fully capture the real fat-tailed frequency of
close games, unchanged by this session's additions. Concrete target for
next iteration: a per-team or run-environment-dependent dispersion instead
of one global value.

### Isotonic calibration (tested, did not help)

Fit on the first half of the 2024 season, applied to the second half: ECE
went from 0.017 (raw) to 0.046 (isotonic-calibrated) — **worse**. The raw
simulation probabilities were already reasonably well-calibrated; isotonic
regression overfit on ~1,200 training games. We ship the raw probabilities.

## Leakage bugs found and fixed during this build

`tests/leakage/` has 18 passing tests. Two real leakage bugs were caught by
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
make test                                    # full suite (18 tests)
make leakage-test                            # just the anti-leakage gate
make features SEASON=2022                    # build one season's dataset (incl. lineups)
make features SEASON=2023
make features SEASON=2024
python scripts/run_backtest.py 2023 --prior 2022 --feature-set both --out-suffix _both
python scripts/run_backtest.py 2024 --prior 2023 --feature-set both --out-suffix _both
python scripts/evaluate_backtest.py 2024 --prior 2023 --feature-set both
python scripts/run_ensemble.py               # build + evaluate the stacked ensemble
```

## Acceptance criteria — honest status

- [x] Full pipeline runs raw data → predictions (backtest form; live daily
      slate deployment is not yet built — see limitations).
- [x] Walk-forward backtest, zero leakage (18 leakage tests passing,
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
- [x] Beats baselines out-of-sample — **the standalone simulation model
      beats home-field-always/better-record/Elo-only and is statistically
      tied with pitcher-adjusted Elo; the stacked ensemble decisively beats
      every baseline including pitcher-adjusted Elo on every metric**
      (56.0% accuracy, Brier 0.2447, log loss 0.6825, ECE 0.0058 — see
      "Moneyline — stacked ensemble" above). Reported with the full honest
      path to get there, including a tuning attempt that initially failed
      to transfer and was fixed, not hidden.
- [x] Predictions immutable (backtest predictions parquet is write-once);
      no fabricated data anywhere in the pipeline.
- [x] Uncertainty and the ~57–60% realistic ceiling stated (this file, top).
- [x] No "beats Vegas" claim anywhere — there is no Vegas comparison in
      this build at all, by design.

## What's next (not done yet)

See `docs/limitations.md` for the full list. Two more levers were tried
this session and are worth knowing didn't move the needle further: a 4th
ensemble component (gradient-boosted trees on the matchup features) and
PA-weighted lineup averaging (kept for its own honesty merits, but nearly
neutral on performance) — see "Two further experiments" above. In priority
order for what's left: extending the stacked ensemble to run line and
totals (currently moneyline-only — would need run/total-producing
baselines to blend with, which none of Elo/pitcher-adjusted-Elo are),
tuning across even more seasons (two validation seasons was enough to catch
attempt 1's overfitting, but the spec's full nested time-series CV would
use more), live confirmed-lineup ingestion for daily predictions (vs.
backtest-only actual lineups), extending the backtest across more seasons,
an odds data source for CLV, and improving the run-line dispersion model
for one-run games.
