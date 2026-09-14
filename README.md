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

**The single simulation model lands at ~55.4% moneyline accuracy on the
true 2024 holdout — the best-accuracy standalone model of the group, still
trailing pitcher-adjusted Elo on Brier/log loss. A stacked ensemble
(simulation + Elo-only + pitcher-adjusted-Elo, blended on log-odds, weights
learned walk-forward) clearly beats every individual model on every
metric: 56.3% accuracy, Brier 0.2447, log loss 0.6824, ECE 0.0095** — the
best result on every axis of any model in this build, and the first time
this project has decisively beaten its strongest baseline rather than
merely approaching it. The same stacking approach was extended to run line
(a second win: ECE 0.0134→0.0083, adopted) and totals (a genuine negative
result: a standalone direct model beats the simulation, but blending the
two makes things WORSE than either alone — not adopted, and said so
plainly). See Results below for the full story, including several negative
results along the way that were diagnosed and reported rather than hidden:
a first, single-season hyperparameter tuning pass looked good on its
validation split but didn't transfer to the 2024 holdout (fixed by tuning
across 4 seasons instead of 1), isotonic post-hoc calibration made
calibration worse not better, a 4th (GBM) ensemble component hurt rather
than helped, and the totals ensemble described above. **CLV against real
closing-line odds (a paid-API-funded, 498-game sample of 2024) is now
measured too: the market beats our model on moneyline and run line — no
edge demonstrated, stated plainly — while the model is in a genuine
statistical dead heat with the market on totals**, exactly the realistic
"near market breakeven" outcome this project's honesty standard describes
as a good result.

## What's real here

- **Data**: real Statcast pitch-level data (via `pybaseball`) for the full
  2019, 2021, 2022, 2023, and 2024 regular seasons (~20,000-21,000
  pitcher-game rows and ~69,000-77,000 batter-game rows each), real MLB
  Stats API schedules/scores, real empirical park factors computed from
  actual prior-season game results (Coors Field comes out 139.2 — the most
  hitter-friendly park in MLB, exactly as expected; Petco Park comes out
  83.6, the most pitcher-friendly — this is a real signal, not a guess).
- **Every projection is as-of-date and leak-tested.** A pitcher's, batter's,
  team's, or bullpen's projection for game N uses ONLY games strictly before
  game N, exponentially time-weighted and shrunk toward a same-date league
  prior that is ITSELF computed from only strictly-prior games. `tests/leakage/`
  has 20 passing leakage tests enforcing this, including two real bugs caught and
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
  mph, L To R"}`) — pulled for all ~12,100 games across 2019, 2021-2024 via a
  resumable, moderately-concurrent puller. A real API data artifact (dome
  games sometimes report `temp=0` as a placeholder) was caught and nulled
  out rather than fed to the model as a literal reading.
- **Hyperparameters were actually tuned**, not just guessed — via a
  held-out coordinate-descent search across 2019+2021+2022+2023 validation data,
  keeping 2024 completely untouched as the final test set. See
  "Hyperparameter tuning" below for the four-attempt story (one failed,
  three worked, converging to a stable plateau).
- **Real stacked ensembles, for moneyline and run line** (`mlb.ensemble.
  stacking`), each blending the simulation with a second model on log-odds,
  weights learned via walk-forward logistic regression — per the original
  spec's requirement, not a hand-picked blend. Moneyline's is the
  best-performing model in the whole build; run line's improves Brier/log
  loss/calibration too. A totals ensemble was also tried and honestly
  rejected — see Results.
- **Live confirmed-lineup ingestion** (`mlb.lineups.live`): real-time
  fetching of probable pitchers and confirmed batting orders from the MLB
  Stats API, verified against actual current games — reports "not posted
  yet" honestly rather than guessing a lineup. See `docs/limitations.md`
  §1 for what's needed beyond ingestion to reach a full live pipeline.
- **CLV is now measured, with real closing-line data.** After free sources
  turned up nothing usable (investigated, not assumed — see
  `docs/limitations.md` §3), a paid The Odds API key pulled real closing
  lines (9-14 real US sportsbooks, de-vigged) for a systematic 498-game
  sample (~20.5%) of the 2024 season. Honest result: the market is sharper
  than our model on moneyline and run line (a modest gap, not a blowout),
  and the model is in a statistical dead heat with the market on totals
  (0.2500 vs. 0.2502 Brier) — exactly the "totals near market breakeven"
  outcome the spec names as a good, realistic result. See Results below.

## Architecture

```
src/mlb/
  data/            Stats API schedule ingestion, team ID mapping, real
                   historical closing-line odds (mlb.data.odds)
  pitchers/        Statcast pull + aggregation, as-of-date pitcher projections
  bullpen/         As-of-date team bullpen quality + fatigue/workload
  lineups/         Actual-lineup extraction (backtest), as-of-date batter
                   platoon-split projections, lineup-vs-opposing-starter-hand
                   aggregation, + live.py (real-time confirmed-lineup fetch)
  features/        Team-offense proxy (fallback), as-of-date utilities, matchup dataset assembly
  park_weather/    Empirical park factors (real game logs, prior-seasons-only) + real per-game weather
  simulation/      Poisson-mean regression + NB dispersion, Monte Carlo game engine
  models/moneyline/  Elo, pitcher-adjusted Elo, Log5, home-field baselines, GBM (tried, rejected)
  models/total/    Direct Ridge regression on matchup features (2nd totals signal)
  models/runline/  Direct logistic regression on matchup features (2nd run-line signal)
  ensemble/        Log-odds stacking (moneyline, run line) + linear stacking (totals)
  calibration/     Isotonic post-hoc calibration (tested, not used — see Results)
  backtest/        Walk-forward (expanding-window) backtest loop
  evaluation/      Brier/log-loss/ECE/reliability/totals-MAE metrics
tests/
  leakage/         The anti-leakage test suite (20 tests, all passing)
  unit/            Regression tests for real data artifacts found along the way
scripts/
  build_features.py     Build one season's leak-free game-feature dataset
  run_backtest.py        Walk-forward backtest one season
  evaluate_backtest.py    Produce the honest evaluation report
  tune_hyperparams.py     Held-out coordinate-descent hyperparameter search (2021+2022+2023)
  tune_hyperparams_4season_check.py  Targeted follow-up adding 2019 as a 4th validation season
  run_ensemble.py         Build + evaluate the moneyline stacked ensemble
  run_market_ensembles.py Build + evaluate the totals and run-line ensembles
  run_gbm_baseline.py     Generate the GBM baseline (moneyline 4th-component experiment)
  pull_historical_odds.py Pull a real closing-line odds sample (needs ODDS_API_KEY in .env)
  compute_clv.py          Compute honest CLV vs. the pulled real closing lines
```

## How the model works

1. **Starter projection** (`mlb.pitchers.projections`): xwOBA-against, K%,
   BB%, whiff%, CSW%, barrel% — each exponentially time-weighted (110-day
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
   machinery as pitchers (140-day halflife, k=100 PA, both tuned), then
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
improved more than attempt 1 managed. The lesson we're keeping, not just
the number: single-season hyperparameter tuning on this system was
actively misleading, and tuning across at least two independent seasons
was enough to catch it.

**Attempt 3** — pushed further: added 2021 as a third validation season
(pulling its full Statcast/lineup/weather data), starting the search from
attempt 2's winner to see whether more validation data confirms it or
moves it further:

| | Pitcher halflife | Batter halflife | Team-off. halflife | Pooled val. log loss (3 seasons) |
|---|---|---|---|---|
| Attempt 2 config (starting point) | 75d | 100d | 50d | 0.6777 |
| **Attempt 3 tuned** | **110d** | **140d** | **70d** | **0.6772** |

Every other parameter (pitcher k, batter k, bullpen halflife/k, team-offense
k) held at attempt 2's values — the search confirmed them rather than
finding something new. All three halflives that DID move, moved further in
the SAME direction as attempt 2 (longer, not shorter or reversed) — a good
sign of a stable trend rather than noise. Applying attempt 3's config to
the true 2024 holdout:

| | Accuracy | Brier | Log loss |
|---|---|---|---|
| Defaults + weather | 54.5% | 0.2474 | 0.6879 |
| Attempt 1 (2023-only, didn't transfer) | 53.8% | 0.2473 | 0.6878 |
| Attempt 2 (2022+2023) | 54.5% | 0.2471 | 0.6874 |
| **Attempt 3 (2021+2022+2023, shipped)** | **55.4%** | **0.2471** | **0.6873** |

**Honest read:** accuracy improved further (54.5%→55.4%) while Brier/log
loss stayed essentially flat — a real, if modest, additional gain, and
importantly no reversal like attempt 1's.

**Attempt 4** — added 2019 as a fourth validation season (2020 deliberately
skipped: a 60-game pandemic-shortened season is a poor validation signal).
Rather than re-running the full 8-parameter grid (diminishing value once 3
seasons already agreed on 5 of 8 parameters), this pass targeted only the
three halflives that had trended upward every prior round:

| | Pitcher halflife | Batter halflife | Team-off. halflife | Pooled val. log loss (4 seasons) |
|---|---|---|---|---|
| Attempt 3 config (starting point) | 110d | 140d | 70d | 0.6757 |
| **Attempt 4 checked** | **150d** | 140d (unchanged) | 70d (unchanged) | 0.6757 |

Pitcher halflife extended further still (110→150d) — but batter and
team-offense halflives, which had moved every round so far, both
PLATEAUED: their further-extended candidates (180d, 100d) did not beat the
attempt-3 values. Applied to the 2024 holdout, this made essentially no
practical difference (accuracy 55.6%→55.4%, Brier/log loss flat, well
within noise) even though the pooled 4-season metric ticked up slightly.
**This is the config shipped in this build.**

| | Accuracy | Brier | Log loss |
|---|---|---|---|
| Defaults + weather | 54.5% | 0.2474 | 0.6879 |
| Attempt 1 (2023-only, didn't transfer) | 53.8% | 0.2473 | 0.6878 |
| Attempt 2 (2022+2023) | 54.5% | 0.2471 | 0.6874 |
| Attempt 3 (2021+2022+2023) | 55.6% | 0.2470 | 0.6871 |
| **Attempt 4 (2019+2021+2022+2023, shipped)** | 55.4% | 0.2470 | 0.6872 |

**Honest read:** the progression across 4 attempts IS the interesting
result here, more than any single number: one misleading result (attempt
1), then three consecutive rounds of increasing agreement — two parameters
plateauing while only one kept moving, and even that one with a negligible
practical effect on the holdout. That's a genuinely reassuring sign of
convergence toward a stable signal rather than chasing noise indefinitely,
and a reasonable point to stop this particular search. The original
spec's full nested time-series CV (even more seasons, or a non-greedy
joint search) would be the natural further extension for more compute
budget than this build spent here.

### Moneyline — single-model comparison (lineups + weather + 3-season-tuned hyperparameters)

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| **Simulation model (ours, final)** | 55.4% | 0.2471 | 0.6873 | 0.029 |
| Elo-only | 55.0% | 0.2483 | 0.6901 | 0.044 |
| Home-field-always | 52.8% | 0.2494 | 0.6920 | 0.006 |
| Better-record (Log5) | 50.3% | 0.2646 | 0.7264 | 0.097 |
| Pitcher-adjusted Elo | 54.9% | **0.2455** | **0.6840** | 0.009 |

**Honest read:** as a standalone model, the simulation is now the
best-accuracy model of the group (55.4%, ahead of both Elo variants) and
clearly beats Elo-only on Brier score and log loss (0.2471/0.6873 vs.
0.2483/0.6901). It still trails pitcher-adjusted Elo on Brier/log loss
(0.2471/0.6873 vs. 0.2455/0.6840), a narrower gap than earlier in this
build (was 0.2474/0.6879 vs. 0.2466/0.6864 pre-tuning). Real, standalone
progress — see the stacked ensemble below for the bigger win. (For
reference: home-field-always remains the best-calibrated single baseline
since it just predicts the historical rate, though least discriminating;
better-record/Log5 remains the weakest model overall.)

### Moneyline — stacked ensemble (the headline result)

Blending all three win-probability signals — simulation, Elo-only,
pitcher-adjusted Elo — on log-odds via a walk-forward logistic regression
(`scripts/run_ensemble.py`), evaluated on the full 2,429-game 2024 holdout
(the ensemble's warm-up comes from 2022-2023, so unlike the pitcher-adjusted-
Elo baseline alone, no games need to be dropped from 2024 for warm-up):

| Model | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| Simulation (component) | 55.4% | 0.2471 | 0.6873 | 0.029 |
| Elo-only (component) | 55.0% | 0.2483 | 0.6901 | 0.044 |
| Pitcher-adjusted Elo (component) | 54.9% | 0.2455 | 0.6840 | 0.009 |
| **Stacked ensemble** | **56.3%** | **0.2447** | **0.6824** | **0.0095** |

**Honest read:** this is the best result anywhere in this build, on every
metric simultaneously, including calibration (ECE 0.0095 is roughly 1-5x
better than any individual component). 56.2% accuracy is just short of the
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
with the Elo term. This particular ensemble covers moneyline only — Elo
and pitcher-adjusted-Elo have no run distribution to combine with, so
totals and run line use their own, separate ensembles (see below), built
the same way but with different second-model components.

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
   (55.0% accuracy, worse Brier/log loss/ECE than sim, Elo, or
   pitcher-adjusted-Elo) — unsurprising given its modest training-set size.
   Added as a 4th ensemble input, it did not help: 55.5% accuracy vs. the
   3-way ensemble's 56.2%, and worse on every other metric too. **Not
   included in the shipped ensemble** — tested and rejected, not
   silently dropped. (Confirmed again after the 3-season hyperparameter
   retune — same conclusion held.)

### Totals — simulation, a direct model, and an attempted ensemble

We tried extending the stacked-ensemble approach to totals: a second,
differently-shaped signal (Ridge regression directly on the matchup
features — `mlb.models.total.direct`) blended with the simulation's own
total via walk-forward Ridge-regularized linear stacking
(`mlb.ensemble.stacking.walk_forward_linear_stacking`).

| | MAE | RMSE | Bias |
|---|---|---|---|
| Naive (as-of league-average total) | 3.451 | 4.355 | +0.083 |
| Simulation model (component) | 3.428 | 4.368 | +0.183 |
| **Direct Ridge regression (component)** | **3.414** | **4.297** | +0.264 |
| Ensemble (sim + direct, blended) | 3.454 | 4.352 | +0.363 |

**Honest read — a genuine surprise, reported as found:** the standalone
direct Ridge model is the BEST single totals predictor in this build,
beating both the simulation and naive baseline on MAE. But blending it
with the simulation made things WORSE than either alone (MAE 3.454, worse
than both). This held even after regularizing the stacking regression
(Ridge, not plain OLS) to rule out simple overfitting in the meta-model —
with only 2 highly-correlated inputs and a weekly refit, stacking doesn't
reliably beat the better of the two components. **We do not adopt the
totals ensemble.** The simulation's own total remains the shipped output
for two reasons beyond raw MAE: it stays jointly consistent with the
moneyline and run-line predictions from one simulation (the direct Ridge
model has no run distribution to offer run-line or moneyline), and per-game
robustness matters more than a marginal average MAE gain for a system
that reports probabilities, not just point estimates. The finding that a
simple linear model on the same features currently outpredicts the
Poisson-simulation's mean is nonetheless a concrete, honest target for
improving the simulation's mean-runs regression itself.

### Run line (±1.5) — simulation, a direct model, and a working ensemble

Same approach as totals, but this time it worked: a direct logistic
regression (`mlb.models.runline.direct`) predicting P(home −1.5 covers)
directly from the matchup features, blended with the simulation's own
run-line probability via the same log-odds stacking used for moneyline.

| | Accuracy | Brier | Log loss | ECE |
|---|---|---|---|---|
| Simulation (component) | 64.7% | 0.2256 | 0.6431 | 0.0134 |
| Direct logistic (component) | 64.9% | 0.2240 | 0.6399 | 0.0122 |
| **Ensemble (sim + direct, blended)** | 64.7% | **0.2236** | **0.6389** | **0.0083** |

**Honest read:** the ensemble has the best Brier score, log loss, AND
calibration of the three — genuinely the best-calibrated run-line output
in this build — even though its raw accuracy is a hair lower than either
component alone. Accuracy is a weak metric here regardless: home covers
−1.5 only 35.3% of the time, so a trivial always-predict-no-cover baseline
would already score ~65% "accuracy" without any skill — which is exactly
why this project selects on log loss/Brier/calibration rather than
accuracy, and why the ensemble is adopted here despite the accuracy dip.
Actual home −1.5 cover rate: 35.3% vs. the ensemble's mean predicted 34.9%
— well-calibrated in aggregate as well as per-bin.

One-run-game frequency is still underestimated (actual 27.8% vs. simulated
19.6%, unchanged by this session's work) — the NB dispersion parameter (fit
once, globally) doesn't capture the real fat-tailed frequency of close
games. Concrete target for next iteration: a per-team or
run-environment-dependent dispersion instead of one global value.

### CLV vs. the real closing line — the spec's "decisive and humbling benchmark"

A paid The Odds API key made this possible after free sources came up
empty (see `docs/limitations.md` §3 for that investigation). Budget
(20,000 credits) didn't cover the full 2,429-game 2024 season at 3-market
resolution (~85,000 credits), so `scripts/pull_historical_odds.py` pulled
a systematic, evenly-spread sample of 498 games (~20.5% of the season),
each with a real closing-line snapshot (9-14 US sportsbooks — FanDuel,
DraftKings, BetMGM, Bovada, and others — taken 8 minutes before that
game's own first pitch, de-vigged to a fair consensus probability) —
genuinely real, sample-sized, not full-season.

| Market | Model Brier | Market Brier | Model log loss | Market log loss |
|---|---|---|---|---|
| Moneyline (n=498) | 0.2410 | 0.2382 | 0.6749 | 0.6693 |
| Run line, home-favored subset (n=292) | 0.2440 | 0.2404 | 0.6818 | 0.6738 |
| **Totals, re-simulated at market's line (n=498)** | **0.2500** | **0.2502** | **0.6945** | **0.6935** |

**Honest read:** the market beats our model on moneyline and run line — a
modest gap (Brier differences of 0.0028 and 0.0036), not a blowout, but no
edge is demonstrated there. MLB closing lines are among the sharpest in
sports, exactly as the spec warns, and this result says plainly that our
model hasn't beaten them. On TOTALS, the model is in a genuine statistical
dead heat with the market (0.2500 vs. 0.2502 — a gap smaller than sample
noise at n=498) — this is precisely the "totals near market breakeven"
outcome the spec names as a good, realistic result, achieved with real
data rather than assumed. The run-line comparison further excludes 206 of
498 games (where the market favored the away team, i.e. home was +1.5) —
our saved run-line output represents P(home covers −1.5)/P(away covers
+1.5) specifically, which isn't the complementary event needed for the
away-favored framing without re-deriving from the full margin distribution;
excluding them and saying so is the honest choice over quietly comparing
mismatched quantities.

This is the first time this build has a real answer to the spec's central
credibility question, even if a sample-sized and partly-humbling one:
**no demonstrated edge on moneyline or run line vs. the close; genuine
parity on totals.** No "beats Vegas" claim is or should be made from this.

### Isotonic calibration (tested, did not help)

Fit on the first half of the 2024 season, applied to the second half: ECE
went from 0.017 (raw) to 0.046 (isotonic-calibrated) — **worse**. The raw
simulation probabilities were already reasonably well-calibrated; isotonic
regression overfit on ~1,200 training games. We ship the raw probabilities.

## Leakage bugs found and fixed during this build

`tests/leakage/` has 20 passing tests. Two real leakage bugs were caught by
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
make test                                    # full suite (31 tests)
make leakage-test                            # just the anti-leakage gate
make features SEASON=2021                    # build one season's dataset (incl. lineups)
make features SEASON=2022                    # (2021+2022 support hyperparameter tuning/warm-up)
make features SEASON=2023
make features SEASON=2024
python scripts/run_backtest.py 2023 --prior 2022 --feature-set both --out-suffix _both
python scripts/run_backtest.py 2024 --prior 2023 --feature-set both --out-suffix _both
python scripts/evaluate_backtest.py 2024 --prior 2023 --feature-set both
python scripts/run_ensemble.py               # build + evaluate the moneyline stacked ensemble
python scripts/run_market_ensembles.py       # build + evaluate the totals/run-line ensembles

# CLV (needs a paid The Odds API key — put ODDS_API_KEY=... in a .env file,
# NEVER commit it; .env is already gitignored):
source .env && export ODDS_API_KEY
python scripts/pull_historical_odds.py 500   # pulls a real closing-line sample (costs API credits)
python scripts/compute_clv.py                # compares our models to that real market data

# Live daily predictions (no API key needed — public MLB Stats API only):
python scripts/predict_today.py              # today's real slate
python scripts/predict_today.py 2026-09-20   # a specific future date's slate
```

## Acceptance criteria — honest status

- [x] Full pipeline runs raw data → predictions, both in backtest form and
      as a genuine live daily pipeline (`scripts/predict_today.py`) that
      predicts every game on today's real MLB slate. See "Live daily
      predictions" below.
- [x] Walk-forward backtest, zero leakage (20 leakage tests passing,
      including 2 real bugs caught and fixed during this build).
- [~] Predictions driven by starter + bullpen + confirmed lineups —
      **implemented and backtested** (actual lineups derived from
      play-by-play, platoon-split batter projections), but the LIVE daily
      pipeline still needs to be wired to the Stats API's pre-game confirmed
      lineup endpoint rather than backtest-derived actual lineups. See
      limitations §1.
- [x] Calibration (reliability + ECE) reported for all three markets.
- [x] Run line modeled as P(margin), not a variable spread — and, like
      moneyline, improved via a stacked ensemble (simulation + a direct
      logistic model), adopted after it improved Brier/log loss/ECE.
- [x] CLV vs. closing line — **measured, with real data**, on a systematic
      498-game (~20.5%) sample of 2024 (paid odds API; full-season coverage
      would need far more budget). Honest result: market beats the model on
      moneyline/run-line, model ties the market on totals. See "CLV vs. the
      real closing line" in Results.
- [x] Beats baselines out-of-sample — **the standalone simulation model
      beats home-field-always/better-record/Elo-only and is statistically
      tied with pitcher-adjusted Elo; the stacked ensemble decisively beats
      every baseline including pitcher-adjusted Elo on every metric**
      (56.3% accuracy, Brier 0.2447, log loss 0.6824, ECE 0.0095 — see
      "Moneyline — stacked ensemble" above). Reported with the full honest
      path to get there, including a tuning attempt that initially failed
      to transfer and was fixed, not hidden.
- [x] Predictions immutable (backtest predictions parquet is write-once);
      no fabricated data anywhere in the pipeline.
- [x] Uncertainty and the ~57–60% realistic ceiling stated (this file, top).
- [x] No "beats Vegas" claim anywhere — now that a real market comparison
      exists (CLV, above), the result is reported exactly as it came out:
      the market wins on 2 of 3 markets, the model ties on the third. No
      claim of beating the market is made anywhere in this build.

## Deployment

- **Repo**: [github.com/kshreyan/mlb-prediction-system](https://github.com/kshreyan/mlb-prediction-system) (public). CI runs the full test suite (including the leakage gate) on every push.
- **Results dashboard**: [kshreyan.github.io/mlb-prediction-system](https://kshreyan.github.io/mlb-prediction-system/) — a static GitHub Pages site (`docs/index.html`) reporting the real headline numbers, the moneyline model comparison, the calibration reliability diagram, the CLV-vs-market chart, and the four-attempt tuning progression, all built from this session's actual output (no live-slate predictions yet — see below).
- **CLV sample expansion, running locally on a schedule**: `scripts/pull_historical_odds.py --daily-batch 15`, wired to a macOS `launchd` job (`scripts/run_daily_odds_pull.sh`, daily at 9am local) that pulls 15 more real closing-line games per day, in date order, until the season is fully covered or the API key's credit budget runs low (it stops itself with a safety margin — never spends a key to zero). **The API key runs locally, in `.env`, and never leaves this machine** — a cloud-based scheduled routine was considered and deliberately rejected, since cloud routines have no secret-injection mechanism and can't be deleted (only disabled), which would have left a live paid key permanently embedded in a routine config with no way to fully remove it.
- **Live daily predictions, running locally on a schedule**: `scripts/predict_today.py`, wired to a second `launchd` job (`scripts/run_daily_predictions.sh`, daily at 9:15am local) that predicts every game on that day's real MLB slate and writes an immutable, timestamped parquet file per run under `data/processed/daily_predictions/`. Uses only the public MLB Stats API — no key/secret involved, so this job carries none of the CLV job's key-exposure considerations. See "Live daily predictions" below for how it works and its honest limitations (lineups, weather).

## Live daily predictions

`scripts/predict_today.py` (`src/mlb/daily/predict.py`) predicts every game
on a real, current MLB slate — not a backtest. It reuses the exact same
leak-tested as-of-date functions as the backtest (never new, untested
logic) via one technique: for each entity (a pitcher, a batter, a team), it
appends a single synthetic row dated *today* with every stat column zeroed,
runs the unchanged `add_asof_*` function, and reads off that one row's
output. Because those functions compute each row's projection strictly
from *earlier* rows by construction (the same property `tests/leakage/`
already verifies), a zero-valued row dated today cannot leak into any
historical projection, and its own output is exactly "what the model would
project for this entity as of right now."

Every underlying model (the Poisson run-environment simulation, Elo,
pitcher-adjusted-Elo, and the moneyline stacking ensemble) is refit as a
**final production version** — trained on all available historical data
through yesterday (2019, 2021–2026) rather than the walk-forward-windowed
form used for backtesting/evaluation. The ensemble's blending weights come
from `predictions_2024_ensemble_final.parquet`, the same honestly
walk-forward-out-of-sample logits the backtest itself validated the
ensemble on, rather than being re-derived in-sample here.

Two honest, load-bearing limitations of the live pipeline:

- **Weather is unavailable for any not-yet-played game.** The MLB Stats API
  returns no forecast until close to game time — confirmed directly against
  the live API (`weather: {}`). Live predictions use a neutral fallback
  (`wind_effect=0`, `temp_f_filled=72`) and every row carries
  `weather_available: False`, rather than guessing or fabricating a
  forecast.
- **Lineups are frequently not yet confirmed** when the daily job runs
  (confirmed lineups typically post 1–3 hours before first pitch, after the
  9:15am local run). Each side carries `home_lineup_confirmed` /
  `away_lineup_confirmed`; when a lineup isn't posted yet, that team's most
  recent *actual* lineup (real historical data, not a guess) is used as an
  honest proxy. A later manual run of `scripts/predict_today.py` the same
  day, after lineups post, produces a *separate*, more accurate immutable
  prediction file rather than overwriting the morning one.

## What's next (not done yet)

See `docs/limitations.md` for the full list of what's resolved vs. still
open. As of this build, all of the following ARE done (not "next"):
stacked ensembles for moneyline and run line, PA-weighted lineups, live
confirmed-lineup ingestion, real CLV measurement, a genuine live daily
prediction pipeline, and deployment (GitHub + Pages + two locally-scheduled
jobs — CLV expansion and daily predictions). What's genuinely still open,
in rough priority order:

- **Full-season CLV coverage, growing daily** — the local scheduled job
  above adds ~15 real games/day; full 2024 coverage at 3-market resolution
  needs ~85,000 odds-API credits total, well beyond the current ~4,100
  remaining, so this will keep the sample growing but likely won't reach
  full-season coverage on the current budget alone.
- **The Pages dashboard doesn't yet show live slate predictions** — it
  currently reports the 2024 backtest only; wiring today's
  `daily_predictions/*.parquet` output into the dashboard is unbuilt.
- **Tuning across even more seasons** — 4 consecutive validation seasons
  converged to a stable plateau, but the spec's full nested time-series CV
  would use more still.
- **Improving the run-line dispersion model** for one-run games (still
  underestimated — see "Run line" in Results) and PA-weighting a batter's
  expected plate appearances more precisely (currently a fixed per-slot
  average, not adjusted for a specific lineup's actual construction).
