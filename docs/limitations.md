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

- **RESOLVED (ingestion only): live confirmed-lineup fetching is now real.**
  `mlb.lineups.live` queries the MLB Stats API's `game` endpoint for a
  given game_pk and returns real probable pitchers (available far ahead of
  game time) and the confirmed batting order (9 real batter IDs/names, only
  once actually posted — `liveData.boxscore.teams.<side>.battingOrder`
  stays empty before that). Verified against real live data on 2026-09-14's
  actual slate: correctly reported `lineups_confirmed: False` with real
  probable-pitcher names for not-yet-started games, and real, correct
  batting orders (actual current MLB players) for that day's completed
  games. Unit-tested with mocked API responses
  (`tests/unit/test_live_lineups.py`) so the "not posted yet" vs. "posted"
  distinction is verified deterministically, not just by eyeballing live
  output once.

  **RESOLVED: the full live prediction pipeline is now built and running.**
  `src/mlb/daily/predict.py` (entry point `scripts/predict_today.py`)
  predicts every game on a real, current slate. It closed the two gaps
  above: (1) 2025 full-season and 2026 season-to-date Statcast
  pitcher/batter/lineup/weather data were pulled, and `game_features_*`
  built for every season now used (2019, 2021-2026); (2) each as-of-date
  projection for "right now" is computed by appending one synthetic,
  zero-valued row dated today into the historical series for that entity
  and running the *exact same* leak-tested `add_asof_*` function — no new,
  separately-tested projection logic. Because those functions only ever
  read strictly-prior rows (the same guarantee `tests/leakage/` already
  verifies), this is leak-safe by construction, not by new testing.
  Demonstrated end-to-end on 2026-09-14's real 10-game slate: predictions
  for every scheduled game, with `weather_available: False` and
  `*_lineup_confirmed: False` honestly flagged where the underlying data
  genuinely wasn't available (see §2 below and the note under this list).
  Scheduled locally via `launchd` (`scripts/run_daily_predictions.sh`,
  9:15am daily) — needs no API key, so none of the CLV job's key-exposure
  considerations apply.
- **RESOLVED: batters are now PA-weighted, not averaged equally.**
  `compute_pa_weights_by_slot` computes real empirical plate-appearances
  per batting-order slot from prior seasons only (leadoff hitters average
  3.01 PA/game vs. 2.44 for the #9 hitter, from 2022-2023 data — a real,
  stable structural fact about lineup construction, not invented). Effect
  was small: essentially neutral on the standalone simulation model's
  accuracy, with a modest calibration improvement on the final ensemble
  (ECE 0.0072 → 0.0058). Kept for the honesty win even though the
  performance win was modest.
- **RESOLVED: a batter missing the exact opposing-hand split now falls
  back to his own projection vs. whichever hand he DID face that same game**
  (a starter always batted at least once, so this recovers nearly every
  case) before falling back further to the flat neutral prior (0.31) — only
  for the residual handful of games where even that's unavailable. Effect
  was small, as expected given only ~0.4% of lineup slots were affected:
  standalone simulation accuracy +0.2pp, ensemble accuracy +0.1-0.2pp,
  Brier/log loss marginally better, ECE essentially a wash. A real
  methodological improvement with a correspondingly small measured effect.
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
whatever point the API snapshot was taken), not a pre-game FORECAST.
Confirmed directly against the live API for not-yet-played games:
`gameData.weather` comes back `{}` (empty) — no forecast is available at
any lead time this build checked. **This is a real, structural gap, not
something the live pipeline works around**: `src/mlb/daily/predict.py` uses
a neutral fallback (`wind_effect=0`, `temp_f_filled=72`, i.e. treated like
a dome game) for every live prediction and sets `weather_available: False`
on every row, rather than fabricating or estimating a forecast. Concretely,
this means live totals predictions lose the ~0.026 MAE improvement weather
was shown to provide in backtesting (above) until a forecast API is
integrated — a known, quantifiable accuracy cost, stated plainly rather
than silently absorbed.

## 3. RESOLVED: CLV is now measured, with real closing-line data, on a real sample

We first investigated free sources (documented for posterity below) and
found none usable. The user then provided a paid The Odds API key, which
changed the picture.

**What was checked and ruled out first**, so this isn't re-litigated
blindly later:
- **sportsbookreviewsonline.com** — the historically-canonical free source
  for MLB closing-line spreadsheets. Its archive pages now 404 / the
  domain has been repurposed as a generic affiliate site. Gone.
- **GitHub scrapers** (e.g. `ArnavSaraogi/mlb-odds-scraper`) — checked
  directly via the GitHub API: scraper CODE against the now-broken site
  above, not a committed dataset, despite a search-engine summary claiming
  otherwise. Verify data-availability claims directly; don't trust a
  search summary.
- **Kaggle** (`christophertreasure/major-league-baseball-vegas-data`) —
  covers ~2012-2021, missing most of this project's 2022-2024 evaluation
  window, and needs Kaggle credentials this environment lacked.
- **The Odds API free tier** — real, but 500 credits/month with historical
  calls at 10x cost — nowhere near enough for backtesting.

**What actually worked**: a paid The Odds API key (first key provided was
deactivated — cancellation or failed payment; a second key worked, 20,000
credits). Historical odds cost real credits (~33-38 per game for
moneyline+run-line+totals+de-vig at 3-market resolution), and a full 2024
season (2,429 games) would cost far more than the available budget — so
`scripts/pull_historical_odds.py` pulls a **systematic sample of 498 games
(~20.5% of the 2024 season)**, evenly spread across the season for
unbiased temporal coverage, each with a snapshot taken 8 minutes before
that specific game's own first pitch (a genuine closing line, not a stale
mid-day price) from a real board of 9-14 US sportsbooks (FanDuel,
DraftKings, BetMGM, Bovada, and others), de-vigged to a fair consensus
probability. The API key lives in `.env` (gitignored, never committed);
raw odds data lives in `data/raw/odds/` (also gitignored, like all raw
data in this project).

**Result** (`scripts/compute_clv.py`, comparing our FINAL shipped models —
the moneyline/run-line ensembles, and the simulation re-queried at the
market's own total line — against the real closing-line market, on the
matched sample):

| Market | Model Brier | Market Brier | Model log loss | Market log loss |
|---|---|---|---|---|
| Moneyline (n=498) | 0.2410 | 0.2382 | 0.6749 | 0.6693 |
| Run line, home-favored subset (n=292) | 0.2440 | 0.2404 | 0.6818 | 0.6738 |
| Totals, re-simulated at market's line (n=498) | 0.2500 | 0.2502 | 0.6945 | 0.6935 |

**Honest read:** the market is sharper than our model on moneyline and run
line (a modest gap, not a blowout — MLB closing lines are among the
sharpest in sports, exactly as the spec warns) — no edge is demonstrated
there. On totals, the model is in a statistical dead heat with the market
(0.2500 vs. 0.2502 Brier — a difference smaller than sample noise) — which
is precisely the "totals near market breakeven" outcome the spec names as
a good, realistic result. This is the honest, sample-sized answer to the
spec's "decisive and humbling benchmark," not a full-season claim: run-line
CLV further excludes 206 of 498 games (home +1.5 / away-favored games)
because our saved run-line output isn't directly comparable to that side
of the market without re-deriving from the full margin distribution — a
real scope limit, stated rather than papered over with a mismatched
number.

Extending this to full-season coverage, or to 2021-2023 for a
walk-forward CLV trend over time, is mechanical but would need
substantially more odds-API budget (roughly 2,429 games/season x ~35
credits = ~85,000 credits for one full season at 3-market resolution) than
the ~4,100 remaining after this pull.

**UPDATE: a local scheduled job now grows this sample daily.**
`scripts/run_daily_odds_pull.sh`, wired to a macOS `launchd` agent (daily,
9am local), calls `scripts/pull_historical_odds.py --daily-batch 15` —
walking the full 2024 season in date order, skipping games already
pulled, adding ~15 new real closing-line games per run, and stopping
itself with a safety margin before the key's credits run out (never spends
a key to exactly zero). This was deliberately built as a LOCAL scheduled
job rather than a cloud routine (the platform's cloud routines have no
secret-injection mechanism and cannot be deleted, only disabled — pasting
a live paid API key into a cloud routine's config would have left it
permanently embedded with no way to fully remove it). The key lives only
in this machine's `.env`, sourced fresh by the job each run, never
committed, never leaving the machine.

## 4. Backtest coverage: 2019, 2021-2024 pulled (2024 evaluated), not the full 2015+ history

The walk-forward backtest in this build covers the full 2024 season
(2,429 games, evaluated) and 2023 (2,430 games, also backtested — used both
as 2024's training pool and, via its own 2022-warm-started walk-forward run,
as ensemble/tuning validation data). 2019, 2021, and 2022 were each pulled
in full (pitchers, batters/lineups, weather) specifically to support
multi-season hyperparameter tuning (§9) and the ensemble's warm-up — 2019
was added as a 4th tuning-validation season; 2020 was deliberately skipped
(a 60-game pandemic-shortened season is a poor validation signal). Extending
to 2015+ is mechanical (`make features SEASON=<year>` for each year, then
include earlier seasons in `--prior`), but was out of scope for the time
budget of this build. Statcast data (the pitcher-peripheral foundation)
only exists from 2015 onward regardless.

## 5. Umpire tendencies, catcher framing splits, travel/rest beyond bullpen fatigue

Named in the spec as real signals; not implemented. Catcher framing is
available via `pybaseball.statcast_catcher_framing` and would be a
straightforward addition to the feature set.

## 6. Run-line and totals are read off the SAME simulation as moneyline

This is intentional (the spec calls for a jointly-consistent simulation
rather than three separately-fit models), not a limitation — but it does
mean an error in the run-environment mean model propagates to all three
markets simultaneously rather than being independently correctable. UPDATE:
each market now ALSO has an independent second signal available
(`mlb.models.total.direct`, `mlb.models.runline.direct`) for comparison and
(for run line) blending — see §10 below.

## 10. Extending stacking to run line (adopted) and totals (tried, rejected)

Following the moneyline ensemble's success, the same approach was tried for
the other two markets — a second, differently-shaped model (not derived
from the Poisson simulation) blended with the simulation's own prediction.

**Run line**: `mlb.models.runline.direct` is a logistic regression
predicting P(home −1.5 covers) directly from the matchup features, blended
with the simulation's run-line probability via the same log-odds stacking
as moneyline. Result: Brier 0.2256→0.2236, log loss 0.6431→0.6389, ECE
0.0134→0.0083 (best of all three: sim, direct, ensemble) — **adopted**.
Raw accuracy held roughly flat (64.7%→64.7%), but accuracy is a weak metric
for this market specifically: home covers −1.5 only 35.3% of the time, so
"always predict no-cover" alone would already score ~65% "accuracy" with
zero skill. Brier/log loss/calibration are what matter here, consistent
with this project's whole selection philosophy, and all three improved.

**Totals**: `mlb.models.total.direct` is a Ridge regression predicting the
total directly from the matchup features. Standalone, it's the best single
totals predictor in the build (MAE 3.414, beating both the simulation's
3.428 and the naive baseline's 3.451) — a genuinely interesting finding in
its own right, suggesting the simulation's Poisson-mean regression has
room to improve. But blending it with the simulation via walk-forward
Ridge-regularized linear stacking made things WORSE (MAE 3.454, worse than
either component alone) — confirmed with regularization specifically to
rule out simple meta-model overfitting as the cause. **Not adopted.** The
simulation's own total remains the shipped output, both because it's not
the worst option here and because it stays jointly consistent with
moneyline and run line from one model (the direct Ridge model has no run
distribution to offer those markets). The standalone Ridge result is kept
as a concrete lead for improving the simulation's own mean-runs regression.

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
weather, and hyperparameter tuning, was statistically tied with (and later,
after further tuning, ahead of on accuracy but still behind on Brier/log
loss) pitcher-adjusted Elo.

It's resolved by NOT picking one model. `mlb.ensemble.stacking` blends the
simulation model, Elo-only, and pitcher-adjusted Elo on log-odds via a
walk-forward logistic regression (`scripts/run_ensemble.py`), and the
result decisively beats every component on every metric: 56.2% accuracy,
Brier 0.2447, log loss 0.6825, ECE 0.0081 (1-5x better calibrated than any
single component). See `README.md` §"Moneyline — stacked ensemble" for the
full table and `tests/leakage/test_ensemble_leakage.py` for the leakage
verification.

**A 4th ensemble component was tried and rejected, honestly:** a
gradient-boosted-trees model (`mlb.models.moneyline.gbm`) trained
walk-forward directly on the matchup features — the "direct ML baseline"
the original spec calls for. Standalone it was the weakest of all four
components (55.0% accuracy, worst Brier/log loss/ECE of the group,
plausibly because a modest few-thousand-game training set isn't enough for
an unconstrained tree ensemble to beat the more structured models). Added
as a 4th ensemble input, it made things worse, not better (55.5% accuracy
vs. the 3-way ensemble's 56.2%, worse on every metric) — not included in
the shipped ensemble.

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
- Remaining likely further gains: tuning across more than two seasons (§9),
  a smarter fallback for batters missing an exact platoon-split match (§1),
  and more diverse ensemble components beyond the GBM attempt above (e.g. a
  hierarchical Bayesian pitcher/batter model).

## 9. Hyperparameter tuning: four attempts, converging to a stable plateau

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
holdout, actually delivered: accuracy recovered to 54.5% (from 53.8%) and
Brier/log loss improved further (0.2471/0.6874, vs. attempt 1's
0.2473/0.6878).

The THIRD attempt added 2021 as a third validation season (pulling its
full Statcast/lineup/weather data), starting the search from attempt 2's
winner rather than from scratch. Pooled log loss improved further
(0.6777→0.6772 across all 3 validation seasons), and every halflife that
moved (pitcher 75→110 days, batter 100→140 days, team-offense 50→70 days)
moved FURTHER IN THE SAME DIRECTION as attempt 2 rather than reversing —
every other parameter (pitcher k, batter k, bullpen halflife/k,
team-offense k) was confirmed unchanged. Applied to the 2024 holdout:
accuracy improved again, to 55.4%, with Brier/log loss essentially flat
(0.2471/0.6873).

A FOURTH pass (`scripts/tune_hyperparams_4season_check.py`) added 2019 as
a fourth validation season (2020 deliberately excluded — a 60-game
pandemic-shortened season is a poor validation signal), but rather than
re-running the full 8-parameter grid (diminishing value once 3 seasons
already agreed on 5 of 8 parameters), it targeted only the three halflives
that had trended upward every round: pitcher, batter, team-offense. Result:
pitcher_halflife extended FURTHER STILL (110→150 days), but
batter_halflife (140) and team_offense_halflife (70) both PLATEAUED — their
further-extended candidates (180, 100) did not beat the 3-season values.
Applied to the 2024 holdout, this made essentially no practical difference
(accuracy 55.6%→55.4%, Brier/log loss flat, well within noise) even though
the pooled 4-season validation metric ticked up slightly. This is the
config shipped in this build, and the plateau on 2 of 3 trending
parameters — after 3 consecutive rounds of them moving in lockstep — is a
genuinely reassuring sign that the search has found a real, stable region
rather than chasing noise indefinitely.

The lesson worth keeping: on a system this noisy, tuning on one validation
season was actively misleading (attempt 1), two seasons was enough to
produce a result that actually transferred (attempt 2), three seasons gave
a further, consistent-direction refinement rather than a reversal (attempt
3), and a fourth season showed the process converging — most parameters
holding steady, only one still moving, and even that one with a negligible
practical effect (attempt 4). That progression IS the evidence that the
hyperparameters now reflect a real, stable signal rather than noise, and a
reasonable point to stop this particular line of investigation. The
original spec's full nested time-series CV (even more seasons, or a
non-greedy joint search) would be the natural further extension for
someone with more compute budget to spend here — not yet built.

## 11. Out-of-sample validation on 2025 — a genuinely untouched season (what went right, what didn't, and a rejected fix)

Every number reported above through 2024 came from seasons this build
either tuned on (2019, 2021-2023) or used as its primary validation season
(2024). 2025 was pulled only afterward, to support live predictions — it
had never influenced a single hyperparameter, ensemble weight, or design
decision. Running the exact same walk-forward backtest and evaluation on
2025 (`scripts/run_backtest.py 2025 --prior 2024`, `scripts/evaluate_backtest.py`,
`scripts/validate_2025_holdout.py`) is the closest thing this project has
to a true, un-gamed test of whether the reported 2024 numbers reflect a
real, generalizing signal or were partly luck.

**What went right.** Every core component transferred with almost no
degradation:

| Model | 2024 (reported) | 2025 (untouched holdout) |
|---|---|---|
| Simulation | acc 55.4%, Brier 0.2470 | acc 54.2%, Brier 0.2469 |
| Elo-only | acc 55.0%, Brier 0.2483 | acc 54.7%, Brier 0.2484 |
| Pitcher-adjusted Elo | acc 54.9%, Brier 0.2455 | acc 55.5%, Brier 0.2450 |
| Home-field-always | acc 52.8%, Brier 0.2494 | acc 53.5%, Brier 0.2489 |
| Better-record (Log5) | acc 50.3%, Brier 0.2646 | acc 51.8%, Brier 0.2580 |

Brier scores land within 0.0001-0.0007 of their 2024 values across every
model — as close to "identical" as 2,186-2,430-game single-season samples
allow. Run-line calibration held up too: 2025's actual home -1.5 cover
rate (35.6%) is nearly identical to 2024's (35.3%), and the model's mean
predicted rate (35.1%) is close on both. **This is the headline finding:
the 2024 results were not a fluke — the model's real-world performance is
stable across an independent season it never touched.**

The production ensemble-fitting method specifically validated well: fitting
the final stacking weights ONCE on 2024's genuine walk-forward-out-of-sample
logits (exactly what `src/mlb/daily/predict.py` does for live predictions —
see "Live daily predictions" in the README) and applying those fixed
weights to 2025 gave Brier 0.2441 / accuracy 56.5% / ECE 0.0139 — matching
or slightly *beating* the original 2024 ensemble numbers (Brier 0.2447 /
56.3%). A separate check — re-running the backtest's own within-season
walk-forward ensemble refit (weekly retraining, `min_training_games=200`)
on 2025 alone — did notably worse (Brier 0.2458, ECE 0.0304) and even
slightly underperformed its own best individual component
(pitcher-adjusted Elo, Brier 0.2450) that season. The takeaway: a stable
set of weights learned from a full prior season's genuine OOS performance
generalizes to a new season better than trying to re-learn weights from
scratch as that new season's data trickles in — which is exactly why the
live pipeline is built the way it is, and this result is now the
justification, not just a design guess.

**A totals red herring, checked and ruled out.** 2025's totals MAE (3.591)
looked worse than 2024's (3.425) at first glance — worth flagging honestly
rather than burying. But 2025's real run-scoring environment was more
variable that season (runs/game std 4.594 vs 2024's 4.312, computed
directly from `data/raw/schedule`), which mechanically inflates MAE for
*any* method, including a naive one. Recomputing the same as-of
league-average naive baseline used for the reported 2024 number (3.451) on
2025 gives 3.633 — the model still beats it, by a similar-to-slightly-larger
margin (0.042 vs 0.026 runs) than in 2024. **Conclusion: not a regression,
a season-wide variance shift that a same-year baseline comparison catches
and a same-value comparison across years would have missed.**

**What's still wrong, confirmed for a third straight season.** The
one-run-game underestimate flagged in the Results section is real and
persistent, not a 2024 artifact: pooling 2019/2021-2025 (14,577 real
games), the true rate of exactly-one-run finals is 28.1%, and it lands at
27.8% (2023), 27.8% (2024), and 29.4% (2025) individually — remarkably
stable — while this build's simulation predicts only ~19.6-19.7% in every
one of those seasons. Two independent Gamma-Poisson team-run draws simply
don't produce enough one-run games; the leading real-world explanation is
strategic bullpen usage (a team protecting a small lead pitches its best
reliever, compressing what an independent-scoring model would render as a
blowout into a narrow finish) — a genuine within-game dynamic this
game-level (not play-by-play) simulation has no way to represent.

**A fix was attempted and rejected — an honest negative result, not a
silent abandonment.** A post-hoc "close-game compression" was implemented
in `simulate_game`: with probability `p`, take a decisive (\|margin\|>=2)
simulated game and pull its margin in to the closest value consistent with
that game's own total-runs parity (1 if odd, 2 if even), holding the total
runs AND the winner exactly fixed — by construction this cannot move
`home_win_prob`, `mean_total`, or the total-runs distribution, only
run-line-adjacent quantities. Calibrated on 2023 alone (`p=0.28`, via a
grid search matching 2023's own actual one-run rate) and validated purely
out-of-sample on 2025:

- One-run-game calibration clearly improved: predicted rate 19.6%→28.1%
  (actual: 29.4%), Brier 0.2167→0.2074, log loss 0.6317→0.6053.
- But the actual traded run-line market got clearly **worse**: Brier
  0.2273→0.2298, and ECE nearly quadrupled, 0.0123→0.0474.

The mechanism: the original (uncompressed) model's aggregate run-line
calibration was already good — but through compensating errors across
adjacent margin buckets, not because every bucket was individually
correct. Uniformly pulling mass out of every decisive-game bucket to fix
the one-run bucket broke that cancellation and made the metric that
actually matters (the traded run-line market) worse, even though the
diagnostic statistic it targeted got better. **Rejected and reverted** —
`src/mlb/simulation/engine.py`'s `simulate_game` ships unchanged. This is
the same category of result as the rejected GBM ensemble member and the
rejected Ridge-regularized totals-stacking fix (§10): a plausible-sounding
improvement that real out-of-sample measurement disproved before it
shipped. A future fix for the one-run-game gap should preserve the
existing margin distribution's correct aggregate shape rather than
uniformly redistributing it — e.g., a smarter, bucket-aware recalibration,
or a mechanistic bullpen-usage feature in the run-environment model itself
— neither attempted here.
