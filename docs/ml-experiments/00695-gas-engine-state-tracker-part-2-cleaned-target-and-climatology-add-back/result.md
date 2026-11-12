---
id: "00695"
slug: gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back
title: >-
  Does a demand forecast trained on D8's reading with the gas engine removed, plus a climatology
  of the engine added back at forecast time, forecast D8's metered apparent power better than a
  forecast trained on the raw reading?
theme: gross-demand-simulation
also_relevant_to:
  - production-reference-and-best-guess
  - lagged-power-and-error-correction
  - differentiable-physics-embedded-generation
status: done
kind: other
protocol: demand-v2
size: >-
  Anything else (a new pipeline step and a new target definition, no new data). Stored: a cleaned
  D8 series and a tracker state per fold, 3 seeds of a neural network per arm and fold (state
  files), and one prediction set per arm, all private under data/ml-experiments/<id>-<slug>/. More
  than one design is defensible (which forecaster, which state cleans the target, what is added
  back), so the choices are fixed below. Callers: this folder's scripts, which import experiment
  694's tracker (e00694_trackers.py, e00694_common.py), experiment 677's step-5 chain, and
  studies.eval_protocol.
hypothesis: >-
  At D8, removing the inferred gas-engine output from the training target and from the lagged-power
  inputs of the demand neural network, and adding the engine back at forecast time as a
  climatological distribution over the number of running units (passed through the apparent-power
  fold), lowers the error of the forecast of D8's metered apparent power, against the same neural
  network trained on the raw reading.
prediction: >-
  Written after a short look at the stored forecasts and before any fit. (P1) On all scored rows at
  D8, arm A1 (causal cleaning) lowers CRPS by 0.3 to 1.0 points of capacity against arm A0 (raw),
  and lowers the normalised MAE of the median by 0.5 to 1.5 points. (P2) The gain in the months
  2026-03 to 2026-06, when the engine fades, is at least as large in CRPS as the gain in 2025-11 to
  2026-02, and sits mostly in the q01 to q25 pinball loss. If P2 holds strongly, C1 loses power:
  the monthly standard deviation of D8's CRPS difference between two stored sister arms is 0.047
  points, so a uniform gain of 0.4 is resolvable, but a difference of about 0.6 between the
  engine months and the fade months makes the t interval on C1 cross zero. (P3) Arm AH (cleaning
  with hindsight) beats A1 by under 0.2 points, so the causal tracker is not the limit. (P4) At the
  host Primary with an injected engine, the perfect-cleaning arm WT beats the raw arm W0, and W1
  recovers at least half of that gain. (P5) Adding back the tracked state at the valid time (an
  unattainable bound) lowers CRPS by 2 points or more, far more than P1, so the larger prize is in
  predicting the engine. (P6) No other Primary moves, by the per-Primary rule below.
decision_rule: >-
  Fixed before any fit. Metric of record: CRPS of the forecast of metered apparent power at D8, in
  points of effective capacity (capacity fixed at 10.379 MVA for every arm), leads 24 to 240 hours
  counted from the 12:00 UTC contract origin, origins 2025-11-01 to 2026-06-30, valid times before
  2026-07-01. Control: arm A0 refitted here on all 8 folds with seeds 0 to 2. Smallest gain worth
  porting: 5.5% of the refitted A0's own D8 CRPS on the scored rows, which is about 0.4 points
  (the stored step 9a scores 7.10 at D8 on experiment 677's shared rows). The change adds a fitted
  tracker with per-era parameters, a daily smoothing pass, a second target definition, and an
  add-back step, for one Primary, and a false engine flag biases the forecast silently; a smaller
  gain does not repay that. The threshold is about 0.05 points on the headline of the MVA-only
  eight, 2.5 times the 0.02 that demand-v2 asks of a new fitted step. Three planned contrasts are
  read at 98.3% (Bonferroni for three) from a paired bootstrap over whole origin months (20,000
  resamples, 170 draws per tail), with a paired t interval at 98.3% on the 8 monthly differences
  beside each, because a month bootstrap under-covers at 8 months. C1: A1 minus A0, D8, all
  months. C4: W1 minus W0 at the host. C5: WT minus W0 at the host. C2 (C1 on 2025-11 to 2026-02)
  and C3 (C1 on 2026-03 to 2026-06) have 4 month blocks each, sit outside the family, and are
  reported at 95% as descriptive. A contrast "passes" when both of its intervals lie below zero,
  and "fails" otherwise. A Primary other than D8 "moves" when the 95% month-block interval of its
  paired difference (A1 minus A0) excludes zero and the absolute estimate exceeds that Primary's
  absolute CRPS difference between the stored sister arms 5w and 5wo of experiment 677 (which
  differ only in the injected shifts: 0.16 at D15, 0.11 at D22, 0.058 at D16). Verdict, first
  branch that applies. (1) `invalid` if a leakage test fails (weather after the origin, or a
  tracker state or training row that used a reading after its cut). (2) `failed` if gate G0 cannot
  be met after investigation. (3) `inconclusive` if gate G1 fails (no network is fitted), or if the
  3 seeds disagree on the sign of C1. (4) `supported` if C1 passes with an estimate at or beyond
  the threshold, C4 passes, the normalised MAE of the median at D8 does not rise beyond its 95%
  interval, and no other Primary moves for the worse. (5) `partly supported` if C1 passes with an
  estimate at or beyond the threshold and any other condition of branch 4 fails (the write-up
  names it; a C5 failure is recorded as "not supported at the host"). (6) `small gain` if C1 passes
  with an estimate under the threshold. (7) `partly supported` if C1 fails and exactly one of C2
  or C3 has its 95% interval below zero, labelled descriptive. (8) `not supported` in every other
  case, which is the default branch.
benchmark: >-
  Arm A0, refitted here: experiment 677's step-5 network at monthly refits (step 9a), seeds 0 to 2,
  on all 8 folds, as the paired control. The stored step 9a file,
  data/ml-experiments/00677-.../backtests/step5w_with_injected_sp1.parquet (3 seeds, no manifest,
  so experiment_store.read_predictions cannot read it), is the G0 reference only, with row keys
  asserted equal. Reference rows on the same half-hours: the stored Candidate (normal-running rows
  of primaries.parquet, experiment 667), the manual heuristic, ClimatologyForecaster on the raw
  reading, and BG2 and Primary demand research default 3 from experiment 677's stored reference
  scores where their rows overlap. Production does not run BG2.
row_set_id: >-
  backtest_year_2025_11_to_2026_10_daily_12utc, restricted to origins 2025-11-01 to 2026-06-30 and
  valid times before 2026-07-01 (the rows of backtest_year_2025_11_to_2026_06_daily_12utc), all 16
  Primaries, with D8 as the decision series
metric: >-
  CRPS (twice the mean pinball loss over 11 quantiles) and normalised MAE of the median, in points
  of effective capacity, both against metered apparent power; twCRPS; coverage and pinball loss at
  q01, q05, q95, and q99; by lead band, by engine-visible and fade months, by tracked engine state
  (on, off), on sunny middays of the fade months, and on cold winter nights; all imported from
  studies.eval_protocol. The avoidable-curtailment cost is not defined at an MVA-only Primary and
  is not reported.
verdict: not supported
best_arm: >-
  None. The raw arm A0 (CRPS 7.232 points of capacity at D8) is the best forecast; the cleaned arm A1
  scored 7.349, a difference of +0.117 (98.3% interval -0.070 to +0.412).
headline: >-
  Pre-registered branch 8, `not supported`. Cleaning D8's target and adding a calendar average of the
  engine back did not lower the forecast error: C1 is +0.117 points of capacity (worse), and the
  lower bound of -0.070 rules out any gain above about 0.07 points, against a threshold of 0.398
  and a predicted gain of 0.3 to 1.0. Limits: D8 only; leads 24 to 240 hours; origins 2025-11 to
  2026-06; 8 monthly folds, 3 seeds.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  Taking the gas engine out of D8's recorded demand before training, and adding the engine's
  typical calendar pattern back when forecasting, does not improve the forecast and is probably
  about 0.1 points of capacity worse, mostly in the months the engine fades. The tracker that
  separates the engine is not the limit: at a substation with an injected engine, perfect cleaning
  gained only 0.1. Knowing whether the engine runs at the forecast's valid time would be worth 1.4
  to 2.3 points, but that knowledge is not available at forecast time, so the next step is to
  forecast the engine's state from information available at the origin, not to refine the cleaning.
lessons: >-
  A calendar climatology cannot say which days the engine runs, so adding it back turns the
  forecast into a wide two-component mixture about 24 points apart that over-covers the low tail and
  raises the error on the rows where the engine runs. The paired design and the identity control
  worked: the raw arm reproduced experiment 677's stored step 9a to 1.6e-6 points, and the network
  with cleaning disabled reproduced the raw arm to within 1e-4 of capacity.
advice:
  - >-
      Do not clean D8's engine out of the training target and add a calendar climatology back; the
      cleaned arm was no better than the raw arm (C1, interval -0.070 to +0.412) and probably about
      0.1 points worse (post hoc seed-averaged interval +0.036 to +0.245).
  - >-
      Spend the next experiment on forecasting the engine's on or off state from origin-time
      information, because knowing the state at the valid time was worth 1.4 to 2.3 points as a
      bound (A1+state and AH+state, a bound and not a forecast).
  - >-
      Read a bound that uses the valid-time engine state as an upper limit only, and never as an arm
      of a verdict, because it reads the scored reading (the bound rows of the table).
  - >-
      Do not refine the tracker to improve this approach: perfect cleaning with a perfect
      climatology at the host gained 0.100 points, and the causal tracker matched it (C5, W1 minus WT).
  - >-
      Read a seed-averaged month bootstrap beside the library interval when a contrast has 3 seeds
      and 8 months, because the library draws one seed per resample and was 2.4 times wider (C1).
  - >-
      Fit the no-fold variant of the cleaning only with a model of the export branch, because the
      plain add-back of the unit output lifted the cleaned level 2.3 MVA too high on running rows
      (A1n minus A1, +0.334).
code: ml-experiments/00695-gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back/
---
**Taking a gas engine's output out of one substation's recorded demand before a neural network
learns from it, and adding the engine's typical output back when forecasting, did not improve the
forecast.** The substation, D8, has a gas engine that switches on and off, so its recorded power
jumps by up to about 50% of the substation's capacity. We used the on and off tracker of [experiment
694](../00694-gas-engine-state-tracker-part-1/result.md) to remove the engine from the neural
network's training data and from its recent-power inputs, refitted the network every month, and
added the engine back to each forecast as a recency-weighted average over the same time of day and
day type in the previous 12 weeks. We compared the result with the same network trained on the raw
reading, for 8 monthly refits from 2025-11 to 2026-06, with 3 random seeds each. The cleaned arm was
no better, and probably about 0.1 points of capacity worse. A point of capacity is one percentage
point of D8's effective capacity of 10.379 MVA. At another substation, with a real engine's record
injected, cleaning gained about 0.1 points and perfect cleaning gained no more, so the tracker is
not the limit. The forecast error of the version that knows the engine's state at the forecast's
valid time is 1.4 to 2.3 points lower, but no forecast has that knowledge.

**The verdict is `not supported`, by branch 8 of the pre-registered table.** The planned contrast C1
(cleaned arm A1 minus raw arm A0, CRPS at D8) is +0.117 points, with a 98.3% interval from -0.070 to
+0.412 and a t interval from -0.037 to +0.268. A gain needs both intervals below zero and an
estimate of at least 0.398 points, which is 5.5% of the raw arm's own score. Neither condition
holds. The prediction P1, a gain of 0.3 to 1.0 points, is ruled out, because the interval's lower
bound is -0.070.

**The decision interval is wide only because the library draws one seed per resample.** A month
bootstrap of the seed-averaged difference, which was not pre-registered and is **post hoc**, gives
+0.036 to +0.245 for C1 at 98.3%. The audit's script computed it, and a second calculation from the
8 monthly differences gives +0.036 to +0.243. The post hoc reading says the cleaned arm is reliably
a little worse, while the pre-registered intervals say it is not distinguishable from the raw arm.
The verdict stays branch 8 because the t interval, the pre-registered pair's second member, crosses
zero.

## Results

**The cleaned arms lose to the raw arm on D8's whole-year CRPS, and only the bounds, which cheat,
win.** Every row below is scored against the metered apparent power at D8 (or, for the host, the
injected reading), on leads 24 to 240 hours from the 12:00 UTC contract origin, origins 2025-11-01
to 2026-06-30, mean of 3 seeds. "MAE" is the normalised mean absolute error of the median forecast.
The rows marked bound read the scored reading, a hindsight fit, or the true injected output, so they
are not forecasts. The reference rows come from [experiment
677](../00677-building-up-the-candidate-demand-network-one-component-at-a-time/result.md) on its
80,688 shared half-hours at D8, a different row set from the 102,310 half-hours scored here, so they
are context and not a paired comparison.

| System | What it is | CRPS | MAE |
|---|---|---|---|
| A0 | Raw reading, the control | 7.232 | 13.252 |
| A1 | Causal cleaning plus the calendar add-back | 7.349 | 13.372 |
| A1n | A1 with the no-fold cleaning and a clamped signed add-back | 7.683 | 13.891 |
| AH (bound, not a forecast) | Hindsight cleaning to 2026-06-30 | 7.146 | 12.942 |
| A1+state (bound, not a forecast) | A1 plus the tracked state at the valid time | 5.819 | 10.386 |
| AH+state (bound, not a forecast) | AH plus the tracked state at the valid time | 4.878 | 8.830 |
| W0 | Host, raw injected reading | 4.489 | 7.447 |
| W1 | Host, causally cleaned plus add-back | 4.396 | 7.390 |
| WT | Host, raw network plus the true engine's calendar add-back | 4.389 | 7.401 |
| WT+truth (bound, not a forecast) | WT plus the true injected output at the valid time | 2.232 | 3.953 |
| Step 9a on 677's rows (reference) | Stored control | 7.100 | 13.002 |
| Candidate, monthly refits (reference) | Experiment 677's step 8 with injected shifts | 7.131 | 13.077 |
| Manual heuristic (reference) | Stored | 7.934 | 14.135 |
| Climatology on the raw reading (reference) | Stored | 8.581 | 16.189 |

![CRPS of the forecast at D8 by system, with bars for the bounds in orange](figure_1_d8_crps_by_system.svg)

**The contrasts show no gain at D8 and a small gain, within noise, at the host.** The planned
contrasts C1, C4, and C5 are read at 98.3%, a Bonferroni split of 5% over three. C2 and C3 are
descriptive and read at 95%. Each row gives the estimate, the interval from resampling whole origin
months with one seed per draw, and the t interval over the monthly differences. A negative value is
a gain.

| Contrast | Estimate | Bootstrap interval | t interval | Seeds |
|---|---|---|---|---|
| C1: A1 minus A0, D8 | +0.117 | -0.070 to +0.412 | -0.037 to +0.268 | +0.182, +0.062, +0.108 |
| C2: C1 in 2025-11 to 2026-02 (engine months) | +0.164 | -0.133 to +0.613 | -0.146 to +0.472 | +0.306, +0.115, +0.070 |
| C3: C1 in 2026-03 to 2026-06 (fade months) | +0.069 | -0.071 to +0.184 | +0.033 to +0.103 | +0.055, +0.007, +0.146 |
| C4: W1 minus W0, host | -0.093 | -0.203 to +0.021 | -0.231 to +0.043 | -0.087, -0.085, -0.108 |
| C5: WT minus W0, host | -0.100 | -0.203 to +0.062 | -0.224 to +0.027 | -0.119, -0.064, -0.116 |
| MAE of the median, A1 minus A0, D8 (95%) | +0.120 | -0.487 to +0.788 | -0.526 to +0.720 | +0.287, -0.019, +0.094 |

![The planned contrasts with both intervals](figure_3_contrasts.svg)

**The bounds, and every contrast that contains a bound, are limits and not forecasts.** AH minus A1
is -0.203 (95%, bootstrap -0.363 to -0.050; bound, not a forecast). A1+state minus A0 is -1.413
(-2.245 to -0.550; bound, not a forecast), AH+state minus A0 is -2.354 (-3.488 to -1.344; bound, not
a forecast), and WT+truth minus W0 is -2.257 (-2.583 to -1.915; bound, not a forecast). W1 minus WT
is +0.007 (-0.052 to +0.069), so the causally cleaned host arm equals the arm with perfect cleaning.
A1n minus A1 is +0.334 (+0.197 to +0.457).

**The cleaned arm is worse exactly where the engine runs.** On the rows where the hindsight tracker
flags the engine as running (35,302 rows per seed), CRPS is 8.453 for A0 and 9.002 for A1. On the
rows where it flags the engine as not running (67,008 rows per seed), CRPS is 6.588 for A0 and 6.478
for A1. The median forecast of A1 sits 12.9 points above the reading on running rows, against 10.0
for A0. A1's 90% band is 62 points wide against 51 for A0, and it over-covers the low tail: the
share of readings below the q05 forecast is 0.013 for A1 and 0.059 for A0, and below q01 is 0.002
and 0.012. A1 is slightly better on tracked-off rows, on sunny middays in the fade months (6.856
against 7.061), and in the q95 pinball loss (1.569 against 1.600). On cold winter nights A1 scores
7.593 against 7.137.

![CRPS at D8 by whether the engine is tracked as running, for each system](figure_2_d8_crps_by_engine_state.svg)

**At the host, cleaning gained about 0.1 points and perfect cleaning gained no more.** The host is
D4, chosen before any fit because its median reading is closest to D8's tracked-off median. A real
engine's record is injected into D4's reading. W1 minus W0 is -0.093 and WT minus W0 is -0.100, and
both intervals include zero. Most of the host's error is in the half-hours where the injected engine
runs (CRPS 11.7 for W0), and the add-back lowers it to 10.9 for W1 and 10.9 for WT.

![CRPS at the host by whether the injected engine is running](figure_4_host_crps_by_engine_state.svg)

**No other Primary moved.** The 95% month-block interval of A1 minus A0 includes zero at all 15
other Primaries. The largest estimates are +0.130 at D10 and +0.108 at D11 (sister-arm differences
0.005 and 0.037), and the intervals are -0.077 to +0.402 and -0.139 to +0.559. The placebos D22
(-0.158) and D2 (-0.003) did not move either.

## Why the add-back does not help

**A calendar average cannot say which days the engine runs.** The average of the tracker's state
over the previous 12 weeks, by half-hour of day and day type, gives the right share of running
engines (0.37 against 0.40 in the engine months). The add-back then turns each forecast into a
mixture of two components about 24 points of capacity apart, weighted by that share. On a day with
the engine off, the upper component is wrong, and on a day with it running, the lower component is
wrong. The audit found that the mixture widens the forecast and raises its median on both kinds of
row, by 2.9 points on tracked-on rows and 1.2 points on tracked-off rows, which fits the table
above. The window also lags the fade: in 2026-03 the average says the engine runs in 0.386 of
half-hours, against a hindsight share of 0.267, which fits C3 being positive.

**The tracker is not the limit, and the engine's state at the valid time is where the value is.** At
the host, W1 equals WT, so a perfect tracker with the same add-back would gain nothing more. The two
bounds that use the state at the valid time, A1+state and AH+state, lower the CRPS by 1.4 and 2.4
points. Neither is available at forecast time, because both read the reading being forecast. [A
short follow-on analysis of experiment 694](#what-the-next-experiment-should-answer) found no
horizon beyond about a day at which any input we can forecast beats a calendar climatology.

**A1n, the no-fold variant, is a mis-specified variant, and its result says nothing against A1.** On
running rows the cleaned level falls below the engine's output for 39% of the 2025-11 fold's rows
and 27% of the 2026-03 fold's rows. The no-fold rule then adds the engine's output to a reading that
was already folded, which lifts the cleaned mean on running rows to 7.66 MVA against 5.40 MVA for
A1. The A1n result shows only that the fold is needed. A1's own cleaned level on running rows (5.40
MVA) is 0.65 MVA above the level on tracked-off rows (4.75 MVA). Either demand is higher when the
engine runs, or the folded level is biased upward. The experiment cannot tell which.

## Checks

- **Median minus mean forecast.** The networks emit 11 quantiles and no mean, so the mean here is
  the mean of the 11 quantile values. At D8 the mean of the quantiles minus the median is +1.627
  points for A0, -2.774 for A1, and -1.766 for AH, and at the host it is -4.627 for W0 and -4.483
  for W1. Across the 16 Primaries on A0's raw output the gap is between +0.155 and +0.996 points at
  13 of them, +1.627 at D8, +2.648 at D12, and +8.562 at D15; the cause of the D15 and D12 gaps was
  not investigated. The add-back moves D8's quantile mean by 4.4 points, from +1.627 for A0 to
  -2.774 for A1. All values are in `printed/e00695_median_mean.txt`.
- **Error of the median beside every CRPS contrast.** C1 on the MAE of the median is +0.120 (95%,
  -0.487 to +0.788), the same sign as the CRPS contrast and also crossing zero. The mean error of
  the median at D8 is +0.24 points for A0, +2.00 for A1, and +1.15 for AH, and it is +10.00, +12.91,
  and +11.11 on tracked-on rows. The MAE and CRPS contrasts agree in direction.
- **Gate G0.** The refitted A0 scores 7.231588 CRPS at D8 against 7.231586 for the stored step 9a on
  102,310 shared rows, a gap of 1.6e-6 points. The audit found A0 bit-identical to the stored
  predictions per seed and per row.
- **Gate G1.** At the host, the causal tracker finds the injected engine with F1 0.785, precision
  0.998, and recall 0.647, and flags 0.03% of the half-hours with no injection (bars: F1 0.60,
  precision 0.50, 3%). G1 passed before any network was fitted.
- **Leakage tests.** Every test gave a change of exactly 0: the synthetic cut tests, and the
  real-series test at each of the 8 refit months. The climatology reads states at or before 24 hours
  before the native origin. The audit found no leakage.
- **Identity control.** Arm A1id sends D8's raw row through the cleaned-row route of arm A1, with
  cleaning disabled (the unit count is 0 everywhere, so the add-back adds nothing), for fold 2025-11
  and seeds 0 to 2. Its forecasts match arm A0's to a maximum of 8.8e-5 of capacity and a mean of
  about 1e-6, which is GPU noise.
- **Origin-time weather.** The `CHECKS PASSED` line of each per-fold prediction log asserts that
  `nwp_init_time` is at or before the origin of every test row. The scorer reads all 8 logs of each
  of the 6 fitted arms, including the host arms, and requires 242 origins (30, 31, 31, 28, 31, 30,
  31, and 30) for each arm. The assertion passes.
- **Which audit triggers fired.** An Opus audit
  (`data/ml-experiments/briefs/e695_audit/e695_audit.md`, private) re-derived every headline from
  the stored predictions and found nothing that changes branch 8. Trigger 6 fired: the C1 interval
  is 0.48 points wide, against 0.10 for the sister arms 5w and 5wo of experiment 677 on the same
  rows. Those sister arms barely differ at D8 (estimate +0.001), and a contrast of A0 seed against
  A0 seed is 0.36 to 0.61 wide, so the width of C1 is seed noise. Trigger 2 fired, because a cleaned
  arm that was predicted to gain 0.3 to 1.0 points lost 0.117. The audit tested an explanation
  (finding 4 below), so the trigger is explained. Trigger 7 arguably fires, because part 3 would
  depend on the sign of C1; it is treated as fired and the result is negative. Triggers 1, 3, 4, and
  5 did not fire: no control failed (G0, the identity control, and the leakage tests passed), the
  MAE and CRPS contrasts agree in direction, no nested arm beat the arm that contains it, and a
  leave-one-site-out check does not apply, because C1 is the D8 contrast.
- **Audit findings and their triage.** (1) None changes the verdict: accepted, every number
  reproduces to 3 decimals. (2) A1 is reliably a little worse: accepted and reported as post hoc,
  with the pre-registered decision interval first. (3) The scorer's per-Primary row for D8 (+4.646)
  scored A1's network output with no add-back: accepted and fixed, the D8 row is dropped from the
  table and C1 is the D8 contrast. (4) The cause of the null, an uninformative calendar climatology:
  accepted, in "Why the add-back does not help". (5) The bounds were listed beside the arms without
  a label: accepted and fixed, the printed report tags every bound as "BOUND, NOT A FORECAST". (6)
  A1n is mis-specified: accepted and stated above. (7) The comparison is paired: confirmed. The
  scorer read only the last fold's weather-check count, so that count (30) was a fold and not the
  run: accepted and fixed, the scorer now reads all 8 per-fold logs. (8) Weather for the host arms
  is aligned: confirmed, and now asserted. (9) No leakage: confirmed. (10) Trigger 6 is explained:
  accepted. (11) The power of the C1 test: accepted, the 8 monthly differences have a standard
  deviation of 0.138, so the 98.3% t half-width is 0.15 points. No audit finding was rejected.

## Deviations from the plan and items not done

- **The scorer changed after the first score.** The first print of the report held the D8 row of the
  per-Primary table, which scored a forecast that is not a system, and the weather count of the last
  fold only. Both are fixed, and the scores of the stored predictions did not change.
- **The sun check was not run.** It compares the tracked running share on the sunniest and the
  dullest third of days, to test whether the flags follow the sun. No script builds the per-day
  sunniness measure, so C3 carries the caveat that the flags may follow the sun, and the
  sunny-midday slice above is the only sun-related evidence.
- **The false-alarm check at the 7 other MVA-only Primaries was not run.** The 15 other Primaries
  show no move in A1 minus A0, which is an indirect check.
- **The twCRPS, the correlation of the engine state with the forecast error on training rows, and
  the climatology row on the cleaned series with the add-back were not run.** None can change the
  verdict. The correlation would test the add-back's independence assumption, which the audit's
  finding 4 already suggests fails.
- **The number of half-hours dropped per fold was not reported.** The standing practice asks for it;
  the cleaning is inherited from experiment 677's cleaned power table, unchanged.
- **The four load-shift scenarios were not reported**, because the experiment does not concern load
  shifts.

## What the next experiment should answer

**Can the engine's on or off state be forecast from information available at the origin well enough
to beat a calendar climatology?** The bounds say a perfect answer is worth 1.4 to 2.3 points at D8.
The follow-on analysis of experiment 694
(`data/ml-experiments/briefs/e694_followon/predictability.md`, private) found that persistence of
the state helps only within about a day. It also found that the calendar is already inside the
climatology, and that no input beyond a day beats a calendar climatology except possibly a wind
forecast out to a few days. Two candidates follow. Part 3 of this series would feed origin-time
engine-state features (the causal state at the two lag times and the trailing running share for the
valid half-hour's cell) to the network. A pre-registered test of a wind forecast as a predictor of
the engine's state at days 2 to 5 would follow the analysis directly. The part 3 idea is a second
hypothesis, described in the plan's git history and not run here.

## Method

### Architecture

- **The network is experiment 677's step-5 chain at monthly refits** (step 9a of [experiment
  677](../00677-building-up-the-candidate-demand-network-one-component-at-a-time/result.md)), a
  neural network pooled over 16 Primaries with a three-component Gaussian mixture output, from which
  11 quantiles (0.01 to 0.99) are read and sorted.
- **The inputs are weather, date-time, and two lagged-power encoders.** The weather encoder reads 7
  weather means, 7 spread columns, 2 lead columns, and 14 thermal-lag and solar-gain columns; the
  date-time encoder reads 21 columns; the lag encoder reads power at 24 hours and 1 week before the
  origin, with block dropout of lags. Every power input is at least 24 hours older than its origin.
- **The widths are an 8-wide Primary embedding, encoders from 64 to 32, a mixer of 64, and dropout
  0.1.** The encoders are trained on ERA5 and CAMS rows and then frozen; a fresh mixer is trained,
  and the whole is fine-tuned on ensemble weather with experiment 581's injected lag-only shifts.
- **The change at arm A1 is to D8's row of the power stack.** The stack row is replaced by a series
  with the engine removed. Where the smoothed most probable unit count is 0, the cleaned value is
  the reading. Where it is 1 or more, the cleaned value is the posterior mean of the demand level
  under the model y = sqrt((L - k u)^2 + q^2), with L the demand level, k the units running, u the
  unit size, and q = 0.14 MVA the reactive floor. All three stages read the replaced row, training
  targets and lagged inputs alike.
- **The add-back is a mixture over the unit count.** For each origin the weights p_j (j = 0 to 3)
  are the weighted mean of the smoothed marginal P(k = j), by half-hour and weekend flag, over the
  84 days of states at least 24 hours before the native origin, with weight 0.5 raised to the age in
  units of 28 days. The 11 quantiles are extrapolated linearly to 0.001 and 0.999, each level c maps
  to c for j = 0 and to sqrt((c - j u)^2 + q^2) for j of 1 or more, and the output quantiles are the
  weighted quantiles of that set. The network's mixture parameters are not exported.
- **The tracker is experiment 694's M2 grid filter** ([experiment
  694](../00694-gas-engine-state-tracker-part-1/result.md)), fitted per fold on readings up to 3.5
  hours before the refit. Era 1 (to 2025-03-23) is fitted once. Era 2 is refitted per fold and never
  split at the fade. A training state never reads a reading from 3.5 hours before the refit onward,
  and at inference a state at time t reads readings up to t + 20 hours.
- **The other arms differ as follows.** AH uses one tracker fit and one smoothed series to
  2026-06-30 for every fold. A1n uses the no-fold cleaning (y + k u) and a signed add-back clamped
  at zero. W0 trains on the host's reading with the injected engine, W1 cleans it as A1 does, and WT
  adds the true injected output's climatology to the network trained on the raw host reading (it
  reuses W0's fits). The bounds A1+state, AH+state, and WT+truth add the tracked or true state at
  the valid time. Arm A1id is the identity control.

### Training regime

- **Training data are ERA5 and CAMS (pre-training rows to 3.5 hours before the refit) and the ECMWF
  ensemble 00Z run of the origin day (fine-tuning and inference, treated as available at 09:00
  UTC).** Power comes from the cleaned power table with `substation_zero` half-hours dropped, and
  D15 and D18 are cut before 2025-03-01.
- **The host is D4**, one of the three MVA-only Primaries D2, D3, and D4, chosen before any fit
  because its median reading over 2025-03-24 to 2025-10-31 is closest to D8's tracked-off median.
  One real engine's 44-day record is resampled by whole days within weekday and weekend classes
  (seed 0) and injected by experiment 694's folded rule. The record is never shown with dates.
- **Refits are monthly, for the test months 2025-11 to 2026-06, with seeds 0, 1, and 2.** The fold
  equals the origin month. Each fold pre-trains (60 epochs in the logs), fits a fresh mixer, and
  fine-tunes with a 7-day embargo. Every fitted state file is saved.
- **GPU time was about 1.5 GPU-hours of wall time for the run** (about 0.27 hours for arm A0 and
  about 0.25 hours for each other arm, with 8 to 10 folds sharing the GPU at 100% utilisation),
  against a cap of 12.

### Measurement

- **Rows:** all 16 Primaries are fitted, and the decision series is D8. Scored rows have leads of 24
  to 240 hours from the 12:00 UTC contract origin, origins 2025-11-01 to 2026-06-30, valid times
  before 2026-07-01, and a finite metered reading (102,310 half-hours per seed at D8, 102,391 at the
  host). The row set is `backtest_year_2025_11_to_2026_06_daily_12utc`.
- **Metrics:** CRPS (twice the mean pinball loss over 11 quantiles) and the normalised MAE of the
  median, in points of effective capacity, with D8's capacity fixed at 10.379 MVA for every arm.
  Truth is the metered apparent power for every arm, at the host the injected reading.
- **Intervals:** each planned contrast is read through
  `studies.bootstrap.bootstrap_difference_at_level` from 20,000 resamples of whole origin months,
  one seed per draw, at 98.3%, with a paired t interval on the 8 monthly differences. C2, C3, and
  the MAE contrast are read at 95% with 2,000 resamples. A Primary "moves" when its 95% interval
  excludes zero and the estimate exceeds the stored sister-arm difference.
- **Decision rule:** the smallest gain worth porting is 5.5% of A0's CRPS at D8 (0.398 points). The
  verdict is the first branch of the pre-registered table that applies, and the first branch is
  `invalid` for a leakage failure.
- **Reproduction gate G0** compares A0 with experiment 677's stored step 9a. **Gate G1** requires
  the causal tracker to reach F1 0.60, precision 0.50, and at most 3% false flags at the host.

## Files

- `ml-experiments/00695-gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back/`
    - `e00695_look.py`: the short look at the stored forecasts that fixed the prediction (step 0).
    - `e00695_tracker.py`, `e00695_engine.py`, `e00695_clean.py`: the tracker import, the engine
      injection, and the build of each arm's cleaned stack row and states per fold.
    - `e00695_addback.py`: the climatology and the add-back of the engine mixture.
    - `e00695_phase_a.py`, `e00695_checks.py`: the host choice, gate G1, the leakage tests, and the
      sign check.
    - `e00695_k1.py`, `e00695_chain.sh`, `e00695_gpu.sh`: the step-5 chain with the stack row
      replaced.
    - `e00695_arm.sh`, `e00695_queue.sh`, `e00695_wait.sh`: the launch and wait scripts, 8 to 10
      single-fold jobs at a time.
    - `e00695_identity.py`: the identity control (writes arm A1id's row and compares its forecasts
      with A0's).
    - `e00695_score.py`: the scorer, which computes every contrast and the verdict branch.
    - `e00695_median_mean.py`: the median-minus-mean check and the median error by tracker state.
    - `e00695_figures.py`: draws the four figures.
    - `printed/`: the printed outputs (`e00695_score_report.txt`, `e00695_median_mean.txt`,
      `e00695_identity_compare.txt`, and the checks and sign-check prints).
- `data/ml-experiments/00695-gas-engine-state-tracker-part-2-cleaned-target-and-climatology-add-back/`
  (private, never committed)
    - `runs/<ARM>/`: the fits (`arms/k1_5w_sp1/work/fits/P5/<month>_d0_s<seed>.parquet`, quantiles
      as fractions of capacity), logs, and saved networks, for A0, A1, A1n, AH, W0, W1, and A1id.
    - `stacks/<ARM>/`, `states/<ARM>/`: the cleaned stack rows and the tracker's state marginals per
      fold.
    - `predictions/a0-d8`, `a1-d8`, `ah-d8`, `w0-host`, and `w1-host`: the immutable prediction
      sets.
    - `scores/report.json`, `d8_slices.csv`, `host_slices.csv`, `references_d8.csv`: the scorer's
      tables.
    - `checks.json`, `host.json`: the gate and leakage results and the host's name.
    - `briefs/e695_audit/` (under `data/ml-experiments/`): the audit and its script.
- Read from earlier experiments: experiment 677's chain, its shared protocol rows, and the stored
  step 9a at `backtests/step5w_with_injected_sp1.parquet`
  ([677](../00677-building-up-the-candidate-demand-network-one-component-at-a-time/result.md), read
  only); experiment 694's tracker and the D8 hindsight flags
  ([694](../00694-gas-engine-state-tracker-part-1/result.md)).
