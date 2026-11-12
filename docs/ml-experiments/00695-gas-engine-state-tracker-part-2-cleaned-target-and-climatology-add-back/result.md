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
status: planned
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
---

**This experiment asks whether a gas engine's output can be taken out of one substation's recorded
demand before a neural network learns from it, and put back as an average when forecasting.** The
substation, D8, has a gas engine that runs on and off, so its recorded power jumps by up to about 50%
of the substation's capacity. We use the on and off tracker of [experiment
694](../00694-gas-engine-state-tracker-part-1/result.md) to remove the engine from the training data
and from the neural network's recent-power inputs, retrain the network at monthly refits, and add the
engine's typical output back to each forecast. We compare the result with the same network trained on
the raw reading. The experiment is planned and has produced no score.

**This plan tests one hypothesis: separating the engine from D8's reading before training, and
adding the engine back as a climatology, improves the forecast of D8's metered apparent power.**
The maintainer's second idea, engine-state features with dropout, is a second hypothesis and is
split off as part 3 (see "The alternative arm"). Every number for D8 carries the label "reactive
power assumed zero" except the tracker's fixed reactive floor of 0.14 MVA.

## What the short look found

**The stored Candidate's median at D8 sits between the engine-on level and the engine-off level.**
`e00695_look.py` joined the normal-running rows of `primaries.parquet` to experiment 694's
hindsight M2 state (`d8.pkl`) and printed every figure below. **Every forecast figure is on the
scored rows only** (leads 24 to 240 hours from the 12:00 UTC origin, origins 2025-11-01 to
2026-06-30, valid times before 2026-07-01, a finite metered reading), so a half-hour counts once per
forecast that covers it. Months are valid-time months.

- The Candidate's CRPS is 7.18 and its normalised MAE of the median is 13.18 points of capacity,
  over 102,310 scored half-hours.
- In 2025-11 to 2026-02 the median is 7.9 points too low on tracked-off rows (28,091 rows) and 13.4
  points too high on tracked-on rows (21,483 rows). The tracker flags 35% to 49% of scored rows in
  those months (33% to 49% of all half-hours with a reading).
- In 2026-03 to 2026-06 the tracker flags 23% to 31% of scored rows and of all half-hours (14% to
  16% for the filtered flag). Experiment 194 found the share of large shortfalls falling to 0.8% in
  April 2026. Either the tracker false-alarms in the fade or the engine runs at a lower level.
- D8's capacity is 10.379 MVA, so three units at 1.75 MW are about 51 points of capacity.
- On experiment 677's 80,688 shared rows at D8, CRPS is 7.100 for step 9a, 7.131 for the Candidate
  at monthly refits (step 8w), and 7.157 for the stored experiment 667 file. The sister arm 5wo
  scores 7.110.
- Step 9a minus its sister 5wo, in absolute CRPS, by Primary: 0.161 at D15, 0.108 at D22, 0.058 at
  D16, and 0.037 to 0.043 at D11 and D14. The 8 monthly differences at D8 have a standard deviation
  of 0.047.

## What I would change in the maintainer's design

- **Add back a distribution, never an expected output.** The engine is off more than half the
  time, so the total is a two-level mixture. Adding a mean output shifts the median to a level the
  reading rarely takes. The add-back mixes the folded forecast over the unit count.
- **Key the climatology by half-hour and weekend flag over a recency-weighted trailing window, not
  by month.** A month key needs a year of history per era and cannot follow the 2026 fade. The fade
  is not known at the forecast origin, so no era boundary is placed there.
- **Clean the lagged-power inputs as well as the target.** The 24-hour and 1-week lags otherwise
  carry yesterday's engine state into a forecast of engine-free demand.
- **Use a causal cleaned series, from a fixed-lag smoother that never passes a cut.** A
  filter-only state is noisier than needed, and a full smoother uses the future.
- **Test on experiment 677's step 9a, not on the Candidate.** Over all Primaries and the whole
  backtest year, step 9a's normalised MAE of the median is 7.39 against 7.48 for the Candidate at
  monthly refits. At D8, on experiment 677's 80,688 shared rows, the CRPS is 7.10 for step 9a,
  7.13 for the Candidate at monthly refits, and 7.16 for the stored backtest file. Every power
  input of step 9a is at least 24 hours older than its origin, which makes the leakage argument
  short. A port to the Candidate is a later `change` experiment.
- **Bound the benefit three ways:** hindsight cleaning at D8, perfect cleaning at a host Primary
  where an engine is injected, and the tracked state at the valid time.

## Arms

| Arm | Training target and lag inputs at D8 | Added back at forecast time | Fits |
|---|---|---|---|
| A0 | Raw reading | Nothing | 8 folds, 3 seeds (paired control) |
| A1 | Causal cleaned series | Causal engine climatology | 8 folds, 3 seeds |
| AH (a bound, never a forecast) | Hindsight cleaned series (tracker fitted and smoothed to 2026-06-30) | Climatology from hindsight states before the origin | 8 folds, 3 seeds |
| A1n (conditional) | Causal series cleaned without the fold | Causal climatology, signed add-back clamped at zero | 8 folds, 3 seeds, only if the sign check below triggers |
| A1+state, AH+state | As A1 and AH | The tracked unit count at the valid time (bound; uses the scored reading) | None |
| W0 | Host reading with an injected engine | Nothing | 8 folds, 3 seeds |
| W1 | Host injected series, cleaned by the causal tracker | Causal engine climatology | 8 folds, 3 seeds |
| WT | Host's original series (perfect cleaning) | Climatology of the true injected output, same trailing window and cells as A1 | 8 folds, 3 seeds |
| WT+truth | As WT | The true injected output at the valid time (bound) | None |

Two reference rows need no network: `ClimatologyForecaster` on the raw reading, and
`ClimatologyForecaster` on the cleaned series with the engine climatology added back.

**The cleaned series.** M2 models the reading as y = sqrt((L - k u)^2 + q^2), with demand level L,
k running units of size u, and reactive floor q = 0.14 MVA. Where the smoothed most probable k is
0, the cleaned value is the reading itself, so a Primary with no flag keeps an identical series.
Where k is 1 or more, the cleaned value is the smoothed posterior mean of L on the tracker's grid,
which averages over the unit count and over the two branches of the fold. The cleaned series is
demand net of every embedded generator except the engine, still read as apparent power: option A
of the review for demand, and "demand minus a non-negative generation term" (experiment 387's
recommendation) for the engine alone.

**Sign check.** A hard choice of branch errs by 2|L - k u|, up to about 40 points, and experiment
694 showed no benefit of the fold. Phase A therefore reports, over rows with k of 1 or more, the
share where the posterior-mean cleaning and a no-fold cleaning (y + k u) differ by more than 1 MVA.
If the share exceeds 10%, arm A1n runs, and the write-up reports A1n minus A1 at 95%, outside the
family.

**Causal state.** The tracker reads the same `substation_zero`-cleaned power table the network
trains on, never experiment 196's `arrays.pkl`. Its parameters (K, u on a 0.05 MW grid, transition
matrix) are fitted per fold on readings up to the fold's `train_end`. The era from 2023-08-15 to
2025-03-23 is fitted once, because it ends before every cut-off. The era from 2025-03-24 is
refitted per fold and is never split at the fade. The state at half-hour t uses readings up to
t + 20 hours, **cut at `train_end` for every training row of every stage** (pre-training, the
fresh mixer, and the fine-tune). The cut is needed because experiment 677's pre-training rows run
to 3.5 hours before the refit and its second stage has a 3-day embargo, so only the fine-tune's
7-day embargo would hide a 20-hour look-ahead. At inference the state at t uses readings up to
t + 20 hours. Step 9a reads power no younger than 24 hours before its native 09:00 UTC origin, and
a reading arrives 3.5 hours late, so every input state uses only readings published by the origin.
Rows before 2023-08-15 are untouched.

**Add-back.** For each origin, p_j (j = 0 to 3) is the weighted mean of the smoothed marginal
P(k = j) over states at least 24 hours older than the native origin, in the cell (local half-hour,
weekend flag), within 84 days, with weight 0.5^(age / 28 days), smoothed over the neighbouring
half-hours. The network does not export its mixture parameters, so the add-back extrapolates the 11 output
quantiles linearly to the levels 0.001 and 0.999 and treats the quantile function as piecewise
linear between them. Each level c maps to c for j = 0 and to sqrt((c - j u)^2 + q^2) for j of 1 or
more, with weight p_j, and the 11 output quantiles are the weighted quantiles of that set. The
signed variant of arm A1n maps c to max(c - j u, 0). The add-back assumes the engine state is
independent of the demand forecast error within a cell, and the write-up reports the correlation
on training rows.

## Data

- Power: the cleaned power table, with the `substation_zero` half-hours dropped everywhere, and
  D15 and D18 cut before 2025-03-01. The tracker treats a dropped half-hour as unobserved.
- Capacity: D8's effective capacity stays 10.379 MVA in every arm, for the network's target
  scaling and for scoring, although the cleaned series is higher than the reading.
- Weather: as experiment 677's step 5. Fine-tuning and inference read the ECMWF ensemble 00Z run of
  the origin day, treated as available at 09:00 UTC; ERA5 and CAMS enter only pre-training rows
  that end before the refit time. **Every weather input is origin-time.** The code review asserts
  that `nwp_init_time` of every scored row is at or before the origin and equals arm A0's, and
  that no feature for a valid time after the origin reads ERA5, CAMS, or a later run.
- Host: the MVA-only Primary among D2, D3, and D4 (no fold on sunny middays, hosts in experiment
  694) whose median reading over 2025-03-24 to 2025-10-31 is closest to D8's tracked-off median.
  Phase A fixes the host before any fit.
- Injected engine: Gas A's 44-day record, resampled by whole days within weekday and weekend
  classes (seed 0) to cover 2023-08-15 to 2026-06-30, with ramps and part load kept, injected by
  experiment 694's folded rule. The record is never published with dates.

## Training

Each fitted arm reruns experiment 677's step-5 chain at monthly refits (pre-training on ERA5 and
CAMS, fresh mixer, fine-tune on the ensemble, injected lag-only shifts, 7-day embargo), for the
test months 2025-11 to 2026-06 and seeds 0 to 2, with one change: the power rows of D8 (or of the
host). The network stays pooled over 16 Primaries. Every fitted state file is saved.

## Scoring

- Truth is the metered apparent power, for every arm. No arm is scored on a cleaned series.
- Seeds are paired through `experiment_store.losses_for_bootstrap`; the climatology rows are
  repeated once per seed.
- Slices: lead bands of demand-v2; engine-visible months and fade months; tracked-on and
  tracked-off rows at D8 (hindsight state, a tracker-defined slice) and truly-on and truly-off
  rows at the host; sunny middays (the protocol's clear-sky-index mask) in 2026-03 to 2026-06;
  cold winter nights; tails at q01, q05, q95, and q99.
- Every Primary is scored and no screening arm is used, as demand-v2's stop rule for blind alleys
  requires. The counted `not supported` experiments on D8's engine are 264, 265, 294, 361, and
  363. **Approval: the maintainer asked for this experiment by name**, which the rule accepts.
- Figures: raw and cleaned D8 with the tracked state (two weeks per era); a day-ahead forecast and
  one full 14-day forecast for A0 and A1 against the reading; C1 by month; the host arms by true
  state. A figure of the host's injected series shows Gas A's output shape, so it follows the
  `dataviz` skill's rules for generators: no dates, no name, no identifier, and a normalised axis.

## Gates, controls, and stop rules

- **G0, reproduction.** The refitted A0's seed-mean CRPS at D8 matches the stored step 9a file
  within 0.10 points on shared rows. A mismatch stops the run until the cause is found. The
  refitted A0 is the control in either case, so drift in data or code cannot enter C1.
- **G1, tracker in the scored era (no network fitted before G1 passes).** On the host's injected
  series, in causal operation, half-hour F1 for "any unit running" is at least 0.60 with precision
  at least 0.50, and at most 3% of un-injected half-hours are flagged. Reported beside G1:
  agreement of causal and hindsight states at D8, M1 against M2, and the flagged share by
  half-hour and by month through the fade.
- **Sun check at D8 (reported, pre-registered).** Within each daylight half-hour cell, the flagged
  share on the sunniest third of days is compared with the dullest third, by era. A gap above 10
  percentage points in the fade months means the flags follow the sun, the independence assumption
  of the add-back fails there, and C3 carries that caveat.
- **Identity control.** The A1 pipeline with cleaning disabled (k = 0 everywhere, p_0 = 1), on one
  fold and seed, reproduces A0's forecasts to floating-point tolerance.
- **False-alarm check.** The causal tracker's flagged share is reported at the 7 other MVA-only
  Primaries.
- **Host ceiling (G2).** The host arms run first. If C5 fails, the result is recorded as "not
  supported at the host", and A1 still runs at D8, because the host's engine runs 17% of
  half-hours, independently of demand and price, and is a weak stand-in for D8.
- **Placebos.** D22 and D2 are the named placebos, and every other Primary is checked, by the
  per-Primary rule in the decision rule.
- **Leakage tests.** Readings after an origin's cut are set to missing and every model input for
  that origin is asserted unchanged. The same test at `train_end` covers the rows of all three
  training stages.
- **Audit.** Trigger 7 will fire if part 3 or a port depends on the result; triggers 1, 3, and 5
  are checked from the gates, the MAE direction, and AH against A1.
- **Budget stop.** 12 GPU-hours and 6 CPU-hours of tracker fits; stop between folds.

## Standing practices

- Followed: monthly refits; cleaned data everywhere, with the dropped half-hours counted per fold;
  the D15 and D18 cut; saved models; stored predictions as references and as the G0 check; a
  climatology row; no valid time on or after 2026-07-01; normalised MAE of the median in every
  table; tails and cold winter nights; example forecast plots; a draft pull request.
- Followed from earlier advice: train an engine term on the state a forecast will see (294);
  score before and after March 2026 separately (194); refit the tracker for the regime after
  2025-03 (196); a 0.05 MW grid for the unit size and a record-like schedule (694); a known answer
  whose injected signal comes from a real record (387).
- Departed from: no four-month screen, because the blind-alley rule replaces a screen. No control
  without calendar columns, because the change adds none and the engine climatology has no month
  or day-of-year key. Experiment 693's advice of 10 refits per arm is not followed: 3 paired seeds
  are used, and all 3 must agree on the sign of C1. Experiment 361's advice not to remove detected
  steps from D8's target is the idea under test, with a better tracker and an add-back that
  experiment 361 lacked. The four load-shift scenarios are not reported, because the experiment
  does not concern load shifts; D8's multi-day drops stay in the scored rows.

## Compute

One fold of the step-5 chain with 3 seeds took about 8.5 minutes in experiment 677's logs, so one
arm over 8 folds takes about 1.1 GPU-hours. Six fitted arms come to about 7 GPU-hours, and A1n
would add 1.1. The tracker needs about 1 CPU-hour on 24 cores. Folds and arms run in parallel
within the slot.

## Choices the code makes where the plan leaves room

These choices are fixed before any fit, and the code review checks each one.

- **Training cut.** A training-time state never reads a reading from 3.5 hours (7 half-hours)
  before the refit onward, because a reading takes 3.5 hours to arrive. States of half-hours before
  the refit come from a forward pass on that hidden series and a backward pass of 20 hours that
  stops at the cut.
- **Era 2 fit.** The fit reads readings up to 3.5 hours before the refit, and the states after the
  refit use the uncut series with the same fixed lag.
- **Hindsight arm.** Arm AH uses one fit and one smoothed series to 2026-06-30 for every fold.
- **Quantiles beyond q01 and q99.** The network's mixture parameters are not exported, so the
  add-back extrapolates the 11 quantiles linearly to the levels 0.001 and 0.999.
- **Host.** The host is the candidate whose median reading is closest to D8's tracked-off median,
  and `e00695_phase_a.py` writes the choice to `host.json` before any fit.
- **Injected days.** The engine record is cut into whole UTC days, and a day with a gap is dropped.
- **Host truth.** The score at the host uses the injected reading, and arm WT uses arm A0's network
  at the host with the true engine's climatology added back.
- **Sunny middays.** The slice is the protocol's `sunny_unit_shape` column in the months 2026-03 to
  2026-06.

**The cleaned series is a hedge wherever the engine's state is uncertain.** Where the most probable
unit count is 1 or more, the cleaned value is the posterior mean of the demand level over all unit
counts, so it includes some probability that no unit runs. Where that probability is not small, the
cleaned value sits between the engine-free and the engine-on level.

**Slices and bootstrap blocks use different months.** The engine-month and fade-month slices, and
the sunny-midday slice, use the month of the valid time. C2, C3, and the bootstrap blocks use the
month of the forecast origin.

## Items of the plan not implemented before the first fit

- The sun check is defined (`sun_check` in `e00695_checks.py`) but nothing calls it yet.
- The false-alarm check at the 7 other MVA-only Primaries, twCRPS, the correlation of engine state
  with forecast error on training rows, and the `ClimatologyForecaster` row on the cleaned series
  with the add-back are not written.
- The network-level identity run (the A1 pipeline with cleaning disabled, on one fold and seed) is
  not written. Only the synthetic identity check exists.
- No report or figure script exists yet.
- The tracker's states for the 40 half-hours before a refit have less look-ahead than later ones.

## Risks

- **The tracker may flag solar or cloud dips as the engine after 2025-03-24.** M2's unit size there
  is 2.25 MW against 1.5 MW for M1, and the hindsight fit of the fade is 2 units of 3.5 MW while
  the causal fit keeps one era. G1 runs on the host and cannot see this; the sun check and the
  sunny-midday slice can.
- **The fade months have no truth.** A gain there may come from a faster-adapting climatology or
  from the tracker's errors cancelling; C3 cannot tell the two apart.
- **The host's engine runs 17% of half-hours, D8's 35% to 47%.** The host validates the pipeline,
  not the size of the gain at D8.
- **Four engine-visible months is a short test,** and the whole year overlaps step 5's development
  months.
- **A flexible network may already learn the mixture.** Then the gain is confined to the lags and
  to the fade, and C1 is small.
- **Eight month blocks give a weak interval,** hence the t interval and the seed-sign rule.

## The alternative arm (part 3, not run here)

**Engine-state features are sane only in their origin-time form, and they are a second
hypothesis.** A state at the valid time is never available at inference. Training with that state
present on some rows and a "missing" flag on others does not leak, provided the flag is always
"missing" at inference, but the network then forecasts in the mode that has no state, which holds
the same information as arm A0. The only gain dropout can add is a regularising one. Experiment
294 showed that a state built from the reading being forecast fits that reading: its coefficient
was 3.6 MW per half-hour with one or more units running, against 0.15 MW per unit of probability
for the last settled day's state. Part 3 should therefore test origin-time features only: the
causal state at the two lag times, and the trailing running share for the valid half-hour's cell.
Part 3 is worth running only if G1 passes here.
