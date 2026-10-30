---
id: "00413"
slug: differentiable-physics-demand-network-part-1
title: >-
  Can one neural network that mixes shared libraries of learned daily shapes and learned temperature
  responses forecast Primary demand about as well as the plain mixture density network?
theme: differentiable-physics-demand-calendar-and-weather
also_relevant_to:
  - production-reference-and-best-guess
  - hierarchy-and-pooling
status: done
kind: other
protocol: demand-v2
size: >-
  complex. Stored: fitted networks, one prediction table per arm, and the learned curves in this
  experiment's private folder only; every earlier experiment is read and never written. Production
  serving path: not touched (this repository holds none). Degradation rule: not touched. Designs:
  more than one is defensible (how the libraries are normalised, what the residual mixture reads,
  how the Primary mix is shared), so the design note's choices are registered and the alternatives
  are named as arms or caveats. Callers: the code copies experiment 397's network head and training
  loop and imports experiments 87, 397, and 405, whose callers sit in those folders.
hypothesis: >-
  A pooled network that forecasts Primary demand as a Primary-specific mix of shared learned daily
  shapes plus a Primary-specific mix of shared learned temperature and solar responses, with a
  plain mixture density head on the residual, scores within 0.03 points of the pooled plain mixture
  density network on the 2026-01 fold, and its learned curves can be read.
prediction: >-
  Written before any fit. P1: D2 (shape and response libraries with the Primary mix) minus D0 (the
  pooled plain network) is between -0.05 and +0.25 points of CRPS at days 1 to 10 at the signed-MW
  six and at the MVA-only eight, because the plain network already reads the same inputs and the
  libraries restrict what the residual may read. P2: D1 (shape library only) minus D0 is between
  -0.05 and +0.10. P3: D3 (one fleet-wide mix) minus D2 is between 0 and +0.30 at both groups,
  because the Primaries differ in level and shape. P4: q95 coverage of every arm at the signed-MW
  six stays between 0.82 and 0.90, as for the networks of experiment 397. P5: the mixed temperature
  response is monotone below the reference temperature at 14 of the 16 Primaries, because each
  library curve is monotone by construction.
decision_rule: >-
  A screen: one fold, one seed, no interval, and every number interim. D2 advances to the five folds
  of experiment 397 (three seeds each) if D2 minus D0 is at most +0.03 points at both groups. This
  advance threshold was not confirmed by the maintainer; the manager's default was used. A
  single-seed difference carries seed noise of about 0.03 to 0.06 points (experiment 334), so a gap
  under 0.03 is within noise and the advance is a decision to look further, not a finding. The
  screen stops if D2 minus D0 exceeds +0.30 at either group, or if the mixed temperature response
  is non-monotone below the reference temperature at 3 or more Primaries. A gap between +0.03 and
  +0.30 stops nothing and advances nothing: the maintainer decides. No claim of benefit rests on this
  screen. The smallest gain worth porting is not set, because the screen asks whether the design is
  viable, and its interpretable curves are the reason to carry it forward.
benchmark: >-
  Arm D0 (the pooled plain network, refitted here under the same rules and rows), and for absolute
  scale the stored GAM stack (experiment 377) and monthly XGBoost (experiment 389) forecasts on the
  same half-hours, read and never refitted.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 336 hours; the headline band is days 1 to 10.
verdict: not supported
best_arm: >-
  D0, the pooled plain network: CRPS 3.322 at the signed-MW six and 3.908 at the MVA-only eight
  (days 1 to 10, five test months, 3 seeds). Of the library arms, D1 scored 3.405 and 3.956, and D2
  scored 3.496 and 4.136.
headline: >-
  The network built from shared learned shapes and temperature responses (D2) scored 0.174 points
  worse than the plain pooled network (D0) at the signed-MW six and 0.228 points worse at the
  MVA-only eight, and was lower in 2 of 5 months at the six and 1 of 5 at the eight. Its one-month
  gain in 2026-01 (-0.195 at the six) came from a yearly term that learned January's level.
lessons: >-
  The libraries made the forecast no better and a little worse, and the learned curves were not
  cleanly separated. D2's upper-tail coverage and upper-quantile loss were better than the plain
  network's, and its lower-tail loss and median error were worse.
production_status: not-for-production
adopted_in: none
take_home_message: >-
  A neural network built from shared, readable parts (learned daily shapes and temperature
  responses) forecast Primary demand worse than the plain neural network on five test months: by
  0.17 points of capacity at the 6 substations that record the direction of flow and by 0.23 points
  at the other 8. It gave better coverage of the highest outcomes, but its typical forecast error was
  larger, and its three heating curves came out nearly identical, so they added little to read. The
  readable-parts design is not worth carrying forward as built.
advice:
  - >-
    Do not carry the shape-and-response library network forward as built: it scored 0.174 worse than
    the pooled plain network at the signed-MW six and 0.228 worse at the MVA-only eight over five
    months (C1).
  - >-
    Judge any design with a yearly term on at least five test months, because on 2026-01 alone the
    term gave D2 a gain of 0.195 that reversed over the other four months (C1).
  - >-
    Give a control the same yearly columns as the arm under test (arm D0Y), because the yearly
    columns alone moved the plain network by +0.014 at the six and -0.015 at the eight (C6).
  - >-
    If the readable-parts idea returns, normalise heating curves so that they cannot duplicate each
    other, because the three heating curves of D2 were nearly the same (learned temperature
    responses figure).
date: 2026-10-30
code: ml-experiments/00413-differentiable-physics-demand-network-part-1/
---

**Take-home message: a network built from shared, readable parts did not forecast Primary demand as
well as the plain pooled network.** We asked whether a network that writes each Primary's demand as
a mix of shared learned daily shapes and shared learned temperature responses, with a plain mixture
head on the leftover, can match the plain pooled mixture density network of [experiment
397](../00397-plain-and-attention-mixture-density-network-for-primary-demand/result.md). Over five
test months with three seeds each, it could not: it scored 0.174 points of capacity worse (CRPS,
days 1 to 10) at the 6 Primaries that record the direction of flow (signed-MW six), and 0.228 worse
at the other 8 (MVA-only eight). **The one-month result we first saw was misleading:** in 2026-01
the library network scored 0.195 better at the six, but the gain came from a yearly term that
learned the level of the one January it had seen, and it reversed in other months. The verdict is
`not supported`. By the default rule (D2 within 0.03 points of D0 at both groups, which the
maintainer has not confirmed) D2 does not advance. All numbers are interim: five test months, no
interval.

![CRPS by test month for each arm](scores_by_fold.svg)

![D2 minus D0 and D2 minus D0Y by test month](contrasts_by_fold.svg)

## Results

CRPS at days 1 to 10, mean over five test months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06),
equal weight per Primary, in points of capacity (smaller is better). Arms D0, D0Y, D1, and D2 use 3
seeds averaged in quantile space.

| Arm | Signed-MW six | MVA-only eight |
|---|---|---|
| D0, pooled plain network | 3.322 | 3.908 |
| D0Y, D0 plus the yearly columns | 3.336 | 3.893 |
| D1, shape library, head reads all inputs | 3.405 | 3.956 |
| D2, shape and response libraries, Primary mix | 3.496 | 4.136 |
| GAM stack (experiment 377, stored) | 3.247 | 4.135 |
| Monthly XGBoost (experiment 389, stored) | 3.544 | 4.019 |
| Pooled plain network of experiment 397 (stored, no D15 and D18 cut) | 3.270 | 3.866 |

- **C1, D2 minus D0:** +0.174 at the six (lower in 2 of 5 months: 2025-10 +0.139, 2025-12 +0.699,
  2026-01 -0.195, 2026-03 +0.248, 2026-06 -0.021) and +0.228 at the eight (lower in 1 of 5).
- **C2, D1 minus D0:** +0.082 at the six and +0.048 at the eight. Adding a shape-library mean to the
  plain network did not help.
- **C3, D2 minus D1:** +0.092 and +0.180. The response library did not help either.
- **C5, D2 minus D0Y:** +0.160 and +0.243. **C6, D0Y minus D0:** +0.014 and -0.015. The yearly
  columns alone do not explain D2's loss.
- **Stop rule:** D2 minus D0 exceeded +0.30 in 2025-12 at the six, and the mean exceeded +0.03 at
  both groups, so D2 does not advance. Arm D3 (one fleet-wide mix) was fitted on 2026-01 only: it
  scored 4.745 at the six and 4.503 at the eight, so D3 minus D2 was +0.883 and +0.565.
- **Median error:** the mean absolute error of the median forecast, in MW at the six and the eight,
  is 5.69 and 7.08 for D0, 5.76 and 7.04 for D0Y, 5.82 and 7.13 for D1, and 6.02 and 7.23 for D2.
  D2's typical error is larger, in the same direction as its CRPS.
- **Tails, mean over months** (coverage nominal 0.95 and 0.99; pinball loss in points):

| Arm, group | q95 coverage | q99 coverage | pinball q01 | pinball q05 | pinball q95 | pinball q99 |
|---|---|---|---|---|---|---|
| D0, six | 0.829 | 0.926 | 0.216 | 0.712 | 1.091 | 0.436 |
| D2, six | 0.880 | 0.975 | 0.323 | 0.954 | 0.932 | 0.243 |
| D0, eight | 0.899 | 0.969 | 0.224 | 0.851 | 1.011 | 0.281 |
| D2, eight | 0.889 | 0.966 | 0.292 | 0.998 | 1.164 | 0.372 |

  At the six, D2's upper quantiles cover more outcomes and cost less (q99 pinball 0.243 against
  0.436), and its lower quantiles cost more. At the eight, D2 is worse in both tails. The
  curtailment-cost measure was not computed for this screen.

![Day-ahead forecasts of D0 and D2 at D5](example_day_ahead.svg)

![A 14-day forecast of D0 and D2 at D5](example_14_day.svg)

![D2's eight learned shapes](learned_shapes.svg)

![D2's learned temperature responses](learned_temperature_responses.svg)

- **Learned curves (D2, 2026-01 fold, seed 0).** The three heating curves are nearly identical, so
  the library did not learn three different responses. The eight shapes are smooth and mix freely,
  and the page makes no claim that any one of them stands for a physical daily profile. The
  comparison with the GAM's fitted curves and the three-seed stability check were not run, because
  D2 did not advance.
- **Per Primary (C1, mean of five months):** D2 is better at D5 (-0.276), D16 (-0.094), D14
  (-0.068), and D21 (-0.053), and worse at D11 (+1.161), D8 (+0.657), D12 (+0.401), and D10
  (+0.249). D15 and D18 are report-only.

## Predictions

- **P1 held, but the advance rule failed.** D2 minus D0 lies between -0.05 and +0.25 at both groups
  (+0.174 and +0.228), yet the rule needs at most +0.03.
- **P2 held:** D1 minus D0 is +0.082 and +0.048, within -0.05 to +0.10.
- **P3 not tested on five months:** D3 ran on 2026-01 only, where D3 minus D2 was +0.883 at the six,
  outside the predicted 0 to +0.30.
- **P4 held:** the six's q95 coverage is 0.829 (D0), 0.827 (D1), and 0.880 (D2), within 0.82 to
  0.90.
- **P5 holds by construction,** because each library curve is monotone.

## Checks

- **Median minus the mean of the 11 quantiles (a weighted average, MW):** D0 -0.028, D0Y -0.026, D1
  -0.033, D2 -0.026. No arm's spread shifts its forecast.
- **Error of the median beside the CRPS contrasts:** the mean absolute error of the median points
  the same way as the CRPS (above).
- **Audit triggers fired:** trigger 2 (D2 minus D0 on 2026-01 was -0.379 against a predicted -0.05
  to +0.25) and trigger 5 (D1, which contains D0's inputs, scored worse than D0 in 2026-01). A fresh
  Sonnet audit found no bug. It re-derived the per-Primary scores, found no leakage, and attributed
  the 2026-01 gap to the yearly term: zeroing D2's yearly coefficients at inference moved its
  2026-01 score at the six from 3.861 to 5.298. The audit could not say why D0 sits 0.1 to 0.2 above
  experiment 397's plain network seeds. The other five months settle the question the audit left
  open: the 2026-01 gain was specific to that month.
- **Leave-one-out and interval triggers:** the screen reports no intervals and no
  leave-one-site-out, so triggers 4 and 6 were not tested.

## Caveats

- **Advance rule not confirmed.** The 0.03 threshold was the manager's default, and the maintainer
  has not confirmed it.
- **Five test months, no interval, seed-averaged.** Every number is interim.
- **Lead range.** The headline band is days 1 to 10, the range NGED care about.
- **Yearly harmonics sit in D1 to D3 and in D0Y, not in D0.** The data start in 2024-04, so a fold
  holds at most two of each calendar month, and the yearly term fits very few years.
- **D0 is not a copy of experiment 397's network.** D0 uses the D15 and D18 cut, its own
  standardiser, and its own seed stream, and scores 0.05 worse than the stored 3-seed pooled network
  at the six.
- **Rows dropped for a missing target: none were found.** The raw matrices from experiment 397 omit
  the flagged half-hours already, so this experiment did not count them.
- **Shapes and the gate are normalised on 4,096 sampled training rows,** not on a grid of ordinary
  days.
- **The monotonicity check cannot fail,** because each curve is monotone by construction.
- **Unlogged load-shift events.** NGED says load-shift events may be missing from their logs, so the
  rows may hold events that no label covers.
- **Resources.** Fits ran on the GPU: D2 took 59 to 257 s per fit (2026 months), peak GPU memory 1.6
  GB, and peak host memory 8.3 GB on the largest fold. Fitted networks are in the experiment's
  private folder under `work/models/`, and the scripts are in
  `ml-experiments/00413-differentiable-physics-demand-network-part-1/`.

## What the next experiment should answer

Whether a network with readable parts can be made to match the plain network at all. The shapes and
heating curves overlap, so the first step would be to force them apart and to read them against the
GAM's fitted curves. No experiment is started by this page.

## Plan as committed before any fit

**Status: planned. Nothing has been fitted.** The plan follows the Design session's note
(`differentiable_physics_demand_design_answer.md`, Opus-reviewed and approved by the maintainer on
2026-10-30) and the manager's programme brief for experiment 7.

## What is asked

**The question is whether a neural network built from shared, readable parts can forecast Primary
demand as well as a plain neural network.** The aim behind the series is a forecaster that keeps the
GAM's (generalised additive model's) readable shapes and temperature curves and gains the
adaptability of the networks. Experiment 397's pooled plain network scored 3.264 points of CRPS
(days 1 to 10) at the signed-MW six, close to the GAM stack's 3.239. The first fit checks, on one
winter month, whether the library design loses much against that plain network. It is a screen to
stop a poor design early, not a ranking.

## Review of the plan against the latest evidence

- **Experiment 397** (merged): the pooled plain network is the control. Its inputs, rows, folds,
  early-stopping rule and training loop are copied. The attention gate was inert, so the design has
  none.
- **Experiment 403** (merged): pooling beat one network per Primary by 0.185 at the signed-MW six
  and 0.161 at the MVA-only eight, so the libraries are shared across all 16 Primaries.
- **Experiment 405** (merged, inconclusive) and the open plan for **experiment 409** ([PR
  66](https://github.com/openclimatefix/nged-ml-research/pull/66)): a network with no day-of-year
  term had no other home for the monthly daylight profile, so a solar term could not be separated
  from it. The design note already keeps fixed yearly harmonics with Primary-specific coefficients,
  and this plan adds them to the physics output of arms D1 to D3 and flags that difference from D0
  (see "Caveats").
- **Experiment 394** (merged): the ensemble mean carries the XGBoost gain, the spread adds little.
  The network reads the mean for the responses and the spread only in the residual head.
- **Experiment 398** (merged) and the standing rule of 2026-10-30: scoring actuals and lags are
  cleaned (flagged `substation_zero` half-hours are nulls in experiment 87's `observed.parquet`),
  D15 and D18 are cut before 2025-03-01 in training, lags, inference and scoring, and retraining is
  monthly. All three are applied; the fold counts of dropped half-hours are printed.
- **Experiments 404, 406, 407, 410, 412** are still open. None changes this plan. Their results may
  change the control (a lagless module, no embedding), and the manager decides whether D0 is then
  replaced before the five folds.
- **Nothing in the design note changed**, except two points the note left open and this plan fixes
  below: which inputs the residual head reads in D1, and how arms are compared across the two
  groups.

## Architecture

**Every arm predicts the Primary's power as a fraction of its effective capacity, standardised per
Primary on the fold's training rows, as a 3-component Gaussian mixture (the head of experiment 397's
`PlainMdn`).** The arms differ in what produces the mixture's means.

- **D0, control.** The pooled plain network of experiment 397: all `summary` inputs, an 8-wide
  Primary embedding, 2 hidden layers of 64 (setting `g01`).
- **Shape library.** 8 learned curves (`K_s = 8`) from a small network (2 hidden layers of 32) that
  reads the calendar and holiday columns of experiment 397 (the hour of day as sine and cosine of
  one and two cycles, the day of the week, the COVID indicator, and the Christmas and bank-holiday
  columns). Each curve is normalised on a fixed grid of ordinary days (48 half-hours by 7 days of
  the week): zero mean and unit root-mean-square over the grid.
- **Response library.** 6 learned monotone curves (`K_r = 6`): 3 heating curves that fall with the
  ensemble-mean temperature below the reference of 15.5 °C and are zero above it, 1 cooling curve
  that rises above the reference and is zero below it, 1 curve of the ensemble-mean shortwave
  radiation (zero at no sunshine), and 1 curve of the cosine of the solar zenith angle (zero when
  the sun is down). Each curve is a piecewise-linear function with 16 knots, whose slopes are
  non-negative or non-positive by a softplus, so monotonicity holds by construction, and each is
  normalised to unit root-mean-square over the training range.
- **Primary mix.** An 8-wide Primary embedding feeds linear layers with softmax outputs, which give
  mixing weights over the shapes and over the responses, one non-negative amplitude per library, and
  a Primary-specific level. A third softmax-free layer gives the gate: the response part is
  multiplied by `gate(t) = exp(c · shape(t))`, divided by its mean over the grid, so the gate has a
  daily mean of 1. This is the interaction between the temperature response and the part of the day.
  The mean physics output is `mu = level + A_s * sum_i a_i shape_i(t) + A_r * sum_j b_j response_j *
  gate(t) + yearly`, where `yearly` is the sine and cosine of the day of the year at 1 and 2 cycles
  with Primary-specific coefficients (the GAM's fixed low-order harmonics).
- **Residual mixture.** The mixture's component means are `mu` plus learned offsets, and its weights
  and widths are learned. In D2 and D3 the mixture reads only the three origin lags and their
  missing flags and ages, the lead, the ensemble spread columns, and `mu`. It does not read the
  weather means or the calendar, because it could then learn the demand itself and leave the library
  curves unidentified.
- **D1, shape library only.** `mu = level + A_s * sum_i a_i shape_i(t) + yearly`. The design note
  says "plain MDN residual" and does not say what it reads. Without a response library the residual
  has no other route to temperature, so in D1 the mixture reads all of D0's inputs, the Primary
  embedding, and `mu`. D1 is therefore D0 plus a shape-library mean, not a restricted network.
- **D2, shape and response libraries with the Primary mix** (the main arm), as above.
- **D3, D2 with one fleet-wide mix.** The Primary embedding is removed everywhere: one set of mixing
  weights, amplitudes, gate, and level for all 16 Primaries.

**Identifiability constraints.** Shapes are zero-mean and unit-RMS on the grid, responses are zero
at the reference and unit-RMS, mixing weights are softmax, and the gate has a daily mean of 1. The
yearly term and the temperature response stay collinear, and so do the solar-zenith response and the
daily shapes in daylight. The first fit does not test whether those pairs separate; the stability
check in the five-fold stage does.

## Folds, rows, and compute

- **Fold and seed.** The 2026-01 fold of experiment 397 (the test month 2026-01, trained on the 24
  months before 2026-01-01), seed 0, 4 arms. Selection of rows, early stopping (one week in 12 held
  out with purging), recency weights, widths and training loop are experiment 397's.
- **Hyperparameters.** `K_s`, `K_r`, the knot count, and the embedding width are fixed above from
  the design note, and are not tuned on any scored month.
- **Cleaning and cuts.** Rows, lags, and scoring actuals from experiment 87's cleaned `observed`
  data; D15 and D18 are cut before 2025-03-01 (training, lags, inference, and scoring); the number
  of dropped half-hours is printed for the fold.
- **Cadence.** One fit per fold, equivalent to a monthly refit.
- **Compute.** One GPU process under 8 GB of GPU memory, 1 feeder core, `MemoryMax=8G`. The timed
  fit is D2 on the fold, on the GPU, plus one epoch on 1 CPU thread for comparison, as the GPU rule
  requires. Reused stored forecasts: none for D0 (its stored forecasts predate the D15 and D18 cut,
  so D0 is refitted); the GAM stack and XGBoost forecasts are references only.
- **Fitted networks** are saved as `.pt` files under the experiment's private `work/models/`.

## Scoring

- **Contrasts**, each on CRPS at days 1 to 10 per group (signed-MW six, MVA-only eight, D15 and D18
  report-only): C1 D2 minus D0, C2 D1 minus D0, C3 D2 minus D1, C4 D3 minus D2. One fold and one
  seed give no interval, and every number is interim.
- **Per-Primary table** of C1, so a gap concentrated at one Primary is visible.
- **Tails**: coverage at q95 and q99 and pinball loss at q01, q05, q95, and q99, per group, with and
  without the 1.15 widening of experiment 87.
- **Interpretation outputs**: the 8 shape curves, the 6 response curves, a table of each Primary's
  mixing weights and amplitudes, and the monotonicity check of the mixed temperature response at
  each Primary. The comparison with the GAM's fitted curves and the three-seed stability check wait
  for the five-fold stage.
- **Example forecasts**: a day-ahead forecast and a 14-day forecast from one run at a Primary, for
  D0 and D2, with median, 50% and 90% bands, and actuals.
- **Checks** (the skill's audit triggers): the median minus the mean forecast per group, the mean
  forecast's error beside the quantile score, and which of the seven triggers fired.

## Code review

**A fresh Sonnet reviewer read the code before any fit.** It found no leakage and no blocker. Six
findings were triaged: D1's mixture now reads the Primary embedding as D0's does (so D1 is D0 plus a
shape-library mean); the gate exponent is clamped to plus or minus 3; D0's initial weights are
seeded; the scorer's mean check is labelled in MW and uses the stored quantile weights; a guard
rejects a reference temperature outside the input clip range. The sixth finding is accepted as a
caveat below.

## Plan caveats

- **The monotonicity check cannot fail.** Each curve is monotone by construction, so P5 and its stop
  rule test the code, not the design.
- **Shapes and the gate are normalised on 4,096 sampled training rows**, not on a grid of ordinary
  days.

- **Yearly harmonics are in D1 to D3 and not in D0.** A gain or loss of D2 against D0 therefore
  mixes the effect of the library design with the effect of the yearly harmonics. If D2 advances,
  the five-fold stage adds D0 with the yearly harmonics as an input.
- **Lead range.** The headline band is 1 to 10 days, the horizon NGED cares about.
- **One winter month, one seed.** No claim of benefit, and cold winter nights are not scored
  separately until the five-fold stage.
- **Unlogged load-shift events.** NGED says load-shift events may be missing from their logs, so the
  rows may hold events that no label covers.
