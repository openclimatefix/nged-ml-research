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
status: planned
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
date: 2026-10-30
code: ml-experiments/00413-differentiable-physics-demand-network-part-1/
---

**Status: planned. Nothing has been fitted.** The plan follows the Design session's note
(`differentiable_physics_demand_design_answer.md`, Opus-reviewed and approved by the maintainer on
2026-10-30) and the manager's programme brief for experiment 7.

## What is asked

**The question is whether a neural network built from shared, readable parts can forecast Primary
demand as well as a plain neural network.** The aim behind the series is a forecaster that keeps the
GAM's (generalised additive model's) readable shapes and temperature curves and gains the adaptability
of the networks. Experiment 397's pooled plain network scored 3.264 points of CRPS (days 1 to 10) at
the signed-MW six, close to the GAM stack's 3.239. The first fit checks, on one winter month, whether
the library design loses much against that plain network. It is a screen to stop a poor design early,
not a ranking.

## Review of the plan against the latest evidence

- **Experiment 397** (merged): the pooled plain network is the control. Its inputs, rows, folds,
  early-stopping rule and training loop are copied. The attention gate was inert, so the design has
  none.
- **Experiment 403** (merged): pooling beat one network per Primary by 0.185 at the signed-MW six
  and 0.161 at the MVA-only eight, so the libraries are shared across all 16 Primaries.
- **Experiment 405** (merged, inconclusive) and the open plan for **experiment 409**
  ([PR 66](https://github.com/openclimatefix/nged-ml-research/pull/66)): a network with no
  day-of-year term had no other home for the monthly daylight profile, so a solar term could not be
  separated from it. The design note already keeps fixed yearly harmonics with Primary-specific
  coefficients, and this plan adds them to the physics output of arms D1 to D3 and flags that
  difference from D0 (see "Caveats").
- **Experiment 394** (merged): the ensemble mean carries the XGBoost gain, the spread adds little.
  The network reads the mean for the responses and the spread only in the residual head.
- **Experiment 398** (merged) and the standing rule of 2026-10-30: scoring actuals and lags are
  cleaned (flagged `substation_zero` half-hours are nulls in experiment 87's `observed.parquet`), D15
  and D18 are cut before 2025-03-01 in training, lags, inference and scoring, and retraining is
  monthly. All three are applied; the fold counts of dropped half-hours are printed.
- **Experiments 404, 406, 407, 410, 412** are still open. None changes this plan. Their results may
  change the control (a lagless module, no embedding), and the manager decides whether D0 is then
  replaced before the five folds.
- **Nothing in the design note changed**, except two points the note left open and this plan fixes
  below: which inputs the residual head reads in D1, and how arms are compared across the two groups.

## Architecture

**Every arm predicts the Primary's power as a fraction of its effective capacity, standardised per
Primary on the fold's training rows, as a 3-component Gaussian mixture (the head of experiment
397's `PlainMdn`).** The arms differ in what produces the mixture's means.

- **D0, control.** The pooled plain network of experiment 397: all `summary` inputs, an 8-wide
  Primary embedding, 2 hidden layers of 64 (setting `g01`).
- **Shape library.** 8 learned curves (`K_s = 8`) from a small network (2 hidden layers of 32) that
  reads the calendar and holiday columns of experiment 397 (the hour of day as sine and cosine of one
  and two cycles, the day of the week, the COVID indicator, and the Christmas and bank-holiday
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
  daily mean of 1. This is the interaction between the temperature response and the part of the
  day. The mean physics output is
  `mu = level + A_s * sum_i a_i shape_i(t) + A_r * sum_j b_j response_j * gate(t) + yearly`, where
  `yearly` is the sine and cosine of the day of the year at 1 and 2 cycles with Primary-specific
  coefficients (the GAM's fixed low-order harmonics).
- **Residual mixture.** The mixture's component means are `mu` plus learned offsets, and its weights
  and widths are learned. In D2 and D3 the mixture reads only the three origin lags and their
  missing flags and ages, the lead, the ensemble spread columns, and `mu`. It does not read the
  weather means or the calendar, because it could then learn the demand itself and leave the library
  curves unidentified.
- **D1, shape library only.** `mu = level + A_s * sum_i a_i shape_i(t) + yearly`. The design note
  says "plain MDN residual" and does not say what it reads. Without a response library the residual
  has no other route to temperature, so in D1 the mixture reads all of D0's inputs, the Primary embedding, and `mu`. D1 is
  therefore D0 plus a shape-library mean, not a restricted network.
- **D2, shape and response libraries with the Primary mix** (the main arm), as above.
- **D3, D2 with one fleet-wide mix.** The Primary embedding is removed everywhere: one set of mixing
  weights, amplitudes, gate, and level for all 16 Primaries.

**Identifiability constraints.** Shapes are zero-mean and unit-RMS on the grid, responses are zero at
the reference and unit-RMS, mixing weights are softmax, and the gate has a daily mean of 1. The
yearly term and the temperature response stay collinear, and so do the solar-zenith response and the
daily shapes in daylight. The first fit does not test whether those pairs separate; the stability
check in the five-fold stage does.

## Folds, rows, and compute

- **Fold and seed.** The 2026-01 fold of experiment 397 (the test month 2026-01, trained on the 24
  months before 2026-01-01), seed 0, 4 arms. Selection of rows, early stopping (one week in 12 held
  out with purging), recency weights, widths and training loop are experiment 397's.
- **Hyperparameters.** `K_s`, `K_r`, the knot count, and the embedding width are fixed above from the
  design note, and are not tuned on any scored month.
- **Cleaning and cuts.** Rows, lags, and scoring actuals from experiment 87's cleaned `observed`
  data; D15 and D18 are cut before 2025-03-01 (training, lags, inference, and scoring); the number
  of dropped half-hours is printed for the fold.
- **Cadence.** One fit per fold, equivalent to a monthly refit.
- **Compute.** One GPU process under 8 GB of GPU memory, 1 feeder core, `MemoryMax=8G`. The timed fit
  is D2 on the fold, on the GPU, plus one epoch on 1 CPU thread for comparison, as the GPU rule
  requires. Reused stored forecasts: none for D0 (its stored forecasts predate the D15 and D18 cut,
  so D0 is refitted); the GAM stack and XGBoost forecasts are references only.
- **Fitted networks** are saved as `.pt` files under the experiment's private `work/models/`.

## Scoring

- **Contrasts**, each on CRPS at days 1 to 10 per group (signed-MW six, MVA-only eight, D15 and D18
  report-only): C1 D2 minus D0, C2 D1 minus D0, C3 D2 minus D1, C4 D3 minus D2. One fold and one seed
  give no interval, and every number is interim.
- **Per-Primary table** of C1, so a gap concentrated at one Primary is visible.
- **Tails**: coverage at q95 and q99 and pinball loss at q01, q05, q95, and q99, per group, with
  and without the 1.15 widening of experiment 87.
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
shape-library mean); the gate exponent is clamped to plus or minus 3; D0's initial weights are seeded;
the scorer's mean check is labelled in MW and uses the stored quantile weights; a guard rejects a
reference temperature outside the input clip range. The sixth finding is accepted as a caveat below.

## Caveats

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
- **Unlogged load-shift events.** NGED says load-shift events may be missing from their logs, so
  the rows may hold events that no label covers.
