---
id: "00449"
slug: pretraining-with-lags-and-surprise-inputs-on-the-s1-network
title: >-
  Does pretraining the S1 demand network on ERA5 and CAMS weather help when the pretraining stage
  itself sees the recent-power lags and the surprise inputs?
theme: production-reference-and-best-guess
also_relevant_to:
  - lagged-power-and-error-correction
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: 36 pretrained networks, 48 fine-tuned networks, 12 cross-fitted lagless tables
  built on ERA5 and CAMS weather, and their forecasts, in this experiment's private folder only;
  experiments 404, 428, 439, and 448 are read and never written. More than one design is defensible
  (how the lagless forecast is built for the pretraining years, whether the lag inputs are masked in
  pretraining), so this plan fixes each choice before any score is read. Production serving path:
  not touched (this repository holds none). Degradation rule: not touched. Callers: the scripts
  copy experiment 448's fit and score code and import the folders of experiments 397, 404, 418, 422,
  428, and the `studies` evaluation code.
hypothesis: >-
  Pretraining on ERA5 and CAMS weather did not help the S1 network in experiment 448 because the
  pretraining stage hid the lag and surprise inputs, so the lag pathway was learned only from the
  shorter ensemble history. If the pretraining stage instead sees the three origin lags and the
  surprise at each lag (the surprise computed from a cross-fitted lagless forecast on ERA5 and CAMS
  weather), the fine-tuned S1 network scores lower days 1 to 10 CRPS than S1.
prediction: >-
  Written before any fit and before any score is read. P0. The reproduction arm PT0 matches
  experiment 448's S1PT: +0.05 CRPS points against S1 at the signed-MW six and at the MVA-only
  eight, within the seed spread. P1. PT1 (lags and surprise live in pretraining) beats S1 by 0.03 to
  0.10 points at the signed-MW six and by 0.05 to 0.15 at the MVA-only eight, which is the size
  experiment 425 found for a network without lags (-0.050 and -0.124). P2. PT1 beats PT0 by at
  least 0.05 points, because PT0 lost the S1 lead. P3. PT2 (token dropout 0.3 on the lag inputs)
  scores within 0.03 points of PT1 on clean rows (C5); its q99 pinball loss on rows with a lag jump
  is reported, not tested. P4. Fine-tuning at the full learning rate (PT1-full) is worse than at
  2e-4 by under 0.03 points (C9). P5. PT1b (lags live, surprise masked in pretraining) recovers part
  of PT1's gain without any lagless table; if PT1 loses to S1 and PT1b does not, the ERA5-built
  surprise is the cause (C8). The
  prediction could fail in either direction: PT1 could also lose to S1, which would mean the
  surprise inputs built from ERA5 differ too much from the ones built from the ensemble.
decision_rule: >-
  Four screen folds (test months 2026-06, 2026-01, 2025-10, 2026-03), seeds 0 to 2, monthly refits,
  1.15 widening unless stated, CRPS at days 1 to 10 in points of effective capacity (a negative
  difference is a gain), per metering group (signed-MW six; MVA-only eight; D15 and D18
  report-only, with the experiment 399 start cut) and per Primary. Every number is interim and
  carries no 99% claim. Contrasts are paired by fold and seed with the month-block bootstrap of
  `studies.eval_protocol`, at a level Bonferroni-corrected over the nine contrasts and the two
  groups (18 tests). C1 PT1 minus S1. C2 PT1-full minus S1. C3 PT2 minus S1. C4 PT1 minus PT0 (the
  effect of showing the lags to the pretraining stage). C5 PT2 minus PT1. C6 PT0 minus S1 (the
  reproduction of experiment 448). C7 PT1b minus S1. C8 PT1 minus PT1b (the effect of the surprise
  inputs in pretraining). C9 PT1-full minus PT1. Pretraining adds a stage to the pipeline, so the
  smallest gain worth porting is 0.03 points. A PT arm advances if its pooled point gain against S1
  is at least 0.03 points and the interval of that contrast lies wholly below zero (the brief's
  rule), at the signed-MW six or at the MVA-only eight, the point gain is negative in at least 3 of
  the 4 folds (the pooled paired difference of the fold's three seeds, no interval per fold), the
  other group is not above +0.02, and q95 and q99 pinball loss on clean rows is no more than +0.02
  above S1's. The minimum detectable
  effect is computed per group from the paired fold-by-seed differences of S1 seeds 0 to 2 and
  printed before any contrast is read. Real-event rows are reported with and without the largest
  event, and so is the cold-winter-night slice. The encroachment sweep of experiment 392 runs
  prediction-only on the saved networks of any arm that advances. The experiment ends at this
  screen; the full scored year is a later experiment.
benchmark: >-
  S1 seeds 0 to 2 from experiment 428 (raw forecasts, 1.15 widening); PT0 is experiment 448's S1PT
  (seeds 0 to 2); the monthly XGBoost recipe, BG2, `manual_heuristic`, and `climatology` as read by
  experiment 448.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS from the 11 quantiles at days 1 to 10; coverage and pinball loss at q01, q05, q95, q99; cold
  winter nights; sunny half-hours; avoidable-curtailment cost; clean rows and rows with a lag jump,
  with and without the largest event. Per group and per Primary.
take_home_message: >-
  Plan, not a result. This experiment tests whether the S1 demand network gains from pretraining on
  ERA5 and CAMS weather when the pretraining stage sees the same recent-power inputs that the
  network sees in use, which the earlier pretraining experiment hid. All numbers will be interim,
  from four test months.
---

**Take-home message (plan, not a result): the earlier pretraining arm may have lost because it hid
the recent-power inputs, and this experiment shows them to the pretraining stage.** Experiment 448
pretrained the S1 network with the lag and surprise inputs masked and then fine-tuned it with them
live; the result was 0.05 CRPS points worse than S1. This experiment pretrains with the inputs live.

## Question

Experiment 448 pretrained the S1 network on ERA5 and CAMS weather with every recent-power input
masked, then fine-tuned it on ensemble rows with the inputs live. The result lost to S1 by 0.05
points at the signed-MW six and at the MVA-only eight. The maintainer expected that pretraining
across all history and all Primaries would add signal. This experiment asks whether the loss came
from the masking: the lag pathway of the pretrained network never saw a lag, so fine-tuning on the
shorter ensemble history could not recover it.

## Plan against the latest evidence

- **Merged since the brief was written (2026-11-01).** Experiments 440, 446, and 447 (the
  hyperparameter sweep) are merged. None changes the S1 recipe this plan uses. Experiment 447's
  width-256 finding is interim and unaudited, so the S1 reference keeps hidden width 64.
- **Reused stored forecasts.** Only seeds 0 to 2 of S1 and PT0 are used (both are stored to seed 4);
  the minimum detectable effect comes from S1's seeds, so the PT arms' own seed spread is read from
  the result. S1 seeds 0 to 2 (experiment 428) and PT0, which is experiment 448's
  S1PT (stored networks and forecasts in experiment 448's private folder). Their rows are the same
  four folds, worlds, and row set, so no S1 or PT0 network is refitted. One PT0 fold-seed is
  refitted
  as a reproduction check, and its validation loss must equal the stored value.
- **Standing practices followed.** Monthly refits; cleaned data (flagged and start-cut half-hours
  dropped, counted per fold); the D15 and D18 cut at 2025-03-01 applied; every network saved in the
  data folder; tails, cold nights, and avoidable-curtailment cost reported; example forecasts and
  event plots in the write-up; draft PR; Sonnet for every review. Process: Sonnet writes the code; a
  fresh Sonnet agent reviews it before the first fit; a fresh Sonnet agent reviews the results and
  re-derives the headline numbers; the write-up carries the "Checks" bullets (median minus mean
  forecast per site, mean-forecast error beside each quantile contrast, which audit triggers fired);
  a fresh Sonnet agent reviews the write-up. Nulled inputs: for PT2 the page reports interval width
  and coverage on rows whose lags are missing against rows with live lags. Avoidable-curtailment
  cost is reported during events, in the 7 days after, and across all rows, per group and per
  Primary, headlined at the scaled limit (f = 0.9, sweep 0.75 and 0.5) against `manual_heuristic` in
  GBP per MW per day with an interval. Pools start with the spawn method.

## Arms

| Arm | Pretraining stage | Fine-tuning stage | New networks |
|---|---|---|---|
| S1 | none | ensemble rows, lr 2e-3 | 0 (stored) |
| PT0 | ERA5 and CAMS, lags and surprise masked | lr 2e-4 | 0 (experiment 448's S1PT) |
| PT1 | ERA5 and CAMS, lags and surprise live | lr 2e-4 | 12 + 12 |
| PT1-full | the PT1 pretrained network | lr 2e-3 | 12 |
| PT1b | ERA5 and CAMS, lags live, surprise masked (zero) | lr 2e-4 | 12 + 12 |
| PT2 | ERA5 and CAMS, each lag masked with probability 0.3 | lr 2e-4 | 12 + 12 |

PT1b is a diagnostic that needs no lagless table: it separates the effect of the lags from the
effect of the surprise. Each lag token (the lag value, its missing flag, its age, and its surprise)
is masked as a unit in PT2,
so the network keeps a pathway that works without the lags. Fine-tuning shows all lags. PT1 and
PT1-full share one pretrained network per fold and seed.

## How the surprise is built for the pretraining years

**The surprise at a lag is the lag minus a lagless forecast of the same half-hour, divided by the
lagless forecast's half band width.** The lagless forecast must not have seen the half-hour it
forecasts, or the surprise shrinks towards zero in pretraining and the network learns a lag pathway
that fails on ensemble rows. The plan therefore builds one cross-fitted lagless table per fold and
seed
on ERA5 and CAMS weather, by the method of experiment 404's table:

- **Span and blocks.** 2019-09-01 to the fold's refit, split into 6 contiguous blocks. A lagless
  module fitted on 5 blocks forecasts the sixth. Nothing at or after the refit enters.
- **Module and inputs.** Experiment 404's lagless module (a pooled mixture density network with a
  Primary embedding, 64 hidden units, 6 epochs, learning rate 2e-3). The inputs are the 7 ERA5 and
  CAMS
  weather means, the solar-position, calendar, and holiday columns of experiment 397, and no lag and
  no lead. Experiment 404's ensemble spread columns are dropped because ERA5 has no spread. Rows
  have
  equal weights.
- **Target.** Cleaned power over capacity (world 0), with flagged half-hours and the D15 and D18
  start
  cut as missing and left out of the fits.
- **Table.** Per `(primary, grid)`: the q50 and half the q05-to-q95 width, in the layout the S1 code
  already reads.
- **Rescale.** The smoke check on 2026-01 seed 0 found the ERA5 table's surprise standard deviation
  (1.245) well above the ensemble table's (0.850), over 2025-02 to 2026-01. The ERA5 half band width
  is therefore multiplied, per Primary, by the ratio of the two surprise standard deviations over
  the half-hours before the refit where both tables exist (the ensemble years), so the pretraining
  surprise has the scale the network sees in fine-tuning. The ratio uses no data at or after the
  refit. The ratios are printed per fold and Primary; a ratio outside 0.5 to 2 stops the run.
- **Overlap years.** Pretraining rows after 2024-04 use the ERA5 table in pretraining and the
  ensemble table in fine-tuning, for the same half-hours.
- **Non-causal and unpurged.** A module forecasting a block is trained on later blocks too, as in
  experiment 404, and the held-out and purged weeks of pretraining are not excluded from the other
  blocks' fits. Neither uses anything at or after the refit.
- **What differs from experiment 404's recipe.** The span is 2019-09 to the refit rather than 24
  months, rows have equal weights, and the inputs are ERA5 and CAMS, so the ERA5 modules are not
  identical in quality to 404's. The further module of experiment 404 (scoring months) is not
  needed.
- **What differs from use.** In fine-tuning, and in use, the surprise comes from experiment 404's
  table, built from ensemble weather. The pretraining surprise comes from ERA5 and CAMS weather,
  which
  is more accurate, so its half-hours are less surprising on average. A smoke check prints the
  surprise standard deviation by source and asserts that the pretraining table differs from the
  fine-tuning table. ERA5 arrives about 5 days late and is not available at
  forecast time; the live service uses the ensemble-based table, as the fine-tuned networks do.

## Compute

| Stage | Networks |
|---|---|
| Lagless tables: 4 folds by 3 seeds by 6 blocks | 72 small modules |
| Pretraining: PT1, PT1b, and PT2, 4 folds by 3 seeds each | 36 |
| Fine-tuning: PT1, PT1-full, PT1b, PT2 | 48 |
| Reproduction of one PT0 fold and seed | 1 |

- **Time.** Experiment 448's pretraining took under a minute per network and its fine-tuning 10 to
  60
  seconds. The first fit is timed, and the GPU-hour estimate is corrected in the PR.
- **Slot.** One GPU process under 8 GB, one feeder core, `MemoryMax=8G`, `-p OOMPolicy=continue`,
  `OMP_NUM_THREADS=1`, `free -g` before each launch. The reproduction check accepts a validation
  loss within 0.002 of the stored value, because cuDNN autotuning and GPU reductions are not bitwise
  repeatable.

## Figures

The headline chart is CRPS by lead day for S1, PT0, PT1, PT1-full, PT2, the monthly XGBoost recipe,
BG2, `manual_heuristic`, and `climatology`, with intervals. Further figures: the contrasts by fold,
tail coverage, the surprise distribution by source, example day-ahead and 14-day forecasts, and six
lag-jump event plots across both metering groups with the unlogged-event caveat.
