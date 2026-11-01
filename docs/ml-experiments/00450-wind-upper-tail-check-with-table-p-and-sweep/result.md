---
id: "00450"
slug: wind-upper-tail-check-with-table-p-and-sweep
title: >-
  Does a recalibration table meet the wind upper-tail rule, and does changing any one setting of
  the plain wind network, or pretraining it on ERA5, lower its quantile score by 0.10 points?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
protocol: wind-v1
date: 2026-11-01
size: >-
  medium. Stored: up to 165 fitted wind networks (about 3 MB each) and their quantile tables in
  this experiment's private data folder only; nothing is written to experiments 429 or 435. Step 1
  needs no fit. Serving path: not touched. Each factor has one design, fixed below before any
  step 2 score. The scripts import the folders of experiments 230, 303, 310, 326, 336, 347, and 429
  and `studies.*`. Sonnet only: one plan review, one code review, one results review, one
  write-up review.
hypothesis: >-
  The wind upper-tail rule that stopped every experiment 429 arm (more than 1.5% of observations
  above q99 at true days 1 to 10) is met once the stage-0 recalibration table P of experiment 435
  is applied, and the plain pooled wind network (429's `member_plain`) has at least one setting,
  among hidden width, depth, encoder width, mixture components, and ERA5 pretraining, whose change
  lowers its quantile score by the 0.10-point wind margin.
prediction: >-
  Step 1 (post hoc: its numbers were seen before this rule was written; interim and unaudited, see
  Results): the 10 neural arms of experiment 429 meet the pooled rule after P, narrowly, and Farm A
  alone does not. Step 2 (written before any step 2
  score): I expect no setting to move `member_plain`. In experiment 447 no pre-registered setting
  of the demand network reached its margin (a width-256 gain found afterwards fell on the hardest
  months). In experiment 429 the architecture contrasts were +0.015, +0.000, -0.004, and +0.018
  points. Experiment 430 found ERA5-and-CAMS pretraining no better for solar. The prediction that
  could fail: at least one of the 9 stage 1 arms moves `member_plain` as defined in the decision
  rule (0.10 points, interval wholly below zero, no tail harm).
decision_rule: >-
  Step 1 (no fits; post hoc, frozen after the numbers were seen). Table P is fitted per (farm, lead
  band) on the earlier months only, so the
  first scored month (2025-07) is never scored. An arm passes the upper-tail rule if the share of
  observations above q99 at true days 1 to 10 (lead 24 to 240 hours) is at most 1.5% as a point
  estimate pooled over the three farms (experiment 429's rule), and q01 and q05 coverage each move
  away from nominal by at most 0.02 (2 percentage points) against the same arm unrecalibrated.
  Reported per farm (Farm A, Farm B, Farm C) with 95%
  month-block bootstrap intervals (11 scored months), with and without experiment 210's idle-band
  mask. If no arm passes, step 2 does not run. Step 2 (pre-registered, frozen at the commit before
  the first step 2 fit). The deciding score is the raw (not table P) quantile score over 11 levels
  in points of capacity, kept rows, true days 1 to 10, on all 53 Tuesday runs of 2025-07 to
  2026-06 (12 months), from seed-mean forecasts; the all-rows score is reported beside it. The
  upper-tail condition is read after table P on its 11 scored months (point estimate, kept rows
  decides, all rows reported). The minimum detectable effect is printed first from the seed
  spread. A setting moves the network if its score is below `member_plain` by at least 0.10 points
  (the agreed wind margin), the month-block 99.4% interval (Bonferroni over the 9 stage 1 contrasts)
  is wholly below zero, the
  pooled q01 and q99 pinball losses rise by no more than 0.02 points, and the arm meets the
  upper-tail rule after P. The margin is the larger of 0.10 and the minimum detectable effect
  re-estimated from the arm's own seed spread. Stage 1 runs the 8 one-factor arms and A6 (5 seeds
  each, 45 fits). If no arm moves
  `member_plain`, the experiment stops: that is a valid result. Otherwise stage 2 runs a random
  search of at most 20 configurations over at most 3 factors, only the factors that moved, 5
  seeds each (random draws cap the cost; with 3 levels per factor the grid has at most 27
  configurations). The top 2 configurations by raw kept-row score and the reference are re-fitted
  on fresh seeds 5 to 9; a configuration is confirmed if it meets all the rules above at the 99%
  level on those seeds. The confirmation reuses the same 12 months, so it checks seed luck and not
  month luck. Every stage 1 and stage 2 number is interim until the confirmation.
benchmark: >-
  `member_plain` of experiment 429 (seeds 0 to 4, stored, reproduced bit-for-bit by this
  experiment's `base` arm), Wind farms research default 2 (429's G0 repeat), and climatology.
row_set_id: fold_a_tuesdays_v1
metric: >-
  Quantile score over 11 levels in points of capacity, kept rows, true days 1 to 10; coverage and
  pinball loss at 0.01, 0.05, 0.95, and 0.99 per farm, with and without the idle-band mask.
code: ml-experiments/00450-wind-upper-tail-check-with-table-p-and-sweep/
---

**Take-home (interim, unaudited): table P brings the pooled share above q99 down to the 1.5% limit
only narrowly, and Farm A misses it for every arm.** Pooled over the three farms, the 10 neural
arms of experiment 429 reach 1.39% to 1.50% after table P, while Wind farms research default 2
stays at 1.64%. Farm A is at 1.7% to 2.1% for every arm.

**This is the plan page.** Step 1 (no fits) has run; step 2 (the sweep) waits for the code review.

## Question

Experiment 429 found that the wind upper tail fails for every arm: 2.2% to 2.8% of observations
exceed q99, against a limit of 1.5%. Experiment 435's table P rescales each quantile's distance from
the median per farm and lead band, using only earlier months. This experiment checks whether that
table meets the limit and, if it does, sweeps the settings of the plain wind network.

## Results (interim, unaudited; step 1 only)

- **Table P brings the pooled share above q99 to the limit, narrowly.** Share of observations above
  q99 at true days 1 to 10, pooled over Farm A, Farm B, and Farm C, kept rows, after table P and
  after clipping every quantile to [0, 1] as the raw quantiles are: 1.39% to 1.50% for the 10
  neural arms (`member_plain` 1.41%, 95% month-block interval 0.85% to 1.97%), 1.64% for Wind farms
  research default 2 (0.98% to 2.29%), against 2.1% to 2.4% unrecalibrated. By experiment 429's
  point-estimate rule the 10 neural arms pass and default 2 fails. All rows: 1.33% to 1.49% and
  1.64%.
- **Farm A fails for every arm.** Farm A is at 1.7% to 2.1% (`member_plain` 1.76%, interval 0.85%
  to 2.78%); Farm B (`member_plain` 1.13%) and Farm C (1.31%) are under 1.5%. The pooled pass comes
  from Farm B and Farm C.
- **Clipping matters.** An earlier run that did not clip the rescaled q99 to 1 gave 0.7% to 1.1%,
  because observations between 1.0 and 1.09 of capacity counted as covered by a q99 above 1. That
  run was wrong; the numbers above replace it.
- **Lower tail.** q01 and q05 coverage stay within 0.02 of the unrecalibrated arm's distance from
  nominal in every arm, with and without the mask.
- **Dependency.** Table P is code-reviewed in experiment 435 (commit `552c3f87`), but its page is
  not merged yet; this page cites that commit.

## Plan for step 2

- **Base.** `member_plain`: member route, no plant, clipped head, kept rows, hidden width 64, depth
  2, encoder width 16, 3 mixture components (the plain network that experiment 429 chose).
- **Stage 1, 9 arms of 5 seeds.** Hidden width 32 and 128; depth 1 and 3; encoder width 8 and 32
  (the member encoder, the only embedding in the network); mixture components 2 and 5; and the
  pretraining arm (experiment 430's A6),
  which starts from weights pretrained on ERA5 weather and observed power before 2024-04-01 (all
  51 members set to the ERA5 value, random lead 12 to 252 hours) and fine-tunes at half the
  learning rate on the ensemble hindcast, as experiment 430 did for solar. Experiment 429 already
  covers the 2 x 2 of plant against plain network, trained with or without
  the idle band, so it is not repeated. Pretraining uses ERA5 power rows before 2024-04-01 with a
  60-day early-stopping split inside that window, standardisation from the ensemble hindcast
  training rows, and one pretraining fit per seed; ERA5 is an estimate of past weather.
- **Fits and cost.** One fit takes 56 seconds on the GPU (measured; `base` seed 0 matches 429's
  `member_plain` seed 0 exactly). Stage 1: 45 fits plus 5 ERA5 pretraining fits (the base arm reuses
  429's seeds 0 to 4), about 0.9
  GPU-hours if the pretraining fits take as long as an ordinary fit; the first one is timed. If a
  factor moves: stage 2 adds at most 100 fits (1.6 GPU-hours) and confirmation 15
  fits (0.25 GPU-hours). Upper bound 165 fits, about 2.8 GPU-hours; one GPU process under 8 GB.
- **Minimum detectable effect.** The seed standard deviation of `member_plain`'s score is 0.027
  points over seeds 0 to 4, so seeds alone give a minimum detectable effect of 0.034 points with 5
  seeds (95%, half power; indicative, since other arms may be noisier); the month blocks make it
  larger (experiment 429 quoted about 0.08). Both figures are below the 0.10-point margin.
- **Reported every time:** with and without experiment 210's mask, per farm, and with the tail
  table (q01, q05, q95, q99 pinball loss and coverage), raw and after table P.

## Problems caused by the move

None so far.
