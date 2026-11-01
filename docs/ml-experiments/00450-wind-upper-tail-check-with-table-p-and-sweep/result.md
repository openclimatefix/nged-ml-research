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
  medium. Stored: up to 160 fitted wind networks (about 3 MB each) and their quantile tables in
  this experiment's private data folder only; nothing is written to experiments 429 or 435. Step 1
  needs no fit. Serving path: not touched. Designs: one defensible per factor, fixed below before
  any score. Callers: the scripts import the folders of experiments 230, 303, 310, 326, 336, 347,
  and 429 and `studies.*`. Sonnet only: one plan review, one code review, one results review, one
  write-up review.
hypothesis: >-
  The wind upper-tail rule that stopped every experiment 429 arm (more than 1.5% of observations
  above q99 at true days 1 to 10) is met once the stage-0 recalibration table P of experiment 435
  is applied, and the plain pooled wind network (429's `member_plain`) has at least one setting,
  among hidden width, depth, encoder width, mixture components, and ERA5 pretraining, whose change
  lowers its quantile score by the 0.10-point wind margin.
prediction: >-
  Written before step 2 and before any step 2 score. Step 1 was run first, and its numbers are
  interim and unaudited (see Results): every 429 arm meets the rule after P. Step 2: I expect no
  setting to move `member_plain` by 0.10 points. Experiment 447 moved no setting of the demand
  network by its margin, 429 found the wind architecture choices tied within 0.03 points, and
  experiment 448 found ERA5-and-CAMS pretraining raised the demand network's CRPS. The prediction
  that could fail: at least one of the 9 screen arms beats `member_plain` by 0.10 points with a
  99% interval wholly below zero.
decision_rule: >-
  Step 1 (no fits). Table P is fitted per (farm, lead band) on the earlier months only, so the
  first scored month (2025-07) is never scored. An arm passes the upper-tail rule if the share of
  observations above q99 at true days 1 to 10 (lead 24 to 240 hours) is at most 1.5% pooled over
  the three farms, and each of q01 and q05 coverage moves away from nominal by at most 0.02
  against the same arm unrecalibrated. Reported per farm (Farm A, Farm B, Farm C) with 95%
  month-block bootstrap intervals (11 scored months), with and without experiment 210's idle-band
  mask. If no arm passes, step 2 does not run. Step 2. Score is the quantile score over 11 levels
  in points of capacity, kept rows, true days 1 to 10, on the 53 Tuesday runs of 2025-07 to
  2026-06, seed-mean forecasts. The minimum detectable effect is printed first from the seed
  spread. A setting moves the network if its score is below `member_plain` by at least 0.10 points
  (the agreed wind margin), the 99% interval over months is wholly below zero, the pooled q01 and
  q99 pinball losses rise by no more than 0.02 points, and the arm meets the upper-tail rule after
  P. Stage 1 runs the 8 one-factor arms and A6 (5 seeds each, 45 fits). If no arm moves
  `member_plain`, the experiment stops: that is a valid result. Otherwise stage 2 runs a random
  search of at most 20 configurations over at most 3 factors, only the factors that moved, 5
  seeds each. The top 2 configurations by score and the reference are re-fitted on fresh seeds 5
  to 9 as confirmation; a configuration is confirmed if it still meets the rules above. Every
  stage-1 and stage-2 number is interim until the confirmation.
benchmark: >-
  `member_plain` of experiment 429 (seeds 0 to 4, stored, reproduced bit-for-bit by this
  experiment's `base` arm), Wind farms research default 2 (429's G0 repeat), and climatology.
row_set_id: fold_a_tuesdays_v1
metric: >-
  Quantile score over 11 levels in points of capacity, kept rows, true days 1 to 10; coverage and
  pinball loss at 0.01, 0.05, 0.95, and 0.99 per farm, with and without the idle-band mask.
code: ml-experiments/00450-wind-upper-tail-check-with-table-p-and-sweep/
---

**This is the plan page.** Step 1 (no fits) has run; step 2 (the sweep) waits for the plan and code
reviews.

## Question

Experiment 429 found that the wind upper tail fails for every arm: 2.2% to 2.8% of observations
exceed q99, against a limit of 1.5%. Experiment 435's table P rescales each quantile's distance from
the median per farm and lead band, using only earlier months. This experiment checks whether that
table meets the limit and, if it does, sweeps the settings of the plain wind network.

## Results (interim, unaudited; step 1 only)

- **Table P brings every arm under the limit.** Share of observations above q99 at true days 1 to
  10, pooled over Farm A, Farm B, and Farm C, kept rows: 0.70% to 1.00% for the 10 neural arms
  (`member_plain` 0.80%, 95% month-block interval 0.48% to 1.13%), 1.19% for Wind farms research
  default 2 (0.71% to 1.73%), against 2.0% to 2.4% unrecalibrated. All rows: 0.77% to 1.12% and
  1.30%.
- **Per farm.** Point estimates are at most 1.5% for every arm and farm (largest: default 2 at
  Farm A, 1.46% on kept rows and 1.43% on all rows). Some per-farm upper bounds exceed 1.5%, because
  the scored year gives 11 months to resample.
- **Lower tail.** q01 and q05 coverage moves closer to nominal or stays within 0.02 of it in every
  arm, with and without the mask.
- **Dependency.** Table P is code-reviewed in experiment 435 (commit `552c3f87`), but its page is
  not merged yet; this page cites that commit.

## Plan for step 2

- **Base.** `member_plain`: member route, no plant, clipped head, kept rows, hidden width 64, depth
  2, encoder width 16, 3 mixture components (the manager and the neural Discuss session approved
  it as the base on 2026-11-01).
- **Stage 1, 9 arms of 5 seeds.** Hidden width 32 and 128; depth 1 and 3; encoder width 8 and 32
  (the member encoder, the only embedding in the network); mixture components 2 and 5; and A6,
  which starts from weights pretrained on ERA5 weather and observed power before 2024-04-01 (all
  51 members set to the ERA5 value, random lead 12 to 252 hours) and fine-tunes at half the
  learning rate on the ensemble hindcast, as experiment 430 did for solar. The H-P 2 x 2 (plant
  against plain, idle band in or out) is already covered by experiment 429, so it is not repeated.
- **Fits and cost.** One fit takes 56 seconds on the GPU (measured; `base` seed 0 matches 429's
  `member_plain` seed 0 exactly). Stage 1: 45 fits plus 5 ERA5 pretraining fits, about 0.9
  GPU-hours. If a factor moves: stage 2 adds at most 100 fits (1.6 GPU-hours) and confirmation 15
  fits (0.25 GPU-hours). Upper bound 165 fits, about 2.8 GPU-hours; one GPU process under 8 GB.
- **Minimum detectable effect.** The seed standard deviation of `member_plain`'s score is 0.027
  points over seeds 0 to 4, so seeds alone give a minimum detectable effect of 0.034 points with 5
  seeds; the month blocks make it larger (experiment 429 quoted about 0.08). Both are below the
  0.10-point margin.
- **Reported every time:** with and without experiment 210's mask, per farm, and with the tail
  table (q01, q05, q95, q99 pinball loss and coverage), raw and after table P.

## Problems caused by the move

None so far.
