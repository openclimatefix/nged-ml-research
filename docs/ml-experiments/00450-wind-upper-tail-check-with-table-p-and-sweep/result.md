---
id: "00450"
slug: wind-upper-tail-check-with-table-p-and-sweep
title: >-
  Does a recalibration table meet the wind upper-tail rule, and does changing any one setting of
  the plain wind network, or pretraining it on ERA5, lower its quantile score by 0.10 points?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: finished
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
verdict: not supported
best_arm: >-
  none. The plain network (`member_plain`, 10.533 points) is not beaten by any setting; the nearest
  are depth 1 (10.521, -0.012) and 2 mixture components (10.522, -0.011).
headline: >-
  After table P, 1.39% to 1.50% of observations fall above q99 for the 10 neural arms of
  experiment 429 pooled over the three farms (Wind farms research default 2: 1.64%), but Farm A
  stays at 1.7% to 2.1%. None of 9 changes to the plain wind network (hidden width, depth, encoder
  width, mixture components, ERA5 pretraining) lowers its quantile score by the 0.10-point margin,
  and none fixes Farm A.
lessons: >-
  The wind network is insensitive to its size and to ERA5 pretraining, as the demand network was in
  experiment 447. The upper-tail failure at Farm A is not a size or pretraining problem and a
  per-farm table P does not remove it. A rescaled quantile must be clipped to the capacity range
  before coverage is counted; without the clip the same table appeared to pass every farm.
take_home_message: >-
  Table P nearly meets the wind upper-tail limit when the three farms are pooled (1.4% of
  observations above q99, limit 1.5%) but not at Farm A (1.8%), and no change to the size of the
  plain wind network or ERA5 pretraining moves its score by the 0.10-point margin or fixes Farm A.
  The wind network is at its plateau on these inputs, so the open problem is Farm A's windy
  half-hours, not the network.
advice:
  - >-
    Clip every rescaled or widened quantile to the capacity range [0, 1] before counting coverage
    (experiment 435's table P code does not).
  - >-
    Keep the plain `member_plain` wind network at its experiment 429 size; the sweep found no
    better setting.
  - >-
    Look at Farm A's windy half-hours directly, for example the share above q99 by forecast
    wind-speed band, before trying more network settings.
production_status: not-for-production
adopted_in: none
code: ml-experiments/00450-wind-upper-tail-check-with-table-p-and-sweep/
---

**Take-home (interim, unaudited): the plain wind network is at a plateau, and Farm A's upper tail is
the open problem.** After table P, 1.39% to 1.50% of observations fall above q99 for the 10 neural
arms of experiment 429 pooled over the three farms (limit 1.5%; Wind farms research default 2:
1.64%), but Farm A stays at 1.7% to 2.1% in every arm and setting. None of 9 changes to the plain
network lowers its quantile score by the 0.10-point margin, so the stop rule ended the sweep after
stage 1 (45 fits plus 5 pretraining fits).

![Quantile score of each setting minus the base network's, with 99.4% month-block intervals](figure_1_contrasts.svg)

![Share of observations above q99 after table P, by farm and setting](figure_2_upper_tail_by_farm.svg)

## Results

- **Step 1, table P (kept rows, true days 1 to 10, after clipping to [0, 1]).** Pooled share above
  q99: 1.39% to 1.50% for the 10 neural arms (`member_plain` 1.41%, 95% month-block interval 0.85%
  to 1.97%), 1.64% for default 2 (0.98% to 2.29%), against 2.1% to 2.4% before table P. Farm A:
  `member_plain` 1.76% (0.85% to 2.78%), default 2 2.07%; Farm B 1.13%; Farm C 1.31%. All rows:
  1.33% to 1.49% and 1.64%. q01 and q05 coverage are not harmed. An earlier run without the clip
  gave 0.7% to 1.1%; observations between 1.0 and 1.09 of capacity counted as covered by a q99
  above 1. Experiment 435's table P code has the same missing clip.
- **Step 2, stage 1 (kept rows, true days 1 to 10; score and difference to `member_plain`,
  99.4% interval).** `member_plain` 10.533. Hidden width 32: -0.001 (-0.046 to +0.062); width 128:
  +0.034; depth 1: -0.012 (-0.046 to +0.033); depth 3: +0.061 (+0.007 to +0.112); encoder width 8:
  +0.077; encoder width 32: +0.003; 2 mixture components: -0.011; 5: +0.007; ERA5 pretraining:
  +0.039 (-0.020 to +0.137). Positive is worse. No setting reaches -0.10.
- **Tails of the settings.** The q99 pinball loss rises by 0.067 points (2 components) and 0.063
  (ERA5 pretraining), above the 0.02 allowance, and by 0.022 (encoder width 8); the share above q99
  after table P is 1.36% to 1.55% pooled. Farm A is at 1.69% (encoder width 32) to 1.88% (2
  components) in every setting.
- **Seed spread.** The `member_plain` seed standard deviation is 0.027 points, a seed-only minimum
  detectable effect of 0.034 for 5 seeds; the month-block intervals above are wider.
- **Cost.** 50 fits of about 56 to 69 seconds each, about 0.8 GPU-hours, one process of 5.5 GB.
  Stage 2 and the confirmation did not run, because no setting moved the network.

## Caveats

- Step 1 was run before its rule was written; the rule is post hoc. The step 2 rule was fixed
  before the first stage 1 fit.
- The scored year gives 12 months to resample; per-farm intervals include 1.5%.
- Table P's page (experiment 435) is not merged; this page cites its code at commit `552c3f87` and
  adds the clip.
- Every number is interim and unaudited; the results review has not run yet.

## Problems caused by the move

None.
