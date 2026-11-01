---
id: "00461"
slug: wind-and-solar-retrain-with-p99-9-effective-capacity
title: >-
  Does retraining the wind and solar networks with effective capacity at the 99.9th percentile of
  earlier power keep their quantile score and bring the upper tail inside the 1.5% limit?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: change
protocol: wind-v1
date: 2026-11-01
size: >-
  medium. Stored: predictions of 5 wind fits and 36 solar refits under
  data/ml-experiments/00461-wind-and-solar-retrain-with-p99-9-effective-capacity/ (private).
  Serving path: not touched. Degradation rule: not touched. Designs: the capacity enters the
  training target, so one rescaling design (divide target and lagged power by the capacity ratio)
  is fixed in this plan. Callers: the folder's scripts, which import experiments 429, 430, 450,
  and 453. One Sonnet plan review, one Sonnet code review before the first fit, one Sonnet results
  review, and two Sonnet write-up reviews.
hypothesis: >-
  If effective capacity is the 99.9th percentile of each series' earlier power instead of the 99th
  percentile of its whole record, then retraining the best wind network and the solar networks
  keeps the quantile score in MW and brings the pooled share of actuals above the table-P q99 to at
  most 1.5%, because the networks no longer clip their upper tail at a capacity the series exceeds.
prediction: >-
  Written before any fit. Wind pooled share above q99 after table P falls from about 1.4% to about
  0.85% (experiment 453's re-scoring), Farm A to about 1.05%, with the score in MW within 0.5% of
  the old rule. Solar `a2_member` falls from about 3.4% to about 1.5%, with Solar B near 1.5% and
  Solar E near 2.4% still above the limit, and the score in MW 0.3% lower. Refitting changes the
  score in MW by less than the margin against experiment 453's re-scoring.
decision_rule: >-
  Pooled means the mean quantile score in MW over all scored half-hours of a technology's series,
  so each series weighs by its capacity. Adopt the p99.9 rule for a family if the upper bound of
  the 95% month-block interval of (new minus old) pooled score in MW lies below the margin (wind
  0.10 points of the old capacity, applied per series as 0.10 percent times its capacity in MW;
  solar 1% of the old pooled score), and its pooled share above q99 after table P is at most 1.5%
  as a point estimate (the interval is reported). If the minimum detectable effect exceeds the
  margin, the verdict is inconclusive on the score. Each series is also reported against the 1.5%
  limit and the margin, and a failing series is reported. The margins are the ones experiments 450
  and 443 set. They are non-inferiority bounds, because the rule costs one number per series to
  port. Experiment 453's re-scoring is a descriptive comparison, not a tested contrast.
benchmark: >-
  The same networks under the old capacity rule: experiment 429's stored `member_plain` forecasts
  (5 seeds) and experiment 430's stored `a2_member`, `g0_onepass`, and `a5_plain` forecasts (3
  seeds), each with table P and experiment 451's clip.
row_set_id: fold_a_tuesdays_v1
metric: >-
  Quantile score in MW (mean over the 11 levels), share of actuals above q99 after table P, and
  coverage and pinball loss at q01, q05, q95, and q99, per series, at true days 1 to 10.
---

**Plan (committed before any fit).** The sections below are fixed before the first fit.

## Question and arms

- **Question.** Experiment 453 re-scored stored forecasts at capacity p99.9 of earlier power. This
  experiment retrains, because capacity also sets the training target and the lagged-power input.
- **Wind arm.** Experiment 429's `member_plain` (plain pooled mixture density network, 51-member
  weather, no plant), 5 seeds (0 to 4), one fit per seed on training before 2025-04-16, scored on
  the 53 Tuesday runs, as in 429. Experiment 210's idle-band mask stays.
- **Solar arms.** Experiment 430's `a2_member` (headline; the arm experiment 453 scored),
  `g0_onepass`, and `a5_plain`, 3 seeds, 4 quarterly refits each (12 refit-seeds per arm).
- **Old versus new capacity.** Old: the stored forecasts, no refit. New: refit with the target (and
  for solar the lagged power) divided by the new capacity. The new capacity of a fit is the 99.9th
  percentile of the series' cleaned power before the end of that fit's training (wind: before
  2025-04-16; solar: before each refit's cutoff, the refit start minus experiment 310's refit gap).
  It is one number per fit, not per scored month, so old and new differ in the training-target
  scale as well as in the divisor, and the old capacity (p99 of the whole record) also used data
  after the fit while the new capacity does not. The page says so and prints, per refit, each
  capacity and the length of earlier history behind it. The page states the capacity of Farm A to C
  and Solar A to F as
  fractions of the old capacity.
- **Table P.** Fitted per (series, lead band) on earlier scored months only, as in experiment 450,
  with experiment 451's upper clip as coded: the larger of 1.0 and the series' largest actual in
  the earlier scored months, in each arm's own units. The floor of 1.0 is the capacity in use, so it
  differs by arm (old p99, new p99.9). The network head clips its quantiles to 0 to 1 of the capacity
  in use, so the new arm's head reaches the new capacity.

## Scoring

- **Rows.** True days 1 to 10 (lead 24 to 240 hours). Wind: the kept rows of experiment 210's mask,
  as experiments 450 and 453 scored (so 453's 0.85% is a direct prior), plus a second row set under
  experiment 432's cleaned-zero rule (exploratory, dropped if the stored old forecasts do not
  cover those rows). Solar: the rows experiment 430 scored. Row counts are equal
  across arms and printed.
- **Contrasts in MW**, because the divisor changes: new minus old quantile score in MW, per series
  and pooled over the series of a technology, month-block bootstrap (1,000 draws). Absolute scores in
  MW and in points of the old capacity sit beside every contrast. The minimum detectable effect is
  stated first, from the spread of single-seed scores of the old arm over all seed pairs. The
  month-block bootstrap is conditional on the fitted seeds (it resamples months only), and the new
  and old seeds are not paired fits, so fit-to-fit spread is also reported.
- **Bonferroni count.** Eleven intervals are reported (2 pooled plus 3 wind plus 6 solar series for
  the headline arm); the adjusted level is 99.55%. The decision rule uses the 2 pooled contrasts.
  The `g0_onepass` and
  `a5_plain` arms and the cleaned-zero row set are exploratory and outside that count.
- **Tails.** Coverage and pinball loss at q01, q05, q95, and q99, raw and after table P, per series;
  the headline is the share of actuals above q99 after table P, flagged where above 1.5%
  (experiment 453 found Solar B at 1.53% and Solar E at 2.45% under p99.9).
- **Charts (private store).** Per series, the q01 to q99 fan against the actual for a week with a
  high actual and a week with a low actual, and the worst 20 half-hours by q99 pinball loss. The
  page shows Farm A and Solar A and B only, and no name or ID.
- **Checks.** Median minus mean forecast per series; the mean-forecast error beside each quantile
  contrast; which audit triggers fired.

## Compute

- **Fits.** 5 wind fits plus 36 solar refit-seeds (3 arms) = 41 fits, about 60 s per wind seed and
  about 36 s per solar refit-seed, so well under 1 GPU-hour. The count and GPU-hours are re-stated
  after timing one fit, before the rest.
- **Limits.** One GPU process under 8 GB (`MemoryMax=8G`, `OMP_NUM_THREADS=1`), one feeder core.
- **Reused.** The old-arm forecasts of 429 and 430 (no refit), the 51-member weather caches, and
  the fit loops `e00429_run.py` and `e00430_fit.py`.

## Caveats stated in advance

- The avoidable-curtailment cost does not apply to wind and solar farms.
- Solar A, Solar B, and Solar D produced more in 2021 to 2023 than now, a level shift in the
  training history that a capacity rule does not remove.
- Wind uses 53 Tuesday runs over 11 months, and the bootstrap treats months as independent.
- Cleaned data throughout: the power is the cleaned series of experiment 453's `series.parquet`.
