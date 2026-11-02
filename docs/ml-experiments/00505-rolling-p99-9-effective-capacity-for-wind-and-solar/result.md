---
id: "00505"
slug: rolling-p99-9-effective-capacity-for-wind-and-solar
title: >-
  Does one effective-capacity rule for wind and solar, the 99.9th percentile of the most recent 365
  or 730 days of power, keep the quantile score and the upper tail at every series?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
date: 2026-11-02
size: >-
  medium. Stored: predictions of 10 wind fits and 24 solar refit-seeds under
  data/ml-experiments/00505-rolling-p99-9-effective-capacity-for-wind-and-solar/ (private). Serving
  path: not touched. Degradation rule: not touched. Designs: the window enters the training target
  through the capacity, so one rescaling design (experiment 461's) is reused and two window
  lengths are run. Callers: the folder's scripts, which import experiments 310, 347, 429, 430, 450,
  453, and 461. One Sonnet plan review, one Sonnet code review before the first fit, one Sonnet
  results review, two Sonnet write-up reviews.
hypothesis: >-
  If effective capacity is the 99.9th percentile of the power in the most recent 730 (or 365) days
  instead of all earlier power, then one rule serves wind and solar: the pooled share of actuals
  above the table-P q99 stays at most 1.5% for both technologies, no series is worse than 1.5%
  (except a series with under 8,000 valid half-hours), and no series loses score beyond its margin,
  because a trailing window follows a level shift in a series' output (Solar A, B, and D produced
  more in 2021 to 2023 according to experiment 453, a hypothesis that covers April to August only).
prediction: >-
  Written before any fit. P-730 changes the capacity ratio (new over old) of Solar A from 1.14 to
  1.04 to 1.11 and of Solar B from 1.10 to 1.12, so Solar A's score loss against the old capacity
  falls from 2.3% to under 1%. The pooled solar share above q99 stays at 1.5% or below, within 0.1
  points of P-all (1.496%). Wind is unchanged within the minimum detectable effect (0.0034 MW
  pooled). P-365 is noisier than P-730 and falls back to P-all at Solar D and Solar E.
decision_rule: >-
  One rule is adopted for both families only if all of these hold for the same window: the pooled
  share above q99 after table P is at most 1.5% for wind and for solar; no series is above 1.5%,
  except Solar E, whose window holds only about 11,000 to 14,000 daylight half-hours (its history is
  shorter than 730 days) and which is reported separately;
  no series score worsens against P-old by more than its margin (wind 0.10 points of capacity,
  stated in MW per farm; solar 1% of the pooled score); and dropping any one solar series does not
  break the pooled share rule. Per-series wind verdicts may be inconclusive when the margin is under
  the minimum detectable effect; the wind decision then rests on the pooled score and the tail
  limit. If neither window passes, wind keeps P-all, solar keeps the old capacity at Solar A and
  Solar E, and the single rule is not supported. The threshold for porting is zero gain: a rolling
  capacity is one number per series per refit, from power the serving path already stores, so it is
  worth porting if it meets the rule at no loss.
benchmark: >-
  P-all (experiment 461's stored `member_plain` and `a2_member` forecasts at p99.9 of all earlier
  power) and P-old (experiment 429's and 430's stored forecasts at the p99 of the whole record), each
  with table P and experiment 451's clip.
row_set_id: fold_a_tuesdays_v1
metric: >-
  Quantile score in MW (mean over the 11 levels), share of actuals above q99 after table P, and
  coverage and pinball loss at q01, q05, q95, and q99, per series, at true days 1 to 10.
protocol: wind-v1
code: ml-experiments/00505-rolling-p99-9-effective-capacity-for-wind-and-solar/
---

**Planned.** This page is the plan, committed before any fit. The results replace it.

## Question

Experiment 461 adopted a p99.9 effective capacity (capacity is the divisor that turns a series'
power into a fraction between 0 and 1 for the neural network) computed from all earlier power for
wind and did not adopt it for Solar A and Solar E. The maintainer asked for one rule for wind and
solar if possible. This experiment tests a rolling capacity: the 99.9th percentile of the power in
the 730 days (P-730) or 365 days (P-365) before the fit's training ends.

## Design

- **Arms**, each applied to wind `member_plain` and solar `a2_member` (experiment 461's headline
  networks), with the same seeds, folds, table P, clip, and rows as experiment 461:
    - P-all: p99.9 of all earlier power (stored, experiment 461; no refit).
    - P-730: p99.9 of the cleaned power in [cutoff - 730 days, cutoff).
    - P-365: the same over 365 days.
    - P-old: p99 of the whole record (stored, experiments 429 and 430), the reference.
- **Valid half-hours** are half-hours that pass cleaning. For solar only daylight half-hours (cleaned
  power above zero) count, for wind every cleaned half-hour counts. The percentile itself is taken
  over every cleaned half-hour of the window, night zeros included, as in experiment 461.
- **Fallback.** A window with under 8,000 valid half-hours (about 45% of a year of daylight solar
  half-hours) falls back to P-all, for both windows. Every series and refit that falls back is
  flagged. Planned from the stored series: P-365 falls back at Solar D and Solar E (6,000 to 7,300
  daylight half-hours in the window), P-730 falls back nowhere, but Solar E's history is shorter
  than 730 days, so its P-730 window is its whole history and P-730 equals P-all exactly there.
  Solar E is therefore not a test of the window, and its verdict is decided separately. Solar A's
  P-365 window holds 8,043 to 8,204 daylight half-hours, just above the threshold, so it does not
  fall back. Wind never falls back.
- **Capacity is fixed per fit,** as in experiment 461, so the arms differ in the training-target
  scale as well as in the divisor.
- **Reported alongside:** each series' capacity as a ratio to P-old at every refit, the count of
  window half-hours above the capacity (the count behind each p99.9), and the trailing p99.9
  against power over time for each series, so a reader sees whether the capacity follows the level
  shift of Solar A, B, and D.
- **Fits:** 5 wind fits (seeds) and 12 solar refit-seeds (3 seeds, 4 quarterly refits) per window,
  34 in all, about 0.44 GPU-hours at experiment 461's timings (wind about 60 s, solar about 36 s
  each). The count is confirmed after timing one fit.
- **Reused:** the scripts of experiment 461, with a window argument added to `capacity()`.
  Experiment 461's P-all forecasts and the P-old forecasts of experiments 429 and 430 are read, not
  refitted. No identity gate is rerun, because experiment 461's gate covers the unchanged fit loop;
  only `capacity()` differs.

## Scoring

- Month-block bootstrap intervals of each rolling arm minus P-all, and minus P-old, pooled and per
  series, plus six leave-one-series-out contrasts for solar. The minimum detectable effect is
  stated first (experiment 461: 0.0034 MW wind, 0.0017 MW solar, pooled). Margins are stated in MW
  beside each series' minimum detectable effect.
- Bonferroni count: 44 intervals (two rolling arms, two references, 11 intervals each: 2 pooled, 3
  wind, 6 solar), level 99.89%.
- Tails per series after table P: share above q99, coverage at q01, q05, q95, and q99, and q99
  pinball loss.
- Absolute scores beside every contrast, rows equal across arms (wind 54,899 half-hours, solar
  345,510), seed counts, and production cost.

## Standing practices

Retraining follows experiment 461's schedule (wind one fit, solar quarterly refits), not weekly;
the contrast could depend on it. The cleaned series of experiment 453 supplies the capacity. The
past-weather caveat does not apply, because these networks read the weather inputs they were
trained with in experiments 429 and 430 and the capacity reads only earlier power. Charts show
Farm A to C and Solar A to F only. The hypotheses about Solar A and Solar E are hypotheses:
experiment 453 covers April to August only, and Solar E's short history is experiment 461's remark
on noise, not an explanation of its 2.4% above q99.

## Problems caused by the move

None so far.
