---
id: "00384"
slug: mva-only-primaries-what-is-special
theme: production-reference-and-best-guess
also_relevant_to: []
status: planned
kind: other
protocol: demand-v1
size: >-
  simple. Stored: small result tables and figures in this experiment's private folder only; no
  forecast is written and no model is fitted. Designs: the question has one defensible design, a
  rank correlation of per-Primary measures with the stored per-Primary gaps, and the measures are
  registered below before any is computed. Callers: the scripts read experiment 380's cached
  features through `studies.dp_features` and the scored gaps of experiments 377 and 378, and
  nothing imports them. Production serving path and degradation rule: not touched (this repository
  has neither).
hypothesis: >-
  The gradient-boosted-free GAM stack (experiment 377) trails BG2 at the eight Primaries that
  report only apparent power (MVA-only) for one or more of four reasons, none assumed true. (A)
  The MVA-only Primaries are older, because newer Primaries have megawatt metering. (B) They are
  bigger or smaller than the signed-MW Primaries. (C) Their demand is more random, for example
  because generators start and stop to chase price signals, which a GAM built on smooth daily and
  weekly patterns cannot follow. (D) They have more embedded generation of any kind.
prediction: >-
  Written before any measure is computed. At n = 16 only a strong relation can show, so the
  prediction is about direction. PA: the MVA-only group has an earlier first reading than the
  signed-MW group (difference of medians negative). PB: no size difference (interval for the
  difference of medians includes zero). PC: the MVA-only group has a lower share of variance in the
  weekly profile, a higher step rate, a higher 99th-percentile step, and a higher lag-1-day residual
  autocorrelation than the signed-MW group, and the Spearman correlation of the gap with the step
  rate is at least +0.4. PD: the MVA-only group has more listed generators behind its grid supply
  point. Most likely outcome: D8 alone drives the gap, and no measure separates the groups once D8
  is set aside.
decision_rule: >-
  Descriptive only; nothing is adopted. A hypothesis is called "shown" only when its direct measures
  have the predicted sign in the difference of group medians with a 95% interval from resampling
  Primaries excluding zero, and the Spearman correlation with the gap (n = 16, then n = 14 without
  the two short-history Primaries D15 and D18) has the predicted sign with |rho| at least 0.5 and a
  permutation p value under 0.05. Otherwise it is "not shown", which at n = 16 is weak evidence
  against, not proof. The place of the finding is judged by the leave-one-out range and the
  shuffled placebo. Measures that cannot be separated from the metering group are named.
benchmark: >-
  BG2 as stored by experiment 87, as scored in experiments 377 and 378 (days 1 to 10 quantile
  score per Primary).
row_set_id: fold_a_daily_gap_v1
metric: >-
  Spearman rank correlation of each per-Primary measure with the per-Primary gap (stack minus BG2,
  quantile score in points of effective capacity, days 1 to 10), and difference of group medians.
code: ml-experiments/00384-mva-only-primaries-what-is-special/
date: 2026-10-28
---

# What is special about the MVA-only Primaries?

**Plan, not yet run.** Experiment 380 measured how regular each Primary's demand is (M1 to M5) and
found no relation to the gap at n = 16. This experiment adds measures that experiment 380 did not
compute, and compares the metering groups directly. The page is rewritten when the analysis
finishes.

## Added to experiment 380's measures

Experiment 380's M1 (share of variance in a half-hour-of-week profile), M2 (extra share explained
by temperature), M3 (lag-1 half-hour residual autocorrelation), M4 (step rate above 5% of
capacity), and M5 (reverse-flow share) are reused unchanged from its stored table. New, all from
readings before 2025-07-01 with hand-filtered events excluded:

- Residual autocorrelation at lags of 1 hour and 1 day.
- Step size: median and 99th percentile of the half-hour change, and the step rate at 2% and 10%.
- Coefficient of variation of power.
- Price-chasing: correlation of the residual after profile and temperature with the Market Index
  Data half-hourly price (a proxy; experiment 25's design note) and with its half-hour change, and
  the same with the price shuffled in time as a placebo.
- Facts per Primary: metering units, effective capacity, peak load, first reading, interconnected
  neighbours, and the count and types of listed generators at its balancing-and-grid supply point.
