---
id: "00492"
slug: two-constant-tail-widening-in-place-of-table-p-for-demand
title: >-
  Can two constants per metering group replace table P for the first-release demand candidates?
theme: calibration-and-uncertainty
also_relevant_to: []
status: planned
kind: other
date: 2026-11-01
size: >-
  medium. Stored: tables under the experiment's data folder only; no neural network is fitted.
  Serving path: not touched. Degradation rule: not touched. Designs: one, because the fit rule and
  the limits come from the series brief. Callers: this folder's scripts, which import experiments
  435 and 483. One Sonnet plan review, one Sonnet code review before the first run, one Sonnet
  results review, and two Sonnet write-up reviews.
hypothesis: >-
  Table P (a factor per metering group, lead band, and tail quantile) can be replaced at the eight
  MVA-only Primaries by two constants per metering group, one for the lower tail (q01 and q05) and
  one for the upper tail (q95 and q99), fitted on earlier folds only, with no loss of calibration.
  It cannot be replaced at the signed-MW six, where table P already misses the q99 pinball clause.
prediction: >-
  Written before any score is read. P1: at the eight, A-pool's and A-gam-in's four clean-row tail
  coverages under the two-constant rule are within 0.01 of table P's, and every tail pinball loss
  is within 5% of table P's. P2: at the six, the two-constant rule misses at least one of those
  limits (probably q05 coverage and q99 pinball loss), so table P stays there. P3: the two-constant
  lower factor is below 1 and the upper factor above 1 at the eight. P4: cold winter nights stay
  under-covered at q95 under both rules.
decision_rule: >-
  The two-constant rule replaces table P for a candidate and metering group if, on the same
  held-out rows, each of q01, q05, q95, and q99 clean-row coverage differs from table P's by at
  most 0.01 and each tail pinball loss is at most 5% above table P's, and both paired
  Bonferroni-adjusted intervals lie inside those limits. An interval that spans a limit is
  inconclusive, not equivalent. Where the rule does not replace table P, table P is kept. The limits
  sit at or below the minimum detectable effect of the stored forecasts (below), so a result inside
  the limits with an interval that crosses one is reported as inconclusive. The cost of keeping
  table P in production is a table of 40 factors and its clip rule; the two-constant rule is four
  numbers, so any pass is worth porting at zero further gain.
benchmark: >-
  Table P fitted on the same folds (leave-one-fold-out and earlier-folds-only), the raw forecasts,
  the symmetric 1.15 widening, and the upper-tail-only 1.15 widening, all from experiment 483.
row_set_id: >-
  The `gam14` row set of experiment 470 (experiment 483's cache), clean rows.
metric: >-
  Coverage and pinball loss at q01, q05, q95, and q99, and CRPS in points of effective capacity.
---

**Plan only.** The question is whether the table of per-lead-band factors that recalibrates the
first-release demand candidates (table P of [experiment
483](../00483-uncertainty-rerun-on-the-first-release-demand-candidates/result.md)) can be replaced
by two constants per metering group. Experiment 483 found that a widening of both tails by 1.15
repairs the upper tail and breaks the lower one, and that a widening of the upper tail alone fixes
the eight MVA-only Primaries as well as table P does. This experiment re-scores the stored
forecasts of experiment 483; it fits no neural network.

## Plan

- **Standing practices followed:** cleaned data (inherited from experiment 483's cache); the
  experiment 399 start cut for D15 and D18 (inherited, report-only rows); the stored forecasts come
  from monthly refits; stored predictions are reused and nothing is refitted; July 2026 stays
  unused; every score is selection-biased because five folds choose and score the candidates.
- **Advice followed:** experiment 483's advice to fit recalibration on earlier folds only, to test
  the K2-425A q99 loss with table P, and not to widen the signed-MW six by 1.15.
- **Reused:** experiment 483's cache (A-pool, A-gam-in, K2-425A, five folds), `e00483_tails.py`
  (`load`, `slices`, `summarise`) and experiment 435's `widen`. The comparators `tablep_tails`,
  `strict_tails`, and `upper_tails` are stored in experiment 483's work folder.
- **Rule:** per candidate and metering group, a lower factor `a` scales the distance of q01 and
  q05 from the median, and an upper factor `b` scales the distance of q95 and q99. Each is the
  value on a 0.50 to 1.80 grid (step 0.01) that minimises the summed pinball loss of its two
  quantiles on the fitting rows (clean rows only). Lead band, cold-night class and Christmas week
  are not distinguished.
- **Two fits:** earlier-folds-only (the planning figure; 2025-10 never scored) against table P fitted
  earlier-only, and leave-one-fold-out against table P leave-one-fold-out (comparable to
  experiment 483's headline table).
- **Gating rows:** clean rows at the eight (A-pool and A-gam-in) and the signed-MW six (A-pool),
  all leads, the 12 to 36 hour band, and cold winter nights. K2-425A, D15, D18, and Christmas week
  are report-only.
- **Intervals:** paired differences in per-row coverage and pinball loss, resampled in blocks of one
  origin day within a month, 20,000 resamples. The Bonferroni count is 24 (four tail quantiles,
  coverage and pinball loss, for A-pool at the eight, A-pool at the six, and A-gam-in at the
  eight). The minimum detectable effects are experiment 483's (eight: q95 coverage 0.010, q99
  coverage 0.003, q99 pinball loss 0.0095 points; six: 0.014, 0.008, 0.018). A limit of 0.01 sits at
  or below those, so an equivalence claim cannot show a smaller difference than the minimum
  detectable effect.
- **Wind and solar:** not re-scored here. Experiment 481 already found a constant widening of 1.10
  best for wind at every number of seeds; this experiment quotes that result, and a wind or solar
  re-score would need the stored forecasts of experiments 429, 430, 450, and 461 and is left to a
  follow-up if the maintainer wants it.
- **Serving table:** the demand recalibration at serving is table P (40 factors per group and a
  clip rule) or two constants per group; the page reports which candidates and groups can use the
  second.
- **Curtailment:** the avoidable-curtailment cost at the six (f = 0.9, 0.75, 0.5) is re-scored only
  if experiment 483's curtailment script accepts the new variant without changes beyond its
  adapter.
