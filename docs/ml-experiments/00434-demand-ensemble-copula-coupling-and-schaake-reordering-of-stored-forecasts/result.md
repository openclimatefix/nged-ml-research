---
id: "00434"
slug: demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts
title: >-
  Which re-ordering of the stored demand forecast's 51 sample paths, by ensemble member rank or by
  past observed paths, makes the paths coherent across the 16 Primaries and across leads?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
  - hierarchy-and-pooling
status: planned
kind: other
protocol: demand-v2
date: 2026-10-31
size: >-
  medium. Stored: tables and figures in this experiment's private folder only; no forecast is
  fitted or written for reuse. Serving path: not touched. Degradation rule: not touched. Designs:
  more than one defensible (which variable ranks the members, which past dates feed the Schaake
  shuffle), so the plan registers each. Callers: this folder's scripts only, which read experiment
  425's stored arm A, the stored 51-member weather frames, and the cleaned actuals, and call no
  reviewed module beyond `studies.*`. One Sonnet plan review, one Sonnet code review before the
  first run, one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  The stored forecast of experiment 425's arm A holds 11 quantiles per Primary and half-hour.
  Drawing 51 values from those quantiles and assigning them to the 51 ensemble members by the rank
  of each member's 2-metre temperature (ensemble copula coupling, ECC) gives sample paths that match
  the observed dependence across Primaries and leads better than independent draws do, by more than
  the 5% coherence margin on the variogram score, without changing the marginal CRPS. A Schaake
  shuffle from past observed paths does at least as well as ECC.
prediction: >-
  Written before any score is read. P1. All four orderings share the same 51 values at each
  Primary, half-hour, and run, so the marginal CRPS is identical across them (a check, not a
  contrast), and the 51-value CRPS is within 0.03 points of capacity of the CRPS the stored 11
  quantiles give. P2. ECC by temperature (ECC-T) lowers the joint variogram score against the
  independent draws (IND) by 5% or more. P3. The Schaake shuffle (SCH) lowers it by more than ECC-T
  does, because experiment 304 found that members correlate 0.95 to 0.97 across farms while errors
  correlate 0.54 to 0.78, so member ranks may overstate dependence. P4. Ranking by shortwave
  radiation (ECC-SW) beats ECC-T in the daylight slice and loses it in the cold-night slice.
decision_rule: >-
  Scores in points of capacity (the series' effective capacity, 1 point = 1%), over the 12 monthly
  folds of experiment 425, runs scored at their `lead_days_1_10` half-hours and at the 12 to 36 hour
  band. The coherence scores are the energy score and the variogram score (exponent 0.5, equal
  weights) of the 51 paths in three blocks: across the 16 Primaries at each half-hour (spatial),
  across the 2-hourly leads of days 1 to 10 for each Primary (temporal), and across both at 6-hourly
  leads (joint). A contrast is the relative difference of the mean score, paired by fold, with a
  month-block bootstrap (20,000 resamples) at 95% Bonferroni-corrected over the four tested
  contrasts (98.75%). The registered margins are the maintainer's proposals: 0.03 points of CRPS and
  5% of the coherence score. C1 ECC-T against IND, C2 SCH against IND, C3 SCH against ECC-T, C4 ECC-
  SW against ECC-T. An ordering counts as coherent if its joint variogram interval lies wholly below
  -5% against IND. C3 or C4 count as a difference only if the interval lies wholly beyond 5%. The
  gating block is the joint variogram score; the spatial and temporal blocks, the energy score, and
  the 12 to 36 hour band are reported beside it as secondary and interim, without a verdict. If an
  interval straddles -5% or +5%, the page says "no difference at the margin" and gives the direction
  of the point estimate; P3 and P4 are directional predictions that fail unless their interval lies
  wholly beyond the margin. The marginal CRPS needs no contrast. The comonotone ordering (CO, all
  paths ranked alike) checks that the scores can see coherence: if CO and IND score within 5% of
  each other on the joint variogram, the scores are blind and the page says so. No verdict is sought
  on the member-wise route, which is step 2.
benchmark: >-
  IND, the independent random draw, scored on the same half-hours; the stored 11-quantile CRPS of
  arm A at the 1.15 widening (experiment 425) for the marginal check; CO as the dependence upper
  bound.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Marginal CRPS of the 51 values (energy form), energy score and variogram score (exponent 0.5) of
  the 51 paths in three blocks, daylight and cold-night slices, and the share of actuals beyond the
  smallest and largest of the 51 values, in points of effective capacity; leads 12 to 36 hours and
  days 1 to 10.
code: ml-experiments/00434-demand-ensemble-copula-coupling-and-schaake-reordering-of-stored-forecasts/
---

**Plan only.** This page holds the pre-registered plan for experiment 434, step 1 of the
member-handling ablation A3 (experiment 13's design note): re-ordering the stored forecasts of
experiment 425's arm A, with no fit. The write-up replaces this section when the scores exist.

- **Stored forecasts used.** Experiment 425's arm A, seeds 0 to 4 averaged in quantile space, 12
  monthly folds, 16 Primaries, widened by 1.15 about the median (experiment 425's headline form).
- **What the stored forecasts hold.** Eleven quantile levels (q01 to q99) of the pooled mixture,
  from one pass on the ensemble mean and spread. They do not hold per-member quantiles, because arm
  A never ran the 51 members through the network. The per-member comparison is therefore not scored
  here and waits for step 2.
- **What is read besides.** The 51-member weather frames of the daily runs (2-metre temperature and
  shortwave radiation at each Primary's H3 cell, `demand_frame_path` of experiment 87) and the
  cleaned actuals of experiment 381's scoring context, minus the flagged exact zeros.
- **The 51 values.** The widened 11 quantiles are interpolated linearly in the level to the 51
  levels k/52 (k = 1 to 51). All 51 levels lie inside q01 to q99 (1/52 = 0.0192), so nothing is
  extrapolated, but the two extreme values are pinned near q01 and q99 and understate the tails
  beyond them. If the 51-value CRPS differs from the stored CRPS by more than 0.03, P1 fails for
  that reason, which the page reports as a discretisation loss and not as an effect of ordering.
- **Orderings.** ECC-T gives the coldest member the highest value (demand falls as temperature
  rises in Great Britain, where cooling is rare). ECC-SW gives the member with the least shortwave
  radiation the highest value. SCH ranks the 51 values by the ranks of 51 past observed paths: the
  dates nearest in day of year to the run, from 2025-03-01 (the D15 and D18 cut, so that all 16
  Primaries have data) to 16 days before the run (so every path of up to 14.6 days ends before the
  run), with a date dropped if any Primary lacks more than 5% of its half-hours, and gaps linearly
  interpolated. The pool size per fold is reported; if fewer than 51 dates qualify, all of them are
  used and the rest of the 51 paths repeat dates cyclically. Ties in any rank (shortwave radiation
  is zero for all members at night, so ECC-SW equals IND in expectation then) are broken by a seeded
  random draw. Each lead is ranked independently. IND is a seeded random permutation per Primary and
  half-hour. CO ranks every path alike.
- **Scoring vectors.** Values and actuals are in points of capacity before any score. Where a
  component of a block has no actual, it is dropped from that block for every ordering alike, and
  the number of complete and partial blocks per fold is reported. Scores are divided by the square
  root of the block's dimension (energy) or the number of pairs (variogram). The energy score is
  dominated by the marginal part, so the variogram score carries the dependence question.
- **Step 2.** Step 2 reuses these 51 values, orderings, and seeds, so its scores are comparable. The
  design note's gate (a route worse than 5% on the coherence score does not advance) is applied in
  step 2, not here. The contrasts are on days 1 to 10; no empirical check of the temperature sign
  is possible, because no observation exists per member.
- **Follows the standing practices.** Cleaned data with the D15 and D18 cut; no fit, so no cadence,
  seeds beyond arm A's, or model files; stored predictions reused; the tails reported as the share
  of actuals beyond the smallest and largest of the 51 values; examples of the paths drawn from one
  run. Does not apply: load-shift event plots, because the question is the dependence of the paths,
  not the response to events; the avoidable-curtailment cost, because it needs the node and its
  generators and the stored forecasts hold neither.
- **Earlier advice followed.** Compare against experiment 425's arm A on the same 12 folds; read
  the D15 and D18 columns apart.
