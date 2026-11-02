---
id: "00523"
slug: lead-awareness-of-the-lag-free-network-widening-and-pretraining-by-lead
title: >-
  Does the lag-free neural network know what lead time means, and do per-lead widening or ERA5
  pretraining explain its long-lead gap to K2-S1?
theme: calibration-and-uncertainty
also_relevant_to:
  - lagged-power-and-error-correction
  - retraining-and-training-history
status: finished
kind: other
protocol: demand-v2
size: >-
  medium. What gets stored: score tables and figures under this experiment's private data folder,
  and the page; no forecast table of other experiments is changed, and no network is fitted.
  Designs:
  more than one is defensible for the per-lead widening (how the factor is estimated), so the plan
  fixes it below. Callers: the scripts read experiments 511, 521, 454, and 448 stored outputs and
  import experiments 454, 435, 521, 446, and 428 for the scorer and the input layout. One Sonnet
  plan
  review, one Sonnet code review before the first number is used, one Sonnet results review, and two
  Sonnet write-up reviews. CPU only, 2 cores, 8 GB.
hypothesis: >-
  The maintainer's hypothesis is that pretraining on ERA5 (where the weather is the truth at every
  lead) never teaches the lag-free network that a longer lead means a larger weather error, so the
  network does not widen its forecast with lead. Experiment 521 already found that L-1d (the
  daily-refit lag-free network) and K2-S1 (the network with lagged and lag-free heads) widen by
  about
  the same amount from 12 to 24 h to days 6 to 10, so this experiment asks the narrower question of
  whether the stored bands are the right width at each lead, and whether the network's output
  changes when its lead input changes.
prediction: >-
  Written before any score is read. P1. The lead swap: setting every row of L-1d to day 10 widens
  the
  90% band by under 10% and moves the median by under 1% of capacity, because the network mostly
  ignores lead. P2. Stratified by hour of day, L-1d's 90% width from 12-24 h to 120-240 h grows by
  within 0.05 (as a ratio) of K2-S1's growth, as 521's pooled bands suggest. P3. Per-band widening
  factors fitted only on earlier held-out forecasts (a point-estimate threshold; the counted test is
  separate, see the decision rule) lower L-1d's CRPS at 120-240 h by at least 0.03 points at
  the six signed-MW Primaries (q95 coverage 0.934, short of 0.95) and by less than 0.03 at the eight
  MVA-only Primaries (q95 coverage 0.960, already above 0.95). P4. ERA5-and-CAMS pretraining helps
  below 2 days and hurts at days 5 to 10: L0PT minus LL (the same lag-free network without
  pretraining, ensemble-only) is negative in the bands below 48 h and positive at 120-240 h (the
  bands from 48 to 120 h are not predicted). P4 fails if L0PT is as good or better in every band.
decision_rule: >-
  Report-only for band width and the probability integral transform (PIT) histogram. A CRPS contrast
  counts only when its Bonferroni interval excludes zero (28 counted CRPS tests: 7 lead bands x 2
  groups x 2 contrasts, the two contrasts being Part 2's per-band widening minus flat 1.15 and the
  pretrained-minus-unpretrained contrast; post hoc, the counted unpretrained control is experiment
  448's own L0, because experiment 454's LL, named in the plan, differs in training recipe, and L0PT
  minus LL is report-only) under origin-day blocks, with whole-month
  blocks (4 folds for each contrast, so near the minimum-to-maximum of the folds) as the
  conservative check. The q95 and q99 pinball contrasts are counted separately, 56 tests each, and
  the 0-12 h non-worsening check below is one more test. P3's 0.03 is a point-estimate threshold,
  and the per-band minimum detectable effect (about 0.12 to 0.27 in experiment 521) is larger, so P3
  can hold as a point estimate without a counted contrast. Per-band widening counts as closing the
  gap only if it closes at least half of L-1d's 120-240 h CRPS gap to K2-S1 (0.174 at the eight,
  0.196 at the six). P3 and P4 are tested at the bands
  named, and the minimum detectable effect (from the seed spread) is printed beside every contrast.
  Part 3 (training with a lead-aware head) is not run here. This experiment only reports whether its
  gate holds: either L-1d's q95 coverage at 120-240 h is below K2-S1's by at least 0.03 more than at
  12-24 h (K2-S1 is the control), or P1 fails because the lead swap shows lead is ignored. The
  smallest gain worth porting per-band widening is zero CRPS (it is a
  table of constants and the serving path already applies a flat factor) but it must not worsen the
  0-12 h bands by more than 0.02.
benchmark: >-
  L-1d (experiment 511, two-seed quantile average, 1.15 widening) against K2-S1 (experiment 454,
  monthly), L-Ww, and experiment 454's monthly LL; for the pretrained contrast, experiment 448's
  L0PT (monthly,
  pretrained, seeds 0 to 2) against experiment 454's LL on the same rows.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by lead band (0 to 6 h, 6 to 12 h, 12 to 24 h, 24 to 48 h, 48
  to 72 h, 72 to 120 h, 120 to 240 h), q01, q05, q95, and q99 coverage and pinball loss, mean 90%
  width, and the PIT histogram.
verdict: partly supported
best_arm: >-
  Not a ranking of arms. At 120 to 240 h at the eight MVA-only Primaries, K2-S1 scores 4.867 points
  of CRPS (continuous ranked probability score), L0PT (pretrained) 5.455, L0 (same network, not
  pretrained) 5.596, and LL 5.084.
headline: >-
  The daily-refit lag-free network L-1d does not widen its bands with lead: replacing its lead input
  changes the mean 90% band width by under 4%, and within one half-hour of the day the band grows
  only 5% from 12-24 h to 120-240 h (K2-S1: 4%). Per-lead widening fitted on earlier held-out
  forecasts changes CRPS by at most 0.024 points. ERA5 pretraining lowers CRPS in every lead band
  (L0PT minus L0: -0.14 at 120-240 h at the eight, -0.05 at the six).
lessons: >-
  The 27 to 30% widening of the pooled bands from 12-24 h to 120-240 h is mostly time of day,
  because every forecast origin is at 09:00 UTC and so lead fixes the half-hour of the target. The
  long-lead gap to K2-S1 is not a band-width problem. Re-fitting the width per lead band recovers
  under 5% of that gap. Pretraining does not cause the gap.
take_home_message: >-
  The lag-free neural network does not make its forecast bands wider at longer lead times, but
  neither does K2-S1, and re-fitting the band width per lead time recovers almost none of K2-S1's
  advantage at days 5 to 10. ERA5 pretraining does not explain the gap either: it lowers the
  error at every lead time. The next thing to test is why K2-S1 and the monthly five-head lag-free
  arm score better at days 5 to 10, with the centre of the forecast (not the band width) as the
  suspect.
advice:
  - >-
    Do not build a per-lead widening table for the lag-free network to close K2-S1's gap: it
    recovers under 5% of the 120-240 h gap (Part 2 table).
  - >-
    Stratify any width-by-lead comparison by the half-hour of the day, because every origin is at
    09:00 UTC and lead fixes the target's half-hour (width growth table).
  - >-
    Keep ERA5-and-CAMS pretraining in the lag-free network: against the same network without it,
    it lowers CRPS in all seven lead bands (figure 1).
  - >-
    Compare pretrained and unpretrained networks only when they share one training recipe: the
    pretrained network against experiment 454's lag-free heads reversed the sign at 120-240 h
    (Checks).
production_status: not-for-production
adopted_in: none
---

**The lag-free network does not widen its bands with lead, but neither does K2-S1, and re-fitting
the band width per lead recovers almost none of K2-S1's advantage at days 5 to 10.** We asked
whether the daily-refit lag-free neural network L-1d has learned what "lead time" (how far ahead
the forecast looks) means, because it starts from weights pretrained on ERA5 weather, where the
weather is the truth at every lead. We read the forecasts that experiments 511, 454, and 448 had
stored, scored them by lead band, and fed experiment 511's stored L-1d weights altered lead inputs.
Replacing the lead input changed the mean 90% band width by under 4%. The band grows only 5% from
12-24 h to 120-240 h once the half-hour of the day is held fixed, against 27% in the pooled
figures. K2-S1, the network with lagged and lag-free heads, grows by 4%. Per-lead widening fitted
only on earlier held-out forecasts changed CRPS by at most 0.024 points in any band. ERA5
pretraining lowers CRPS at every lead. Every number on this page is interim and unaudited until a
human has read it.

Here "the eight" are the eight MVA-only (megavolt-ampere-metered) Primary substations and "the six"
are the six signed-MW (megawatt-metered) Primary substations, as in experiment 521. CRPS (the
continuous ranked probability score, lower is better) is in points of effective capacity. This
experiment fitted no neural network and used no GPU.

![Pretrained minus unpretrained](figure_1_pretrained_minus_unpretrained.svg)

![Width growth](figure_2_width_growth.svg)

![Lead swap](figure_3_lead_swap.svg)

![Per-band widening](figure_4_per_band_widening.svg)

![PIT at 120 to 240 h](figure_5_pit.svg)

![Day-ahead forecast, D4, winter](figure_6_example_winter_day_ahead.svg)

![10-day forecast, D4, winter](figure_6_example_winter_10_days.svg)

![Day-ahead forecast, D4, summer](figure_6_example_summer_day_ahead.svg)

![10-day forecast, D4, summer](figure_6_example_summer_10_days.svg)

## Predictions

- **P1 held for the width and failed for the median.** Setting every row to day 10 changes L-1d's
  mean 90% width by -4% to +3% across lead bands (under the predicted 10%). It moves the median by a
  mean absolute 0.7 to 4.1 points of capacity (day 10), against the predicted 1% of capacity or
  less.
  The network ignores lead for the band and uses it for the centre of the forecast.
- **P2 held.** Within one half-hour of the day the 90% width grows by 4.9% (L-1d) and 3.7%
  (K2-S1) at the eight and by 5.0% and 4.6% at the six, differences of 0.012 and 0.003 as ratios.
- **P3 failed.** Per-band widening lowers CRPS at 120-240 h by 0.008 points at the six
  (Bonferroni interval -0.012 to -0.003, excludes zero) and by 0.002 at the eight (includes
  zero). The predicted 0.03 at the six was not reached. It closes 4% of the six's 0.196 gap and 1%
  of the eight's 0.174.
- **P4 failed.** Against L0, the same network without pretraining, the pretrained L0PT has the
  lower CRPS in every band (see Results). Against LL, experiment 454's lag-free heads, L0PT is
  worse at 120-240 h by 0.37 (eight) and 0.53 (six); that contrast is kept below as
  recipe-confounded.

## Results

- **Pretraining helps at every lead (like-for-like).** L0PT minus L0, in points of CRPS (negative
  means pretraining is better; Bonferroni intervals over origin-day blocks, 28 counted tests).
  Eight: -0.39 [-0.58, -0.22] at 0-6 h, -0.14 [-0.23, -0.05] at 120-240 h, -0.13 [-0.21, -0.06]
  pooled. Six: -0.18 [-0.33, -0.04] at 0-6 h, -0.05 [-0.09, -0.01] at 120-240 h, -0.08
  [-0.11, -0.05] pooled. The whole-month intervals (4 blocks, near the range of the folds) exclude
  zero in 1 of the 7 bands at the eight and in 6 of the 7 at the six.
- **The contrast with LL is confounded.** L0PT minus LL is +0.37 [+0.22, +0.53] at 120-240 h at
  the eight and +0.53 [+0.33, +0.73] at the six, but LL is the cross-fitted lag-free table of
  experiment 404, trained with another recipe. LL scores 5.084 and L0 5.596 at the eight, so the
  difference lies in the recipe and not in pretraining.
- **Per-band widening gains under 0.03 everywhere.** Per-band minus flat 1.15, CRPS: -0.002
  (eight) and -0.008 (six) at 120-240 h; -0.018 and -0.007 at 0-6 h; -0.005 and -0.009 pooled
  (folds 2025-12 onward). The flat 1.15 was chosen on the scored rows, so it is not held-out. The
  per-band factors, fitted on earlier forecasts, leave q95 coverage at 0.920 (eight) and 0.938 (six)
  pooled, below the 0.95 target, because the earlier folds were better covered than the later
  ones.
- **Hour of day and lead are tied together.** Every origin is at 09:00 UTC, so lead fixes the
  half-hour of the target. The pooled 90% width grows 27% (eight) and 30% (six) from 12-24 h to
  120-240 h, and 5% once the half-hour is held fixed.
- **The Part 3 gate does not hold.** L-1d minus K2-S1 q95 coverage at 120-240 h is 0.015 (eight)
  and 0.027 (six) lower than at 12-24 h, short of the 0.03 gate, and L-1d covers more than K2-S1
  in both bands. Part 3 is not run.
- **PIT histograms.** At 120-240 h the top PIT bin holds 0.124 (eight) and 0.172 (six) of L-1d's
  rows against 0.1 if calibrated, and the bottom bin holds 0.084 and 0.051. The actuals fall above
  the bands more often than below.

## Tail calibration at 120-240 h (L0PT, L0 in the like-for-like pair; K2-S1 for reference)

| Arm | Group | q95 coverage | q99 coverage | q99 pinball |
|---|---|---|---|---|
| L0PT | eight | 0.938 | 0.987 | 0.302 |
| L0 | eight | 0.934 | 0.984 | 0.306 |
| K2-S1 | eight | 0.952 | 0.989 | 0.287 |
| L0PT | six | 0.899 | 0.975 | 0.215 |
| L0 | six | 0.896 | 0.972 | 0.225 |
| K2-S1 | six | 0.911 | 0.979 | 0.196 |

## Checks

- **Median minus mean forecast.** The only change to a forecast is widening about the median, which
  leaves the median unchanged. The median minus the atom-weighted mean of the 11 stored quantiles
  is -0.6 (L0PT) and -0.5 (L0) points at the eight, and -0.3 at the six.
- **Error of the median beside the CRPS contrast.** The mean absolute error of the median,
  L0PT minus L0, is -0.22 at the eight and -0.08 at the six at 120-240 h, the same sign as the CRPS
  contrast.
- **Leave one Primary out.** The L0PT minus L0 CRPS at 120-240 h is between -0.17 and -0.10 at the
  eight and between -0.06 and -0.03 at the six, whichever Primary is left out.
- **Triggers.** Trigger 2 fired (P1's median part, P3, and P4 contradict their predictions); the
  explanation is the 09:00 UTC origin (P1), non-stationary coverage (P3), and the recipe
  difference between L0PT and LL (P4), tested by adding L0. Triggers 1, 3, 4, 5, and 6 did not
  fire. Trigger 7 may fire: the manager's next lead experiments could depend on the pretraining
  sign, which rests on one pair of experiments (448's L0 and L0PT, 3 seeds, 4 folds); no audit has
  run.
- **Reviews.** A Sonnet plan review, a Sonnet code review (no blocking findings; the hour
  stratification, the width comparison rows, the Bonferroni tail resamples, and unused code were
  fixed), and a Sonnet results review that re-derived the L0PT minus LL contrast, the width ratios,
  one widening factor, and the L-1d reproduction without a mismatch. The like-for-like L0 contrast
  was added after the manager objected to the LL contrast, and is covered by the same code and
  panel; the results review ran before it was added.
- **Reproduction.** The panel reproduces experiment 511's stored L-1d CRPS to 5e-7 in every month.
  The lead-swap "true" condition reproduces experiment 511's stored seed-0 quantiles to 5.4e-5 of
  capacity. Experiment 511's 2,736 rows of 2026-06 with NaN `cov_*` and `pin_*` are in no row set
  and are excluded.

## Method and data

- **Arms and rows.** Rows are experiment 511's scored rows (row set
  `experiment_470_intersection_v1`;
  folds 2025-10, 2025-12, 2026-01, 2026-03, 2026-06; D15 and D18 cut at 2025-03-01; cleaned data).
  L-1d is the quantile average of experiment 511's two seeds. K2-S1 and LL are experiment 454's
  three cells each. L0PT and L0 are experiment 448's seeds 0 to 2 (world 0; no 2025-12 fold, so
  four folds). All scores use the 1.15 widening, and the cells are scored one by one and averaged.
- **Cadence.** Monthly for experiments 448 and 454, daily for L-1d. No refit here.
- **Part 2 factors.** Estimated per daily cutoff from the unwidened L-1d quantiles whose target is
  at
  least 3.5 h before the cutoff; floor 0.5, cap 3, fallback 1.15 when a band holds under 2,000
  rows (100 of 6,832 day, set, band, and tail cells; `part2_fallbacks.csv`).
- **Past-weather caveat.** No weather enters this experiment; it re-reads stored forecasts, whose
  inputs are the stored ensemble mean and spread.
- **Minimum detectable effect.** From the seed spread, for L0PT and L0 at 120-240 h: 0.13 and 0.20
  (eight), 0.21 and 0.09 (six). At 120-240 h the L0PT minus L0 contrast of -0.14 (eight) sits
  below the L0 arm's own 0.20, so it is a point estimate with 3 of 4 folds negative at the eight (4
  of 4 at the six), and its Bonferroni interval excludes zero over origin-day blocks only.
- **Lead swap.** L-1d, fold 2026-01, seed 0, 31 daily models, CPU. K2-S1's lag-free heads cannot be
  swapped: experiment 454 stored their forecasts, not their weights.
- **No load-shift event plots.** This experiment scores no event slice.
- **Not done.** Part 3 (L-Pshare waits for experiment 519; L-lead-head is not supported by the lead
  swap and Part 2). Comparison with experiment 435's table P factors, which were fitted on other
  arms.

## What the next experiment should answer

**The long-lead gap is in the centre of the forecast, not the band.** The lead swap moves L-1d's
median by up to 4 points of capacity while its band stays put, and a per-band width correction
recovers under 5% of the gap. The next experiment can test whether K2-S1's and LL-ens5's advantage
at days 5 to 10 comes from averaging several heads (the median) by scoring the mean absolute error
of the median of each arm by lead.
