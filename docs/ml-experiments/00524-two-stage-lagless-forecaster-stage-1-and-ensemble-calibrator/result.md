---
id: "00524"
slug: two-stage-lagless-forecaster-stage-1-and-ensemble-calibrator
title: >-
  Can a weather-to-power network that never sees lead time, plus a small ensemble calibrator, match
  the single-stage lag-free demand network?
theme: retraining-and-training-history
also_relevant_to:
  - calibration-and-uncertainty
  - weather-products-and-blending
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. What gets stored: stage-1 and stage-2 weights, cached mixture parameters, and forecast
  tables under this experiment's private data folder; no other experiment's table changes. Designs:
  several are defensible (frozen, joint, member-wise, and stage-0 variants), so the plan fixes them
  below. Callers: the scripts import experiments 428, 448, 454, 470, 510, and 511 for rows, weights,
  and the scorer. One Sonnet plan review, a Sonnet code review before each batch of numbers, a Sonnet
  results review, and two Sonnet write-up reviews. The CPU gate and CPU-only arms run on 2 cores and
  8 GB; every GPU fit waits for the manager's "GPU go".
hypothesis: >-
  A lag-free demand network can be split in two. Stage 1 maps ERA5 and CAMS weather, calendar, solar
  geometry, and the Primary embedding to a three-component Gaussian mixture of power. It has no lead
  input and no ensemble-spread input. Stage 2 is a small network that takes stage 1's mixture, the
  lead, the ensemble spread, and the half-hour of day, and returns a calibrated mixture. The split
  is a simpler and more explainable forecaster: one part says what the weather does to demand, the
  other says how much to trust the ensemble weather at each lead. The claim is that the split loses
  no CRPS against the single-stage network L-Ww.
prediction: >-
  Written before any score is read. Equivalence margins are the minimum detectable effects (MDE) of
  the two-seed pool, 0.164 at the eight MVA-only Primaries and 0.123 at the six signed-MW Primaries,
  read per band from experiment 521's `scores/mde.csv` (recomputed for two seeds). P1. S2-F beats
  L-Ww at 120-240 h by at least 0.087 at the eight and 0.098 at the six CRPS points. P2. S2-J is
  within the MDE of L-Ww in every coarse band (equivalence), and closes at least half of the
  120-240 h gap to K2-S1 only if P1 also holds. P3. S2-R is not run unless P1 or P2's gain at days
  5 to 10 exceeds the MDE; if it runs, it beats S2-F below 2 days by more than the MDE. P4. Form A
  (per-lead scale and shift on the mixture) recovers at least 70% of form B's gain over S2-0, where form B's
  gain exceeds the MDE; below that gate the ratio is descriptive. P5. Injected-shift leak (experiment
  511's stress world) of S2-F is at most L-Ww's 7.2 points the day after the shift, fading to 3.0;
  S2-R's is at least S2-F's. P6. S0-A form 2 (the multilayer perceptron) lowers CRPS against S2-0 by
  0.03 to 0.07 (experiment 14's size, below the MDE, so paired intervals only and descriptive).
  P7. S2-M without stage 2 is narrower than L-Ww and under-covers at the six; stage 2 restores q95
  coverage to within 0.03 of nominal and moves CRPS by less than the MDE.
decision_rule: >-
  Main control: L-Ww, the weekly-refit lag-free network, on the same weekly harness, two seeds, the
  five folds of experiment 511. L-1d (daily refit) is a report-only reference. Counted family: 3
  arms (S2-F, S2-J at 0.4x, S2-J at 0.1x) x 2 groups x 3 coarse lead bands (under 2 d, 2-5 d, 5-10
  d) = 18 CRPS contrasts against L-Ww; the seven bands of experiment 521 are report-only (42 if
  counted). Intervals are Bonferroni-adjusted origin-day block bootstraps (153 origin days as the
  main block), with the 5 whole-month blocks (near the minimum-to-maximum of the folds) as the
  check, and per-fold signs beside every contrast. A gain counts only when its adjusted interval
  excludes zero and the point estimate is at or above the MDE. Every arm without a threshold at or
  above the MDE (the mixture-input forms, the NLL and CRPS losses, S0-A, S0-S2, S0-E2E, S2-M) is
  labelled descriptive. The smallest gain worth porting is zero CRPS if the result is a simpler
  recipe with equivalent scores (serving-cost equivalent: one more small network and no
  lead input to stage 1), and 0.087 and 0.098 at 120-240 h if the case rests on beating L-Ww.
  All numbers are interim and unaudited until the audit and the reviews have run.
benchmark: >-
  L-Ww (experiment 510, two seeds, weekly refit, 90-day recency weight) on experiment 454's rows
  (`experiment_470_intersection_v1`); K2-S1 (experiment 454); L-1d (experiment 511); experiment
  454's ensemble-only lag-free heads LL and LL-ens5; BG2, `manual_heuristic`, and `climatology`.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by lead band (the seven bands of experiment 521: 0 to 6 h, 6
  to 12 h, 12 to 24 h, 24 to 48 h, 48 to 72 h, 72 to 120 h, 120 to 240 h; three coarse bands for the
  counted contrasts), q01, q05, q95, and q99 pinball loss and coverage by lead, mean 90% width, the
  probability integral transform by lead, avoidable-curtailment cost, and serving cost.
---

**Plan: split the lag-free demand network into a weather-to-power stage and an ensemble calibrator,
and test whether the split loses any accuracy.** This page is the plan; the write-up replaces it.
The CPU gate (S2-0) runs first and is reported before any GPU use.

## Why the split is worth testing

**A weather-to-power network that never sees lead time is simpler to explain and has fewer parts
that depend on live recent data.** Stage 1 sees only ERA5 and CAMS weather, so it cannot learn a
false idea of lead. Stage 2 sees lead only where lead matters, in a small network that is easy to
inspect. Evidence from earlier experiments sets the expectations:

- **Pretraining helps at every lead.** [Experiment
  418](../00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens/result.md) found that
  pretrained minus ensemble-only
  CRPS is negative at every lead band, most at day 1. The split therefore cannot be justified as a
  fix for a harm pretraining does.
- **The eight's long-lead gap looks like head count.** [Experiment
  521](../00521-lag-advantage-by-lead-time-k2-s1-vs-daily-lag-free/result.md) found that a
  monthly five-head lag-free
  ensemble matches K2-S1 at 120-240 h at the eight (4.984 against 4.967) and not at the six
  (3.724 against 3.523). The split is not expected to close that gap by calibration alone.
- **Experiment 523** ([PR 184](https://github.com/openclimatefix/nged-ml-research/pull/184)) asks
  whether L-Ww uses its lead input. Its Part 3 (a lead-aware head) is merged into this plan as
  arm S2-F with the lead going only to stage 2.

## Arms

All arms use the five folds of experiment 511 (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), two
seeds, the weekly harness (refit cutoffs at days 1, 8, 15, 22, 29), and the experiment-510 row
rule: stage-2 labels at a cutoff have valid time at most cutoff minus 3.5 h, origin at most the
cutoff, valid time within 730 days, and a 90-day half-life recency weight.

### Core (CPU gate, then GPU)

- **S2-0 (stage 1 alone, the denominator of every gain).** CPU gate: the stored L0PT pretrained
  weights (`E448_PRETRAINED/{month}_s{seed}.pt`; 510's own `L0PT/2025-12_s*.pt`) forecast the ENS
  rows with no fine-tune. These networks keep `lead`, `lead_sqrt`, and the `std_*` columns, so S2-0
  bounds a frozen stage 1 from below. The 10 lead-free, spread-free stage-1 networks (5 folds x 2
  seeds, same recipe and rows, those 9 columns removed) are the GPU part.
- **S2-F (frozen stage 1).** Stage 1 trained once per fold on ERA5 and CAMS rows to the month's
  start, then frozen. Stage 2 trains on the 730-day ensemble window with the 90-day recency weight
  and refits at each cutoff. Form A: per-lead monotone scale on the mixture's standard deviations
  and shift on its means, per metering group. Form B: a multilayer perceptron (2 hidden layers of
  32) on the mixture parameters. Both forms take a group input and the Primary embedding.
- **S2-J (joint fine-tune).** Same two stages; stage 1's weights change after stage 2 is plugged
  in. Primary: the L-1d rate ratio (0.4x of the pretrain rate of 5e-4, that is 2e-4, so only the
  missing lead input differs from L-1d). Second arm: 0.1x. The base rate is the pretrain rate.

### CPU-only arms on cached stage-1 outputs (minutes)

- **Mixture-input stage-2 forms.** Stage 2 takes the 9 mixture parameters (3 logits, 3 means, 3 log
  standard deviations), the lead, two spread terms, two half-hour terms, and the group; it returns
  a mixture of the same form, trained with the negative log-likelihood and, as a second arm, the
  closed-form CRPS of the mixture. The 11-quantile input is a control, reporting its crossing rate
  before sorting; its output is monotone by cumulative softplus steps.
- **S0-A (stage 0, weather-space adaptor, no stage 2).** Maps ensemble weather (mean, spread, lead,
  half-hour) to ERA5 and CAMS weather at the same valid time; pairs use ERA5 up to the cutoff minus
  5 days and CAMS up to minus 1 day. Form 1: per-variable, per-lead-band bias and scale. Form 2: a
  multilayer perceptron on all seven weather variables with the lead. Then frozen in front of frozen
  stage 1.
- **S0-S2.** The better stage-0 form of S0-A plus stage 2.

### Later arms (after the core, and only as stated)

- **S2-M (member-wise).** Each of the 51 ensemble members goes through stage 1; the 51 mixtures are
  combined as an equal-weight mixture of mixtures and collapsed to 3 components by moment matching
  (the choice is reported, with the fixed-quantile route as a check on the tails), with and without
  stage 2. Runs after the core, sharing its cache, regardless of the earlier results.
- **S2-R and the daily L-1d harness.** Run only if S2-F or S2-J beats L-Ww at days 5 to 10 by more
  than the MDE. S2-R restarts from the month's pretrained weights each week and never warm-starts.
- **S0-E2E.** Stage 0 trained end to end through frozen stage 1 against the power negative
  log-likelihood on ensemble rows: labels cut at cutoff minus 3.5 h, pairs cut at 5 days (ERA5) and
  1 day (CAMS). Tested against S2-J.

### Controls

L-Ww; L-W and L-M (experiment 510); L-1d (experiment 511); experiment 454's ensemble-only lag-free
heads LL and LL-ens5; K2-S1; the 11-quantile stage 2. A stage 2 with no stage 1 is the stored LL
and LL-ens5 of experiment 454 and arm R of experiment 425, unless a like-for-like small network is
cheap.

## Stage 2 training design

**Stage 1 has already fitted the power in stage 2's window, so stage 2 uses held-out stage-1
outputs.** Stage 1 trains on ERA5 and CAMS rows to the month's start and sees the power at every
valid time of stage 2's 730-day window. A stage 2 trained on those in-sample outputs would learn a
calibration that does not hold at test. The plan uses one of two cuts, fixed before any score is
read: cross-fit stage-1 outputs in the 6 contiguous blocks of experiment 449, or hold out 1 week in
12 (`HELD_WEEK_PERIOD` of experiment 397) from stage 1's training and train stage 2 on those
weeks. The CPU gate does not need either, because it fits no stage 2. The ensemble archive covers
only 18 to 26 months before the five folds, so early folds train stage 2 on under 730 days.

## Step 0 (read from the code and from the earlier experiments, before any fit)

- **Stage-1 inputs.** The 37 non-lag columns of the `summary` scheme (`e00397_prep.feature_names`
  minus the 9 lag columns of `e00428_data.non_lag_layout`) are 7 `mean_` and 7 `std_` weather
  columns, 3 solar, 12 calendar (4 hour terms, 7 `dow_`, `covid`), 6 holiday, `lead`, and
  `lead_sqrt`. Stage 1 drops `lead`, `lead_sqrt`, and the 7 `std_*`, leaving 28 columns plus the
  8-wide Primary embedding. `covid` stays (experiment 517 was inconclusive). The network is
  `PlainMdn` (64 x 2, 3 components), about 7,241 weights.
- **ERA5 and CAMS rows carry a fabricated spread and a real lead.** Their `std_*` is the fold's
  median ensemble spread by 24-hour lead band, and their pseudo-runs have leads 0.5 to 351 h
  ([experiment
  418](../00418-mdn-pretrain-on-era5-and-cams-then-fine-tune-on-ens/result.md)). Stage 1 drops
  both.
  This is the first arm that is lead-free and spread-free in the pretraining role; the only
  precedent is experiment 449's cross-fitted ERA5 and CAMS module, which was never scored on
  ensemble rows.
- **Serving grid.** Lead steps are 3-hourly to 144 h and 6-hourly after, upsampled to half-hourly.
- **Why stage 2's rows do not overlap stage 1's.** Stage 1 never sees ensemble rows, so no row is
  shared; the stacking leak above is the one that remains, and the cuts above address it.
- **Earlier weather-space corrections.** Experiment 9 (CAMS- against ERA5-trained stage 1a, refuted)
  -0.035 (-0.070 to +0.001); experiment 14 (empirical quantile mapping of ensemble shortwave to
  CAMS, inconclusive because monthly daylight bias checks failed up to 18.6%) -0.067 (-0.094 to
  -0.039) for a CAMS-trained arm and -0.025 for an ERA5-trained arm; experiment 111 fitted per-site,
  season, lead, and hour calibration of ensemble temperature (shifts -1.15 to +2.05 K) and
  irradiance (ratio 0.80 to 1.11) without a stored effect size for the calibration alone;
  experiment 418's arm P3 (ensemble-sized noise on ERA5 rows) was +0.012 (-0.003 to +0.026) at the
  eight and -0.003 (-0.022 to +0.013) at the six, a null. The new stage 0 differs by covering all
  seven variables, taking lead and spread as inputs, using a network and not a marginal mapping,
  and (S0-E2E) training through stage 1.

## Standing practices

- **Followed:** monthly-fold screen on the five folds of experiment 511 with weekly refits (the
  harness the control L-Ww uses; the cadence is not the question), cleaned data, the D15 and D18 cut
  at 2025-03-01, stored predictions reused (L-Ww, L-W, L-M, L-1d, K2-S1, LL, references), July 2026
  unused, saved weights (stage-1 and stage-2 state files in the private data folder), the tail table
  (q01, q05, q95, q99 pinball and coverage, by lead), avoidable-curtailment cost beside CRPS, example
  forecasts (a day-ahead and a 14-day horizon, a winter and a summer Primary), the draft pull
  request, GPU use through one process under 8 GB with `OMP_NUM_THREADS=1`.
- **Load-shift events.** The injected-shift leak uses experiment 511's stress world. The incomplete
  event-label caveat applies: clean rows may hold unlabelled events, so the leak table is optimistic
  and real-event slices undercount. Metered generators appear only as Farm A to C and Solar A to F.
  Rows with no labelled event are `tgt_real` false. Charts use the OCF palette, never `ocf.GREY_1`,
  and no mark in the background colour `#fffbf5`.
- **Advice followed:** experiment 521 (quote K2-S1 as the three-cell mean); experiment 511
  (origin-day block bootstrap, per-fold signs). **Departure:** the brief sets the counted family
  against L-1d; the screen runs on the weekly harness, so the counted control is L-Ww and L-1d is
  report-only.

## Cost

About 2.3 to 3.6 GPU-hours for the whole set; GPU only for the 10 stage-1 pretrains, S2-J, S2-R,
S0-E2E, and the S2-M passes. The CPU gate and the CPU-only arms use 2 cores and 8 GB. GPU runs
wait for the manager's "GPU go", which waits for experiments 519 and 515/516 to merge.
