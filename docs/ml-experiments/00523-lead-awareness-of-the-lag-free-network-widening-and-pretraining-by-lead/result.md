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
status: planned
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
  pretrained-minus-unpretrained contrast L0PT minus LL) under origin-day blocks, with whole-month
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
---

**Plan: three CPU-only reads of stored output, and one inference check on stored weights, ask
whether the lag-free network uses its lead input.** This page is the plan; the write-up replaces it.

## What the experiment does

- **Part 1a, width by lead, stratified by hour of day.** From 521's `work/refs_{month}.parquet`
  and experiment 511's row tables, the mean 90% width of L-1d, K2-S1 (cell c0 and the three-cell
  mean), L-Ww, and 454's LL, LL-ens3, LL-ens5, by the seven lead bands, within each of six
  four-hour blocks of target hour (so a wide 0-12 h band that comes from time of day is not
  mistaken for lead). The 2,736 L-1d rows in 2026-06 with NaN `cov_*` and `pin_*` are counted and
  excluded with `fill_nan(None)`; the count is reported.
- **Part 1b, PIT histogram.** Ten bins from the stored 11 quantiles (the piecewise-linear
  interpolation of the quantile function, tails clamped at q01 and q99), by lead band, for L-1d and
  K2-S1.
- **Part 1d, width growth against the ensemble's error growth.** The ratio of width at days 6 to 10
  to width below 12 h, beside experiment 418's stored statement of the ensemble's root-mean-square
  error to spread (1.2 to 3.3 at day 1, 1.1 to 1.5 at day 4; no stored value for days 6 to 10, which
  the page says).
- **Part 1c, lead swap (stored weights, no refit, CPU).** L-1d seed 0 of fold 2026-01 uses
  `00511-*/work/models/L-1d/2026-01_d{day}_s0.pt` for every daily cutoff; K2-S1's lag-free heads use
  `00454-*/work/heads/A/2026-01_s0.pt`. The same 2026-01 test rows are scored with `lead` and
  `lead_sqrt` (columns found by `e00446_models.lead_columns()`, set on the fold's input scale) set
  to
  (a) the true lead, (b) day 1 for every row, (c) day 10 for every row, (d) a shuffled lead
  (within each Primary and origin). The page reports the change in median forecast and in 90% width
  by
  true lead. K2-S1's lag-free heads are not pretrained, so the swap contrasts a pretrained network
  with unpretrained heads; the page says this. If rebuilding the fold's inputs on 8 GB does not
  fit, the swap runs on a subset of origin days and the page says which.
- **Part 2, per-band widening on held-out fits only.** Folds 2025-12 onward (four). For each daily
  cutoff, the factor on each tail quantile's distance from the median (q01, q05, q95, q99) is
  estimated per lead band and metering group from the stored, unwidened L-1d quantiles whose target
  is at least 3.5 h before the cutoff. The pool is every earlier stored fold plus the earlier days
  of the same fold, with no trailing-window limit. This departs from the brief's "730-day training
  window" on purpose: only the stored held-out forecasts are free of the fit's own training rows.
  The standardised residual of an upper tail is (actual - median) / (q_tail - median), and of a
  lower tail (median - actual) / (median - q_tail); the factor is its empirical quantile at the
  tail's level (upper) or at one minus the level (lower), which restores nominal coverage, floored
  at 0.5 and capped at 3. A band and group with fewer than 2,000 pool rows uses the flat 1.15, and
  the count of such cases is reported. The factor is applied to the next cutoff's rows, as table P
  (experiment 435) does. A post hoc variant fits one factor per group with no band, to separate the
  effect of banding from the effect of re-estimating on held-out forecasts. K2-S1 is not re-widened
  per band: its stored bands are the control, scored at 1.15, and the page says so. The
  comparison is flat 1.15, no widening, and table P's factors (experiment 435's stored table, which
  were fitted on other arms). The flat 1.15 was selected on the scored rows, so it is not held-out,
  which the page states. The 2025-10 fold has no earlier fold; it is shown with flat 1.15 only.
- **Pretrained against unpretrained, by lead.** Experiment 448's L0PT (pretrained, ensemble
  fine-tuned, monthly, seeds 0 to 2, world 0) minus the `LL` arm of experiment 454's stored
  predictions (the lag-free heads, ensemble only, monthly, three cells) on the rows of the four
  folds where both exist (2025-10, 2026-01, 2026-03, 2026-06; experiment 448 stored no 2025-12
  fold), joined on `(primary, origin, target)` to experiment 511's rows. CRPS and q95/q99
  pinball by the seven bands, the minimum detectable effect, and the per-fold signs.
- **Comparability of the pretrained contrast.** The page confirms from the code that L0PT and LL
  share the architecture, the 1.15 widening, the row set, and the monthly cadence, and differ in
  pretraining, and states any other difference found.
- **Gate for Part 3.** The page states whether L-1d minus K2-S1 q95 coverage at 120-240 h falls
  short by 0.03 more than at 12-24 h. Part 3 (L-Pshare waits for experiment 519; L-lead-head waits
  for this result) is not run.

## Standing practices

- **Followed:** cleaned data (the stored rows are cleaned; dropped half-hours per fold are copied
  from experiment 511's checks), the D15 and D18 cut at 2025-03-01 (experiment 511's rows), stored
  predictions reused, July 2026 unused, the tail table (q01, q05, q95, q99), example forecasts
  (day-ahead and 10-day, a winter and a summer Primary, from 521's example script), and the
  draft pull request.
- **Not applicable:** no refit (CPU, stored output), no new weights saved, no load-shift event plots
  (this experiment scores no event slice; it says so).
- **Advice followed from earlier experiments:** experiment 521 (quote K2-S1 as the three-cell mean,
  which this plan does) and experiment 511 (origin-day block bootstrap).

## Step 0 (read from the code, before any fit)

- **Pretraining rows carry a real lead.** L-1d starts from experiment 448's L0PT weights
  (`e00510_fit.pretrained_path` falls back to them), which `e00448_run.pretrain_rows` and
  `spread_table` build with the logic of experiment 418's `e00418_prep.pretrain_frame`, which builds
  pseudo-runs (a
  pseudo-origin every `STRIDE_DAYS` days, leads of 0.5 to 351 h) and stores `lead_hours` as
  `(target - origin) / 2`; `lead` and `lead_sqrt` come from `feature_matrix` (experiment 397's
  summary scheme) as functions of that value. Lead is neither a fixed value nor drawn at random.
- **The `std_*` columns are the ensemble's median spread by 24-hour lead band.** `fold_pretrain`
  overwrites the raw zeros with `spread_tables`' median standardised `std_*` of the fold's ensemble
  training rows, so the `std_*` inputs grow with lead in pretraining as they do in fine-tuning.
- **Only the noisy variant of the pretraining rows could tell the network that long lead means
  larger error, and L-1d does not use it.** The pretraining targets are power observed against ERA5
  weather, so the label noise does not grow with lead. Experiment 418's P3 arm adds noise at the
  ensemble spread by lead band; `e00448_run.pretrain_rows` has no noise option and adds none.
- **K2-S1's lag-free heads are not pretrained**, so the heads see only ensemble fine-tuning rows,
  where lead and error do move together.

## Statistics

- **Counted tests:** 28 for CRPS (7 bands x 2 groups x 2 contrasts) and 56 with q99 pinball.
  Width, PIT, and the lead swap are report-only.
- **Blocks:** origin days within month (as experiment 511 and 521) and, as the conservative
  check, whole months (4 blocks for each contrast, so near minimum-to-maximum intervals; the page
  says so).
- **Minimum detectable effect:** from the seed spread, beside every contrast; we expect the per-band
  MDE to exceed the pooled values of experiment 511 (0.164 at the eight, 0.123 at the six).
- **Interim numbers:** every number is interim and unaudited until the reviews have run.
