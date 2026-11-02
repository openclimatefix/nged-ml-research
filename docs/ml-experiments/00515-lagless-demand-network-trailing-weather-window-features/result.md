---
id: "00515"
slug: lagless-demand-network-trailing-weather-window-features
title: >-
  Do trailing weather windows lower the error of a lag-free demand network that cannot see recent
  power, on ten winter folds?
theme: demand-inputs
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: pretrained and fine-tuned lag-free networks, their raw quantile forecasts, and
  one table of window columns per Primary, under this experiment's private data folder only; every
  earlier experiment is read and never written. Serving path: not touched (this repository holds
  none). Degradation rule: not touched, but the window columns carry a degradation flag counted in
  missed runs. Designs: more than one defensible (how a window is built at the seam before the
  forecast origin, how the training windows are made as smooth as the ensemble windows, how the
  placebo day is drawn), so the plan fixes each before any score is read. Callers: this folder's
  scripts import experiments 397, 418, 422, 428, 437, 448, 510, and 513, and the `studies`
  evaluation code. One Sonnet plan review, one Sonnet code review before the first fit that
  produces quoted numbers, one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  A demand network that reads no lagged power sees only the weather at the valid time, but a
  building warms and cools slowly, heating is switched on late, and sunshine through windows offsets
  heating on a cold day. Trailing means of temperature and shortwave over the previous 6 to 72
  hours, or the 24-hour product of the two, therefore let the lag-free network forecast cold
  winter nights and cold sunny days better. Earlier neural tests of trailing weather (experiments
  419, 437, 441, 442) all fed the network three origin lags of power, which may have carried the
  same information, so their nulls may not transfer.
prediction: >-
  Written before any score is read. P1 (W-T): the gain over the 37-column base is at least 0.05
  points of CRPS on cold winter nights at the eight; P1 fails if the gain over the ten winter folds
  is under 0.03. P2 (W-S): the gain on cold sunny days is at least 0.05 points larger than on cold
  overcast days; P2 fails if the difference is under 0.03. P3 (W-X): the product helps only on the
  cold-sunny and warm-sunny slices; P3 fails if the all-rows change is above +0.02. P4 (W-null): the
  shuffled-day copy of the nine columns scores within the minimum detectable effect of the base; P4
  fails if it scores worse than the base by more than that effect. P5 (W-seam): the two seam builds
  of W-T are within 0.02 points of each other in a paired same-seed contrast; P5 fails if they
  differ by more.
decision_rule: >-
  See "Pre-registered rules". A gain counts only through CRPS at the six and the eight and through
  the cold-night tail slices (q95 and q99 pinball loss), scored with ensemble windows built as at
  serving. The stop rule continues to W-ALL when a Bonferroni interval excludes zero at either group
  and the point estimate is at least 0.03 points better than the base. A stop reads "not resolved",
  not "no effect".
benchmark: >-
  The 37-column base (arm W0: the lag-free network of experiment 448's `L0PT` and experiment 513's
  `P0`, monthly refit, pretrained and fine-tuned in this experiment's own loop) on the same folds,
  rows, and seeds.
row_set_id: experiment_515_winter_v1
metric: >-
  CRPS in points of effective capacity at the signed-MW six and the MVA-only eight, q95 and q99
  pinball loss and coverage on cold nights, and CRPS by horizon band (day 1, days 2 to 3, days 4 to
  10).
---

**This is the plan, written before any fit that produced a quoted number.** All numbers produced
later are interim and unaudited until the reviews are recorded in the pull request.

## Question

The lag-free demand network (37 input columns plus an 8-wide embedding of the Primary) sees the
weather only at the valid time. We ask whether it forecasts better when it also sees the weather of
the previous hours. The maintainer's argument is that every earlier neural test of trailing,
forward, or daily-mean weather fed the network three origin lags of power, so those nulls may not
transfer. The lag-free tests that exist are experiment 6 (XGBoost, with a 0.1-point floor) and
nothing else: no experiment has tested a trailing shortwave window or a temperature-by-shortwave
product in a lag-free network.

This pull request holds the **core arms** (W-T, W-S, W-X, W-null, and the seam control W-seam).
The learned-window arms (W-CNN) are a separate later pull request. W-ALL, and W-DH as a drop-one
inside W-ALL, run here only if the stop rule passes.

## Arms

Every arm adds columns to the 37 and is re-pretrained from scratch with them, using one loop for
all arms (experiment 513's `P0` loop). The pretrained weights of experiment 448 do not load, and no
new column is zero-initialised.

| Arm | Added columns | What it tests |
|---|---|---|
| W0 | none | The control (experiment 513's `P0`, reused for three folds). |
| W-T | Means of temperature over the 6, 24, and 72 hours ending at the valid time (3 columns). | Thermal lag. |
| W-S | Means of shortwave over the 6 and 24 hours ending at the valid time, and over the 24 hours ending 24 hours before the valid time (3 columns). | Stored solar gain. |
| W-X | One column: (24-hour mean temperature minus 15.5 °C) times the 24-hour mean shortwave. | Cold-sunny and warm-sunny against cold-dull and warm-overcast. |
| W-null | The 9 columns of W-ALL (W-T, W-S, W-X, and the two degree-hour columns) taken from a shuffled other day. | A gain from nine extra columns alone. |
| W-seam | W-T built with persistence fill before the forecast origin instead of the previous run's ensemble mean. | Whether the seam, not the physics, drives W-T. |
| W-ALL, W-ALL minus W-DH | Conditional on the stop rule. W-DH is heating degree-hours below 15.5 °C and cooling degree-hours above 20 °C accumulated over 24 hours (2 columns). | Whether the arms add up, and whether W-DH is needed. |

The valid-time product of temperature and shortwave is dropped: the network can learn it from two
existing columns.

## How a window is built

- **One rule at every lead (maintainer, 2026-11-02).** Every window is built on the half-hourly
  weather series that the stored ensemble summaries already are, with no lead-dependent definitions.
  The ensemble mean is a linear interpolation of 3-hourly steps to 144 hours and of 6-hourly steps
  from 144 to 360 hours, so a 6-hour window holds only two distinct samples on day 1 and one beyond
  day 6. The write-up says so beside every result.
- **Ensemble rows (fine-tuning and scoring).** A row belongs to a run with origin `O` (09:00 UTC).
  For half-hours after `O` the series is that run's ensemble mean. For half-hours at or before `O` it
  is the "analysis proxy" of experiment 437: the first day of the freshest earlier run that covers
  that half-hour (leads of about 9.5 to 33.5 hours). No later run is used. This is available at
  serving, because every earlier run was published by `O`. The 72-hour window reaches back three
  days, so it uses up to three earlier runs. I copy experiment 437's `e00437_features.py`, add the new
  columns, and re-run its check, which rebuilds sampled runs with every later run deleted and
  requires the same result. That check covers only the 22 columns of experiment 437, so it is
  re-run on the new columns.
- **Training rows (pretraining on ERA5 and CAMS).** ERA5 temperature is the `:00` rows of the
  half-hourly `era5_features.parquet` (the `:30` rows are interpolation and contain the next hour),
  and shortwave is the hourly CAMS file with its hour-end stamps. Both are subsampled to the
  ensemble's step grid, at 3-hourly UTC stamps for half-hours up to 144 hours after the run's 00 UTC
  start and at 6-hourly stamps after that, and then linearly interpolated to half-hours. The pretraining
  windows are therefore as smooth as the serving windows, and the grid change follows the lead of
  the window's half-hour, not the lead of the valid time.
- **Missing earlier runs.** A window silently shortens when an earlier run is missing. Each row
  carries a degradation count, the number of the three days before the origin whose run is absent,
  and the scorer reports it. Windows at the start of the ensemble history (April 2024) are truncated
  to what exists and flagged.
- **Calibration of the ensemble level.** The 2 K level offsets between the ensemble mean and ERA5
  are not corrected in the core arms. Arms that pass the stop rule get the noise arm of the brief
  (below).
- **Scoring uses ensemble windows built exactly as at serving**, never ERA5 windows. A gain that
  exists with ERA5 windows but not with ensemble windows does not count.
- **Standardisation.** Each new column is standardised with the mean and standard deviation of the
  fold's fine-tuning training rows (the rows that fit the fold's existing standardiser), so a test
  month never sets its own scale.

## Placebo and seam control

- **W-null.** For each local day, a seeded draw (`seed = 515`) picks another day of the same calendar
  month, the same day type (Monday to Friday, Saturday, Sunday), and the same length, never the day
  itself, without replacement, and from the same side of the fold's refit (training rows take donors
  that start before the refit, scored rows take donors in the test month). The row takes the nine
  window columns of the donor day at the same half-hour of day and the same lead, which is the donor
  day's own run for ensemble rows. If the donor's run is missing the row takes the next donor in the
  seeded order. The number of rows with no donor is printed, and those rows take the column means.
  The same draw serves training and scoring.
- **W-seam.** Persistence fill holds the run's first value (the value at its first half-hour after
  `O`) for every half-hour at or before `O`. W-seam shares W-T's pretrained weights, because ERA5
  windows have no seam, so it adds 30 fine-tunes and no pretrains. P5 uses a paired contrast on the
  same seed and fold, because the 0.02 limit sits below the minimum detectable effect.

## Folds, seeds, and the minimum detectable effect

- **Folds.** Ten winter months: 2024-11, 2024-12, 2025-01, 2025-02, 2025-03, 2025-11, 2025-12,
  2026-01, 2026-02, and 2026-03. Ensemble summaries exist for every daily run from 2024-04-01 to
  2026-06-30. The five early months have 7 to 11 months of ensemble fine-tuning, no `L0PT`
  pretrained weights, and no D15 or D18 (their cut is 2025-03-01; report-only). 2025-11 and 2026-02
  need new pretrains. Experiment 437 scored 2025-10, 2025-12, 2026-01, and 2026-03, and experiment
  442 added 2025-11 and 2026-02, so four of the ten months were read before this question; the
  five early months and the exact weather windows are new. Seeds 0, 1, and 2. July 2026 is unused.
  Christmas week is inside 2024-12 and 2025-12.
- **Control reuse.** Experiment 513's `P0` weights and forecasts give W0 for 2025-12, 2026-01, and
  2026-03, because `P0` is the same network, loop, rows, and seeds. W0 is fitted here for the other
  seven folds.
- **Minimum detectable effect.** Stated first, from the seed spread of the base and the null arm
  over the ten folds, before any contrast is read (`e00515_score.py baseline`). References from
  earlier experiments: 0.13 at the eight and 0.10 at the six on all rows, and 0.034 at the eight and
  0.085 at the six on cold nights in 2026-01 alone (a lower bound), all at 3 seeds. The limits of
  0.05 and 0.03 are at or below the minimum detectable effect on all rows, so a pass at those limits
  reads "not shown worse", and only the pooled ten-month cold-night slice can resolve them.
- **Intervals.** Month-block bootstrap over ten folds, with the fold count printed beside every
  interval and the sign per fold beside it. Few blocks give near min-to-max intervals.

## Scoring

- **Rows and groups.** Rows with no labelled event, leads up to 240 hours, cleaned data, the D15
  and D18 start cut, and the eight MVA-only Primaries and six signed-MW Primaries as the gating
  groups. D15 and D18 are report-only. Experiment 470's row set covers only five months, so this
  experiment builds its own, `experiment_515_winter_v1`: every forecast row of the ten months whose
  actual is cleaned power with no labelled event, that every scored arm holds. The row counts per
  fold and group are printed.
- **Widening.** Forecast quantiles are scored raw and widened by 1.15, as experiment 510 does. The
  widening is a fixed table for every arm.
- **Slices (pre-registered).** All rows; cold nights (temperature below 5 °C, 18:00 to 08:00);
  cold sunny and cold overcast days, with experiment 441's definitions; Christmas week; horizon
  bands day 1, days 2 to 3, and days 4 to 10. Tail results are q95 and q99 pinball loss and
  coverage on cold nights beside CRPS (experiment 441 found signed-six cold-night coverage of 76% and
  83%).
- **Bonferroni count.** The primary family is **20** tests: 5 hand-built arms (W-T, W-S, W-X,
  W-ALL, W-ALL minus W-DH) times 2 groups times 2 slices (all rows, cold nights). The core
  arms contribute 12 of them (3 arms times 2 groups times 2 slices); intervals use 20 throughout so
  arms that run later do not change an interval. The **reported family** is **84** tests: 7 arms times
  2 groups times 4 slices (all rows, cold nights, cold sunny, cold overcast) is 56 CRPS tests, plus
  28 q95 and q99 cold-night tests. Both numbers are printed by the scorer.
- **Quoted earlier effects.** Experiment 437's -0.022 at the eight is all rows. The -0.099 is
  experiment 441's cold-night figure (inconclusive, 2 of 3 folds). Experiment 442 gave -0.014 at the
  eight and -0.086 at the six (D16 drives it).

## Pre-registered rules

- **Stop rule.** Continue to W-ALL and W-ALL minus W-DH if a Bonferroni interval of a core arm
  against W0 excludes zero at either group and its point estimate is at least 0.03 points better.
  Otherwise stop and report "not resolved". W-CNN does not wait for this rule.
- **Noise arm (conditional).** Only for an arm that passes. The noise has one level per window column
  and lead band: the root mean square of (ensemble window minus ERA5 window) over the training
  window before the refit, temporally coherent (independent noise averages out in a 24-hour mean).
  The level offset between ensemble and ERA5 is one offset per column and lead band, fitted on data
  before the refit and never on the scored month.
- **What counts.** A gain counts through CRPS at both groups and through the cold-night tail
  slices, not through a feature's importance. A simple hand-built window that passes is preferred
  to W-CNN for the first release.

## Folds, fit count, and cost

- **Fit count.** Pretrainings: 4 core arms (W-T, W-S, W-X, W-null) times 10 folds times 3 seeds is
  120, plus W0 on 7 folds times 3 seeds is 21, so **141 pretrainings**. Fine-tunes: the same 141,
  plus 30 for W-seam, so **171 fine-tunes**. W-ALL and W-ALL minus W-DH, if they run, add 60
  pretrainings and 60 fine-tunes. The noise arm adds pretrains only for passing arms.
- **Cost.** To be timed on the first arm and reported to the manager before the remaining fits. The
  estimate from experiment 513's logs (pretrain 18 to 24 s, fine-tune 3 to 6 s, rows 13 s per fold)
  is 0.9 to 1.4 GPU-hours for the core fits, plus building the window columns (not yet timed). One GPU
  process under 8 GB, with `MemoryMax=8G` and `OMP_NUM_THREADS=1`; launches are staggered against the
  other workers' GPU jobs.

## Standing practices followed

Cleaned data and the D15 and D18 start cut; rows with no labelled event; July 2026 unused; no
XGBoost work; metered generators are not shown (substations only); titles say "load-shift", never
"switching"; the Plan, a fresh Sonnet code review before the first fit that produces quoted numbers,
and Sonnet reviews of the results and the write-up follow the `ml-experiment` skill.
