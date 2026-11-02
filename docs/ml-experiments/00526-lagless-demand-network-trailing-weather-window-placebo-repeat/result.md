---
id: "00526"
slug: lagless-demand-network-trailing-weather-window-placebo-repeat
title: >-
  Do trailing temperature windows beat a shuffled-day placebo on cold nights in the lag-free demand
  network, with five seeds, four summer folds, and the checks experiment 515 did not run?
theme: demand-inputs
also_relevant_to: []
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: pretrained and fine-tuned lag-free networks, their raw quantile forecasts, and a
  mixture-mean column, under this experiment's private data folder only; every earlier experiment is
  read and never written. Serving path: not touched (this repository holds none). Degradation rule:
  not touched. Designs: more than one defensible (how the placebo matches the three real columns, what
  the column-count control is made of, how seeds are added without refitting), so the plan fixes each
  before any score is read. Callers: this folder's scripts import experiments 397, 428, 510, and 515
  and the `studies` evaluation code. One Sonnet plan review, one Sonnet code review before the first
  fit that produces quoted numbers, one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  Experiment 515 found that trailing temperature windows (W-T) lower cold-night CRPS at the eight
  MVA-only Primaries by 0.150 points, but a shuffled-day placebo over nine columns (W-null) lowered
  it by 0.071, and W-T minus W-null was -0.079 [-0.290, +0.089]. That comparison mixes real against
  shuffled windows with three columns against nine. The claim here is that W-T beats a shuffled-day
  placebo built from exactly W-T's three columns (W-Tnull) on cold nights at the eight, so the gain
  comes from the content of the windows and not from extra input columns or from the fold in which
  W0 scores badly.
prediction: >-
  Written before any score is read. Sign convention for every prediction: a contrast is the first
  arm's CRPS minus the second arm's CRPS in points, so negative favours the first arm. P1 (primary,
  W-T minus W-Tnull, cold nights, the eight MVA-only Primaries, 10 winter folds, 5 seeds): the
  point estimate is -0.05 or lower. P1 fails if it is above -0.03. The observed 515 contrast against
  the nine-column placebo was -0.079. P2 (descriptive, no pass or fail): the column-count control
  W-noise (frozen independent standard-normal columns) gains nothing on cold nights beyond the seed
  spread, and W-null minus W-Tnull shows whether the nine-column placebo gains more than the
  three-column one. P3 (seam, all rows, both groups): the paired contrast between the two seam builds
  of W-T, now at 5 seeds, lies within plus or minus 0.02. P3 fails if it lies outside, as 515's
  3-seed contrast did (+0.025 at the eight, +0.023 at the six).
decision_rule: >-
  Pre-registered before any number is read. The primary contrast is W-T minus W-Tnull, cold nights,
  the eight, 10 winter folds, 5 seeds, with a 95% interval (one test, level 0.05). The result is worded
  as a tighter interval and not as a verdict, unless the point estimate is below -0.12 (the 95%
  half-width at 5 seeds is about 0.12). A contrast inside the minimum detectable effect is worded
  "not shown", never "no effect". The adoption rule for the first-release recipe: adopt W-T if (i) the
  cold-night intervals of W-T minus W-Tnull and of W-T minus W-noise (3 columns), at the eight,
  both exclude zero, (ii) the all-rows contrast of W-T minus W0, at both groups, is no worse than
  +0.03, and (iii) the seam check (P3) passes. The secondary family is 21 tests, Bonferroni-corrected.
benchmark: >-
  The 37-column base (arm W0) on the same folds, rows, and seeds, with seeds 0 to 2 read from
  experiment 515 (and experiment 513's `P0` for 2025-12, 2026-01, and 2026-03) and seeds 3 and 4 fitted
  here. The placebo comparators are W-Tnull and W-null.
row_set_id: experiment_526_v1
metric: >-
  CRPS in points of effective capacity at the eight MVA-only and the six signed-MW Primaries, on all
  rows, cold nights, cold sunny, and warm sunny slices; q95 and q99 pinball loss on cold nights; the
  absolute error of the mixture-mean forecast and the median minus the mean forecast; CRPS by horizon
  band (day 1, days 2 to 3, days 4 to 10).
---

**Take-home: to be written after the results. Interim and unaudited until then.**

## Question

Experiment 515 ([result page](../00515-lagless-demand-network-trailing-weather-window-features/result.md))
gave a lag-free neural network the trailing means of temperature over 6, 24, and 72 hours (W-T) and
found a cold-night CRPS gain of 0.150 points at the eight MVA-only Primaries. A placebo that gave
every row the nine window columns of a randomly drawn other day (W-null) gained 0.071, so the
real windows did not resolve from the placebo. That comparison confounded the content of the
columns with their number (three real columns against nine shuffled ones), used three seeds, and had
no summer fold. It also did not run the median-minus-mean forecast check, the mean-forecast error
beside each contrast, or the example forecast graphs.

This experiment repeats the comparison with a placebo matched to W-T's three columns, five seeds, a
column-count control, four summer folds, and the missing checks. It adds files only: no
`e00515_*` file is edited, and every output goes to this experiment's own folder, because the 515
scorer lists arms by scanning its own `work/seed_preds/*`.

## Plan changes made after reading the brief

- The brief's primary contrast was W-T minus W-null. The manager's queue-order check showed that the
  two differ by column count, so the primary is W-T minus **W-Tnull**, and W-null is secondary.
- The brief's W-const is replaced by **W-noise**, because a constant column is zero after
  standardisation and would equal W0 reseeded.
- The seam check of the adoption rule needs W-seam at five seeds, which 515 ran at three. The plan
  adds W-seam seeds 3 and 4 on the 10 winter folds. It costs 20 fine-tunes and no pretraining, because
  W-seam starts from W-T's pretrained weights. The manager has not yet approved this addition.
- The mean-forecast refit needs only fine-tunes, because 515 stored the pretrained weights of W0,
  W-T, and W-null for seeds 0 to 2 (and experiment 513 stored `P0`'s for the three reused folds).

## Arms

Every arm starts from the 37 non-lag columns and is pretrained from scratch with its added columns,
in experiment 513's `P0` loop, as in 515.

| Arm | Added columns | Role |
|---|---|---|
| W0 | none | Control |
| W-T | 3 trailing temperature means (6, 24, 72 hours) | The candidate |
| W-Tnull | The same 3 columns, taken from a shuffled other day (515's donor rules) | Primary comparator |
| W-null | 515's 9 shuffled columns | Secondary comparator |
| W-noise3, W-noise9 | 3 and 9 frozen independent N(0,1) columns | Column-count control |
| W-S, W-X | 515's trailing shortwave means; the temperature-by-shortwave product | Carried over from 515 |
| W-seam | W-T with persistence fill before the origin | Seam check |

W-noise columns are drawn once per fold and stage from a fixed seed and are the same for every
training seed, so they carry no information about demand.

## Folds and seeds

- **Winter folds (10):** 2024-11, 2024-12, 2025-01, 2025-02, 2025-03, 2025-11, 2025-12, 2026-01,
  2026-02, 2026-03, as in 515. Each reports with and without 2024-11.
- **Summer folds (4):** 2025-05, 2025-07, 2025-09, 2026-04. The refit cutoff is the start of the
  test month. Every fold scores only valid times before 2026-07-01, and July 2026 is unused.
  Summer folds add no cold nights, so they serve the warm-sunny slice only.
- **Seeds:** 5 (0 to 4) for W0, W-T, W-Tnull, W-null, W-noise3, and W-noise9 on the winter folds.
  Seeds 0 to 2 of W0, W-T, W-null, W-S, W-X, and W-seam are read from 515 (W0 for 2025-12, 2026-01, and
  2026-03 from experiment 513's `P0`) and are not refitted. Summer folds fit every arm at all 5 seeds.
  The harness skips any forecast that exists, so more seeds can be added later without refitting.

## Statistics (pre-registered)

- **Primary:** 1 test, W-T minus W-Tnull, cold nights, the eight, level 0.05. The month-block
  bootstrap resamples the 10 winter folds, each carrying its 5-seed mean.
- **Secondary family: 21 tests**, Bonferroni-corrected. W-T minus W-null at five slice-and-group
  pairs: eight and six on all rows, eight and six on cold nights, and six on cold-sunny. W-T minus
  W-noise3 and W-T minus W-Tnull at eight and six on all rows, cold nights, cold sunny, and warm sunny
  (8 + 8 tests). D15 and D18 are report-only.
- **Minimum detectable effect at 5 seeds**, from the single-seed spread of 515 (0.215 all rows, 0.300
  cold nights per fold): one fold 0.267 and 0.371; pooled over 10 folds 0.084 and 0.118. The scorer
  recomputes them from the five seeds and prints them beside every contrast.
- **State plainly:** the primary is not resolvable at 5 seeds. Its standard error is about 0.06, so
  the 95% half-width is about 0.12 and the 80%-power minimum detectable effect about 0.17, against an
  observed 515 contrast of -0.079. An 11-seed extension for W0, W-T, and W-Tnull would cost about
  1.7 to 1.9 GPU-hours and is not part of this plan.
- **Lead bands:** day 1, days 2 to 3, days 4 to 10, as 515 scored.
- **Rows:** `experiment_526_v1` repeats 515's row rules (cleaned data, the D15 and D18 start cut,
  leads up to 240 hours, no labelled event) and keeps the rows every core arm holds.
- **Widening:** 1.15 about the median for CRPS, as in 515.

## Checks that 515 did not run

- **Median minus mean forecast, per arm.** The mixture mean is `sum(exp(log_w) x mean)` as in
  `e00510_common.py`, converted to a fraction of capacity. Every new fit saves the mean column beside
  the quantiles, and the seeds 0 to 2 forecasts of W0, W-T, and W-null are refitted from the stored
  pretrained weights (90 fine-tunes). The refit's quantiles are compared with 515's stored quantiles,
  so the check also reports whether the fine-tune reproduces.
- **Mean-forecast error beside each CRPS contrast** (absolute error of the mean, in points), through
  the same interval code.
- **Example forecast graphs**: a cold night, a cold sunny day, and a warm sunny day at a Primary
  (substations only; no metered generator).
- **Peak resident memory** of the first fold, measured with `/usr/bin/time -v`.

## Run order and cost

The GPU runs in stages, each started only on the manager's "GPU go". One GPU process under 8 GB
(`MemoryMax=8G`, `OMP_NUM_THREADS=1`). A fit pair (pretrain plus fine-tune) costs about 22 seconds, plus
13 to 35 seconds of row building per fold. This experiment runs before experiment 516's GPU fits.

| Stage | Contents | Pairs | Estimate (GPU-hours) |
|---|---|---|---|
| 1 | Winter seeds 3 and 4 of W0, W-T, W-null, plus W-seam seeds 3 and 4 | 60, plus 20 fine-tunes | 0.4 |
| 2 | Winter W-Tnull, W-noise3, W-noise9 at 5 seeds | 150 | 1.0 |
| 3 | Four summer folds, 6 arms at 5 seeds | 120 | 0.8 |
| 4 | W-S and W-X: winter seeds 3 and 4; summer at 5 seeds | 80 | 0.5 |
| 5 | Mean refit of seeds 0 to 2 of W0, W-T, W-null (winter) | 90 fine-tunes | 0.1 |

The total is about 2.8 GPU-hours. The brief priced about 2.3: the difference is the second noise width
(70 pairs, 0.45 GPU-hours), the 20 W-seam fine-tunes, and a mean refit that needs no pretraining. W-noise9
is the first item to drop if the manager wants the lower figure.

## Reuse and standing practices

Stored forecasts reused: 515's seeds 0 to 2 of W0, W-T, W-null, W-S, W-X, and W-seam, and 513's `P0`.
Cleaned data and the D15 and D18 start cut apply. Refits are monthly. Fitted networks are saved in
the experiment's private data folder. July 2026 is unused. Generators are not scored.
The reviews follow the `ml-experiment` skill: a Sonnet plan review, a Sonnet code review before the
first fit, a Sonnet results review, an audit if a trigger fires, and two Sonnet write-up reviews.
