---
id: "00514"
slug: lagless-demand-network-lockdown-flag-removal
title: >-
  Is the 0/1 lockdown flag still needed by the lag-free demand network now that pretraining reaches
  back to 2019?
theme: demand-inputs
also_relevant_to: []
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: pretrained and fine-tuned lag-free networks and their raw quantile forecasts, under
  this experiment's private data folder only; every earlier experiment is read and never written.
  Serving path: not touched (this repository holds none). Degradation rule: not touched. Contract:
  not touched. Designs: one defensible design (three arms that differ in one column and one set of
  pretraining rows). Callers: this folder's scripts import experiments 397, 422, 428, 448, 510, and
  the `studies` evaluation code. One Sonnet code review before the first quoted number, one Sonnet
  results review, and one Sonnet write-up review.
hypothesis: >-
  The flag is 0 in every scored month and in every fine-tune row, so it can only matter through the
  pretraining rows from 2019-09, which hold the 2020-03-23 to 2021-07-19 lockdown. A lag-free network
  pretrained on those rows without the flag may fit them worse, or may be pulled off its normal
  behaviour by them. Dropping the lockdown rows may then help the network more than keeping the flag.
prediction: >-
  Written before any quoted score. P1: C1 (flag removed, rows kept) is not worse than C0 by more than
  the non-inferiority margin of 0.05 points of CRPS at either group. P2: C2 (flag removed, lockdown
  rows dropped) is no worse than C1, and better than C1 by 0.00 to 0.05 points. P1 fails if C1 minus
  C0 has an upper Bonferroni bound above 0.05 at a group. P2 fails if C2 minus C1 is positive at both
  groups.
decision_rule: >-
  See "Pre-registered rules". The flag is "not shown to be needed" if, at both groups, the upper
  Bonferroni bound of C1 minus C0 is below the margin of 0.05 points. It is "needed" if the lower
  Bonferroni bound of C1 minus C0 is above zero at a group. Otherwise the experiment cannot tell.
benchmark: >-
  C0 (the flag as in experiments 425, 448, 449, and 510, pretrained here with the same loop) for every
  contrast.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity, q99 pinball loss, and CRPS by horizon band.
---

## Plan (status: planned; no score has been read)

**The plan was written before any score was read.** One fold and one seed (2025-10, seed 0) were
timed first; their forecasts are kept as part of the final fits and are not scored until the code
review has run. The take-home message, the verdict, and the results are added when the experiment
finishes. The plan covers Part 2 of the brief (the lockdown flag); the whole-day price shape is
[experiment 513](../00513-lagless-demand-network-whole-day-price-shape-at-training-time/result.md).

## Question

The lag-free demand network takes a 0/1 input that is 1 from 2020-03-23 to 2021-07-19. The network is
pretrained on ERA5 and CAMS rows from 2019-09, which include that period, and fine-tuned on ensemble
forecast rows from 2024-04, which do not. The flag is 0 in every scored month. Is the column still
needed, and do the lockdown rows hurt a network that cannot see the flag?

## Arms

All arms use experiment 510's `L-M` recipe (monthly refit, 182.6-day weight, 3 mixture components,
hidden width 64, 2 layers, dropout 0.1), pretrained here by experiment 513's loop. They differ only
in the flag column and the pretraining rows.

| Arm | Flag column | Pretraining rows | Network inputs |
|---|---|---|---|
| C0 | kept | all | 37 non-lag columns |
| C1 | removed | all | 36 |
| C2 | removed | the flagged rows (2020-03-23 to 2021-07-19) dropped, from training and early stopping | 36 |

**C2 is a pretraining-only change.** The fine-tune rows hold no flagged row, and the scripts assert
this for every fold. **C1 changes the fine-tune too**, because the network has one input fewer, so
the pretrained weights of experiment 448 do not load and every arm, C0 included, is pretrained here.
In the first fold 1,395,832 of 7,005,809 pretraining rows are flagged (19.9%).

## Folds, seeds, and cost

- **Folds and seeds.** The five folds of experiments 448, 449, and 510 (2025-10, 2025-12, 2026-01,
  2026-03, 2026-06), seeds 0, 1, and 2. July 2026 is unused.
- **Fit count.** 3 arms x 5 folds x 3 seeds: 45 pretrainings and 45 fine-tunings.
- **Timed cost.** Fold 2025-10, seed 0: feeder 13 s per fold; pretraining 20 s (C0), 16 s (C1), and
  18 s (C2); fine-tuning 7 s, 4 s, and 5 s. About 0.3 GPU-hours for everything (13 s x 5 + 45 x
  (18 + 5.3) s = about 19 minutes). One GPU process under 8 GB, with `MemoryMax=8G` and
  `OMP_NUM_THREADS=1`.

## Scoring and contrasts

- **Rows and scorer.** Experiment 510's pipeline on experiment 470's row set, rows with no labelled
  event, leads up to 10 days, raw quantiles (no widening). "The eight" are the eight Primaries that
  report only apparent power (D15 and D18 report-only); "the six" are the six that report signed
  megawatts.
- **Contrasts (negative favours the first arm).** C1 minus C0, C2 minus C0, C2 minus C1, at CRPS
  over days 1 to 10, at the eight and the six: 6 tests. The Bonferroni count is the family of the
  whole price-and-lockdown brief, 18.
- **Slices.** Horizon bands day 1, days 2 to 3, and days 4 to 10 beside days 1 to 10, and Christmas
  week (22 to 28 December, fold 2025-12).
- **Minimum detectable effect.** From the seed spread of the three arms pooled (three seeds, five
  folds), scaled to a 3-seed mean, printed by `e00514_score.py baseline` before any contrast.
  Earlier experiments report 0.057 to 0.164 points for single-seed or two-seed comparisons.
- **Intervals.** Month-block bootstrap with five blocks gives near min-to-max intervals; the page says
  so and gives the sign per fold beside every interval.

## Pre-registered rules

- **Not shown to be needed.** At both groups the upper Bonferroni bound of C1 minus C0 is below the
  non-inferiority margin of 0.05 points. "Within the minimum detectable effect" is not used: below
  the minimum detectable effect, no difference can be proved.
- **Needed.** The lower Bonferroni bound of C1 minus C0 is above zero at a group.
- **Lockdown rows hurt.** The upper Bonferroni bound of C2 minus C1 is below zero at a group.
- **Otherwise** the experiment cannot tell, and says so.

## Standing practices followed

Cleaned data and the D15 and D18 start cut (through experiment 428's rows); rows with no labelled
event; July 2026 unused; no XGBoost work; titles say "load-shift", never "switching"; every interim
number is unaudited.

## Implementation details (deleted when this ships)

Code in `ml-experiments/00514-.../`: `e00514_fit.py` (pretrain, fine-tune, forecast) and
`e00514_score.py` (experiment 510's scorer pointed here). Run with `PYTHONPATH` set to the
`ml-experiments/*` folders and `DATA_PATH_INTERNAL=/home/jack/dev/nged-substation-forecast/data`.
