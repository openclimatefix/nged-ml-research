---
id: "00569"
slug: modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1
theme: demand-inputs
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  medium, close to complex. Stored: fitted network weights and window forecasts per arm under this
  experiment's private data folder; no earlier experiment's files are written. Serving path: not
  touched (this repository holds none). Degradation rule: not touched. More than one design is
  defensible (which encoder reads each group, how a group with no history before 2023 enters
  pre-training, how the school-proximity input is gated), so the plan fixes each before any score
  is read. Callers: this folder's scripts import experiments 397, 418, 422, 428, 437, 448, 510, 511,
  513, 515, 554, and 562. One Sonnet review of the code before the first fit, one Sonnet review of
  the write-up against the printed report, and one Sonnet prose review.
hypothesis: >-
  Experiment 562 added five input groups to the modular lag-free network at fine-tuning only. Each
  group may help once it is also pre-trained on ERA5 and CAMS rows and enters through the encoder
  that matches its content: thermal-lag and solar-gain windows (B1) and look-ahead windows (B2)
  through the weather encoder, school terms and holidays (B3a) through the datetime encoder, national
  signals of the previous day (B3b) and the whole-day price shape (B4) through a small parallel
  encoder. A sixth group (B3c) adds, to B3a, the number of schools within 2 km of each Primary as a
  gate on the school columns.
prediction: >-
  Written before any fit. P1: B1 beats its shuffled-day placebo on cold winter nights at the six MW
  Primaries by at least 0.05 points of CRPS (562 found 0.231 from 3 monthly blocks) and does not
  actively hurt any other slice of the keep rule; it fails if the cold-night gain is within the
  minimum detectable effect or any slice is hurt. P2: B3a gains at least 0.03 on all rows at one
  group when entered at pre-training and fine-tuning (562: -0.027 at fine-tuning only); it fails if
  it is within 0.02 of its placebo. P3: B2, B3b, and B4 are each within 0.03 of their placebos; each
  fails if it gains 0.03 or more on all rows. P4: B3c beats B3a on all rows at the eight MVA Primaries
  by at least 0.02; it fails otherwise (the step-1 rank correlation of the school-term effect with
  school proximity was +0.28 with an interval that includes zero, so P4 is expected to fail).
decision_rule: >-
  The keep rule of addendum 14, part C: a group is kept if (a) real minus placebo improves all rows at
  a group by at least that contrast's minimum detectable effect (MDE) with a Bonferroni interval
  excluding zero, or (b) it improves cold nights, Christmas week, or the high-spread tercile by at
  least that slice's MDE with a Bonferroni interval excluding zero, and does not actively hurt
  any other slice. A slice is actively hurt when the CRPS difference is worse than the larger of
  0.03 points and the slice's MDE with the 5-block month interval excluding zero on the harmful side,
  or when the sign is harmful in at least 4 of 5 folds. The 0.03 floor is experiment 562's threshold
  because an added column needs a production feed. The leak (injected 7-day shift) is reported beside
  every contrast and does not enter the rule.
benchmark: >-
  Each real arm's own shuffled-day placebo, and plain M0 refitted on the same coarse schedule
  (experiment 554's pre-trained weights, seed 0). Experiment 562's fine-tuning-only scores are
  reference rows.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at the eight MVA-only Primaries and the six signed-MW
  Primaries, on all rows and on the slices of addendum 14, part C; D15 and D18 report-only.
---

*Status: planned. Nothing below has a result yet.*

**Take-home message.** To be written after the scores are read.

## Plan

- **Base network.** Arm M0 of [experiment 554](../00554-m0-replication-five-seeds-leak-mitigations-and-leave-d8-out/result.md), refitted once per fold (coarse refit) on experiment 511's five folds (July 2026 unused), seed 0 first and seed 1 if the GPU budget allows.
- **Pre-training.** Every arm is pre-trained per fold on ERA5 and CAMS rows (experiment 448's rows, with experiment 437's `dense_arrays` for B1 and B2) with its extra columns; the added input units start random, not zero.
- **Routes.** B1 and B2 enter the weather encoder; B3a and B3c enter the datetime encoder; B3b and B4 enter a parallel encoder (32 hidden units, 16 outputs) concatenated into both mixers.
- **Missing history.** A school calendar exists from 2021-09, national signals from 2023-01, and the price shape from 2023, while pre-training history starts in 2019-09. Pre-training rows without a source get zero columns and a presence flag of 0.
- **Placebos.** Each real arm has a shuffled-day placebo of the same width (experiment 515's donor days, same month, day type and length, at least 3 days away, same side of the refit), in both stages.
- **School proximity (B3c).** From the Department for Education's "Get Information about Schools" register (open data, Open Government Licence v3.0, downloaded 2026-11-05 into this experiment's private folder; [source](https://get-information-schools.service.gov.uk/Downloads)): the count of open schools within 2 km of each Primary, as a share of the largest count, multiplies the five school columns.
- **Weather timing.** Fine-tuning and scoring rows assert, row by row, that the weather run is issued at or before the forecast origin and print one example row per fold. Pre-training rows use the same ERA5 and CAMS stand-ins as the 37 base inputs, by construction.
- **Leak.** Each arm refits once at day 14 with and without the injected 7-day shift of experiment 511's harness; the leak is the signed change in the median forecast, as in experiment 554.
- **Statistics.** Month-block bootstrap with 5 blocks (near the minimum-to-maximum of the folds), per-fold signs, the MDE beside every contrast, and the Bonferroni count as a number.
- **Budget.** At most 4 GPU hours; one GPU process under 8 GB of host memory; a fit starts only when available memory exceeds 20 GB and fewer than 4 GPU processes run.

## What step 1 found (school terms in experiment 562)

![School-term effect per Primary against seed noise](figure_1_school_term_per_primary.svg)

- **The spread of B3a's effect across Primaries is smaller than seed noise.** Real minus placebo ranges from -0.230 to +0.314 points (standard deviation 0.137 over 16 Primaries); plain M0 seed 0 minus seed 1 on the same rows has standard deviation 0.224. Only D8 has a 5-block interval that excludes zero, on the harmful side. Interim and unaudited.
- **School proximity does not explain the spread.** Spearman correlation of the effect with open schools within 2 km is +0.28 (bootstrap over Primaries -0.30 to +0.72) and with schools within 5 km is -0.09. The pre-registered expectation (a rank correlation of at most -0.3 with an interval excluding zero) fails. D15 and D18 share one location, so their counts are identical.
