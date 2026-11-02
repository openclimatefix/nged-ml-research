---
id: "00525"
slug: metering-type-flag-lagless-demand-network
title: >-
  Does telling the lag-free demand network whether a Primary reports signed MW or apparent power in
  MVA, with one 0/1 input column, improve its forecast?
theme: hierarchy-and-pooling
also_relevant_to:
  - demand-inputs
status: planned
kind: change
protocol: demand-v2
size: >-
  medium. Stored: fitted network weights, window forecasts, and score tables under this experiment's
  private data folder only; experiments 448, 510, and 511 are read, never written. Serving path: not
  touched (this repository holds none). Degradation rule: not touched. Designs: more than one is
  defensible (what the constant-flag control holds, and how the mixture's mass below zero is read),
  so the plan fixes both before any score is read. Callers: this folder's scripts import experiments
  428, 448, 470, 510, 511, and 519 and the `studies` evaluation code. One Sonnet plan review, one
  Sonnet code review before the first fit, one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  The 8-wide Primary embedding of the lag-free network is mostly its random start, so it may not
  carry the one fact that separates the eight MVA-only Primaries from the six signed-MW Primaries:
  apparent power cannot be negative, and signed power can. One extra 0/1 input column (1 for
  apparent power in MVA) tells the network that fact directly. If the embedding already carries it,
  the column adds nothing; if the column improves the forecast or removes the mixture's probability
  mass below zero at the eight, the embedding is too thin.
prediction: >-
  Written before any score is read. P1: the flag does not move CRPS at either group, so the
  paired interval of flag minus constant flag contains zero at the six and at the eight (a flag
  duplicates information the embedding may already carry); the page words this as "not shown to move
  CRPS (interval a to b)". P2: at D8, D12, and D22 (the only ones of the eight whose stored q01 falls
  below zero in a proxy check, which the first inference-only step re-verifies) the flag cuts the
  mixture's probability mass below zero by at least half against the paired control C: the mean mass
  of M divided by that of C is at most 0.5 in both seeds in at least 4 of 5 folds. The stored L-Ww
  mass is shown for scale only. P2 is tested per Primary, with per-fold values, and is untestable at
  a Primary whose baseline mass is below 0.1% of rows. P3: the flag
  moves q99 pinball loss by less than 0.01 at both groups (experiment 519's flag moved it by 0.011
  to 0.012, so a larger move falsifies P3). P4 is description, not a test: at D10 (the one signed-MW
  Primary with
  reverse flow on 15 of 117 days since 2025-10) the flag is 0 in both M and C, so any difference
  between them comes from interference through shared weights; the page reports the share of
  forecast medians and q05 below zero on those days. P5 (falsifier): M is worse than C at either
  group, with an interval excluding zero.
decision_rule: >-
  A contrast counts only if its Bonferroni interval excludes zero. The counted family is 8 tests:
  2 contrasts (flag minus constant flag, the core; flag minus L-Ww, the 37-input anchor) x 2 groups
  (six, eight) x 2 metrics (CRPS, q99 pinball loss). q05 and q95 pinball loss and coverage at all
  four tails are report-only. Porting the column needs a maintained metering label per series (a
  data-contract change), so the smallest gain worth porting is 0.03 points of CRPS, the margin
  experiment 519 used. This experiment's own paired-block minimum detectable effect is computed from
  its seed-paired fits and printed beside every contrast (experiment 511's 0.164 and 0.123 are for a
  larger family and are shown for scale only). A null is worded "an effect smaller than X could not
  have been detected" and "not shown to move CRPS (interval a to b)". If CRPS improves by
  at least the minimum detectable effect, the embedding is too thin and the descriptor-initialised
  embedding, higher embedding learning rate, and static-descriptor options move up the queue.
benchmark: >-
  The constant-flag control C is the counted control. C is experiment 519's stored arm F0z
  (38 inputs, the column 0 in pretraining, fine-tuning, and scoring, seeds 0 and 1, the same five
  folds, CRPS 4.875 at the eight and 3.556 at the six). L-Ww (experiment 510,
  weekly, 90-day half-life, 37 inputs) is the anchor; K2-S1, BG2, climatology, and `manual_heuristic`
  are linked for scale.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at the six signed-MW and the eight MVA-only Primaries, q01,
  q05, q95, and q99 coverage and pinball loss, the mixture's probability mass below zero per Primary
  at D8, D12, and D22, and the share of half-hours at D10 whose forecast median or q05 is below zero.
verdict: ""
---

**Planned, not yet run. Every number on this page will be interim and unaudited until a human has
read it.**

## Plan (written before any fit)

- **The metering flag is the 38th input, not the 39th.** Experiment 519 (merged) keeps the 37-input
  L-Ww recipe as the base: its best arm F0 carried a source flag only as a constant column (0 in every
  ERA5-and-CAMS pretraining row, so its first-layer weights never trained), and its advice is to
  pretrain on ERA5 and CAMS rows alone. The recipe therefore has no source flag, and the metering
  flag is the only added column.
- **Arms.** M: 38 inputs, the 37 lag-free columns plus a metering column read from the Primary index
  (1 for apparent power in MVA, 0 for signed MW), pretrained with it, fine-tuned weekly from those
  weights. M is pretrained with experiment 519's `train_share` at share 0 (8,192 ERA5 and CAMS rows
  per step), with the column derived from the Primary index, so M and C share the recipe, the width,
  and (with the same `torch.manual_seed`) the initial weights. C: experiment 519's stored F0
  pretrains and F0z fits, reused unrefitted; its column is a source flag held at 0, which is a constant
  column here. A sanity check runs M's code path with the lookup forced to 0 and must reproduce F0z to
  float tolerance, and the row counts must match 519's. L-Ww: experiment 510's stored forecasts. The
  M minus L-Ww contrast is confounded with the width and batch difference that 519 documented (F0
  minus L-Ww is -0.043 at the eight). P (report-only): the same network with a fixed random
  8/6 assignment of the 16 Primaries to the flag, one draw, to separate "metering type" from "any
  extra per-Primary column". At the six, M and C have identical inputs (flag 0 in both), so the
  six's contrast tests interference and noise, not information.
- **Recipe.** The L-Ww harness: ERA5-and-CAMS pretraining (experiment 448's rows and loop), weekly
  refit, 90-day half-life, restart from the month's pretrained weights, five folds (2025-10,
  2025-12, 2026-01, 2026-03, 2026-06), seeds 0 and 1, weekly cadence because L-Ww is the weekly
  anchor. July 2026 stays unused. Cleaned data throughout, with the flagged exact-zero half-hours
  dropped (the count is printed per fold), and the D15 and D18 rows before 2025-03-01 cut.
- **Group of each Primary.** The flag takes the metering unit of each of the 16 Primaries. The eight
  are D2, D3, D4, D8, D12, D19, D21, and D22; the six are D5, D9, D10, D11, D14, and D16; D15 and
  D18 take the flag of their own metering unit (checked against the metering table before any fit)
  and are report-only. The code review checks the Primary-index-to-label mapping against `sites()`
  before any per-Primary number is quoted, because the earlier proxy inferred it.
- **Mass below zero.** For each row, the sum over the three components of the softmax weight times
  the normal CDF of (z0 - mean_k) / std_k, with z0 = -target_mean / target_std of the Primary; no
  outlier component. It is not zero for a correct network, because demand sits near zero at the eight.
- **Inference-only baseline first (CPU, no GPU go needed).** Load experiment 510's stored L-Ww
  fine-tuned state dicts (`models/hl90`), run them on the scored rows, and compute for each Primary
  the mixture's probability mass below zero (the mixture CDF at 0 MW after undoing the Primary's
  target standardiser), before any clipping. Report it per Primary at D8, D12, and D22. A Primary
  whose baseline mass is below 0.1% of rows cannot show a halving.
- **Reverse-flow case study at D10.** Take the stored reverse-flow days of experiment 506
  (`part1_days_six`, level 0: 15 of 117 days since 2025-10 at D10; D5 has 5 days in all, not enough)
  and report, for M, C, and L-Ww, the share of those days' half-hours whose forecast median or q05 is
  below zero beside the share of half-hours whose actual is below zero. It is a single-Primary case
  study.
- **Statistics.** Paired contrasts over origin-day blocks as in experiment 519's scorer, Bonferroni
  family of 8 (above), the two-seed minimum detectable effect printed beside every contrast, per-fold
  signs reported, intervals worded as "not shown to move CRPS (interval a to b)". The Part 2 probe
  of experiment 522 (the leave-one-Primary-out linear probe of metering type from the embedding) is
  reported beside the result, not as a gate, once experiment 522 merges. Tail table: coverage and
  pinball loss at 0.01, 0.05, 0.95, and 0.99. The load-shift reporting rule is not triggered,
  because no labelled event is scored.
- **Figures.** CRPS and q99 contrast of M minus C and M minus L-Ww with intervals (headline), the
  per-Primary mass below zero at D8, D12, and D22 for L-Ww, C, and M, the D10 reverse-flow days with
  the three forecasts, two example forecasts (day-ahead and 14-day, one Primary), and the per-fold
  signs.
- **Compute.** About 0.4 GPU-hours: 20 pretrains (arms M and P x 5 folds x 2 seeds, 31 to 46 s
  each) and 100 weekly fine-tunes (2 arms x 5 folds x 2 seeds x 5 weeks, about 5 s each) plus the
  month builds; C costs nothing because 519 stored it. One GPU process under 8 GB (`MemoryMax=8G`, `OMP_NUM_THREADS=1`).
  No fit starts before
  the manager's "GPU go". Experiment 510's L-Ww forecasts and weights and 519's F0 and F0z are reused,
  which saves 10 pretrains and 50 fine-tunes. Code changes: a 16-long flag lookup in `sites()` order
  (asserted), a `with_flag` variant that appends `lookup[p]` in pretraining, validation, `GpuMonth`,
  and the test rows, and a mixture-mass function.
- **Reviews.** One Sonnet plan review, one Sonnet code review before the first fit, a results review
  that re-derives the headline numbers, and two write-up reviews.
