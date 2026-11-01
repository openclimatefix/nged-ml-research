---
id: "00487"
slug: era5-and-cams-replay-rows-during-demand-network-fine-tuning
title: >-
  Does mixing ERA5-and-CAMS rows into the fine-tuning batches improve the demand network on cold
  winter nights and in the tails?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
date: 2026-11-01
size: >-
  medium. Stored: 45 fine-tuned networks (the state files and one 11-quantile forecast per fit) in
  the experiment's private folder. More than one design would defensibly test the question (how a
  replay share is defined, which rows count as cold nights), so the plan fixes each before any
  score is read. Production serving path: not touched (this repository holds none). Degradation
  rule: not touched. Callers: the scripts import experiments 397, 418, 425, 454, and 470, whose
  callers sit in those folders. Compute: 45 fits plus one reproduction fit, about 65 minutes of
  wall time with one GPU process under 3 GB, one feeder core. One Sonnet plan review, one Sonnet
  code review before the first run, one Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  Fine-tuning experiment 425's arm A on the ensemble (ENS) rows alone may make the network forget
  what its ERA5-and-CAMS pretraining taught it about rare cold winter nights, which the ENS history
  (from 2024-04) holds few of. Replaying ERA5-and-CAMS rows during fine-tuning, especially cold-night
  rows, lowers the CRPS on cold winter nights.
prediction: >-
  Written before any score is read. P1: Rcold lowers cold-night CRPS by at least 0.05 points of
  effective capacity at the eight and at the six; because the cold-night minimum detectable effect
  is likely above 0.05, a gap below the printed cold-night minimum detectable effect is called
  inconclusive. P2: R25 and R50 leave CRPS on all gating rows (the eight, and the six on rows with no
  labelled event) equivalent to R0, meaning the Bonferroni interval of the contrast lies inside the
  minimum detectable effect printed before any contrast (this replaces the earlier "moves CRPS by
  less than 0.03", which is below the noise and cannot fail). P3: no arm raises q99 pinball loss by
  more than 0.02 points at the eight or at the six on rows with no labelled event. Experiment 452
  found that a longer pretraining did not help, so the null is the likely outcome.
decision_rule: >-
  Score = CRPS in points of effective capacity, 1.15 widening unless stated, three seeds averaged
  per arm, month-block bootstrap (month by origin day, 20,000 resamples) with Bonferroni intervals.
  The count of tested contrasts is 27, fixed before any score is read: 3 contrasts (R25, R50, and
  Rcold, each against R0) times 9 tests (CRPS on 7 row sets: the eight; the six on rows with no
  labelled event; the six on real-event rows against normal-running-arrangement actuals; cold winter
  nights at the eight and at the six; the 12 to 36 hour band at the eight and at the six; plus the
  q99 pinball loss at the eight and at the six on rows with no labelled event). Christmas week, D15
  and D18, the other tail quantiles, the widening sweep, and curtailment are descriptive and enter
  no count. The minimum detectable effect (experiment 428's formula on the seed spread of arm A's
  five single seeds) is printed first, at the eight, the six, and on cold nights at each. An arm is
  a candidate if it meets its prediction and loses at most 0.02 points at the eight and at the six on
  rows with no labelled event. Stop rule: if no replay arm lowers cold-night CRPS by 0.05 points at
  both the eight and the six, report that the network is not forgetting the rare rows and close the
  idea. The smallest gain worth porting is 0.05 points on cold nights without loss elsewhere:
  replay needs the ERA5-and-CAMS rows at every monthly refit, so it adds a stored table and
  roughly 1.3 to 2 times the fine-tuning time.
benchmark: >-
  R0, which is experiment 425's stored arm A fit at seeds 0 to 2 on the same folds (not refitted
  except once to prove it reproduces), and experiment 470's A-pool (five-seed quantile average) as
  context.
row_set_id: >-
  The `gam14` row set of experiment 470 (the eight and the six; D15 and D18 are report-only on
  `main16`).
metric: >-
  CRPS in points of effective capacity, q01, q05, q95, and q99 coverage and pinball loss, and
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the six, per metering group and slice.
---

**Plan only: no fit has produced a score yet.** This page is the plan of experiment 487, committed
before any replay arm is scored.

**Question.** Fine-tuning on the ENS rows alone could let the network forget rare cold winter nights
that the ERA5-and-CAMS pretraining rows hold in larger numbers. The experiment mixes pretraining
rows back into the fine-tuning batches and asks whether the hard rows improve.

**Arms** (all fine-tune experiment 425's arm A from its stored pretrained weights
`work/pretrained/A/<month>_s<seed>.pt`, learning rate 2e-4, five folds of experiment 449: 2025-10,
2025-12, 2026-01, 2026-03, 2026-06; seeds 0 to 2):

- **R0:** no replay. This is experiment 425's stored fit (`seed_preds/A`, `models/A`), not
  refitted. One fit (2026-01, seed 0) is refitted with the new loop to prove it reproduces within
  noise.
- **R25:** replay rows carry 25% of each step's loss weight.
- **R50:** replay rows carry 50% of each step's loss weight.
- **Rcold:** as R25, but replay rows are drawn only from cold nights (sun more than 5 degrees below
  the horizon, 2 m temperature of 3 degrees Celsius or below).

**Definitions fixed before any score is read.**

- **Cold nights.** The brief defined cold nights as the lowest temperature decile, December to
  February, sun below -6 degrees. This plan uses the scorer's definition (experiment 470's
  `cold_nights` slice, from `studies.eval_protocol.slices`: sun more than 5 degrees below the
  horizon and temperature of 3 degrees Celsius or below), so the replay rows and the scored slice
  match. In the 2026-01 fold 332,535 of 7,122,133 replay training rows (4.7%) are cold nights.
- **Replay rows.** The fold's ERA5-and-CAMS pretraining rows from `e00425_prep.fold_pretrain`, with
  the held-out weeks and their purge excluded and both standing cuts applied (the D15 and D18
  start cut and the flagged exact-zero drop). Replay rows carry the same 46 input columns as the
  ENS rows, including the three origin lags with their missing flags and ages. The ENS-spread
  columns (`std_*`) are filled with the fold's median ENS spread by lead band, as in the
  pretraining stage. Replay rows are not recency weighted (the pretraining used equal weights).
  ERA5 rows after 2024-04 share valid times with the ENS rows; both sets are kept.
- **Epoch and batch.** An epoch is one pass over the ENS training rows in batches of 16,384 ENS
  rows, the same steps per epoch as R0. Each step adds `round(16384 * r / (1 - r))` replay rows
  (5,461 for R25 and Rcold, 16,384 for R50), drawn uniformly with replacement. R25 therefore sees
  about 1.33 times and R50 about 2 times the rows per epoch that R0 sees.
- **Loss.** `(1 - r) * ENS loss + r * replay loss`. The ENS loss is the recency-weighted mean
  negative log-likelihood of experiment 425 (weight `0.5 ** (age_days / SHORT_HALF_LIFE_DAYS)`,
  multiplied by nothing new). The replay loss is the unweighted mean. Early stopping reads the
  unweighted mean negative log-likelihood of the ENS held-out weeks only, as R0 does; replay rows
  never enter the validation. Patience 6 and the 40-epoch cap are unchanged.
- **Memory plan.** The fold's 4.9 million ENS rows (0.9 GB) and 7.9 million replay rows (1.5 GB)
  sit on the GPU for the whole month, 2.9 GB peak measured on one fit. The host builds both arrays
  once per month (about 6 GB resident) and frees them after the copy. One GPU process, one feeder
  core, `MemoryMax=14G`.
- **Past-weather caveat.** The replay rows use ERA5-and-CAMS weather, which is not available at
  forecast time. It enters training only, as in the pretraining; scoring inputs are ensemble forecasts.

**Scoring.** Experiment 470's rows and slices (the eight, the six on rows with no labelled event, the
six against normal-running-arrangement actuals, cold nights, the 12 to 36 hour band, and the report-only
Christmas week and D15 and D18), each replay arm against R0 with the same seeds, and A-pool as
context. Tails (q01, q05, q95, q99 coverage and pinball loss, raw and with the 1.00 to 1.15 widening
sweep) and avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the six are descriptive. Every
score is selection-biased, because the five folds that choose the candidate also score it. July 2026
stays unused. Flagged exact-zero half-hours are dropped; the start cut for D15 and D18 applies.
Rows with no labelled event are the clean rows; NGED's event labels are incomplete.

**Production cost.** One serving pass runs the same single network (or the five-seed pool) as arm A.
Replay changes only the monthly refit: it needs the ERA5-and-CAMS pretraining table at refit time and
adds about 1.3 times (R25) or 2 times (R50) the fine-tuning steps' rows. The serving path is unchanged.

**Compute.** 45 fits (3 arms times 5 folds times 3 seeds) plus one reproduction fit. Measured on one
R25 fit: 22 seconds of training, 55 seconds of forecasting, 2.9 GB peak GPU. About 65 minutes of
wall time, under 1 GPU-hour.

**Stored predictions reused.** R0 is experiment 425's `seed_preds/A` (copied through experiment 470's
`A-s0`, `A-s1`, `A-s2`, the same fits on the same rows); A-pool, `A-s3` and `A-s4` are experiment 470's.
Experiment 470 applied the 1.15 widening in its scorer and so does this plan.
