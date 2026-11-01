---
id: "00489"
slug: distill-the-five-network-pool-into-one-network
title: >-
  Can one network, trained to match the five-network pool, forecast Primary demand as well as the pool?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: student weights and predictions under the experiment's data folder only. Serving
  path: not touched (this repository holds none). Designs: more than one defensible, so the plan
  fixes the loss, the teacher, and the rows before any fit. Callers: this folder's scripts, which
  import experiments 397, 425, 449, and 470. One Sonnet code review before the first fit, one
  Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  The pool of five pretrained-then-fine-tuned networks (A-pool, experiment 425's arm A) can be
  replaced at serving by one network of the same shape, fine-tuned to match the pool's forecast
  distribution, without losing forecast skill beyond the seed noise. This moves the seed averaging
  from the serving path into the training regime.
prediction: >-
  Written before any student score is read. P1: D-pool (the 0.5 and 0.8 teacher mixes) is within
  the printed minimum detectable effect of the pool, 0.065 points of effective capacity at the eight
  MVA-only Primaries and 0.100 at the signed-MW six, on rows with no labelled event, and its q99
  pinball loss is within +0.02 points of the pool's. P2: D-pool-ERA5 does no better than D-pool by
  more than the minimum detectable effect. P3: held-out-week coverage of the students at q05, q95,
  and q99 is within 0.01 of the teacher's. The fraction of the gap between a single seed and the
  pool that a student closes is reported as a descriptive number only.
decision_rule: >-
  A student is within the minimum detectable effect of the pool when the upper Bonferroni bound of
  its CRPS difference to the pool is at most the printed minimum detectable effect, at both the
  eight and the six, and the upper Bonferroni bound of its q99 pinball difference is at most +0.02
  points. Bonferroni count 27 (3 arms x 9 tests). Recommend D-pool if it passes: serving falls from
  5 networks to 1, and refits rise from 5 to 6 fits (the teacher plus the student) or to 1 if the
  teacher is retained between refits. If the point estimate closes 40 to 70% of the gap to a single
  seed but fails the bound, report the trade-off. If it closes under 40%, report that the pool's gain
  is not learnable by one network on these rows.
benchmark: >-
  The pool (A-pool) and a single seed of arm A (S0, the mean of seeds 0 to 2 scored separately),
  both from experiments 425 and 470.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity, q01, q05, q95, and q99 coverage and pinball loss, and
  avoidable-curtailment cost at f = 0.9, 0.75, and 0.5 at the signed-MW six.
production_status: not-for-production
adopted_in: none
---

## Plan

Status `planned`. The sections below are filled in as the experiment runs.

### Design

- **Teacher.** The five same-month fine-tuned arm A networks of experiment 425
  (`work/models/A/<month>/s0..s4`), all trained before the scored month. The teacher target of a row
  is the mean of the five networks' 11 sorted standardised quantiles, which is the quantile average
  that A-pool serves. Using another month's teacher would leak the scored actuals, so none is used.
- **Student.** Experiment 425's pooled plain network, started from the stored pretrained arm A
  network of the same seed and fine-tuned on experiment 425's ENS fold rows at 2e-4 with experiment
  397's loop (batch 16,384, at most 40 epochs, patience 6, gradient clip 1.0). The cuts of
  experiments 399 and 425 apply (the D15 and D18 start cut, the flagged exact-zero drop).
- **Loss.** For a row, in standardised units:
  `NLL(y) + alpha * P(q_s, teacher) + (1 - alpha) * P(q_s, y)`, multiplied by the fold's recency weight `0.5 ** (age_days / SHORT_HALF_LIFE_DAYS)` and normalised
  by the batch's weight sum. `NLL` is the student's own mixture likelihood. `P` is the pinball loss
  summed over the 11 levels. Against the teacher it is the expected pinball loss over the teacher's
  quantiles as atoms (the scorers' level masses), because a pinball loss against one teacher value
  is asymmetric and widened the student's tail quantiles in the timing fit (q05 coverage 0.027
  against the teacher's 0.049). Student quantiles are differentiable through the implicit function
  theorem on the mixture CDF. Early stopping sees the same composite loss on the held-out weeks,
  unweighted.
- **Arms** (3 seeds, 5 folds of experiment 449: 2025-10, 2025-12, 2026-01, 2026-03, 2026-06):
    - `Dpool50`: alpha 0.5;
    - `Dpool80`: alpha 0.8;
    - `DpoolERA5`: alpha 0.5, plus one batch per step (16,384 rows) of the ERA5-and-CAMS rows that
      pretrained the teacher, with the teacher term only and equal row weights. The teacher
      outputs on those rows come from the same five networks.
    - References: A-pool (the teacher, stored), and S0 (single seeds 0 to 2 of arm A, stored).
- **Not run.** D-mix (distilling the mixture of arm A seeds and lagless networks): dropped on the
  manager's queue-order check. The variant that distils only from rows of months before the
  student's validation month: not run. EMA/SWA is read from the weight-averaging experiment's page
  when it merges.
- **Teacher in sample.** Every teacher is in sample on the fine-tuning rows, so its quantiles are
  sharper there than on new months. The 0.5 true-target mix and the student's own likelihood limit
  the copying. A student cannot exceed an honest teacher. The log under `work/logs/` stores
  coverage at q05, q95, and q99 of the teacher and each student on the held-out weeks and on
  training rows.
- **Fits.** 45 student fits (3 arms x 5 folds x 3 seeds), plus five teacher passes per fold
  (inference). One fit takes about 128 s on the GPU (experiment 425's fit, for comparison, takes
  30 to 37 s with the forecast). Estimated 2.3 GPU-hours in total.
- **Monthly refits.** The stored folds are the monthly refits of experiment 425. The students are
  scored at the same cadence.

### Scoring

Rows are experiment 470's intersection (`experiment_470_intersection_v1`). Forecasts are widened by
1.15 as in experiment 470. Slices: the eight, the signed-MW six on rows with no labelled event,
cold winter nights, and the 12 to 36 hour band; Christmas week and D15 and D18 are report-only.
The month-block bootstrap with Bonferroni 27 intervals, absolute CRPS beside every contrast, July
2026 unused, every score selection-biased.
