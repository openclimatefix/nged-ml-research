---
id: "00489"
slug: distill-the-five-network-pool-into-one-network
title: >-
  Can one network, trained to match the five-network pool, forecast Primary demand as well as the pool?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: finished
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
verdict: not supported
best_arm: >-
  Dpool50 (alpha 0.5): CRPS 4.762 points at the eight and 3.528 at the signed-MW six, against 4.680
  and 3.451 for the pool and 4.764 and 3.517 for a single seed.
headline: >-
  No student recovers the pool's gain. Dpool50 closes 0% of the 0.084-point gap between a single
  seed and the pool at the eight (4.762 against the single seed's 4.764) and none at the six, and
  its upper Bonferroni bound against the pool is 0.102 at the eight, above the 0.065 minimum
  detectable effect.
lessons: >-
  A student started from the same pretrained network and trained on the teacher's in-sample outputs
  scores the same as one more single seed. The tails match the teacher: held-out-week coverage is
  within 0.01 of the teacher's at q05, q95, and q99.
take_home_message: >-
  One network trained to copy the five-network pool forecasts no better than a single network
  trained normally, so the pool's gain cannot be moved into training with this recipe. The student
  copies the pool's tails but not its accuracy, probably because the teacher is in sample on every
  row it teaches from. Serving the pool (5 networks) stays the way to get that gain.
advice:
  - >-
    Do not distil the five-network pool into one network with a loss on the teacher's in-sample
    outputs; it scores like a single seed (Results table).
  - >-
    Use a pinball loss against the teacher's whole quantile distribution, not against one teacher
    value, when distilling quantiles; the point-target form widened the tails in the timing fit
    (Design).
production_status: not-for-production
adopted_in: none
---

## Take-home message

**One network trained to copy the five-network pool scores like a single seed, not like the pool.**
At the eight MVA-only Primaries the single seed is 0.084 CRPS points worse than the pool, and every
student sits within 0.006 points of the single seed. The stop rule applies: under this recipe the
pool's gain is not learnable by one network. The students do copy the pool's tail coverage.

## Question and answer

Can one student network, fine-tuned to match the pool's quantile forecast, score within the
minimum detectable effect of the pool? No. The pool's advantage over a single seed comes from
averaging five independently fitted networks. A student that starts from the same pretrained
weights and sees the teacher only on rows the teacher was fitted on gains nothing from it. The
explanation is a likely cause, not a tested one.

## Results (CRPS in points of effective capacity, 1.15 widening, 5 folds, rows with no labelled event)

| Slice | Pool (5 networks) | S0 (single seed) | Dpool50 | Dpool80 | DpoolERA5 |
|---|---|---|---|---|---|
| Eight | 4.680 | 4.764 | 4.762 | 4.770 | 4.765 |
| Six, no labelled event | 3.451 | 3.517 | 3.528 | 3.561 | 3.557 |
| Cold nights, eight | 4.161 | 4.286 | 4.292 | 4.331 | 4.320 |
| Cold nights, six | 5.931 | 5.955 | 5.944 | 5.997 | 5.973 |
| 12 to 36 h, eight | 4.126 | 4.211 | 4.217 | 4.230 | 4.227 |
| 12 to 36 h, six | 3.543 | 3.584 | 3.600 | 3.637 | 3.630 |

- **Minimum detectable effect**, printed before any student score: 0.065 at the eight and 0.100 at
  the six (experiment 470's five single seeds); from this experiment's three single seeds 0.077 and
  0.126. The cold-night minimum detectable effect is not separately estimated; cold-night contrasts
  below the gate are descriptive.
- **Decision rule (Bonferroni 27)**: the upper bound of student minus pool must be at most the
  minimum detectable effect. Dpool50 gives +0.082 (upper 0.102) at the eight, which fails, and
  +0.077 (upper 0.094) at the six, which passes. Dpool80 (+0.090, upper 0.113 at the eight; +0.110,
  upper 0.135 at the six) and DpoolERA5 (+0.085, upper 0.109; +0.106, upper 0.128) fail at both. No
  student passes both.
- **q99 pinball loss** (+0.02 rule): every student is within +0.015 of the pool (upper bounds
  0.008 to 0.015), so the tail rule passes. Coverage at q95 is 0.945 to 0.948 at the eight against
  the pool's 0.952.
- **Gap closed** (descriptive): Dpool50 about 2% and Dpool80 and DpoolERA5 below 0% at the eight;
  the stop rule threshold is 40%.
- **Held-out-week coverage** (mean over folds and seeds) of the students at q05, q95, q99:
  0.045 to 0.046, 0.949 to 0.950, 0.991 to 0.992, against the teacher's single network 0.049, 0.950,
  0.991. On training rows the coverage is sharper (q05 0.034 to 0.035), as the in-sample teacher is.
- **Avoidable-curtailment cost at the six, f = 0.9** (units of experiment 470's table, 95%
  intervals): pool 1.89 (0.18 to 4.98), single seed 1.87, Dpool50 1.91, Dpool80 2.00, DpoolERA5
  1.94, manual heuristic 2.02. Student minus pool is +0.019 (-0.032 to 0.093) for Dpool50. At f = 0.5
  Dpool80 is +0.79 (0.34 to 1.35) and DpoolERA5 +0.49 (0.19 to 0.89) above the pool.
- **Epochs**: Dpool50 11.5, Dpool80 13.5, DpoolERA5 17.3 on average.

## Serving table

| Forecaster | Networks per serving pass | Stored tables at serving | Fits per monthly refit | Score cost against the pool (eight / six) |
|---|---|---|---|---|
| A-pool | 5 | none | 5 (pretrained weights reused) | 0 |
| Student (Dpool50) | 1 | none | 6 (teacher pool, then student) or 1 if the teacher is kept | +0.082 / +0.077 |
| Single seed | 1 | none | 1 | +0.084 / +0.066 |

The student removes four networks from serving and scores the same as the single seed that costs
the same to serve, so it offers nothing over a single seed.

## Checks and caveats

- **Recipe**: monthly refits of experiment 425's folds (2025-10, 2025-12, 2026-01, 2026-03,
  2026-06); the D15 and D18 start cut and the flagged exact-zero drop apply; the teacher is the
  same-month pool, so no scored actuals reach the student; the students start from the stored
  pretrained weights of the same seed, so student seed s shares its initialisation with teacher
  s. Code reviewed by one fresh Sonnet agent before the first fit (no bugs; three small fixes).
- **Design change from the brief**: pinball loss against the teacher's quantile distribution, not
  against one value, because the point-target form made the timing fit's tails too wide (q05
  coverage 0.027 against 0.049).
- **Early stopping** sees the composite loss, so `val_loss` differs by alpha and is not compared
  across arms.
- Not run: the mixture-teacher arm (dropped on the queue-order check), the earlier-months-only
  variant, EMA/SWA as a reference, example-forecast figures, and event plots. Every score is
  selection-biased; July 2026 is unused. NGED's event labels are incomplete, so "rows with no
  labelled event" may hold unlabelled events.
- Christmas week and D15 and D18 are in `work/scores/contrasts.csv` (report-only): the pool's
  advantage over a single seed is larger there (0.231 and 0.454 points).

## What to try next

- If the pool's gain is worth keeping, serve the pool; if one network is wanted, spend the effort
  on better single-network training, not on copying the pool.
- Distillation could be retried with a teacher that is out of sample on the student's rows (a
  different month's pool on rows the teacher did not train on), at the cost of a leakage review.

## Plan (as registered)

The plan as committed before any fit; the loss paragraph was updated once after the timing fit.

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
  `NLL(y) + alpha * P(q_s, teacher) + (1 - alpha) * P(q_s, y)`, multiplied
  by the fold's recency weight `0.5 ** (age_days / SHORT_HALF_LIFE_DAYS)` and normalised by the
  batch's weight sum. `NLL` is the student's own mixture likelihood. `P` is the pinball loss
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
