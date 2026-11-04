---
id: "00559"
slug: m0-leak-ladder-three-step-schedule-frozen-encoders-and-penalty
title: >-
  Can the modular lag-free demand network (M0) keep its gain while an injected shift in the recent
  data leaks less into its forecasts, using a three-step training schedule, frozen encoders, an L2
  penalty, gentler fine-tuning, weight averaging, or a metering-type output head?
theme: retraining-and-training-history
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: demand-v2
size: >-
  medium. Stored: step-2 weights, daily fine-tuned weights, window forecasts, and stress-run
  forecasts under this experiment's private data folder only; experiments 545 and 554 are read
  only. Serving path: not touched (this repository holds none). Degradation rule: not touched.
  Designs: more than one defensible for the L2 penalty's strength and for arm A8's non-negative
  head, so the plan fixes each before any score is read. Callers: the scripts import experiments
  397, 428, 448, 510, 511, and 545 and copy the code of experiment 554. One Sonnet code review
  before the first scored run.
hypothesis: >-
  M0's injected-shift leak (7.3 to 7.7 points of capacity, against 4.49 for the single network
  L-1d) comes from the fresh mixer that is trained only on the recent days at every daily refit.
  Training that mixer once on the whole ensemble history (step 2) and then fine-tuning gently
  (step 3) lowers the leak while keeping most of M0's gain at 24 to 120 h.
prediction: >-
  Written before any score is read. P1: A2 (encoders frozen in step 3) or A3 (L2 penalty to the
  step-2 weights) lowers the leak (all leads, mean of the five folds, seed 0) to at most 6.0 points
  of capacity while keeping at least 70% of M0's CRPS gain over L-1d at 24 to 120 h at the eight.
  P2: A1 alone lowers the leak by at least 1.0 point. P3: A6 (180-day half-life) lowers the leak
  but loses gain, as in experiment 536. P4: A8 changes CRPS by under 0.03 at both Primary groups.
  Stop rule: if none of A1, A2, and A3 lowers the leak by at least 1.0 point of capacity against M0
  (seed 0, same five folds), arms A4 to A8 are not fitted.
decision_rule: >-
  A gain or loss counts when the Bonferroni month-block interval (5 blocks, near the minimum to
  maximum of the five folds; 16 counted tests: arms A1 to A8 against M0, pooled over 24 to 120 h,
  at the eight and at the six) excludes zero. The leak is judged against the release checklist's
  limit of 6.0 points of capacity and against M0's own leak, and the retained share of M0's gain is
  judged with the leave-D8-out figure beside it. Any change that only alters how the daily fits
  start costs nothing to port beyond one stored step-2 file per Primary group, so its threshold on
  the leak is 1.0 point (the stop rule's size) and on CRPS is "not worse than M0 by more than the
  minimum detectable effect".
benchmark: >-
  M0 and M0lr with seeds 0 and 1 (experiment 545's stored fits), L-1d with seeds 0 and 1
  (experiment 511's), R-DW (experiment 539) and A-pool (experiment 425) report-only. The 5-seed
  M0 of experiment 554 is read for the leave-D8-out comparison only.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity by the seven lead bands of experiment 521, at the eight
  MVA-only Primaries and the six signed-MW Primaries, with D15 and D18 report-only; injected-shift
  leak in points of capacity.
---

**Planned.** Write-up follows the scoring. Arms (2 seeds, 0 and 1, five folds of 511; the leak uses
seed 0 and experiment 511's stress world):

- **M0 and M0lr (A0):** experiment 545's stored fits. M0 trains a fresh mixer at every daily refit.
- **Step 2 (shared by A1 to A5, saved):** start from the fold's pretrained M0 weights
  (experiment 554's, read only), attach a fresh mixer, and train the encoders and the mixer on every
  ensemble row with a valid time up to the fold's first cutoff minus 3.5 h and an origin up to the
  cutoff, uniform weights, experiment 510's loop and early-stopping weeks.
- **A1:** daily fine-tune of the step-2 weights at the 2e-4 rate with a 90-day half-life.
- **A2:** A1 with both encoders frozen (and in evaluation mode) in step 3.
- **A3:** A1 plus an L2 penalty `lambda * sum (w - w_step2)^2` to the step-2 weights. `lambda` is
  chosen from {1e-3, 1e-2, 1e-1, 1} on fold 1 (2025-10) only: the largest value whose early-stopping
  NLL at the fold's first cutoff is within 0.002 of the best.
- **A4:** A1 with a rate of 5e-5 and at most 5 epochs.
- **A5:** A1 with the weights of the last 5 daily fine-tunes averaged (inference only on A1's
  stored weights).
- **A6 and A7:** M0 (fresh mixer at each daily refit) with a 180-day half-life, and refitted every
  third day (experiment 554's code, copied).
- **A8:** A1 with a folded-normal mixture head for the MVA-only Primaries and the usual mixture for
  the signed-MW Primaries.

Compute: about 4 GPU-hours (one process under 8 GB, one thread). If the stop rule or the budget
changes the plan, the write-up says which arms ran.
