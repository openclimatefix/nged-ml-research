---
id: "00425"
slug: best-mdn-architecture-assembly-with-ablations
title: >-
  Does the pooled mixture density network that combines every component the earlier experiments
  supported beat its parts and the references?
theme: production-reference-and-best-guess
also_relevant_to:
  - hierarchy-and-pooling
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: forecasts and fitted networks in this experiment's private folder only; every
  earlier experiment is read and never written. Production serving path: not touched (this
  repository holds none). Degradation rule: not touched. Designs: more than one defensible (which
  components enter, and which leave-one-out arms to spend fits on), so the arms register each.
  Callers: the scripts import experiments 87, 48, 397, 407, 418, and 419 and the `studies`
  evaluation code, whose callers sit in those folders.
hypothesis: >-
  The pooled mixture density network that adds ERA5-and-CAMS pretraining (experiment 418) to the
  pooled plain network with the Primary embedding, the ensemble-summary inputs, the hand-made
  calendar columns, and the 1.15 widening (experiments 397, 403, 406, 407, 423) scores lower CRPS
  at days 1 to 10 than the same network without pretraining, and each leave-one-out arm scores
  worse than the assembled network.
prediction: >-
  Written before any fit and before any score is read. P1. The assembled network (A) beats the
  plain reference (R) by 0.04 to 0.08 points of effective capacity at the signed-MW six and 0.10 to
  0.20 at the MVA-only eight, repeating experiment 418's P1 (-0.061 and -0.156) with five seeds. P2.
  Removing the Primary embedding from A (A-noemb) is worse than A by at least 0.15 points at both
  groups (experiments 403 and 406 found 0.185 and 0.216). P3. Removing the 1.15 widening is worse
  than A on q95 and q99 coverage and on CRPS by less than 0.03 points. P4. A is within 0.05 points
  of the best monthly XGBoost recipe at the signed-MW six and better at the MVA-only eight. P5. The
  solar summary added to A (A+sol) changes the CRPS by less than the minimum detectable effect, as
  in experiment 419.
decision_rule: >-
  CRPS at days 1 to 10 in points of effective capacity, 1.15-widened as in experiment 419's scorer,
  read per group (signed-MW six; MVA-only eight; D15 and D18 report-only), paired by fold and seed,
  with the month-block bootstrap of `studies.eval_protocol`, 20,000 resamples, at a 95% level
  Bonferroni-corrected over the tested contrasts. C1 A minus R. C2 A-noemb minus A. C3 A unwidened
  minus A widened (free, scoring only). C4 A minus the monthly XGBoost recipe of experiment 389
  (arm S0M). C5 A+sol minus A (4 screen months only). Seeds 0 to 4 for every arm, so the minimum
  detectable effect is printed before any contrast is read: twice the standard deviation over the
  folds of the difference in the signed-MW six CRPS between seeds 0 to 2 and 3 to 4 of R, scaled
  to five-seed means. A contrast smaller than that is "inside the noise". The assembly adds a
  pretraining stage to the pipeline, so the smallest gain worth porting is 0.03 points at the
  signed-MW six or 0.10 at the MVA-only eight (the 418 threshold). C1 supports the hypothesis
  if its interval is wholly below the threshold at the signed-MW six; the verdict is partly
  supported if only the MVA-only eight meets it. No adoption is sought; a numbered release needs the
  weekly re-check of the `release-model` routine.
benchmark: >-
  R, the ENS-only pooled plain network of experiment 418 (seeds 0 to 2 stored, seeds 3 and 4
  fitted here); BG2, the monthly XGBoost recipe of experiment 389, `manual_heuristic`, and
  `climatology` as absolute-skill references, scored on the half-hours they share.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, coverage and pinball loss at q01, q05,
  q95, and q99, and the avoidable-curtailment cost, raw and at the 1.15 widening, in points of
  effective capacity, from `studies.eval_protocol`; leads 0.5 to 336 hours; the headline band is
  days 1 to 10.
code: ml-experiments/00425-best-mdn-architecture-assembly-with-ablations/
---

*Status: planned. No fit has run and no score has been read.*

## Plan

**The assembly is mostly experiment 418's P1 arm, so the new work is five seeds, three
leave-one-out arms, and the reference table.** Experiments 397 to 423 tested one component at a
time on the same pooled network. Of the components the maintainer's list names, only ERA5-and-CAMS
pretraining sits on top of the plain reference. The rest (ensemble-summary inputs, Primary
embedding, hand-made calendar columns, three origin lags, monthly refits, 1.15 widening) are
already in the reference.

| Component | In A? | Supporting experiment and claimed gain |
|---|---|---|
| Ensemble mean and spread inputs, applied once | yes | 397: pooled summary scheme 3.264 against XGBoost 3.542 at the signed-MW six (five months) |
| Pooled network with 8-wide Primary embedding | yes | 403: -0.185 and -0.161; 406: embedding -0.216 against per-Primary |
| ERA5-and-CAMS pretraining, fine-tune all weights | yes (P1) | 418: -0.061 at the six, -0.156 at the eight, 12 monthly folds |
| Hand-made calendar columns | yes | 423: encoder arms 0.27 to 0.37 worse |
| 1.15 widening about the median | yes | 407: best of four tail fixes |
| Monthly refits, three origin lags | yes | standing practice; 397 |
| Solar summary | ablation only (A+sol) | 419: not supported (-0.012) |
| Quantile head | excluded | 407: q99 pinball -15.8% but CRPS +0.066 |
| Export gate for signed MVA output | not in A | 420: gain is on reverse-flow half-hours only; 424's serving test has not merged |
| E1_es2 inputs, B-mix, lag trust head, varied origin hour | excluded | 396, 404, 422 have not merged; none has a page to support it |
| Explicit solar term, shared calendar encoder, shapes network, capacity prior | excluded | 409, 416, 423, 413, 412 failed |

**Arms.** R (reuse 418 `ens-only`, seeds 0 to 2; fit seeds 3 and 4). A (reuse 418 `P1`, seeds 0 to
2; fit seeds 3 and 4, including their pretraining). A-noemb (pretrain and fine-tune with no
Primary input, five seeds). A unwidened (scoring only). A+sol (A plus experiment 419's two
daily-mean shortwave columns, five seeds, four screen months). References: BG2, the XGBoost recipe
(389 S0M), `manual_heuristic`, `climatology`. The three leave-one-out components are pretraining
(that is R), the Primary embedding, and the widening, chosen as the three with a claimed gain at
or above 0.06 that can be removed without rebuilding the input pipeline.

**Compute.** About 230 new fits, 1 to 2 minutes each, so 4 to 8 hours on one GPU process under 8 GB
with one feeder core (`MemoryMax=8G`, `OMP_NUM_THREADS=1`). R: 24 fits. A: 24 pretrain and 24
fine-tune. A-noemb: 60 and 60. A+sol on four months: 20 and 20. Stored predictions from 418 save
about 72 fits. The stop line is 12 GB available in `free -g`.

**Standing practices followed.** Monthly refits; cleaned data with the flagged half-hours counted
per fold; D15 and D18 cut before 2025-03-01; every fitted network saved to the data folder; reuse
of 418's stored predictions (same keys, same recipe, 1.15 applied at scoring); tails and
avoidable-curtailment cost beside CRPS; example day-ahead and 14-day forecasts; no load-shift
claims. Earlier advice followed: use at least five seeds and print the minimum detectable effect
(419); score the references on shared half-hours (418).

**What changed against the brief.** The brief's list assumed more components than have evidence.
Experiments 396, 404, 422, and 424 have no merged page, so they enter no arm. Experiment 420's gate
changes the target the eight is scored on, so it stays out of A and is a follow-up.
