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
  at days 1 to 10 than the same network without pretraining, and removing the Primary embedding or
  the widening makes it worse.
prediction: >-
  Written before any fit and before any score is read. All arms are refitted with the D15 and D18
  cut before 2025-03-01, so experiment 418's numbers are a guide only (they used experiment 103's
  cuts and no widening). P1. The assembled network (A) beats the plain reference (R) at 12 monthly
  folds by 0.03 to 0.10 points of effective capacity at the signed-MW six and 0.08 to 0.20 at the
  MVA-only eight (experiment 418 unwidened: -0.061 and -0.156). P2. A without the Primary embedding
  (A-noemb) is worse than A by 0.5 to 1.0 points at both groups (experiment 406: no embedding minus
  embedding +0.789 and +0.809, without pretraining, on five screen months). P3. Widening A by 1.15
  lowers CRPS by less than 0.03 points and moves q95 coverage towards 0.95 (experiment 407: 0.854
  to 0.891 at the signed-MW six). P4. A scores better than the monthly XGBoost recipe (S0M of
  experiment 389) at both groups; experiment 418 already found P1 2.846 against 3.018 for BG2 on
  shared half-hours.
decision_rule: >-
  CRPS at days 1 to 10 in points of effective capacity, read per group (signed-MW six; MVA-only
  eight; D15 and D18 report-only), at the 1.15 widening unless stated, paired by fold and seed, with
  the month-block bootstrap of `studies.eval_protocol`, 20,000 resamples, at a 95% level
  Bonferroni-corrected over four tested contrasts (alpha 0.0125). C1 A minus R. C2 A-noemb minus A.
  C3 A unwidened minus A widened (scoring only). C4 A minus S0M, widened. Seeds 0 to 4 for every
  fitted arm. Before any contrast is read the scorer prints the minimum detectable effect: the
  standard deviation over folds of the difference in the signed-MW six CRPS between the mean of R
  seeds 0 to 2 and the mean of R seeds 3 and 4, divided by the square root of two, as in
  experiments 419 and 423 (a lower bound). A contrast smaller than that is "inside the noise". The
  assembly adds a pretraining stage to the pipeline, so the smallest gain worth porting is 0.03
  points at the signed-MW six (experiment 418's threshold), and the same 0.03 is applied at the
  MVA-only eight. C1 supports the hypothesis if its interval is wholly below -0.03 at the
  signed-MW six; it is partly supported if only the eight meets that bar or the point estimate
  passes with an interval reaching zero (experiment 418's six-Primary interval ended at -0.030, so
  this is close to a coin flip); not supported otherwise. C2 and C3 are checks: if C2's interval
  reaches zero or C3 reverses sign, the page reports an audit trigger and does not claim the
  component helps. No adoption is sought.
benchmark: >-
  R, the ENS-only pooled plain network refitted here under the standing D15 and D18 cut; S0M (the
  monthly XGBoost recipe, experiment 389's stored arm S0M read through experiment 397's references,
  with its cut status and widening
  stated on the page), BG2, `manual_heuristic`, and `climatology` as absolute-skill references,
  scored on the half-hours they share.
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

**The assembly is experiment 418's P1 arm plus the standing practices, so the new work is a refit
under the D15 and D18 cut, five seeds, one fitted leave-one-out arm, and the reference table.**
Experiments 397 to 423 tested one component at a time on the same pooled network. Only
ERA5-and-CAMS pretraining sits on top of the plain reference; the other components are already in
it.

| Component | In A? | Supporting experiment and claimed gain |
|---|---|---|
| Pooled network with 8-wide Primary embedding | yes | 403: -0.185 and -0.161 against per-Primary networks; 406: no embedding costs +0.789 and +0.809; 421 (open PR) also bears on it |
| ERA5-and-CAMS pretraining, fine-tune all weights | yes (P1) | 418: -0.061 at the six, -0.156 at the eight, 12 monthly folds, unwidened |
| 1.15 widening about the median | yes | 407: q95 coverage 0.854 to 0.891 at the signed-MW six; the CRPS effect is unmeasured |
| Ensemble mean and spread inputs, applied once | yes | 397: one-pass summary scheme; the pooled summary network scored 3.264 against 3.542 for XGBoost on five screen months |
| Hand-made calendar columns | yes | 423: the encoder arms were 0.27 and 0.37 worse |
| Monthly refits, three origin lags | yes | standing practice; 397 |
| Solar summary | excluded | 419: not supported (-0.012) |
| Quantile head | excluded | 407: q99 pinball -15.8% but CRPS +0.066 |
| Export gate for signed MVA output | excluded | 420: gain is on reverse-flow half-hours only; 424 has not merged |
| E1_es2 inputs, B-mix, lag trust head, varied origin hour | excluded | 396, 404, 422 have not merged |
| Explicit solar term, shared calendar encoder, shapes network, capacity prior | excluded | 409, 416, 423, 413, 412 failed |

**Arms.** R and A (pretrain, then fine-tune all weights) as in experiment 418's `ens-only` and `P1`
arms, refitted at 12 monthly folds with seeds 0 to 4, because 418's stored forecasts lack the
standing D15 and D18 cut and are unwidened. A-noemb is A built with experiment 406's `pooled_noemb`
network in both the pretraining and the fine-tuning stage (new pretraining code, since 418's saved
networks carry the embedding). The widening contrast C3 is a scoring-only comparison. Pretraining
is the third leave-one-out component and is R itself. 418's stored forecasts serve as a
cross-check: the plan reports R and A without the cut beside 418's scores only if seed 0 of a
fold is also fitted without the cut (3 fits), to size the cut effect.

**Compute.** 300 fits plus 3 cross-checks. R 60; A 60 pretrain and 60 fine-tune; A-noemb 60 and
60. Experiment 418 timed a pretrain at 30 to 63 s and a fine-tune at 27 to 55 s, so about 1 minute
each, 5 to 8 hours on one GPU process under 8 GB with one feeder core (`MemoryMax=8G`,
`OMP_NUM_THREADS=1`). GPU is used because 418 ran it there at under 3 GB. Stop line: 12 GB
available in `free -g`. Stored predictions from 418 save no fits here, because the recipe differs.

**Scoring and reporting.** The references are scored on the half-hours all arms share, and the
page states that row count and the flagged half-hours dropped per fold. Reported per group and per
Primary: CRPS, q01, q05, q95, q99 coverage and pinball loss, cold winter nights, sunny half-hours,
and the avoidable-curtailment cost. The headline chart is CRPS by lead for A, R, A-noemb, BG2, S0M,
`manual_heuristic`, and `climatology`. The page also carries a day-ahead and a 14-day example
forecast. Fitted networks are saved under the experiment's data folder and the path is stated.

**Standing practices followed.** Monthly refits; cleaned data; D15 and D18 cut; saved networks;
tails and cost beside CRPS; example forecasts. Earlier advice followed: at least five seeds and a
printed minimum detectable effect (419); references scored on shared half-hours (418). No
load-shift claims.

**What changed against the brief and after the plan review.** The brief's list assumed more
components than have merged evidence, so 396, 404, 422, and 424 enter no arm. The review showed that
418's stored forecasts cannot be reused (different D15 and D18 cut, no widening), that the embedding
cost was measured by 406 (+0.789 and +0.809), and that 'monthly XGBoost' is arm S0M of 397. The
solar-summary arm is dropped because 419 did not support it and it needs new pretraining columns.

## Review log

- **Plan review (Sonnet).** Accepted: refit all arms under the cut (418's forecasts lack it and the
  widening); cite 406's embedding cost; S0M source; decision-rule wording; drop the solar arm.
- **Code review (Sonnet).** No leakage or cut defect. Accepted: log writes made atomic, `--no-cut`
  needs explicit months and seeds, C2 trigger also fires when its interval reaches zero. Stated
  limits for the write-up: flagged zeros among the origin lags still reach the network at inference
  (as in experiment 407); contrast intervals pool half-hours while tables average over Primaries.
