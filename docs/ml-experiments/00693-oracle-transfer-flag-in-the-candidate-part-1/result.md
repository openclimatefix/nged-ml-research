---
id: "00693"
slug: oracle-transfer-flag-in-the-candidate-part-1
title: >-
  Would a perfect, causal flag for load transfers improve the Candidate demand forecast at the 16
  Primaries?
theme: switching-labels-in-the-forecast
also_relevant_to:
  - lsc-detectors
  - lagged-power-and-error-correction
status: planned
kind: other
size: >-
  Anything else. (1) It stores fitted networks and prediction files per arm, seed, and fold; (2)
  more than one design would defensibly test the hypothesis (flag in the lags only, in the
  prevailing head only, or both), so the arms are fixed in the plan; (3) it imports code from
  experiments 613, 647, 650, 660, 667, 677, and 681; (4) it touches no serving path or degradation
  rule; (5) it changes no contract.
hypothesis: >-
  A perfect telemetry detector, which knows that a reading lay inside a load transfer (and when the
  transfer began, once a reading after the onset exists), lowers the CRPS of the Candidate demand
  network on transfer-affected rows by at least 0.3 points of capacity against a placebo flag,
  without raising CRPS on event-free rows by more than 0.05.
prediction: >-
  Written before any score. No-retrain-arm (NRA) output: the gain over the placebo is under 0.3
  points, because the Candidate already ignores most of a shifted lag. Prevailing output: the gain
  is 0.3 to 1.0 points inside a transfer, because the R-haz head otherwise infers the onset from
  residuals and is held at p = 0 until its heuristic age passes 0.5 days. Real-event slices hold
  about two verified episodes before July 2026 and are descriptive.
decision_rule: >-
  Smallest gain worth porting: 0.3 points of CRPS on affected rows, because a detector adds a
  pipeline step, a model to maintain, and a false-alarm failure mode (the best real detector of
  experiment 208 recovered 53% of the perfect-knowledge gain). All bars apply to the point estimate,
  and the Bonferroni interval (five tests: C1, C2, C3, C4-NRA, C4-prevailing) must exclude zero in
  the stated direction. Supported - C1, C2, or C3 gains at least 0.3 points; both C4 point estimates
  rise by no more than 0.05; and C5 (placebo equivalence, A2 minus A0, 90% interval inside plus or
  minus 0.05) passes. Small gain - a C1 to C3 gain whose interval excludes zero but whose point
  estimate is under 0.3, with C4 and C5 as above. Not supported - no C1 to C3 interval excludes zero
  in the predicted direction. Veto - C4 above 0.05, or C5 failing, makes the verdict inconclusive
  pending an audit. A null is read as "no detector is worth building" only if the ceilings are
  computed (A0-clean on bank L, oracle B-haz on bank F) and the oracle age is shown to lift the age
  floor (share of P1 rows with p > 0, per arm). Stage 0 gate - if CRPS(A0) minus CRPS(A0-clean) on
  bank L's affected rows is under 0.3 points, the NRA retraining is not run.
benchmark: >-
  Experiment 677's quarterly step-8 Candidate (arm A0, near 7.99 MAE and 4.57 CRPS), experiment 667's
  monthly file (7.53 MAE) for scale, ClimatologyForecaster, and NGED's manual heuristic.
row_set_id: oracle693_contrasts_per_row_set
metric: >-
  CRPS (twice the mean pinball loss over 11 quantiles) in points of capacity, normalised MAE of the
  median, q05-to-q95 coverage, q01, q05, q95, and q99 coverage and pinball loss, cold winter nights,
  avoidable-curtailment cost, copied share of an injected lag-only shift, reaction time.
---

# Would a perfect flag for load transfers improve the Candidate demand forecast?

**Status: planned. The design is fixed in the brief at
`data/ml-experiments/briefs/oracle_switching_flag_design.md` (design v2, reviewed by an adversarial
Opus reviewer and revised). No code has run.**

We ask whether telling the Candidate demand neural network exactly which of its power lags lay
inside a load transfer would lower its forecast error. The answer bounds what a real switching
detector could add to the forecast. We compare a network given the true flags against the same
network given placebo flags, on injected transfers and on real rows, with quarterly refits.

## Plan

- **Arms.** A0 (default: experiment 677's step-8 Candidate and the adopted R-haz head), A1 (default
  plus oracle flags), A2 (placebo flags with matched span lengths, overlaps, and per-Primary
  counts). Stage 0 adds the inference-only ceilings A0-mask, A0-clean, and oracle B-haz.
- **Folds.** Quarterly refits at the maintainer's request (cut-offs 2025-11-01, 2026-02-01,
  2026-05-01), scored on valid times 2025-11-01 to 2026-06-30 only. A one-fold screen (fold 2,
  cut-off 2026-02-01) comes first. Seeds 0 to 2.
- **Deciding banks.** Bank L (no-retrain output) and bank F (prevailing output), both from the
  experiment 660 generator; check banks are experiment 581's generator and experiment 650's
  held-out passes.
- **Contrasts.** C1 to C5 as in the decision rule. Also reported: A1 minus A0-mask, A0 minus
  A0-clean, oracle B-haz minus B-haz, and every contrast on the check banks.
- **Standing practices followed:** cleaned data, D15 and D18 cut before 2025-03-01, every network
  saved, stored predictions reused, origin-time weather only (asserted per row). **Departures:**
  quarterly instead of monthly refits (the maintainer's request; experiment 677's advice favours
  monthly, and any gain here applies to quarterly refits), and a one-fold screen instead of four
  seasonal months.
- **Earlier advice.** Follows experiment 677's advice to judge on copied share and event-free error
  together; departs from its advice to refit monthly.

## Method

To be written with the results.

## Files

To be written with the results.
