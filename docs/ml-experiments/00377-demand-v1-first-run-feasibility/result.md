---
id: "00377"
slug: demand-v1-first-run-feasibility
title: >-
  How does Primary demand research default 1, extended to 240 hours, score against BG2, the manual
  heuristic, and climatology at all 16 Primaries under the demand-v1 protocol?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v1
size: >-
  medium. Stored: forecasts and small result files in this experiment's private folder only;
  experiments 87, 111, 152, 374, and 376 are read and never written. Production serving path: not
  touched. Degradation rule: not touched. Designs: more than one would test the hypothesis (a
  pooled spread across Primaries, or one per Primary); the demand-v1 protocol fixes the per-Primary
  design. Callers: the scripts are copies of experiment 376's, whose callers are all in that
  folder, and `studies.eval_protocol`, whose callers are the protocol's experiments.
hypothesis: >-
  Primary demand research default 1, extended to 240 hours (the 376 stack, with its stage 2 refitted
  for every scored month on earlier months only and a fifth lead band for 168 to 240 hours), is
  not worse than BG2 by more than 0.20 points of effective capacity at days 1 to 10 at the six
  signed-MW Primaries, and at the eight MVA-only Primaries the gap to BG2 is a D8 gap. The
  forecast is NOT Primary demand research default 3, because it lacks the two-number solar summary
  and the daylight heuristic blend at 168 to 240 hours.
prediction: >-
  Written before any score is read. P1. Signed-MW six, default 1 extended minus BG2 on the shared
  rows, days 1 to 10, pooled: between -0.20 and +0.20 points. P2. Eight MVA-only Primaries, days
  1 to 10, pooled with D8: between +0.30 and +0.90 points; without D8: between -0.05 and +0.25.
  P3. Days 8 to 10 alone trail BG2 by more than days 1 to 10 do, in the signed-MW group, because
  stage 2's fifth band is fitted on far fewer rows than the first four. P4. The stack beats
  climatology at every scored Primary at days 1 to 10. P5. Coverage at q05 is above 0.08 at the
  signed-MW six (the lower tail is too narrow, as in experiment 376).
decision_rule: >-
  No adoption: this is the first run under the protocol, so the verdicts are descriptive. Contrasts,
  all registered before any score is read, all CRPS in points of effective capacity on the shared
  rows, by the paired month-resampled bootstrap of `studies.eval_protocol`, 99% interval with
  20,000 resamples as the headline and 95% interval with 2,000 resamples per Primary. C1: default
  1 extended minus BG2, days 1 to 10, signed-MW six pooled. C2: the same, MVA-only eight pooled,
  with and without D8. C3: the same at D18 (report only, not counted in a tally). C4: default 1
  extended minus climatology and minus the manual heuristic, per group. Per-Primary contrasts and
  leave-one-out contrasts are reported beside C1 and C2. One verdict per group, never one for all
  16. No Primaries-won tally counts D15 or D18.
benchmark: >-
  BG2, climatology, and `manual_heuristic` as stored by experiment 87 on `fold_a_daily_gap_v1`
  (the 292 daily runs of 2025-07 to 2026-06), which already hold leads up to 351 hours. Production
  does not run BG2.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles (the quantile score), mean
  absolute error of the median, and twCRPS, in points of effective capacity, from
  `studies.eval_protocol`; leads 0.5 to 240 hours.
code: ml-experiments/00377-demand-v1-first-run-feasibility/
---

# Primary demand research default 1, extended to 240 hours, under the demand-v1 protocol (plan)

This page is the plan. It is replaced by the write-up when the scores exist.
