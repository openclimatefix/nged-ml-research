---
id: "00372"
slug: mva-fold-rule-chosen-fit-at-all-primaries
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
size: >-
  simple. Stored: small result files in the private folder only. Production serving path: not
  touched. Degradation rule: not touched. Designs: more than one would test the hypothesis (a
  re-run of experiment 152's engine to get tuning forecasts for stage 2, or the stored fits as
  they are); the stored fits are chosen because re-running the engine needs hours of compute and a
  maintainer decision. Callers: reuses the scoring helpers of experiment 252, whose callers are all
  in that folder.
hypothesis: >-
  At the eight MVA-only Primaries, the fit that the rule of experiment 366 chooses (the fit through
  the meter at Stickney, D12, and the naive fit that reads the meter reading as signed power at the
  other seven) scores better than the naive fit at D12 on sunny daylight half-hours, and scores no
  better than BG2, the production-style forecast, at most of the eight.
prediction: >-
  P1. At D12, the fit through the meter minus the naive fit on sunny daylight half-hours is between
  -1.0 and 0.0 points of effective capacity (experiment 152 found about -0.4 for a variant of the
  fit through the meter with folded quantiles), with a month-resampled 95% interval wholly below
  zero. P2. At the seven Primaries where the rule chooses the naive fit, the chosen fit equals the
  naive fit by construction, so their contrast is exactly zero and is not evidence. P3. The chosen
  fit scores worse than BG2 over all rows at 6 or more of the 8 Primaries, because experiment 252
  found that raw fits without stage-2 corrections trail BG2 at most MVA-only Primaries; this can
  fail. P4. The two Primaries without a stack fit (D15 and D18) are scored only against the three
  references, so no prediction applies.
decision_rule: >-
  P1 bears the verdict: `supported` when the interval at D12 lies wholly below zero, `not
  supported` when it includes zero or lies above zero. The threshold on the gain is zero, because
  switching to a stored fit at one Primary adds no pipeline step. P3 and P4 are descriptive. Control
  gate: the three references rescored here must reproduce the scores that experiment 252 prints
  for the same Primaries and rows. All results are for zero reactive power, and the rule fails at
  power factor 0.97 (experiments 358 and 366). This experiment is close to a repeat of experiment
  152, applies no stage-2 correction, and makes no claim against experiment 252's arm A1.
benchmark: >-
  The naive fit (arm N of experiment 152); BG2, climatology, and manual_heuristic as stored by
  experiment 87
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity,
  scored year 2025-07-02 to 2026-06-30, leads 0.5 to 168 hours, rows outside the hand-filtered
  events; slices sunny daylight, winter nights, cold nights
verdict: ""
production_status: not-for-production
adopted_in: none
date: 2026-10-28
---

# MVA fold: the rule-chosen fit at all 16 Primaries

**Planned. Written before any score of this experiment is read.**

This is step 4 of the MVA sequence, in its cheap form. Re-running experiment 152's ensemble engine
into a new folder, to get the tuning forecasts that experiment 252's stage-2 corrections need, is
the follow-up the maintainer may choose.

## Plan

- **Eight MVA-only Primaries.** Score the stored forecasts of experiment 152: arm N (the naive
  fit) and arm A (the fit through the meter). The chosen fit is arm A at D12, where the rule of
  [experiment 366](../00366-mva-fold-irradiance-correlation-reverse-flow-rule/result.md) picks the
  fit through the meter on the training rows before 2025-07-01, and arm N at the other seven.
- **All 16 Primaries.** Score BG2, climatology, and the manual heuristic on the rows they share.
  The six signed-MW Primaries have no stored stack scores on these rows, so the stack is not scored
  there. The stack has never run at D15 or D18, whose usable histories are about 0.8 and 2.6 months.
- **Hand-filtered event lists.** The number of masked half-hours of the scored year is reported for
  every Primary, so the Primaries without a list are visible.
- **Reactive power.** Every result holds for reactive power zero.
