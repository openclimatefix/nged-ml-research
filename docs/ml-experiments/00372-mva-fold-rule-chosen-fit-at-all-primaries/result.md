---
id: "00372"
slug: mva-fold-rule-chosen-fit-at-all-primaries
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
size: >-
  medium. Stored: small result files in the private folder only. Production serving path: not
  touched. Degradation rule: not touched. Designs: more than one would test the hypothesis (a full
  refit of the stack with the rule's fit, or the stored fits of experiment 152 plus the stage-2
  corrections of experiment 252); the stored fits plus stage 2 are chosen because experiment 353
  already runs that pipeline at D8 and it needs no new fit. Callers: reuses the scripts of
  experiments 252 and 353, whose callers are all in those folders.
hypothesis: >-
  At the eight MVA-only Primaries, a forecast that uses the fit the rule of experiment 366 chooses
  (the fit through the meter at Stickney, D12, and the naive fit that reads the meter reading as
  signed power at the other seven), followed by the stage-2 corrections of experiment 252, scores
  no worse than experiment 252's arm A1 pooled over the eight Primaries, and scores better than the
  naive fit at D12.
prediction: >-
  C1 (chosen minus A1, quantile score over all rows, pooled over the eight Primaries) lies between
  -0.30 and +0.30 points, because experiment 353 found the fold changes D8's score by -0.23 points
  and experiment 252's A1 differs from the stored fits in having a fitted solar term and clip. C2
  (D12, chosen minus naive fit, sunny daylight half-hours) is at most -0.30 points, in line with
  experiment 152's gain of about 0.4. C3 (the chosen arm minus A1 at the seven naive-fit
  Primaries) is positive at 4 or more of 7, because experiment 252 found that dropping the fitted
  solar term costs 0.19 to 0.34 points; this is the prediction most likely to fail.
decision_rule: >-
  C1 bears the verdict. Paired month-resampled bootstrap, 20,000 resamples, 95% interval, as in
  experiment 353. `supported` when the interval lies wholly below zero and the point estimate is at
  most -0.10 points (a stored meter fit is existing reviewed code, so the threshold covers only the
  added choice between two fits); `small gain` when wholly below zero but above -0.10; `not
  supported` when the interval includes zero and C2 is not at most -0.30. C2 and C3 are descriptive.
  Control gate: rescoring A1 and the three references must reproduce experiment 252's stored
  scores. Reactive power zero only; the rule fails at power factor 0.97 (experiments 358, 366).
benchmark: >-
  Experiment 252's arm A1; BG2, climatology, and manual_heuristic as stored by experiment 87
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity,
  scored year 2025-07-02 to 2026-06-30, leads 0.5 to 168 hours, rows outside the hand-filtered
  events; slices sunny daylight, winter nights, cold nights; tail coverage at q01, q05, q95, q99
verdict: ""
production_status: not-for-production
adopted_in: none
date: 2026-10-28
---

# MVA fold: the rule-chosen fit at all 16 Primaries

**Planned. Written before any score of this experiment is read.**

This is step 4 of the MVA sequence. The 16 Primaries split into six with signed-MW readings, eight
with MVA-only readings, and two (D15 and D18) with short histories.

## Plan

- **Eight MVA-only Primaries.** For each, take the fit the rule of
  [experiment 366](../00366-mva-fold-irradiance-correlation-reverse-flow-rule/result.md) chooses
  (L0 at D12, N elsewhere) from the stored fits of experiment 152, apply the stage-2 corrections of
  experiment 252 as experiment 353 did at D8, and score it against A1 and the references.
- **Six signed-MW Primaries.** Not refitted: they need no fold. Their stack scores come from the
  research-default log, quoted and not recomputed; BG2, climatology, and the manual heuristic are
  rescored on the same rows.
- **D15 and D18.** The stack has never run at either, so only BG2, climatology, and the manual
  heuristic are scored. D15 (about 0.8 months of usable history) and D18 (about 2.6 months) have
  short histories; running the stack there would need new fits and is not done here.
- **Hand-filtered event lists.** The number of masked half-hours of the scored year is reported for
  every Primary, so the Primaries without a list are visible.
- **Reactive power.** All results hold for zero reactive power. Real MVA-only meters have unknown
  reactive power, and the rule fails at power factor 0.97.
