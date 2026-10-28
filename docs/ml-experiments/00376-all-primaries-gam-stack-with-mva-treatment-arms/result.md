---
id: "00376"
slug: all-primaries-gam-stack-with-mva-treatment-arms
theme: production-reference-and-best-guess
also_relevant_to:
  - differentiable-physics-embedded-generation
status: planned
kind: other
size: >-
  medium. Stored: forecasts, fits, and small result files in this experiment's private folder only;
  experiments 111, 152, 252, and 374 are read and never written. Production serving path: not
  touched. Degradation rule: not touched. Designs: more than one would test the hypothesis (a
  pooled stack across Primaries, or one stack per Primary); the brief fixes the per-Primary stack
  because a pooled design needs new machinery. Callers: reuses the scripts of experiments 252 and
  374, whose callers are all in those folders.
hypothesis: >-
  One recipe, experiment 252's stack (stage 1 plus stage 2) with the MVA treatment chosen by
  experiment 366's rule, scores as well as BG2 at the 6 Primaries that report signed megawatts, and
  at the 8 Primaries that report only apparent power the gap to BG2 comes from the MVA treatment
  and from Marsh Lane (D8), not from the architecture. Every result holds for reactive power zero.
prediction: >-
  Written before any score is read. P1. At the 6 signed-MW Primaries (D5, D9, D10, D11, D14, D16),
  stack minus BG2 over all half-hours, pooled, lies between -0.20 and +0.20 points of effective
  capacity. P2. At the 8 MVA-only Primaries, the chosen fit minus the naive fit is between -0.10 and
  +0.05 pooled (experiment 374: the fit through the meter scores 0.06 better pooled, almost all of it
  at D12 and D8). P3. At the 8 MVA-only Primaries the gap to BG2 is +0.53 with D8 and +0.07 without
  D8, as in experiment 374. P4. The stack's q05 coverage is above 0.08 at the 6 signed-MW Primaries
  (the lower tail is too narrow, as at the 8 MVA-only Primaries).
decision_rule: >-
  Per group, by the paired month-resampled bootstrap of experiment 374 (20,000 resamples, 95%).
  Signed-MW group: `supported` when the upper end of the interval of stack minus BG2 over all
  half-hours, pooled over the six, is below +0.10 points; `not supported` when the lower end is at
  or above +0.10 or the whole interval lies above zero by more than 0.10; otherwise
  `inconclusive`. MVA-only group: the contrast of the chosen fit minus the naive fit is
  descriptive, and the gap to BG2 is reported with and without D8. The threshold of +0.10 is zero
  cost to port, because the stack already exists; it follows experiment 374's non-inferiority
  margin. D15 and D18 are scored against references only if the stack cannot be fitted. One verdict
  per group, never one for all 16.
benchmark: >-
  BG2, climatology, and manual_heuristic as stored by experiment 87; experiment 374's chosen fit
  and A1 at the MVA-only Primaries
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity,
  scored year 2025-07-02 to 2026-06-30, leads 0.5 to 168 hours, rows outside the hand-filtered
  events; slices sunny daylight, winter nights, cold nights, Economy 7 step; tail coverage at q01,
  q05, q95, q99
code: ml-experiments/00376-all-primaries-gam-stack-with-mva-treatment-arms/
date: 2026-10-28
---

# One GAM stack at all 16 Primaries, with the MVA treatment as an explicit arm

Planned. The plan is in the front matter; the write-up replaces this page when the run finishes.

- **Arms.** Experiment 252's stack with stage 2, unchanged, at every Primary where it can be fitted.
  At the 8 MVA-only Primaries: the chosen fit and the naive fit of experiment 374, read from its work
  folder. At the 6 signed-MW Primaries: experiment 252's `control` chain (D10 and D11 exist; D5, D9,
  D14, and D16 are fitted here).
- **Reactive power zero** is stated in the headline.
- **Past-weather caveat.** The stack uses the calibrated ensemble mean at forecast time, so the
  weather is available at forecast time.
- **Schedule.** Monthly refits on 24 months, as in experiments 252 and 374, not weekly production
  refits. The contrast between groups does not depend on this.
