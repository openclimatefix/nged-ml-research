---
id: "00376"
slug: all-primaries-gam-stack-with-mva-treatment-arms
theme: production-reference-and-best-guess
also_relevant_to:
  - differentiable-physics-embedded-generation
status: finished
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
verdict: inconclusive
production_status: not-for-production
adopted_in: none
best_arm: >-
  signed-MW group: stack minus BG2 is -0.02 points over all half-hours (95% interval -0.17 to +0.13)
headline: >-
  Verdicts per group, for reactive power zero. Six signed-MW Primaries: inconclusive (stack minus
  BG2 -0.02, -0.17 to +0.13; the registered bar was an upper end below +0.10). Eight MVA-only
  Primaries: descriptive; the rule-chosen fit trails BG2 by +0.53 (+0.35 to +0.73), +0.07 (-0.03 to
  +0.17) without D8, and beats the naive fit by 0.03 (-0.08 to +0.01), all of it at D12. D18: stack
  level with BG2 (+0.04, -0.60 to +1.01). D15: references only.
take_home_message: >-
  One recipe gives the same average score as BG2 at the six Primaries that report signed megawatts,
  but one year of data cannot rule out a gap of up to 0.13 points, and the stack is worse than BG2
  on sunny half-hours and better on cold nights. At the eight Primaries that report only apparent
  power, the gap to BG2 is a Marsh Lane (D8) gap, and the rule that picks the fit adds only 0.03
  points over the naive fit, all of it at Stickney (D12). This holds for reactive power zero only.
lessons:
  - >-
    The architecture is not what separates the MVA-only Primaries from the signed-MW ones: without
    D8 the MVA-only gap to BG2 (+0.07) is about the size of the signed-MW interval's upper end.
  - >-
    The stack's lower tail is too narrow at both groups (q05 coverage 0.072 at the six signed-MW
    Primaries, 0.112 at the eight MVA-only ones, against 0.041 and 0.040 for BG2).
advice:
  - >-
    Fix the stack's lower tail (q05 coverage 0.07 to 0.11 against 0.04 for BG2) before comparing
    its tail scores with BG2's (Tail calibration bullet).
  - >-
    Treat the sunny-half-hour gap to BG2 (+0.46 at the six signed-MW Primaries) as the stack's
    common weakness, not an MVA effect (Results).
  - >-
    Keep reading the MVA rule's benefit as a D12-only gain of 0.27 points until another Primary
    with reverse flow is available (c3).
date: 2026-10-28
---

# One GAM stack at all 16 Primaries, with the MVA treatment as an explicit arm

**One stack, run unchanged, scores level with BG2 on average at the six Primaries that report signed
megawatts, but one year of data cannot rule out a gap of 0.13 points, and the stack is worse than
BG2 on sunny half-hours.** BG2 is the production-style forecast. Stack minus BG2 over all
half-hours, pooled over the six, is -0.02 points of effective capacity (percent of a Primary's
capacity; 95% interval -0.17 to +0.13). The registered bar was an upper end below +0.10, so the
verdict for that group is inconclusive. The sunny-daylight gap is +0.46 (+0.20 to +0.70). Winter and
cold nights go the other way (-0.29 and -0.44). Every result holds for reactive power zero.

We asked whether the best-guess stack of [experiment
252](../00252-gam-stack-at-eight-more-primaries/result.md), with stage 2, scores as well as BG2 at
every Primary, and how much of the gap at the eight Primaries that report only apparent power
(megavolt-amperes, MVA) comes from the MVA treatment. The six signed-MW Primaries are D5, D9, D10,
D11, D14, and D16. We ran the unchanged chain of experiment 252 there (its `control` source), took
the eight MVA-only Primaries from [experiment
374](../00374-mva-fold-stage-two-at-eight-primaries/result.md), and scored all three groups with the
rows, metric, and bootstrap of experiment 374.

**At the eight MVA-only Primaries the gap to BG2 is a Marsh Lane (D8) gap, and the rule adds 0.03
points.** The rule-chosen fit (the fit through the meter at Stickney, D12, and the naive fit
elsewhere) minus BG2 is +0.53 (+0.35 to +0.73), and +0.07 (-0.03 to +0.17) without D8. The naive
fit at all eight minus BG2 is +0.57 (+0.40 to +0.75). The rule-chosen fit minus the naive fit is
-0.03 (-0.08 to +0.01), and it is exactly zero at seven Primaries. At D12 it is -0.27 (-0.61 to
+0.04). The rule changes the forecast at D12 only.

![Figure 1: stack minus BG2 by Primary, 95% interval](figure_1_minus_bg2_by_primary.svg)

![Figure 2: quantile score by Primary](figure_2_score_by_primary.svg)

![Figure 3: tail coverage at q05 and q95](figure_3_tail_coverage.svg)

## Results (quantile score in points of effective capacity; lower is better)

- **Signed-MW group (D5, D9, D10, D11, D14, D16): inconclusive.** Stack minus BG2 is -0.02 (-0.17
  to +0.13). Per Primary: D5 +0.14, D9 -0.02, D10 -0.05, D11 -0.15, D14 -0.15, D16 +0.13; every
  interval includes zero. Stack minus the manual heuristic is -0.70 (-0.96 to -0.44), and minus
  climatology -0.97 (-1.31 to -0.65).
- **Slices, stack minus BG2, signed-MW group.** Sunny daylight +0.46 (+0.20 to +0.70); winter
  nights -0.17 (-0.30 to +0.03); cold nights -0.44 (-0.95 to -0.00); Economy 7 step +0.10 (-0.02 to
  +0.22). Registered prediction P1 (between -0.20 and +0.20) holds.
- **Absolute scores, all half-hours (stack, BG2, manual heuristic, climatology).** D5 3.52, 3.38,
  3.90, 4.26; D9 2.38, 2.40, 2.91, 3.00; D10 3.22, 3.28, 4.13, 4.31; D11 2.35, 2.51, 3.39, 3.55;
  D14 2.09, 2.25, 2.95, 3.45; D16 2.67, 2.55, 3.15, 3.49.
- **MVA-only group (D2, D3, D4, D8, D12, D19, D21, D22): descriptive.** Chosen fit with stage 2
  minus BG2 +0.53 (+0.35 to +0.73); naive fit with stage 2 minus BG2 +0.57 (+0.40 to +0.75); A1
  minus BG2 +0.56 (+0.39 to +0.75). By Primary, chosen minus BG2: D8 +3.77 (+2.42 to +5.27), D12
  +0.65, D22 +0.37, and -0.07 to -0.16 at D2, D3, D4, D19, and D21. Without D8: chosen +0.07
  (-0.03 to +0.17), naive +0.11 (-0.00 to +0.22). P2 and P3 hold.
- **D8 on its own.** Chosen and naive fits score 11.37, BG2 7.59, climatology 9.76, manual
  heuristic 8.47. D8 is the only Primary where the stack trails climatology.
- **Slices, chosen minus BG2, MVA-only group.** Sunny daylight +0.65 (+0.41 to +0.90); winter nights
  +0.28 (+0.16 to +0.39); cold nights -0.12 (-0.26 to +0.13); Economy 7 step +0.71 (+0.29 to
  +1.26).
- **D18 (third group).** The stack fits with 12 monthly fits and borrows its spread from the six
  signed-MW Primaries. Stack 2.81, BG2 2.78, manual heuristic 4.09, climatology 12.13; stack minus
  BG2 +0.04 (-0.60 to +1.01) on 61,519 rows, one Primary and few months, so the interval is wide.
- **D15 (third group).** No ensemble store exists and fetching one was out of scope, so only
  references are scored: BG2 11.27, manual heuristic 17.02, climatology 35.14 (experiment 372).
- **Pooled training.** Not run: it needs machinery that does not exist (the brief allowed it only
  if the plan found no new machinery).
- **Tail calibration** (share of readings at or below each quantile, mean over the Primaries of
  the group; ideal 0.01, 0.05, 0.95, 0.99; pinball loss not computed). Signed-MW stack: 0.024,
  0.072, 0.946, 0.982; BG2: 0.010, 0.041, 0.944, 0.983. MVA-only chosen fit: 0.054, 0.112, 0.916,
  0.964; BG2: 0.005, 0.040, 0.957, 0.995. P4 (signed-MW q05 coverage above 0.08) fails: the mean is
  0.072, and D16 alone is above 0.08 (0.105). The stack's lower tail is still wider than BG2's at
  five of six Primaries (D10 equal), and sunny-half-hour q05 coverage is 0.157 against 0.104.

## Checks

- **Control gate.** This experiment's stage-2 arm d equals experiment 155's stored arm d to a
  maximum difference of 0.0 in q50 at all six signed-MW Primaries. The MVA-only numbers reproduce
  experiment 374's (+0.53, -0.03, +0.07 without D8).
- **Median against mean forecast.** 0.00 at every Primary for every arm.
- **Mean-forecast error beside the contrasts.** Error of the mean forecast against BG2's median, in
  megawatts: D5 6.01 against 5.88; D9 4.21 against 4.33; D10 5.31 against 5.62; D11 4.21 against
  4.48; D14 3.69 against 3.96; D16 4.48 against 4.45. The direction agrees with the quantile score
  at all six Primaries (at D16 the gap is 0.03 MW).
- **Leave one Primary out (signed-MW).** The stack-minus-BG2 interval has an upper end of +0.09
  without D5 and +0.11 to +0.17 without any other Primary. The point estimate flips sign without
  D11 or D14 (+0.01), so trigger 4 fired for the sign. The verdict is inconclusive in all six
  leave-one-out runs, except that leaving out D5 would meet the +0.10 bar.
- **Triggers.** Trigger 4 fired as described. Triggers 1 (the gate passed), 3, 5 (no nested arm in
  the signed-MW group; D8's stage-2 worsening is experiment 374's finding), 6, and 7 did not fire.
  Trigger 2: P4 contradicts its prediction, and the explanation is untested. No Opus audit ran,
  because the brief allows Sonnet only and the verdict does not depend on trigger 4: the
  registered rule uses the upper end of the pooled interval, which is above +0.10 with every
  Primary except D5.

## Caveats and what to try next

- Reactive power zero only; experiment 366's rule fails at power factor 0.97. For D8, it is
  unproven whether the generator explains the steps and the gap.
- One scored year and twelve month-blocks give wide intervals. Monthly refits on 24 months were
  used, not weekly production refits.
- The signed-MW stack pins the solar capacity to the stored capacity vintages, and the MVA-only
  stack has no capacity vintage (experiment 252), so the two groups do not share every design
  choice.
- The Economy 7 slice is the four half-hours ending at 00:30, 01:00, 07:30, and 08:00 UK local
  time, defined here.
- Next: the stack's sunny-half-hour gap to BG2 (+0.46 at the six signed-MW Primaries) is the
  largest cross-group weakness. A narrower lower tail (q05 coverage 0.07 to 0.11) is the second.
