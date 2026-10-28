---
id: "00372"
slug: mva-fold-rule-chosen-fit-at-all-primaries
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - production-reference-and-best-guess
status: finished
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
verdict: not supported
production_status: not-for-production
adopted_in: none
date: 2026-10-28
best_arm: >-
  none: at Stickney (D12) the fit through the meter minus the naive fit is -0.57 points on sunny
  daylight half-hours (95% interval -2.42 to +1.31)
headline: >-
  At D12 the fit through the meter scores 0.57 points better than the naive fit on sunny daylight
  half-hours, but the interval (-2.42 to +1.31) includes zero. Chosen fits trail BG2 by 0.56 points
  over all half-hours, pooled over the 8 MVA-only Primaries (0.39 to 0.75), and most of that gap
  comes from Marsh Lane (D8) and Stickney (D12).
take_home_message: >-
  Using the fit through the meter at Stickney, the one MVA-only Primary where it is chosen, lowers
  the score by 0.57 points on sunny half-hours, but twelve months of data cannot tell that from
  zero. At five of the eight MVA-only Primaries the stored raw fits already score within 0.1 points
  of
  BG2 over all half-hours, so the large gaps sit at D8, D12, and D22, not across the board.
  Experiment 372 repeats most of experiment 152, and the stack with its stage-2 corrections was not
  rerun.
lessons:
  - >-
    The stored raw fits of experiment 152 match BG2 within 0.1 points over all half-hours at 5 of
    the 8 MVA-only Primaries, so the gaps to BG2 are at D8 (+3.36), D12 (+0.76), and D22 (+0.42).
  - >-
    One year of data gives the D12 contrast an interval 3.7 points wide on sunny half-hours, so a
    gain of 0.6 points needs more months or more Primaries with reverse flow to be resolved.
advice:
  - >-
    Look for the gap to BG2 at D8, D12, and D22, the three Primaries where it is widest, before
    building a fix for all eight (figure 2).
  - >-
    Re-run experiment 152's engine for stage-2 corrections only if the maintainer asks, because the
    raw fits are already level with BG2 at 5 of the 8 Primaries (figure 2).
---

# MVA fold: the rule-chosen fit at all 16 Primaries

**At the Primary substation D12 (Stickney), the fit through the meter scores 0.57 points better
than the naive fit on sunny daylight half-hours, but twelve months of data cannot distinguish that
gain from zero.** A Primary is a substation that steps 33 kV down to 11 kV. Eight Primaries report
apparent power in megavolt-amperes (MVA), which carries no direction of flow, so reverse flow
(power flowing from the substation back into the grid) is hidden. The naive fit treats the meter
reading as signed power. The fit through the meter models the lost sign. We asked whether the fit
that the rule of
[experiment 366](../00366-mva-fold-irradiance-correlation-reverse-flow-rule/result.md) chooses
scores better than the naive fit. The rule picks the fit through the meter when the training
readings rise with irradiance or come close to zero, and it picks the fit through the meter at D12
only. Scores are quantile scores in points of effective capacity (percent of the Primary's
capacity). The scored year is 2025-07-02 to 2026-06-30. We scored the stored forecasts of
[experiment 152](../00152-dp-mva-fold-aware-gam-k-at-mva-only-primaries/result.md) over
2025-07-02 to 2026-06-30, leads 0.5 to 168 hours, beside BG2 (the production-style forecast),
climatology, and the manual heuristic at all 16 Primaries. Verdict: not supported, because the D12
interval includes zero.

**Experiment 372 is close to a repeat of experiment 152.** It applies no stage-2 correction and
makes no claim against the arm A1 of
[experiment 252](../00252-gam-stack-at-eight-more-primaries/result.md). Stage 2 is the correction
step fitted on tuning-period forecasts. Experiment 152 did not store
those forecasts. Applying stage 2 means re-running experiment 152's engine into a new folder.
Every result holds for reactive power zero. The rule fails at power factor 0.97 (experiments 358
and 366).

![Figure 1: quantile score of the chosen fit, BG2, the manual heuristic, and climatology at the 16 Primaries](figure_1_score_by_primary.svg)

![Figure 2: chosen fit minus BG2 at the 8 MVA-only Primaries, 95% interval](figure_2_chosen_minus_bg2.svg)

## Results (quantile score in points of effective capacity; lower is better)

- **P1 fails its interval.** At D12, the fit through the meter minus the naive fit is -0.57 on
  sunny daylight half-hours (-2.42 to +1.31) and -0.30 over all half-hours (-0.76 to +0.18). The
  point estimate is inside the predicted -1.0 to 0.0, and the interval includes zero.
- **P2 holds by construction.** At the seven other Primaries the chosen fit is the naive fit. The
  fit through the meter and the naive fit also score within 0.001 points of each other at D2, D3,
  D4, D19, D21, and D22, and differ only at D8 (-0.21 over all half-hours; interval -0.74 to +0.20) and D12.
- **P3 fails.** The chosen fit scores worse than BG2 over all half-hours at 4 of 8 Primaries: D8
  (+3.36; 2.19 to 4.65), D12 (+0.76; 0.40 to 1.15), D22 (+0.42; 0.17 to 0.68), and D19 (+0.07; -0.01
  to +0.15). It is level with or below BG2 at D2, D3, D4, and D21. Pooled over the eight it is +0.56
  (0.39 to 0.75), or +0.16 without D8.
- **Sunny daylight half-hours.** The chosen fit trails BG2 at all 8 Primaries on the point estimate
  (pooled +0.73; 0.48 to 1.01), by 0.01 to 1.70 points; the gaps at D19 (+0.01), D3 (+0.10), and D4
  (+0.28) have intervals that include zero.
- **Control gate passes.** The rescored BG2 scores match experiment 252's stored values (D12 6.47,
  D2 2.18).
- **D15 and D18.** The stack has never run at either. BG2 scores 11.27 at D15 (0.8 months of usable
  history, 53,525 rows) and 2.74 at D18 (61,463 rows) over all half-hours; climatology scores 35.14
  and 12.09, and the manual heuristic 17.02 and 4.04.
- **Hand-filtered half-hours in the scored year.** D5 2,107; D18 1,317; D15 1,124; D16 947; D3 456;
  D14 352; D12 310; D22 209; D10 117; D4 106; D19 106; none at D2, D8, D9, D11, and D21, which
  have either no event list or no event in the scored year.
- **Six signed-MW Primaries.** No stack scores exist on these rows. BG2 scores are in figure 1.

## Checks

- **Median against mean forecast.** This check does not apply, because only quantiles are scored.
- **Triggers.** Trigger 2 applies to P3, which failed against experiment 252's reading that raw
  fits trail BG2 at most MVA-only Primaries; the raw fits are level with BG2 (within 0.1 points) at
  five of eight, and
  the reason (the lead range and the stage-2 correction) was not tested. No other trigger fired.
  The pooled contrast without D8 stays positive (+0.16), so leaving out D8 does not flip its sign.
- **Tail calibration** (coverage and pinball loss at q01, q05, q95, q99) was not computed here;
  experiment 152 holds it for these fits.

## Caveats and what to try next

- One scored year and twelve month-blocks give wide intervals.
- No stage 2, so no comparison with experiment 252's arm A1 or with research default 3 is possible.
- The rule's choice rests on experiment 366's thresholds, one of which was set after seeing
  experiment 365's numbers.
- Re-running experiment 152's engine into a new folder would give stage-2 tuning forecasts at seven
  Primaries. It needs hours of compute and the maintainer's yes.
