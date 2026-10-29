---
id: "00386"
slug: bsp-gsp-demand-v1-top-down-bg2-gam-simple-xgboost
theme: hierarchy-and-pooling
also_relevant_to:
  - production-reference-and-best-guess
  - calibration-and-uncertainty
status: finished
kind: other
protocol: demand-v1
size: >-
  complex in scale, medium in risk. Stored: forecasts and small result files in this experiment's
  private folder only; experiments 87, 161, 377, 379, and 381 are read and never written.
  Production serving path: not touched. Degradation rule: not touched. Designs: more than one would
  test the question (fit the GAM stack on the BSP and GSP readings as stand-alone series, or reuse
  the bulk supply points' older chain); the plan fixes the first. Callers: scripts copy experiment
  377's stack and experiment 161's bulk-supply-point inputs, and call `studies.eval_protocol`.
hypothesis: >-
  At the four bulk and grid supply points that NGED forecasts, fitted and scored top-down on the
  aggregate series, the best Primary-demand GAM stack is not worse than BG2 (the weekly-refitted
  XGBoost model alone, `b_weekly`) by more than 0.20 points of effective capacity at days 1 to 10,
  separately for the two bulk supply points (BSPs: D13, D17) and the two grid supply points (GSPs:
  D1, D6).
prediction: >-
  Written before any score is read. P1. BSPs, stack minus BG2, days 1 to 10, pooled over the two:
  between -0.10 and +0.40 points. P2. GSPs, stack minus BG2: between +0.00 and +0.80 points, because
  a GSP aggregates wind, solar, and storage that the stack has no term for (experiment 36 found
  regional wind and solar columns helped there). P3. The stack beats climatology at all four series
  and `manual_heuristic` at least at three. P4. Coverage at q05 is above 0.07 for the stack at the
  BSPs (the lower tail is too narrow, as in experiment 377). P5. The simple weekly XGBoost arm
  equals BG2 until experiment 381 stores a better arm, because BG2 at these series is already the
  weekly member alone.
decision_rule: >-
  Descriptive: the BSP and GSP family has no research default, so nothing is adopted. Contrasts,
  registered before any score is read, CRPS in points of effective capacity on the shared rows
  (`fold_a_daily_gap_v1`, 292 daily runs of 2025-07 to 2026-06), paired month-resampled bootstrap of
  `studies.eval_protocol`: C1 stack minus BG2 pooled over the BSPs, C2 the same pooled over the
  GSPs, never one pooled number across both; each at the 99% level (20,000 resamples) with the
  largest contributor left out beside it, and per series at 95% (2,000 resamples). C3 stack minus
  `manual_heuristic` and minus `climatology`, per series. C4 simple weekly XGBoost minus BG2 and
  minus the stack, once experiment 381's best simple arm is stored. Lead bands as in the protocol;
  days 1 to 10 headline, days 11 to 14 report only. Tail calibration (pinball loss and coverage at
  0.01, 0.05, 0.95, 0.99) is reported beside CRPS and twCRPS and named in the take-home.
benchmark: >-
  BG2 (`b_weekly`, experiment 87's stored predictions, which hold leads up to 351 hours at these
  four series), `manual_heuristic`, `climatology`, and, as a labelled extra, the bottom-up sum of
  Primary forecasts where experiment 161 or 204 built it.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score estimated from the 11 quantiles, mean absolute error of the
  median, and twCRPS, in points of effective capacity, from `studies.eval_protocol`; leads 0.5 to
  336 hours; headline band days 1 to 10.
code: ml-experiments/00386-bsp-gsp-demand-v1-top-down-bg2-gam-simple-xgboost/
production_status: not-for-production
adopted_in: none
verdict: refuted
best_arm: >-
  BG2 (the weekly-refitted XGBoost model alone): CRPS 4.68 (D13), 7.06 (D17), 14.06 (D1), and
  13.68 (D6) at days 1 to 10
headline: >-
  GAM stack minus BG2, days 1 to 10, CRPS in points of effective capacity: BSPs +1.02 (99%
  interval +0.68 to +1.43), GSPs +2.01 (+1.41 to +2.60). The decision rule's margin was +0.20, so
  the hypothesis is refuted for both groups. The stack scores level with the manual heuristic and
  climatology (-0.04 and -0.33 at the BSPs, -0.34 and -0.37 at the GSPs, all intervals including
  zero).
take_home_message: >-
  At the two bulk supply points and two grid supply points, the GAM stack fitted on each aggregate
  reading does much worse than BG2 (the weekly XGBoost model), by 1.0 points at the bulk supply
  points and 2.0 at the grid supply points, and scores about level with the manual heuristic and
  climatology. The gap is largest on day 1 and shrinks with lead, so the stack is missing what
  recent power gives BG2 at these aggregates, not the weather response. The lower tail is not the
  problem here: BG2's 5% quantile covers 2.8% to 4.4% of readings, and the stack's covers 5.5% to
  10.4%, so the stack's lower tail is too narrow, most clearly at D1.
lessons:
  - >-
    Days 1 to 10 gap to BG2 is smallest at D13 (+0.55) and largest at D1 (+2.54); no series is
    close to the +0.20 margin.
  - >-
    The gap falls from day 1 (+0.9 at D13, +2.4 at D17, +5.0 at D1, +1.6 at D6) to days 8 to 10
    (+0.4, +0.9, +0.5, +1.3), which fits the stack lacking a way to use recent power. This
    explanation was read from the lead pattern and not tested with a fit.
advice:
  - >-
    Use BG2 (the weekly XGBoost model alone) as the bulk and grid supply point forecast to beat,
    and do not port the Primary GAM stack to these series (C1, C2).
  - >-
    Give a stack at an aggregate series a recent-power input before testing its weather response
    (day-1 gap in the lead table).
production_status: not-for-production
adopted_in: none
date: 2026-10-29
---

# BG2, the GAM stack, and the references at the bulk and grid supply points

**The GAM stack of experiment 377, fitted separately on the aggregate readings of two bulk supply
points and two grid supply points, scores much worse than BG2 and about level with the manual
heuristic and climatology.** A bulk supply point (BSP) connects the 132 kV network to the 33 kV
network, and a grid supply point (GSP) connects the transmission network to the 132 kV network.
The series are D13 and D17 (BSPs) and D1 and D6 (GSPs). The stack is a generalised additive model
with a correction stage, here labelled "stack without S below 168 h" (it has no two-number solar
summary below 168 hours and no daylight blend). BG2 at these series is the weekly-refitted XGBoost
model alone (experiment 87's `b_weekly`). Scores are the quantile score (the continuous ranked
probability score, CRPS, estimated from 11 quantiles) in points of effective capacity, and negative
differences favour the stack. No research default exists for this family, so nothing is adopted,
and a "simple weekly XGBoost" arm is added once experiment 381's weekly forecasts have been scored
at these series (stage D, a second pull request).

![Figure 1: stack minus BG2 at days 1 to 10, per series and pooled](headline.svg)

![Figure 2: absolute CRPS of the four arms at each series](absolute_scores.svg)

## Results (CRPS in points of effective capacity; lower is better)

- **Verdict.** Refuted for both groups. Stack minus BG2, days 1 to 10, 99% interval from 20,000
  resamples of whole months: BSPs +1.02 (+0.68 to +1.43); GSPs +2.01 (+1.41 to +2.60). The
  registered margin was +0.20 and both lower ends are above it. The BSPs and GSPs are never pooled
  together.
- **Per series (95%, 2,000 resamples).** D13 +0.55 (+0.11 to +0.98); D17 +1.49 (+1.12 to +1.89); D1
  +2.54 (+1.51 to +3.41); D6 +1.51 (+1.04 to +1.94). Leaving out one series leaves every pooled
  interval above zero (BSP without D13 +1.49, without D17 +0.55; GSP without D1 +1.51, without D6
  +2.54).
- **Absolute scores, days 1 to 10 (BG2, stack, manual heuristic, climatology).** D13 4.68, 5.23,
  5.47, 5.62; D17 7.06, 8.55, 8.39, 8.82; D1 14.06, 16.60, 16.97, 16.98; D6 13.68, 15.19, 15.50,
  15.55.
- **The stack against the references (99%).** Against the manual heuristic: BSPs -0.04 (-0.44 to
  +0.30), GSPs -0.34 (-1.02 to +0.13). Against climatology: BSPs -0.33 (-0.90 to +0.11), GSPs -0.37
  (-1.27 to +0.56). All four intervals include zero. The stack wins 0 of 4 series against BG2, 3 of
  4 against the manual heuristic, and 4 of 4 against climatology (days 1 to 10).
- **BG2 against the references (99%).** BG2 minus the manual heuristic is -1.06 (BSPs) and -2.35
  (GSPs); BG2 minus climatology is -1.35 and -2.38. All four intervals exclude zero.
- **By lead (stack minus BG2, pooled).** Days 8 to 10: BSPs +0.68 (+0.39 to +1.01), GSPs +0.94
  (+0.23 to +1.72). Days 11 to 14 (beyond NGED's horizon, descriptive): BSPs +0.62, GSPs +1.14. On
  day 1 the gap per series is D13 +0.90, D17 +2.39, D1 +4.94, D6 +1.64.
- **Slices (stack, BG2).** The gap is spread over every slice read: sunny half-hours (BSPs 11.9
  against 9.8, GSPs 15.2 against 14.2), winter nights, Economy 7 steps, and worst on Christmas and
  bank holidays (GSPs 22.0 against 15.9 on Christmas).
- **Tails, days 1 to 10 (stack; BG2).** Pinball loss at q01, q05, q95, q99 and coverage at the
  same quantiles are printed per series by `e00386_score_b.py`. Coverage at q05 is 0.063 (D13), 0.061
  (D17), 0.104 (D1), and 0.055 (D6) for the stack against 0.028, 0.037, 0.042, and 0.044 for BG2;
  coverage at q95 is 0.935, 0.927, 0.971, and 0.949 against 0.927, 0.960, 0.955, and 0.935. The
  stack's lower tail is too narrow at D1 and slightly at the other three; BG2's lower tail is too
  wide.
- **Curtailment cost** is not scored: the metric needs the node and generator tables, and the
  readings here are the aggregates alone.

## Checks

- **Median minus mean.** The stack's median equals its mean exactly (difference 0.0 at all four
  series), because the spread is added to the corrected mean as a centred table. The check cannot
  expose a shifted forecast here. BG2 stores no mean.
- **Mean-forecast error beside the quantile score.** The mean-forecast error was not computed; the
  error of the median forecast (MAE of the median, days 1 to 10) is printed and agrees in direction
  with CRPS at all four series (BG2 8.33, 12.62, 25.91, 24.89 for D13, D17, D1, D6).
- **Triggers fired.** Trigger 2: the result contradicts its registered predictions P1 and P2 (stack
  minus BG2 expected at most +0.40 and +0.80). The lead pattern supplies an untested explanation
  (the stack lacks BG2's recent-power input). No later experiment depends on this result yet. An
  audit by a fresh agent is requested in the pull request; its findings are triaged there.

## Setup and caveats

- **Code and reuse.** `ml-experiments/00386-*/` copies experiment 377's stack (`e00386b_*` and
  `e00386bs*_*`) and experiment 161's input rows; the ensemble arrays are rebuilt from the local
  ECMWF archive (read only, no fetch), and the rebuild reproduces D10's stored arrays exactly. BG2,
  climatology, and the manual heuristic are experiment 87's stored forecasts on `fold_a_daily_gap_v1`.
- **What could not be fitted.** The PV capacity is pinned to zero at all four series (no vintage
  exists), so the stack has no explicit solar term; the stack has no regional wind or solar input,
  which experiment 36 found helped at GSPs.
- **Protocol.** `demand-v1` metrics, folds by month, 12-month minimum history (all four series have
  more than 12 months). The protocol's group table knows only the Primaries, so the scoring script
  adds the labels BSP and GSP. The stack's monthly refit follows the new monthly rule.
- **Plan change.** The plan put the GSPs in stage C; the ensemble arrays for the GSPs were cheap to
  build, so the stack at the GSPs is in this pull request. Stage D (simple weekly XGBoost) remains.
- **What to try next.** A stack with a recent-power term at these aggregates, and a stack with the
  embedded generation and storage under each node as explicit terms.
