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
  climatology. The gap is largest on day 1 and shrinks with lead, which is consistent with the
  stack lacking the recent-power input that BG2 has; this explanation is untested. Both forecasts have
  tail errors: BG2's 5% quantile covers 2.8% to 4.4% of readings (too wide), and the stack's covers
  5.5% to 10.4% (too narrow, most clearly at D1).
lessons:
  - >-
    Days 1 to 10 gap to BG2 is smallest at D13 (+0.55) and largest at D1 (+2.54); the lower end of
    D13's 95% interval (+0.11) is the only one below the +0.20 margin.
  - >-
    The gap falls from day 1 (+0.9 at D13, +2.4 at D17, +4.9 at D1, +1.6 at D6) to days 8 to 10
    (+0.4, +0.9, +0.5, +1.3; falls least at D6), which fits the stack lacking a way to use recent
    power. This explanation was read from the lead pattern and not tested with a fit.
advice:
  - >-
    Use BG2 (the weekly XGBoost model alone) as the bulk and grid supply point forecast to beat,
    and do not port the Primary GAM stack to these series (C1, C2).
  - >-
    Give a stack at an aggregate series a recent-power input before testing its weather response
    (the day-1 gaps in the 'By lead' bullet).
production_status: not-for-production
adopted_in: none
date: 2026-10-29
---

# Top-down forecasts at four bulk and grid supply points: a GAM stack against a weekly XGBoost model

**A generalised additive model (GAM) with a correction stage, fitted on the aggregate readings of
two bulk supply points and two grid supply points, scores 1.0 to 2.0 points of capacity worse than
BG2 and about level with two simple benchmarks.** A bulk supply point (BSP) connects the 132 kV
electricity network to the 33 kV electricity network, and a grid supply point (GSP) connects the
transmission network to the 132 kV electricity network. The four series are D13 and D17 (BSPs) and
D1 and D6 (GSPs), labelled as in earlier experiments. "The stack" is the GAM of experiment 377,
labelled "stack without S below 168 h": the stack has no two-number summary of solar output below
168 hours and no daylight blend. BG2 is the best-guess forecast of experiment 87, which at these
series is its weekly-refitted XGBoost model alone (`b_weekly`). The two benchmarks are
`manual_heuristic` (NGED's rule of thumb) and `climatology` (the typical value for the month,
half-hour, and weekday type). A score is the continuous ranked probability score (CRPS), estimated
from 11 quantiles, in points of effective capacity (percent of the series' capacity); a negative
difference favours the stack. No research default exists for this family, so nothing is adopted.
The simple weekly XGBoost arm of experiment 381 is not yet scored here.

![Figure 1: stack minus BG2 at days 1 to 10, per series and pooled](headline.svg)

![Figure 2: absolute CRPS of the four arms at each series](absolute_scores.svg)

## Results (CRPS in points of effective capacity; lower is better)

- **Verdict.** Refuted for both groups. Stack minus BG2, days 1 to 10, 99% interval from 20,000
  resamples of whole months: BSPs +1.02 (+0.68 to +1.43); GSPs +2.01 (+1.41 to +2.60). The
  registered margin was +0.20 and both lower ends are above it. The BSPs and GSPs are never pooled
  together.
- **Per series (95%, 2,000 resamples).** D13 +0.55 (+0.11 to +0.98); D17 +1.49 (+1.12 to +1.89); D1
  +2.54 (+1.51 to +3.41); D6 +1.51 (+1.04 to +1.94). Leaving out one series leaves every pooled
  estimate above zero (BSP without D13 +1.49, without D17 +0.55; GSP without D1 +1.51, without D6
  +2.54).
- **Absolute scores, days 1 to 10 (BG2, stack, manual heuristic, climatology).** D13 4.68, 5.23,
  5.47, 5.62; D17 7.06, 8.55, 8.39, 8.82; D1 14.06, 16.60, 16.97, 16.98; D6 13.68, 15.19, 15.50,
  15.55.
- **The stack against the references (99%).** Against the manual heuristic: BSPs -0.04 (-0.44 to
  +0.30), GSPs -0.34 (-1.02 to +0.13). Against climatology: BSPs -0.33 (-0.90 to +0.11), GSPs -0.37
  (-1.27 to +0.56). All four intervals include zero. The stack scores better than BG2 at 0 of 4
  series, than the manual heuristic at 3 of 4, and than climatology at 4 of 4 (days 1 to 10).
- **BG2 against the references (stage A rows).** The references were scored in a first script on
  the half-hours shared by BG2, climatology, and the manual heuristic; the stack contrasts use
  the half-hours shared by all four arms. BG2's score differs by less than 0.001 points between the
  two row sets.
- **BG2 against the references (99%).** BG2 minus the manual heuristic is -1.06 (BSPs) and -2.35
  (GSPs); BG2 minus climatology is -1.35 and -2.38. All four intervals exclude zero.
- **By lead (stack minus BG2, pooled).** Days 8 to 10: BSPs +0.68 (+0.39 to +1.01), GSPs +0.94
  (+0.23 to +1.72). Days 11 to 14 (beyond NGED's horizon, descriptive): BSPs +0.62, GSPs +1.14. On
  day 1 the gap per series is D13 +0.90, D17 +2.39, D1 +4.94, D6 +1.64.
- **Slices (stack, BG2).** The stack is worse than BG2 in every slice read, and level at BSP cold
  nights (5.21 against 5.20): sunny half-hours (BSPs 11.9 against 9.8, GSPs 15.2 against 14.2),
  winter nights, and Economy 7 steps. The gap is largest at the GSPs on Christmas (22.0 against
  15.9) and bank holidays (22.4 against 14.6).
- **Tails, days 1 to 10 (stack; BG2).** Pinball loss at q01, q05, q95, q99 and coverage at the
same quantiles are computed per series by `e00386_score_b.py`. Coverage at q05 is 0.063 (D13), 0.061
(D17), 0.104 (D1), and 0.055 (D6) for the stack against 0.028, 0.037, 0.042, and 0.044 for BG2;
coverage at q95 is 0.935, 0.927, 0.971, and 0.949 against 0.927, 0.960, 0.955, and 0.935. The
stack's lower tail is too narrow at D1 (0.104) and a little at D13 and D17; D6 is near nominal
(0.055). The stack's upper tail is too wide at D1 (q95 coverage 0.971). BG2's lower tail is too wide
at all four.
- **Curtailment cost** is not scored: the metric needs the node and generator tables, and the
  readings here are the aggregates alone.

## Checks

- **Median minus mean.** The stack's median equals its mean exactly (difference 0.0 at all four
  series), because the spread is added to the corrected mean as a centred table. The check cannot
  expose a shifted forecast here. BG2 stores no mean.
- **Mean-forecast error beside the quantile score.** The mean-forecast error was not computed; the
  error of the median forecast (MAE of the median, days 1 to 10) is printed and agrees in direction
  with CRPS at all four series (BG2 8.33, 12.62, 25.91, 24.89 for D13, D17, D1, D6).
- **Triggers fired.** Review trigger 2 (a result contradicts a registered prediction): the result
  contradicts its registered predictions P1 and P2 (stack minus BG2 expected at most +0.40 and
  +0.80). The lead pattern supplies an untested explanation (the stack lacks BG2's recent-power
  input). No later experiment depends on this result yet. An audit by a fresh agent is requested in
  the pull request; its findings are triaged there.

## Setup and caveats

- **Code and reuse.** `ml-experiments/00386-*/` copies experiment 377's stack (`e00386b_*` and
  `e00386bs*_*`) and experiment 161's input rows; the ensemble arrays are rebuilt from the local
ECMWF archive (read only, no fetch), and the rebuild reproduces the stored arrays of another
substation (D10) exactly. BG2, climatology, and the manual heuristic are experiment 87's stored
forecasts on `fold_a_daily_gap_v1`.
- **What could not be fitted.** The PV capacity is pinned to zero at all four series (no record of
  when each installation's capacity changed exists), so the stack has no explicit solar term; the
  stack has no regional wind or solar input, which experiment 36 found helped at GSPs.
- **Protocol.** `demand-v1` metrics, folds by month, 12-month minimum history (all four series have
  more than 12 months). The protocol's group table knows only the Primaries, so the scoring script
  adds the labels BSP and GSP. The stack refits monthly.
- **What to try next.** A stack with a recent-power term at these aggregates, and a stack with the
  embedded generation and storage under each node as explicit terms.
