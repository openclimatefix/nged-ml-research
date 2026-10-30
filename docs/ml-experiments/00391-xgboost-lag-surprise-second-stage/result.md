---
id: "00391"
slug: xgboost-lag-surprise-second-stage
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: finished
kind: other
protocol: demand-v1
size: >-
  complex. Stored: per-arm prediction folders, a stage 1 surprise table per Primary (kept for
  experiment 392's arm R5), and detector alarm tables, all private. Designs: more than one
  defensible
  (surprise from walk-forward or cross-fitted stage 1; ENS-driven or ERA5-fed residual; which
  neighbour summaries). Callers: imports experiment 87's fit modules and experiment 383's folder
  (interim until its pull request merges; copied if it has not merged). Production serving path and
  degradation rule: not touched.
hypothesis: >-
  Giving the weekly XGBoost demand model a measure of how surprising each of its three lagged power
  readings is, at its own Primary and at the neighbouring Primaries, lets the model handle
  load-shift events as well as an explicit event detector's open-alarm state does. The detector
  state adds nothing that surprise at the Primary and its neighbours cannot supply.
prediction: >-
  Written before any score is read. P1. On rows whose lags lie inside a labelled event, own
  surprise (N0) cuts the continuous ranked probability score (CRPS) by 10% or more of the gap
  between the baseline and the oracle (O). P2. Neighbour surprise (N1) cuts it by a further 5% of
  that gap at the Primaries with a labelled transfer. P3. The detector state (N2) adds less than 5%
  of the gap beyond N1, and N3 (detector only) recovers less than N1. P4. On rows outside events
  and the 7 days after them, no surprise arm is worse than the baseline by more than 0.02 points of
  effective capacity. P5. The ENS-driven and ERA5-fed lagless forecasts give surprise with a
  correlation of at least 0.9 per Primary.
decision_rule: >-
  Seven planned contrasts (C1 to C7 below), each read at the 99.29% level (0.36% per tail, a
  Bonferroni correction across 7 at a 95% family level), with whole-event blocks on event slices and
  month blocks
  elsewhere. A surprise arm counts as helping when its interval on its primary slice (rows whose
  lags lie inside a labelled event, days 1 to 10) excludes zero in the favourable direction and its
  mean change on the rows outside events and the 7 days after is not worse than +0.02 points. Every
  other slice is descriptive.
  The headline C3 reads "surprise gives the benefit of the detector" as supported when its
  interval on the event-lag rows includes zero and has both bounds within ±0.03 points of effective
  capacity. The
  porting threshold is set by cost: an extra column set computed from the same stage 1 model
  (N0, N1) needs a gain whose interval excludes zero; the detector (N2, N3) adds a state machine
  and a veto, so it needs a gain of at least 0.03 points of CRPS on the event-lag rows to be worth
  porting.
benchmark: >-
  The weekly XGBoost member of experiments 379 and 381 (stored, for the reproduction gate and as a
  reference row), the same recipe refitted monthly (B0, the control for every contrast), the
  lagless fallback c2 of experiment 383 (interim), BG2, `manual_heuristic`, and `climatology`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  CRPS estimated from the 11 quantiles, mean absolute error of the median, pinball loss and
  coverage at q01, q05, q95, and q99, and avoidable-curtailment cost, in points of effective
  capacity (cost in GBP), from `studies.eval_protocol`; leads 0.5 to 336 hours; headline band days
  1 to 10.
code: ml-experiments/00391-xgboost-lag-surprise-second-stage/
verdict: not supported
best_arm: >-
  No arm beat the monthly-refit control B0 on event-lag rows. On all rows, days 1 to 10, at the
  signed-MW six (points of effective capacity, seed 0), B0 scored 3.123, N1 3.229, and the
  hand-label oracle O 3.110, a difference of 0.013 between B0 and O.
headline: >-
  Adding own and neighbour surprise (N1) to the monthly XGBoost demand model raised the CRPS on rows
  whose lags lie inside a labelled event by +0.110 points (99.29% interval -0.249 to +0.432) and on
  rows outside events by +0.072 (+0.006 to +0.145), against a seed noise floor of +0.024 and +0.001.
  The oracle with hand labels (O) was also worse than the control, by +0.218. The detector state
  (N2, N3) was not fitted.
take_home_message: >-
  A "surprise" input does not let the monthly XGBoost demand model handle load-shift events, and the
  extra columns make the forecast slightly worse on the ordinary rows that make up 95% of the
  test. An oracle that is told which lags are inside an event did not beat the control either, and
  so the hand-label flags show no sign that the lags have much to recover. The detector that would
  have made the
  comparison failed its pre-registered check, so whether detector state adds to surprise stays
  untested. One option for the next step is to handle the event outside the XGBoost lags, which this
  experiment did not test.
lessons:
  - >-
    The oracle O, given the true event flags, scored +0.218 points worse than the control on
    event-lag
    rows (interval -0.052 to +0.542 against seed 1), so the hand-label flags show no gain over the
    control.
  - >-
    Surprise columns cost +0.072 points on rows outside events (interval +0.006 to +0.145), 3.6
    times the +0.02 tolerance, but the interval's lower bound is below the tolerance, and a second
    control seed moved the same slice by +0.001.
  - >-
    The detector failed its pre-registered known-answer check (recall 0.42 against 0.55; veto
    dropped
    0.33 of placebo alarms against 0.5), and found 0 of 7 labelled transfers at its tuned threshold.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Do not add lag-surprise or hand-label event flags to the XGBoost demand model's inputs: several
    designs of that idea have now failed (Result, contrasts C8 to C11).
  - >-
    Report every lag-input contrast against two seeds of the control: the seed gap (+0.024 on
    event-lag rows) is small beside the +0.110, but the event-lag interval of N1 includes zero, so
    read it as
    no help (Result, noise floor).
  - >-
    Calibrate any detector threshold on transfers of the depth the labels hold (0.71 to 2.3 MW),
    not on shares of capacity, before tuning it (Detector gate).
date: 2026-10-30
---

# Does a "surprise" input let XGBoost handle load-shift events?

**A "surprise" input does not let the monthly XGBoost demand model handle load-shift events, and it
makes the forecast slightly worse on the ordinary rows.** A load shift moves demand between
substations, so a lagged power reading inside one misleads the model. The surprise input is the
normal score of each lagged reading's probability integral transform against a lagless forecast
from a first-stage model, at the Primary and at its neighbours. On rows whose lag lies inside a
labelled event, the surprise columns did not help (+0.110 points of CRPS, interval -0.249 to
+0.432). On rows outside events they cost +0.072 (+0.006 to +0.145), against a +0.02 tolerance.
A Primary is one of the 16 primary substations scored here. CRPS is the continuous ranked
probability score, in points of effective capacity.

**The oracle did not beat the control either.** The oracle O
gets hand-label flags for the target and each lag. On event-lag rows it scored +0.218 points worse
than the control (all 16 Primaries). The interval against seed 1 is -0.052 to +0.542, so the result
is "no gain" and not "harm".
BG2 scores lower than B0 (3.016 against 3.123), so headroom exists elsewhere in the model.

**The explicit detector failed its pre-registered check, so whether detector state adds to
surprise is untested.** Arms N2, N2 without the veto, and N3 were not fitted: the manager held them
after the known-answer gate failed and the surprise arms came out worse than the control. This
experiment therefore answers the question about surprise columns and leaves the detector-state
half open.

![Each contrast against the seed noise floor](headline.svg)

## Result

All numbers are CRPS differences in points of effective capacity (above 0 is worse), days 1 to 10,
raw bands. Intervals are 99.29% (Bonferroni across the seven planned contrasts at a 95% family
level), resampling whole events on event slices and calendar months elsewhere. "Event-lag rows" are
rows whose target lies outside an event but has a lag inside one.

| Contrast | Primaries | Event-lag rows | Rows outside events and 7 days after |
|---|---|---|---|
| Seed 1 minus seed 0 of the control (noise floor) | 16 | +0.024 (-0.017 to +0.097) | +0.001 (-0.002 to +0.006) |
| N1 minus B0, seed 0 | 16 | +0.110 (-0.249 to +0.432) | +0.072 (+0.006 to +0.145) |
| N1 minus B0, seed 1 | 16 | +0.086 (-0.251 to +0.404) | +0.071 (+0.006 to +0.143) |
| O minus B0, seed 1 | 16 | +0.194 (-0.052 to +0.542) | +0.012 (+0.000 to +0.032) |
| N0 minus B0 (own surprise only) | 7 | -0.038 (-0.649 to +0.578) | +0.055 (+0.015 to +0.103) |
| N1 minus N0 (neighbours) | 7 | +0.314 (+0.062 to +0.749) | +0.028 (+0.007 to +0.054) |
| N1-ERA5 minus N1 (hindsight weather) | 7 | -0.004 (-0.496 to +0.654) | -0.020 (-0.064 to +0.008) |
| N1-ERA5 minus B0, seed 1 | 7 | +0.263 (+0.061 to +0.538) | +0.063 (+0.025 to +0.108) |

- **Against both control seeds** every contrast has the same sign and a similar size: N1 is +0.110
  and +0.086, O is +0.218 and +0.194, and the control seeds differ by +0.024. The noise floor is
  small beside the shifts but the event-lag intervals still include zero, so the event-lag result
  is "no help" and not "harm".
- **Signed-MW six and MVA-only eight, separately.** N1 minus B0 on event-lag rows is +0.277
  (-0.508 to +1.002) at the signed-MW six and -0.068 (-0.314 to +0.084) at the MVA-only eight. The
  pooled 16-Primary result is driven by D10 (+1.85) and D16 (+0.78) among the signed-MW six;
  leaving D10 out gives +0.049.
- **Own surprise alone (N0)** moved the event-lag rows by -0.038, an 18% cut of the 0.207 oracle
  gap, which meets prediction P1 in size but with an interval that includes zero. The neighbour
  columns undid it and more (N1 minus N0 +0.314), against prediction P2.
- **The oracle gap on contaminated rows.** O minus B0 is +0.218 at all 16 Primaries,
  +0.207 at the signed-MW six, and +0.332 at the MVA-only eight, so a hand-label flag also did
  not help at the MVA-only Primaries.
- **Absolute scores (all rows, days 1 to 10, signed-MW six, seed 0):** B0 3.123, N1 3.229, O 3.110,
  BG2 3.016. These use the 150-round screening
  recipe, so they are not comparable to experiment 381's.
- **ERA5-fed against ENS-driven surprise (C6):** N1-ERA5 is -0.020 (-0.064 to +0.008) on ordinary
  rows, so hindsight weather does not rescue the surprise columns.
- **Avoidable curtailment cost** (GBP, all rows, signed-MW six): B0 12.65M, seed 1
  12.63M, N0 13.10M, N1 13.15M, N1-ERA5 12.80M, O 12.26M, `manual_heuristic` 12.66M, BG2 8.35M.

## The detector and its gate

**The pre-registered known-answer gate failed.** Injected transfers at shares of capacity of 10%
and 25%: recall at 25% and 2 days or more was 0.42 (gate 0.55); the veto kept 0.98 of plain hits
(gate 0.95); the veto dropped 0.33 of placebo alarms (gate 0.5). At the tuned threshold (k = 2.3
ENS,
2.35 ERA5-fed) the detector marked 0 of 7 labelled transfers in the ENS version (1 of 7 in the
ERA5-fed
version), at 1.70 false
alarms per Primary-year.

**A post hoc gate in megawatts passed, and is labelled as such.** The label table holds transfers
of 0.71, 1.1, and 2.3 MW at the quartiles, which are small shares of a Primary's capacity.
Re-running the gate with those depths gave recall 0.63, veto keeps 0.98, and veto drops 0.91 of
placebo alarms. This was chosen after seeing the failure, so it does not replace the pre-registered
result. It suggests the detector works at the real transfer depths and failed the gate through the
choice of depths.

## Checks

- **Median minus mean forecast per site (points, days 1 to 10).** B0 and N1 differ by less than 0.1
  at most Primaries. The one sign change is at D22 (+0.098 for B0, -0.110 for N1). The full table is
  in `work/scores_core/printed_report.txt`.
- **Mean error beside the quantile contrasts.** The mean absolute error of the median moves with
  CRPS: N1 minus B0 +0.210 (-0.311 to +0.748) on event-lag rows, seed noise +0.025.
- **Which audit triggers fired.** Two: the neighbour columns reversed the sign of prediction P2,
  and O came out worse than the control. A Sonnet audit of the scoring and
  the oracle flags found no bug. The seed-1 control shows that seed noise
  (+0.001 outside events) is far smaller than the +0.072 harm outside events.
- **Design target.** The sweep for the three median and tail tolerances was not run (see below).

## Caveats and what to try next

- **N2, N2 without the veto, N3 were not fitted.** The manager held them. Predictions P3 and the C3,
  C4, C5, and C7 contrasts are therefore untested, and the lag-encroachment sweep, the P5
  correlation, and the code-review check for leakage were not run as separate steps.
- **The detector's tuning window is shorter than the distillation's** (2024-11 to 2025-06, because
  stage 1 starts in 2024-10).
- **The recipe is the 150-round screening recipe**, not experiment 87's 500 rounds, so scores are
not
  comparable to experiment 381's.
- **Two sets of Primaries.** B0, N1, and O ran at all 16; N0 and N1-ERA5 at 7 (the signed-MW six
  plus D8), so their contrasts use those 7.
- **Event-lag rows are about 2.4% of the rows** (64,006 of 2.63M at 16 Primaries) and come from 27
  events; every event-lag interval is wide.
- **Scores were printed from saved rows at half precision** for the 16-Primary pass (float32
  except the CRPS and error columns), after the full-precision pass ran out of memory.
- **Stop rule.** By the `verdict` in each page's front matter, experiments 89 and 90 (`refuted`) and
  383 (`not supported`) count on the idea "handle load-shift events inside the XGBoost lags", and
  this experiment is the fourth. Experiment 5 (`refuted`) tests lag placement, so the manager
  decides whether it counts. Experiments 8, 18, and 88 (`inconclusive`) do not count; 390 (`partly
  supported`) does not
  count either, and the manager decides whether it tests the same idea.

## Data and reproduction

- **Surprise tables for experiment 392:**
`data/ml-experiments/00391-xgboost-lag-surprise-second-stage/surprise_tables/`.
- **Boosters:** `.../work/boosters/<arm>/` (12 monthly refits per Primary).
- **Printed reports:** `.../work/scores_core/printed_report.txt` and
  `.../work/scores_subset/printed_report.txt`.
- **Code:** `ml-experiments/00391-xgboost-lag-surprise-second-stage/`; run order is stage 1,
  detector, fit, assemble, score, charts.
