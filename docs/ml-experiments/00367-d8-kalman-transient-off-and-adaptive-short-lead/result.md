---
id: '00367'
slug: d8-kalman-transient-off-and-adaptive-short-lead
title: >-
  Does switching the Kalman transient off, or scaling it by the substation's own recent benefit,
  recover the short-lead skill it costs at D8?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
size: >-
  simple. Stored: private files under data/ml-experiments/00367-d8-kalman-transient-off-and-adaptive-short-lead/
  only. Serving path: not touched. Degradation rule: not touched. Designs: two arms fixed by the
  brief (off; adaptive gain). Callers: scoring only, reading the stage columns stored by experiment
  364; no model is refitted.
hypothesis: >-
  The Kalman transient adds error at D8 at leads under 24 hours (experiment 364: +0.444 points). A
  transient whose gain is scaled by the substation's own trailing out-of-sample benefit, with no
  per-site switch, recovers much of that loss without hurting D2 or D4.
prediction: >-
  Written before any score is read. Arm KA recovers at least 0.2 of the 0.444 points at D8 at 0 to
  24 hours, and is no worse than the control (the stored stack, transient on) by more than 0.02
  points at D2 and D4 at 0 to 24 hours and over all leads.
decision_rule: >-
  Supported when both parts of the prediction hold: the point estimate of the D8 recovery at 0 to 24
  hours reaches 0.2 points and its month-resampled 95% interval excludes zero.
  Stop rule: if KA is worse than the control by more than 0.05 at any placebo
  (D2, D4), report K0 only and stop. The threshold is set by cost: KA needs a trailing-window
  gain per substation, a small pipeline step.
benchmark: >-
  The control: the stored stack (stage S4, transient on) at D8 weekly (8.338), D2 and D4 monthly,
  from experiment 364. Reproduction gate: stage S4 reproduces 8.338 at D8 on the same 96,986 rows
  as experiment 354.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles) in points of effective capacity;
  Pearson correlation of the median with the actual demand; by lead band 0 to 24, 24 to 72, 72 to
  168 hours.
code: ml-experiments/00367-d8-kalman-transient-off-and-adaptive-short-lead/
production_status: not-for-production
adopted_in: none
date: 2026-10-28
---

**Plan (committed before any score was read).**

This experiment extends [experiment 364](../00364-d8-stage-blame-and-analogue-recent-power/result.md),
which found that the Kalman filter's short-lived correction (the "transient") adds 0.444 points of
quantile score at D8 at leads of 0 to 24 hours. It asks whether turning the transient off, or
scaling it by how much it has helped that substation recently, removes that loss without hurting
other substations. The Kalman level and transient follow the idea of a Kalman filter on a
generalised additive model's coefficients; see the
[Kalman-GAM explainer](../kalman-gam-explainer.md).

The five plan items:

1. **Arms (no refit, scored from the stage columns stored by experiment 364).**
    - Control: stage S4, the stored stack.
    - K0: the transient is removed at leads up to 24 hours. At stage level K0 equals S1 (the plain
      GAM plus the Kalman level) at those leads and S2 elsewhere. At stack level K0 is S4 minus the
      transient (S2 minus S1) and its quantiles move with it. The stack-level form assumes the winter
      night offset and recipe R add the same amounts without the transient (post hoc approximation,
      labelled as such).
    - KA: as K0 at leads up to 24 hours, but the transient is multiplied by a gain g. For each
      forecast run, g is the trailing 90-day mean of (|error of S1| minus |error of S2|) over rows
      with lead up to 24 hours whose valid time is at or before the run's forecast initialisation
      time, divided by the trailing mean absolute size of the transient, clipped to the range 0 to 1.
      Until 14 days of such rows exist, g is 1 (the control). The same rule applies at every
      substation. Beyond 24 hours the transient is untouched, because experiment 364 found it
      neutral or helpful there.
2. **Row set and sites.** The 96,986 D8 rows of experiment 354 (weekly stage columns), plus D2 and D4
   (monthly stage columns) as placebos. The six decision Primaries are scored only if stage columns
   for them are stored in the same layout. Otherwise the page says they were not run.
3. **Scores.** Quantile score and Pearson correlation by lead band, each with a month-resampled 95%
   interval, plus day types as numbers: days touched by a detected step, other days, cold winter
   nights, sunny April to June half-hours. Also tail calibration (coverage and pinball loss at
   q01, q05, q95, q99).
4. **Prediction that can fail.** KA recovers at least 0.2 of the 0.444 points at D8 at 0 to 24 hours
   and is within 0.02 of the control at D2 and D4.
5. **Stop rule.** If KA is worse than the control at D2 or D4 by more than 0.05, report K0 only and
   stop.

Caveat on every headline number: interim, one substation, one scored year; it is unproven whether
the generator explains the steps and the gap. The aim is better uncertainty estimation at D8 without
hurting other substations. Schedule: weekly refits at D8 and monthly at D2 and D4, as stored.
