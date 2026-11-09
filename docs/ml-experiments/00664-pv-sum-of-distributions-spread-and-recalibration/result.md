---
id: "00664"
slug: pv-sum-of-distributions-spread-and-recalibration
theme: solar-disaggregation
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
size: >-
  medium. Stored: nothing new is fitted; the scripts re-load the stored fine-tuned networks of
  experiments 644 (arm a) and 655 (arm d) and write test-window quantiles and probability integral
  transform values to this experiment's private folder. Serving path: not touched. Degradation
  rule: not touched. Designs: more than one way to recalibrate and to couple the two draws is
  defensible, so both are fixed below before any score is read. Callers: the scripts import the
  modules of experiments 397, 428, 510, 554, 634, 638, 639, 644, and 655 and edit none of them.
  Asset graph: not touched. One GPU process at a time, inference only; ceiling 1.5 GPU-hours.
hypothesis: >-
  The job of the explicit solar term is as much to calibrate the spread of the net-demand forecast
  as to shift its centre. In experiments 644 and 655 the explicit term subtracted one number (the
  member-average solar curve) from every forecast quantile, so the solar uncertainty (capacity
  times cloud uncertainty) never reached the output. Building the net-demand distribution as the
  gross-demand distribution minus the solar curve of a randomly drawn ensemble member beats the
  free network on sunny middays, and the gain is more than a recalibration of the free network's
  quantile widths would give.
prediction: >-
  Fixed before any score was read. The free network is already roughly calibrated (5% to 95%
  coverage on sunny middays 88.0% against a nominal 90%), so I expect the recalibration (b) to
  gain at most 0.05 points of capacity at day 1-2, below the bar. I expect the sum-of-distributions
  arm (c) to widen the sunny-midday interval, raise coverage to 89% to 92%, and gain 0.05 to 0.15
  points over (a) at day 1-2 with a smaller gain at days 3-5 and 6-10, because member spread
  carries more of the uncertainty near the forecast time than at long lead. I give the chance that
  (c) beats (b) by the bar at day 1-2 as 25%.
decision_rule: >-
  Fixed before any score was read. CRPS is in points (percent of effective capacity), pooled over
  the six MW Primaries, the three cutoffs (2025-07, 2026-03, 2026-06), and the sunny half-hours
  (10:00 to 15:00 UTC, observed clear-sky index at least 0.7) of the 60 days after each cutoff,
  averaged over five seeds; a contrast is the control's CRPS minus the arm's, so a positive value
  favours the arm. Intervals are day-block bootstraps (experiment 644's convention, Bonferroni
  level over the planned contrasts). The bars are experiment 644's: 0.084, 0.093, and 0.123 points
  at day 1-2, 3-5, and 6-10. The solar physics adds something beyond spread only if (c) or (d)
  beats (b) by at least the bar with an interval that excludes zero. If (b) beats (a) by the bar
  and (c) does not beat (b), the gain is spread calibration, and the recommendation is the
  recalibration (named as an issue for the maintainer, not implemented). If neither (b) nor (c)
  beats (a), stop: the free network is calibrated enough.
benchmark: >-
  Arm (a), the free network of experiment 644, re-scored from its stored fits on the same rows, and
  arm (b), the free network with its quantile widths recalibrated. Climatology and the manual
  heuristic are not re-scored; experiment 644 scores them on these rows.
row_set_id: >-
  Experiment 644's test rows (the 60 days after 2025-07, 2026-03, and 2026-06, cut at 2026-07-01,
  real world only, every ensemble run at lead 0 to 240 hours, six MW Primaries, cleaned power,
  farm-gap rows dropped). The code asserts that the row keys equal those stored by experiment 644.
metric: >-
  Continuous ranked probability score (CRPS) in points from the 11 quantiles of each forecast, on
  sunny middays and on all half-hours, by lead band; coverage of the 5% to 95% interval and its mean
  width; pinball loss and coverage at the 0.01, 0.05, 0.95, and 0.99 quantiles; the probability
  integral transform (PIT) histogram and its distance from uniform.
verdict: ""
production_status: not-for-production
adopted_in: none
---

# Does the solar term mainly calibrate the spread? (planned)

Planned. The arms, the recalibration, the coupling, the placebos, and the decision rule are fixed
in the front matter and in the maintainer's brief of 2026-11-09.
