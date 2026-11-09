---
id: "00661"
slug: pv-ens-time-resolution-shortwave-interpolation-audit-and-bias-adapter
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  complex. Stored: re-built degraded-weather input matrices, pretrained and fine-tuned
  neural-network weights, and quantiles in this experiment's private folder. Serving path: not
  touched. Degradation rule: not touched. Designs: more than one test of the time-resolution
  hypothesis is defensible (shortwave only or all variables degraded, 3-hour or 6-hour windows), so
  the choices are fixed below before any Part 2 or Part 3 result is read. Callers: the scripts
  import the modules of experiments 48, 87, 397, 405, 418, 428, 448, 510, 554, 634, 638, 639, 644,
  and 655 and edit none of them. Asset graph: not touched. One GPU process at a time; ceiling 3
  GPU-hours (Part 2 at most 1.5, Part 3 at most 1.5).
hypothesis: >-
  The explicit solar term gains about 0.75 CRPS points with CAMS and ERA5 weather (experiment 639)
  and loses 0.25 to 0.37 with ECMWF ensemble weather (experiments 644 and 655). One cause is the
  ensemble's coarse time step, which is 3 hours to lead 144 hours and 6 hours after. A second cause
  would be a faulty interpolation of the ensemble's shortwave flux to half-hours. A third is
  forecast cloud error. This experiment tests the first two directly and tests a bias adapter as a
  remedy for the third.
prediction: >-
  Fixed before any Part 2 or Part 3 result was read. Part 1 (the audit) had run before this plan
  was committed, as the brief orders, and is reported below. Part 2: the explicit gain on known
  weather degraded to 3-hour windows falls to between 0.4 and 0.6 CRPS points (a fall of 20% to
  45%), not to half or below, because the interpolation error at 3-hour windows is 5% of the midday
  peak and the explicit term still sees the sunny-day shape. At 6-hour windows I expect 0.2 to 0.4.
  Resolution is therefore a partial cause and not the main one. The 644 loss at 3-hourly leads
  (day 1 to 6) and 6-hourly leads (day 6 to 10) is of similar size (within 0.1 points), so the loss
  does not grow when the step coarsens. Part 3: the adapter arm does not beat the free network at
  day 1-2 by the bar of 0.084, because it must correct a bias that depends on the forecast cloud
  field, which the adapter's five inputs only partly carry.
decision_rule: >-
  Fixed before any Part 2 or Part 3 result was read. CRPS is in points (percent of effective
  capacity), pooled over the six MW Primaries and the sunny half-hours (10:00 to 15:00 UTC,
  clear-sky index at least 0.7) of the 60 days after each cutoff, as in experiments 639 and 644.
  Intervals are day-block bootstraps at 99.7% (Bonferroni over this page's contrasts). Part 1 rule:
  if the audit finds a mismatch with production or a time-stamp shift of one window or more, fix
  the pipeline copy and re-score 644's two headline contrasts on the corrected half-hours, and say
  whether the sign changes. Part 2 step 5 rule: if the explicit-minus-free gain on known weather
  degraded to 3-hour windows is at most half of the same cutoff's undegraded gain (experiment 639's
  stored value for the same seeds), report that time resolution is a main cause and name the remedy
  to test next (an ensemble product at finer time resolution, or a learnt sub-window shape);
  otherwise report that resolution is not the main cause. The 6-hour variant is reported beside it
  and does not change the rule. Part 3 rule: the adapter arm A is kept only if it beats the free
  network at day 1-2 by the experiment 644 bar (0.084 points) with an interval that excludes zero.
  The learned DC-to-AC ratio and the fleet-shape mixture are reported and are not gated.
benchmark: >-
  Part 2: experiment 639's stored free and calibrated-explicit predictions at the same cutoff and
  seeds. Part 3: experiment 644's stored free network (arm a), calibrated explicit network (arm b),
  14-day tracker (arm d), and placebos, on 644's rows, plus climatology and the manual heuristic as
  the floors of 644.
row_set_id: >-
  Part 2: experiment 639's test windows (the 60 days after each cutoff, cut at 2026-07-01). Part 3
  and step 6: experiment 644's rows (every ensemble run in 644's test months at lead 0 to 240
  hours). Cleaned power, farm-gap rows dropped. Nothing dated 2026-07-01 or later is used.
metric: >-
  CRPS in percent of effective capacity (points) from 11 quantiles, on sunny middays and on all
  half-hours, by lead band; mean absolute error of the median forecast normalised by effective
  capacity; coverage of the 5% to 95% interval; tail pinball loss at 0.01, 0.05, 0.95, and 0.99
  for the Part 3 arm.
---

# Plan: is the ensemble's time step the reason the explicit solar term loses?

This page is the plan. The write-up replaces it when the fits have run.

**Part 1 has run, and it found no fault in the interpolation.** The experiment pipeline builds every
half-hourly ensemble shortwave value with the same resampler that production uses
(`ml_core.features._radiation`). On 15 runs between 2024-06 and 2026-06 the stored summary tables
and a fresh call of the resampler differ by 0.0 W/m². Each ensemble value is a mean over the window
that ends at its time stamp, and the half-hourly values keep that convention. Shifting the ensemble
series against CAMS (hourly values stamped at the end of the hour) gives the best correlation at a
shift of zero for all six Primaries (r between 0.966 and 0.986). The other CAMS stamp convention
would give a shift of two half-hours and a lower correlation. Rebuilding known weather from 3-hour
windows by production's method leaves a root-mean-square error of 5.3% of the CAMS midday peak
(6.6% at 6-hour windows) and a peak bias of -2.5% (-4.6%). Linear interpolation of the flux would
give 15.5% (30.8%) and step-holding 10.4% (19.5%). The numbers are in
`e00661_audit.py`'s printed output; the figures come with the write-up.

## Plan

- **Step 6 (CPU)** splits 644's explicit-minus-free contrast into the 3-hourly leads (day 1 to 6)
  and the 6-hourly leads (day 6 to 10) at 644's cutoffs, from 644's stored quantiles.
- **Step 5 (GPU, at most 1.5 GPU-hours)** degrades CAMS shortwave to 3-hour and 6-hour windows on
  the ensemble schedule, rebuilds half-hourly values with production's resampler, rebuilds the
  pretraining and test input matrices of experiment 639, and retrains the free and calibrated
  explicit networks on them. It starts with the cutoff 2026-06 and seeds 0 to 2, and extends to the
  other two cutoffs and seeds 3 and 4 if the first cutoff costs under 0.15 GPU-hours. Only shortwave
  is degraded, because the question is about shortwave; ERA5 temperature stays as it was.
- **Part 3 (GPU, at most 1.5 GPU-hours)** adds arm A to 644: the calibrated explicit network plus a
  small adapter that maps the ensemble inputs to a corrected clear-sky index before a differentiable
  physics curve, fine-tuned on the same 24 months of ensemble rows.

## Method

(To be completed with the write-up.)

## Files

(To be completed with the write-up.)
