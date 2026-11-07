---
id: "00616"
slug: lag-free-network-pv-capacity-probe-on-stored-weights
theme: solar-disaggregation
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
size: >-
  medium. Stored: small private result files only. Serving path: not touched. Degradation rule: not
  touched. Designs: the probe has more than one defensible form, so its regression, its donors, and
  its two placebos are fixed below before any capacity is read. Callers: this folder's scripts, which
  read experiment 565's and experiment 554's stored weights and the helpers of experiments 269 and
  448 without editing them. Asset graph: not touched. No fit and no GPU.
hypothesis: >-
  A stored lag-free neural network that was pretrained on ERA5 and CAMS estimates of past weather has
  learnt how net demand at a Primary falls when the sky clears, with the calendar held fixed. A
  regression of the network's probed response on experiment 269's unit PV curve, over many coherent
  donor days near the probe date, therefore returns an effective alternating-current (AC) PV capacity
  at each of the six MW Primaries that is stable across seeds and close to the estimates of experiments
  474 and 479.
prediction: >-
  Fixed from the Opus design review of 2026-11-07 before any capacity was read. (1) The standard
  deviation of the capacity across the 10 seeds is under 0.3 MW at 4 or more of the 6 Primaries. (2)
  The response falls monotonically in the unit curve at all 6. (3) The hour profile of the
  sunny-minus-dull contrast correlates 0.9 or more with the PV unit curve's. (4) The capacity lies
  within 0.6 to 1.4 times experiment 479's estimate at 4 or more of the 6. (5) Both placebos (a
  profile-only sky and a shuffled sky) read under 0.20 of the real capacity. (6) The capacity moves
  by less than its interval when donors are limited to those within 2 degrees Celsius of the median
  temperature.
decision_rule: >-
  Fixed before any capacity is read. The probe is called usable when predictions (1), (2), and (4)
  hold. It is called not usable when the seed standard deviation exceeds 0.5 MW at 3 or more
  Primaries, or the response is non-monotone at 3 or more, which are the stop rules of the design
  review. Step 2 of the review's plan (the injected known-answer test, 2 to 4 GPU-hours) is
  recommended only if the probe is usable and both placebos pass. No gate is placed on agreement with
  experiments 474 and 479, because both are themselves unvalidated; their numbers are reported
  beside the probe's. The threshold for acting on a probe capacity in production is not set here,
  because nothing is adopted.
benchmark: >-
  Experiment 474's rung R4 with cloud terms (median AC capacity by season-year with 10th and 90th
  percentiles) and experiment 479's anomaly regression with cloud terms (5th to 95th percentile).
row_set_id: >-
  Not a forecast score. Probe dates are 12 Tuesdays from 2025-04-15 to 2025-09-16 and 3 Tuesdays in
  April and May 2026; no input is dated 2026-07-01 or later. Donors are the same half-hour of every
  day within 21 days of the probe date's day of year, in every year of the fold's pretraining window.
metric: >-
  Effective AC capacity in MW per Primary, per probe date and per summer pooled; seed standard
  deviation; day-block bootstrap interval; placebo share of the real capacity.
production_status: not-for-production
adopted_in: none
---

# Probing a lag-free neural network for unmetered PV capacity, on stored weights

*Planned. The write-up replaces this page when the probe has run.*

The probe holds every datetime input of a stored neural network fixed, swaps in coherent weather from
donor days, and reads how much the network's net-demand forecast falls as the sky clears. Step 1 of
the Opus design review (2026-11-07) asks for this on the stored weights only: no fit, no GPU.

**Development notes (not part of the pre-registration).** The pipeline was debugged on the 2026-06
fold with one run that averaged the seeds and fitted the temperature variant only. That run showed
capacities of the same size as experiments 474 and 479 (between 1.5 and 4 MW) and was used to find a
missing-row bug. All thresholds above come from the design review and were not changed afterwards.
