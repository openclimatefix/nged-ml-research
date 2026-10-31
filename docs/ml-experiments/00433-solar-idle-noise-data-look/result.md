---
id: '00433'
slug: solar-idle-noise-data-look
title: >-
  Do the six metered solar farms show the wind farms' idle-noise texture when they are not
  generating?
theme: metered-solar-farms
also_relevant_to:
- generator-forecasting
status: planned
kind: other
size: >-
  simple. Stored: private tables and a private gallery under
  data/ml-experiments/00433-solar-idle-noise-data-look/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: one defensible design (a descriptive analysis of six
  series, no fit). Callers: this folder's scripts only, which read the stored power through
  `studies.power.scan_power` and the stored CAMS site points.
hypothesis: >-
  H0: the six metered solar farms read exact zeros, or smooth small values, when they are not
  generating, so the wind farms' idle-noise texture (exact zeros and repeats, quantised values, and
  a floor above zero that does not follow the season) is absent and no detector or cleaned-zero
  score is needed for solar.
prediction: >-
  Written before any night-time reading is summarised. The metadata lists all six farms in MW, not
  MVA, so a reactive-power floor is not expected. At least five of the six farms show no floor above
  zero (night median under 0.1% of capacity) and fewer than 20% of night readings that are
  non-zero repeats. The prediction fails for a farm if either part fails for it.
decision_rule: >-
  Definitions fixed before any number is read. Night: solar elevation below -6 degrees at the
  mid-point of the half-hour (pvlib, the farm's coordinates). Repeat: a reading equal to the
  previous half-hour's reading. Quantisation step: order the distinct night readings (rounded to 1e-6 MW), take the gaps between adjacent distinct readings, and choose the smallest gap such that readings above gaps no larger than it make up at least 1% of night readings; the lattice share is the share of night readings that are an integer multiple of that step. Night rows need the previous half-hour present, so repeats are defined. Floor: the median of night readings. Texture signature
  present at a farm when all four hold, each with a year-block bootstrap 90% interval (blocks are
  calendar years, 2,000 resamples) lying wholly on the stated side: (1) exact zeros plus non-zero
  repeats are at least 20% of night readings; (2) at least 80% of night readings sit on a lattice
  of the quantisation step; (3) the night median is at least 0.1% of capacity, or at least 5% of
  night readings are negative; (4) the summer (May to August) night median is within a factor of
  two of the winter (November to February) night median, so the floor does not follow the season.
  Report "the same noise" at a farm only when the signature is present. Report "not the same" (on point estimates) when
  at least 90% of night readings are exact zeros, or when fewer than 20% are zeros or repeats and
  no lattice of 80% exists. Anything else is "mixed", with the failing criterion named. Overall
  verdict `supported` (H0) if "not the same" holds at five or more farms, `refuted` if "the same
  noise" holds at three or more, `partly supported` otherwise. Daytime idle check (run-length test; maintenance can disable a farm for weeks and has no weekly cycle): a qualifying half-hour has CAMS clear-sky index at least 0.7 between 10:00 and 15:00 UTC; it is idle when the reading is under the farm's own night 95% value plus 2% of capacity. A day with at least 3 qualifying half-hours is an idle day when all its qualifying half-hours are idle; a run is consecutive idle days, skipping days with fewer than 3 qualifying half-hours. Report the run-length distribution, the 5 longest runs, and the share of idle qualifying half-hours in runs longer than 24 hours and longer than 7 days. The outage class is present at a farm when at least 50% of its idle qualifying half-hours lie in runs longer than 24 hours. The weekday-versus-weekend and hour-of-day profile of idle half-hours is a secondary check and decides nothing. If the noise is present,
  report the share of scored half-hours it touches (night half-hours inside the signature, as a
  share of all half-hours and of daylight half-hours). No threshold on gain applies, because no
  score is computed.
benchmark: >-
  The wind farms' idle floor in experiment 210 (exact zeros 19% to 41% and repeats 27% to 30% of the
  idle-band rows), quoted as stored; the wind data look has not merged.
row_set_id: none
metric: >-
  night-time shares of exact zeros, repeats, negative readings, and lattice readings; floor level by
  year; lag-1 autocorrelation of first differences; daytime idle run lengths
---

**This page is the plan; the write-up replaces it when the run finishes.**

## Plan

- **Question.** The maintainer believes the wind farms' MVA meters read noise, probably reactive
  power, while turbines are not generating. This experiment asks whether the six metered solar
  farms (Solar A to Solar F) show the same texture when not generating, so that the same detector
  and cleaned-zero scoring would apply.
- **Data.** Cleaned-or-raw choice: raw power from `studies.power.scan_power` for the six farms of
  `studies.pv_dataset.pv_sites`, because the cleaning flags drop the very zeros under study, and the
  count of rows each flag would drop is reported. Clear-sky index from the CAMS site points,
  hourly, paired with the half-hours ending at the hour's stamp and 30 minutes before. CAMS is a
  satellite retrieval available after the event, not at forecast time, and this experiment makes no
  forecast. Metering unit from the `units` column of the metadata table.
- **Search first.** Experiments 179 and 218 and `docs/ml-experiments/data-processing.md` are read
  for earlier statements on night-time and zero readings; experiment 179 found 0 to 30 night-time
  half-hours above 2% of capacity per farm and 0 to 24 negative readings.
- **Outputs.** Per-farm tables (night distribution, zeros and repeats, quantisation step, lag-1
  autocorrelation and rolling standard deviation of first differences, floor by year, negative
  readings by hour, summer versus winter floor, daytime idle run lengths, agreement
  between farms that share a CAMS cell), a private gallery of 20 farm-days per farm (10 sure-night
  days, 10 daytime days where the reading disagrees with the irradiance), and published figures that
  show only normalised, aggregated shares.
- **Standing practices.** Cut-off for short-history Primaries, cleaned-data, and retraining rules do
  not apply: no demand series is used and nothing is fitted. Gallery stays in the private data
  folder. Farms are cited by Solar A to F only, with no capacity in MW beside a label.
- **Compute.** CPU only, one core, low priority, no GPU.
- **Reviews.** Sonnet plan review, Sonnet code review before the first run, Sonnet results review,
  Sonnet write-up and prose review.
