---
id: '00433'
slug: solar-idle-noise-data-look
title: >-
  Do the six metered solar farms show the wind farms' idle-noise texture when they are not
  generating?
theme: metered-solar-farms
also_relevant_to:
- generator-forecasting
status: done
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
verdict: supported
best_arm: none (data look, no fitting)
headline: >-
  Under the pre-registered rule five of six metered solar farms are "not the same" as the wind
  farms: 95.8% to 99.9% of their night readings are exact zeros. Solar B alone meets the rule's
  "the same noise" label, through 8.1% negative night readings (median -0.010 MW across all its negative readings) with a median of
  zero, which the post hoc reading calls a small import, not a noisy floor.
take_home_message: >-
  The metered solar farms read exact zeros at night, unlike the wind farms' noisy floor, so solar
  forecasts need no idle-noise detector and no cleaned-zero scoring. One farm (Solar B) shows small
  negative night readings on 8% of night half-hours, and a few farms sit idle for 2 to 3 weeks on
  clear days (Solar D and Solar E), which is an outage question, not a noise question.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Do not build a solar idle-noise detector or score solar forecasts against cleaned zeros; night
    readings are 95.8% to 99.9% exact zeros at the six metered farms (summary.csv).
  - >-
    Treat negative night readings at Solar B (8.1% of night half-hours, median -0.010 MW) as a small
    import, and keep them in scoring unless a later check finds a reason (Figure 2).
  - >-
    Test the outage question for Solar D and Solar E separately: their longest full idle days run 24
    and 17 days on clear-sky days (daytime_runs.csv).
---
**The six metered solar farms read exact zeros at night, so they do not show the wind farms' idle
noise.** The question was whether the wind farms' noise while not generating, which the maintainer
suspects is reactive power on an MVA meter, also appears at the metered solar farms. It does not. The
metadata lists all six farms in MW, not MVA, and 95.8% to 99.9% of their night readings (solar
elevation below -6 degrees) are exact zeros. The wind farms' idle floor in
[experiment 210](../00210-wind-idle-noise-mva-artefact/result.md) held 19% to 41% exact zeros and a
median above zero at the two MVA farms. This narrows the next step: no solar detector, and no
cleaned-zero scoring for solar, unless the outage check below finds a reason.

![Figure 1: composition of night readings per farm](figure_1.svg)

**Solar B meets the pre-registered "the same noise" label, but the label is a flaw in the rule.**
The rule counted exact zeros towards criterion 1, so any farm that is nearly all zeros passes it,
and Solar B's 91.8% zeros also satisfy the "not the same" clause (at least 90% zeros). The rule
listed "the same noise" first. Solar B's other readings are 8.1% negative (interval's upper end
10.9%), and across all hours of the record its negative readings have a median of -0.010 MW and a minimum of -0.192 MW; its night median is zero.
Read post hoc, that is a small import (station load), not a floor above zero. The year-by-year
negative share moves between 0.1% in 2019 and 15.6% in 2023 (Figure 2), which a reactive-power floor
would not do. Solar E shows the same pattern at 4.1% negative.

![Figure 2: negative night readings by year](figure_2.svg)

**Idle runs on clear days are short, except at Solar D and Solar E.** Of the half-hours with CAMS
clear-sky index at least 0.7 between 10:00 and 15:00, 2.8% to 14.4% read near the night floor (3.2%
at Solar F). Whole idle days form runs of at most 4 days at Solar A, B, C, and F, 24 days at Solar D,
and 17 days at Solar E. No farm has a run longer than 24 days, and no farm has the pre-registered
outage class (at least 50% of idle half-hours in runs over 24 hours; the highest share is 41.6%, at
Solar E). Most idle half-hours lie outside whole idle days.

![Figure 3: idle half-hours in whole idle days, by run length](figure_3.svg)

## Checks

- **Prediction.** Night median under 0.1% of capacity and non-zero repeats under 20%: holds at all
  six farms (prediction asked for five).
- **Pre-registered labels.** Solar A, C, D, E, and F "not the same"; Solar B "the same noise"
  (flawed rule, above). Overall verdict `supported` (five or more "not the same").
- **Texture statistics** (`summary.csv`): non-zero repeats are 0.01% to 0.14% of night readings. The
  quantisation step is 0.001 MW at Solar A, B, and E, 0.011 at C, 0.024 at F, and 0.035 at D, so each
  farm reports on the meter's own lattice, zero included. Lag-1 autocorrelation of night first
  differences is -0.51 to -0.03. Median rolling standard deviation of first differences is zero at
  every farm.
- **Season.** Summer and winter night medians are both zero at every farm, so the season criterion
  holds trivially (`season.csv`). Zero shares differ by under 1 point between seasons.
- **Flags.** The cleaning table flags 0 night half-hours at these six customer-metered series. The
  night statistics use all rows.
- **Share touched.** Where the label was "the same noise" (Solar B), night half-hours are 39.3% of
  all half-hours and 71.9% of half-hours with the sun up; the negative ones are 8.1% of those.
- **Daytime idle** (`daytime_runs.csv`, `daytime_run_bins.csv`): a day needs at least 3 qualifying
  half-hours. The run test sees only whole idle days; partial-day idling is counted in the idle
  share and not in a run.
- **Agreement between farms** (`pair_agreement.csv`): of the three pairs within 10 km, Solar A and
  Solar C share 5 idle days of 20 on either (Jaccard 0.25); the others share none. Idle days are
  mostly farm-specific, not weather.
- **Weekday profile** (`daytime_weekday_hour.csv`): a secondary table; it decides nothing.
- **What the data cannot show.** No MVAr channel, outage log, or curtailment record exists, so idle
  days cannot be told from curtailment or a station fault. CAMS clear-sky index is hourly and
  satellite-derived, so cloud inside the hour can make a "clear" half-hour dark. Rolling statistics
  are degenerate when readings are all zero. Gallery PNGs (20 farm-days per farm) are in the private
  data folder, not published.
- **Audit triggers:** the result contradicts nothing it predicted; the one flawed label (Solar B)
  is reported as pre-registered and read post hoc. No control or leave-one-out check applies.

## What to try next

- Check Solar D's 24-day and Solar E's 17-day idle runs against NGED's records for outages or
  export limits.
- Decide Solar B's small negative readings with NGED (station import), only if the solar scores
  show a farm-level bias at night.
