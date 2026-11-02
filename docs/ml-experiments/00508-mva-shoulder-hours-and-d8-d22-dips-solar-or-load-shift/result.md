---
id: "00508"
slug: mva-shoulder-hours-and-d8-d22-dips-solar-or-load-shift
title: >-
  Do dips of apparent power in the sun-position shoulders show reverse flow at D12, and are the D8
  and D22 midday dips solar output, load transfers, or a gas engine running?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
  - lsc-detectors
status: planned
kind: other
date: 2026-11-02
size: >-
  medium. Stored: small result tables under data/ml-experiments/00508-*/work/ (private). Serving
  path: not touched. Degradation rule: not touched. Designs: more than one statistic could test each
  question, so the plan fixes the windows, the arms, and the thresholds before the eight are read.
  Callers: reads the stored results of experiments 186, 291, 412, 506 and the 506 data layer. No
  model is fitted. Reviews: one Sonnet plan review, one Sonnet code review before the first run, one
  Sonnet results review, and two Sonnet write-up reviews.
hypothesis: >-
  (A) A strongly exporting Primary such as D12 crosses zero in the shoulders of the sun's path, so
  the minimum apparent power in the shoulder windows, calibrated on the six signed-MW Primaries
  with injected solar farms, separates sign-change days once export is several times the night
  import, and the midday window stops working at that export-to-import ratio. (B) The short midday
  dips at D8 and D22 are a mix of causes, and the gas engine behind D8 explains a large share of the
  D8 dips.
prediction: >-
  Written before any number is read. A1: the midday window stops being usable (area under the
  curve below 0.8, or a lower end below 0.6) in the ratio bin from 3 to 6 and above, while at least
  one shoulder window stays usable there at floor 0; at power factor 0.97 no shoulder window is
  usable in those bins. A2: D12's observed ratio (about 5.1) lies in a bin where the midday window
  is not usable. A3: D12's shoulder dips fail the placebo, because D12's dark-hour daily minimum
  is below 6.9 points on 20% of days: the lower bound of the sunny-to-cloudy ratio is not above
  1.5, or fewer than 3% of the sunny days without a dark-window dip show a shoulder dip. B1: the
  engine detector flags more of the D8 dip half-hours than of the matched placebo half-hours, and
  at least half of the dip days have most dip half-hours flagged. B2: the cloudy-to-sunny
  Mantel-Haenszel ratio of dip rates has an upper bound under 0.2 at D22 and not at D8. B3: the
  shape test labels at least 80% of injected boxcars and at least 80% of injected solar-shaped
  drops correctly; otherwise it carries no weight in the cause classes.
decision_rule: >-
  A window counts as usable at an export-to-import ratio if its area under the curve is at least 0.8
  with a month-block interval whose lower end is above 0.6 (as in experiment 506). D12's shoulder
  dip is consistent with reverse flow only if all three hold: the area under the curve of a
  shoulder window at D12's ratio is usable, the sunny-to-cloudy ratio of the shoulder dip rate has a
  lower 95% bound above 1.5, and at least 3% of sunny days with no dark-hour reading below the
  threshold still show a shoulder dip. A D8 or D22 dip day is called solar if the clear-sky shape
  explains the drop better than a boxcar, the cloudy-day dip rate is under a fifth of the sunny
  rate, and no engine state is flagged; an engine-running dip is neither solar nor a load transfer.
  Where the three tests disagree, the page says the evidence cannot tell and writes the question to
  NGED. No gain in forecast score is claimed, because no forecast is fitted. The threshold for
  porting is not applicable: nothing here enters production.
benchmark: >-
  Experiment 506 (the 10:00 to 15:00 window: area under the curve 0.995 at real crossings, 0.916 at
  injection level 2.0 at floor 0), its audit, and experiments 186 and 291 (engine states and spark
  spread at D8).
row_set_id: not_applicable_no_forecast
metric: >-
  Area under the curve of the dip statistic per window, arm, and injection level; shoulder dip rate
  on sunny days, cloudy days, and dark hours; shares of dip days by cause; each with a month-block
  bootstrap interval (20,000 resamples).
code: ml-experiments/00508-mva-shoulder-hours-and-d8-d22-dips-solar-or-load-shift/
production_status: not-for-production
adopted_in: none
---

**Plan (status `planned`).** Experiment 506 tested the dip of apparent power only in a fixed 10:00
to 15:00 window and left two items open. This experiment follows the brief
`mva_shoulder_hours_and_d8_d22_dips_brief.md` with its edits of 2026-11-02. No model is fitted.
Counts of events are lower bounds, because NGED's event labels are incomplete. Interim and
unaudited until the reviews and the audit have run.

## Plan

The first version of this plan took `u` over the clear-sky index as the clear-sky shape. A Sonnet
plan review (16 findings, `briefs/508_plan_review.md`) showed that `u` is computed from the actual
sky, so the plan below replaces that choice and the choices that depended on it.

- **Rows.** Cleaned power (`scan_power`, flagged exact-zero half-hours dropped), the experiment 399
  start cut (D15 and D18, neither in the six nor the eight), July 2026 untouched, the experiment 506
  data layer (folder `ml-experiments/00506-*`). Part A uses April to September only, so that the
  midday, shoulder, and whole-day windows describe the same days.
- **Sun position.** The shoulders are the half-hours in which the sun's elevation (pvlib, at the
  half-hour midpoint) lies between the two elevations at which the clear-sky fleet shape `u` of
  experiment 412 reaches 0.05 and 0.3 on rising-sun half-hours with a clear-sky index of 0.9 or more
  at D12. Morning is rising sun, evening is setting sun. The midday window is 10:00 to 15:00 UK
  time, and the fourth window is the whole day.
- **Qualifying days and one label.** A day qualifies when every window has enough finite
  half-hours (6 midday, 3 per shoulder, 40 whole-day), so every window's area under the curve uses
  the same days and any sky. The label is the same for all windows: the day's signed power is both
  below and above zero while the sun is above the horizon.
- **Injection.** signed = p - level x median|p| x farm shape, levels 1, 2, 4, and 8, farm rotation
  of experiment 408 (Solar A to C), the median taken over sunny middays as in experiment 506. The
  recorded farm shape is zero below 15 degrees of elevation, which is where the shoulders sit, so
  there the farm shape is the site's own `u` times the median ratio of the farm shape to `u` where
  `u` is 0.3 to 0.5. The share of half-hours extended this way is printed.
- **Ratio bins.** The export-to-import ratio is the median clear midday reading over the median
  dark-hour reading (sun more than 6 degrees below the horizon), May to July, from each arm's own
  reading in each injected world, never a fitted capacity. Worlds are pooled into the bins 0 to
  1.5, 1.5 to 3, 3 to 6, and above 6, per arm, and the area under the curve is computed per bin.
  D12's observed ratio (about 5.1) is read off the bin that contains it.
- **Arms, counts, and what counts as usable.** Two arms (floor 0 and power factor 0.97; floors 0, 2,
  and 4 give identical areas). Bonferroni families, counted at the end of the run: 32 area-under-
  the-curve intervals (4 bins x 4 windows x 2 arms), 16 named contrasts (each shoulder minus
  midday, per bin and arm, named in advance with no choice of the better shoulder), and 16 placebo
  ratios (8 Primaries x 2 shoulders), 64 in all, plus the Part B tests counted separately. The
  decision rule uses plain 95% intervals on purpose; the named contrasts are also shown at the
  0.025/64 level. A cell with fewer than 30 crossing or 30 ordinary days, or fewer than 10 month
  blocks holding both, is reported as not estimable. Every area carries its crossing and ordinary
  counts, the number of month blocks, and every contrast carries 2.8 standard errors as its minimum
  detectable effect. A window is usable at a ratio if its area is at least 0.8 with a lower end
  above 0.6 (as in experiment 506); the same criterion is used throughout.
- **Thresholds.** Youden-optimal per bin, window, and arm on the six, frozen to disk before the
  eight are read. Each of the eight uses the threshold of the bin that holds its own observed
  ratio; because thresholds for every bin are computed from the six alone, reading the eight's
  ratios only selects a row. For the eight, the power factor 0.97 arm is the matched one and floor
  0 a lower bound, because the meters report apparent power.
- **Placebos.** Sunny days (mean clear-sky index of the window 0.7 or more) against cloudy days
  (below 0.3) for each Primary and shoulder, with the ratio of the dip rates (0.5 added to each
  count) and a month-block interval, and the Primary's own dark-hour reading over a window of the
  same number of half-hours immediately on the dark side of the shoulder. Cloudy-day maintenance
  patterns are a confound that the placebo cannot remove, and the page says so.
- **Part B.** A dip day is a sunny day (clear-sky index 0.7 or more) whose 10:00 to 15:00 minimum
  reading is below 6.9 points of capacity (experiment 506). The run stops unless it finds 71 dip
  days at D8, 13 at D22, and 13 at D12. Cause classes are solar export, load transfer, and gas
  engine running. Engine running is scored against matched placebo days (the same half-hours on the
  14 nearest same-day-type non-dip days after 2023-08-15) with D5, D10, and D11 as controls, using
  experiment 196's engine states as stored by experiment 291, and cannot be assessed at D22. The
  shape test fits one amplitude to the drop (baseline of the 14 nearest non-dip days minus the
  reading), once with the fleet shape `u` and once with a boxcar over the half-peak run around the
  deepest half-hour; injected boxcar and solar-shaped drops of 40 to 50 points on D8 and D22 days
  before 2023 give the known answer, and the shape test enters the cause classes only if at least
  80% of each injected class is labelled correctly. The dip rate on cloudy against sunny middays of
  March to October is a Mantel-Haenszel risk ratio over year-month strata with a stratum bootstrap;
  with few cloudy dips it is reported as underpowered. Spark spread (experiment 186) compares dip
  days with other sunny days after 2023-08-15. The neighbour check reads the other Primaries under
  the same supply point and is a low-power diagnostic. The count of dip days outside every labelled
  event with times is a stand-alone result.
- **Prediction and rule changes.** Prediction A1 and the usable criterion now share one threshold
  (area 0.8, lower end above 0.6). The decision rule for D12 requires: a usable shoulder area in the
  bin that holds D12's ratio; a sunny-to-cloudy ratio of dip rates with a lower bound above 1.5
  (sensitivity at 1 and 2 reported); and, among sunny days with no dark-window reading below the
  frozen shoulder threshold, a shoulder dip share of at least 3% (sensitivity at 1% and 2%).
- **Deliverables.** Part C of the brief (what each answer changes, using "consistent with") and the
  exact question for NGED about what changed at D8 and D22 from 2023 to 2026, including the engine,
  new connections, and load transfers.
- **Reuse.** No predictions are reused or fitted. CPU only, 2 cores. Cadence: not applicable.
  Past-weather caveat: the clear-sky index comes from CAMS and describes the day observed, so the
  sunny and cloudy day labels are not available at forecast time. Advice followed: experiment 506
  says to treat the dip as a screen, not a signed label.

## Results

To be filled after the code review and the run.
