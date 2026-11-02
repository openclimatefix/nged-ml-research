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
  Written before any number is read. A1: the midday window's area under the curve at floor 0 falls
  below 0.7 once the observed export-to-import ratio of the injected world passes 3, and the better
  shoulder window stays at or above 0.8 there; at power factor 0.97 the shoulder windows stay below
  0.8. A2: D12's observed ratio (81.6 over 15.6, about 5.2) lies where the midday window fails. A3:
  D12's shoulder dips fail the low-demand placebo (the sunny-to-cloudy ratio is below 2 or the dips
  occur only on days with low dark-hour readings), because D12's dark-hour daily minimum is below
  6.9 points on 20% of days. B1: at least half of D8's dip days have the engine detector flagging
  most of the dip half-hours. B2: the dip rate on cloudy days from March to October is under a
  fifth of the rate on sunny days at D22 and not at D8. B3: the boxcar-versus-solar shape test
  recovers injected drops of 40 to 50 points for at least 80% of injections; otherwise it stays a
  labelled diagnostic with no weight.
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

- **Rows.** Cleaned power (`scan_power`, flagged exact-zero half-hours dropped), the experiment 399
  start cut (D15 and D18, neither in the six nor the eight), July 2026 untouched, the 506 data layer
  and its sunny-day filter (clear-sky index 0.7 or more). Everything is recomputed from half-hours.
- **Sun position.** The clear-sky shape `u_cs` is the fleet shape `u` over the clear-sky index `k`
  (floored at 0.1), which removes the cloud and keeps the sun's path. The shoulder windows are the
  half-hours with `u_cs` from 0.05 to 0.3, split into morning (UK clock before 12:00) and evening.
  The midday window is 10:00 to 15:00 as in experiment 506, and the fourth window is the whole day.
  A day needs 3 shoulder half-hours, 6 midday half-hours, or 40 whole-day half-hours.
- **Injection.** signed = p - level x median|p| x farm shape, with levels 1, 2, 4, and 8 and the
  farm rotation of experiment 408 (Solar A to C), the median taken as in experiment 506. Real
  crossings (level 0) are reported without intervals. The observed export-to-import ratio of each
  world is the median clear midday reading over the median dark-hour reading from May to July (D12
  = 81.6 / 15.6), never a fitted capacity.
- **Arms and counts.** Two arms (floor 0 and power factor 0.97; floors 0, 2, and 4 give identical
  areas). Part A has 4 windows x 2 arms x 4 levels = 32 area-under-the-curve intervals plus 8 named
  contrasts (the better shoulder window minus the midday window, per level and arm): 40 intervals,
  plain 95% intervals with the count stated, and the 8 contrasts also at the 0.05/40 level.
- **Thresholds.** Youden-optimal per window and arm on the six, frozen to disk before the eight are
  read: at level 2.0 (comparable to experiment 506) and at the level whose observed ratio is
  closest to the Primary's own observed ratio (computed from raw readings, not from dips).
- **Placebos.** Sunny versus cloudy days (shoulder-window mean `k` below 0.3, same months and
  Primary), and the Primary's own dark-hour daily minimum on the same days. The decision rule above
  uses both.
- **Part B.** The 71 D8 and 13 D22 dip days are re-derived from the 506 definition. Tests: (1)
  shape within the day (drop against the 14 nearest non-dip days, clear-sky shape times the
  half-hourly index against a boxcar), calibrated on D10's real reverse-flow days and on injected
  boxcar and solar-shaped drops of 40 to 50 points; (2) dip rate on cloudy against sunny days from
  March to October, within the same year; (3) engine states (experiment 291's detector, flagged at
  probability 0.5) and spark spread (experiment 186) at D8; the neighbouring metered Primaries'
  change over the dip half-hours against placebo days, as a low-power diagnostic. The three causes
  are solar export, load transfer, and gas engine running. The D8 generator-output test and the
  growth comparison with experiment 370 are dropped. The count of dips outside NGED's labelled
  events is a stand-alone result.
- **Reuse.** No predictions are reused or fitted. Compute: CPU only, 2 cores. Cadence: not
  applicable. Past-weather caveat: the clear-sky index comes from CAMS and describes the day
  observed, so the sunny and cloudy day labels are not available at forecast time.
- **Advice followed.** Experiment 506 advice: treat the dip as a screen, not a signed label.

## Results

To be filled after the code review and the run.
