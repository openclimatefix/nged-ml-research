---
id: "00698"
slug: gas-engine-state-tracker-part-5-does-pv-harm-the-disaggregation
title: >-
  Does solar generation behind D8's meter make the tracker's labels of the gas engine worse, beyond
  the harm from the lower level of the reading?
theme: gross-demand-simulation
also_relevant_to:
  - differentiable-physics-embedded-generation
  - solar-disaggregation
status: planned
kind: other
size: >-
  Anything else (no forecast; new analysis code; no new data). Stored: one file per tracker run
  (at most 72 runs on injected readings, of which the curve gate removes some, and 10 runs on D8's
  own readings), the audit tables, and the score tables, all private. More than one design is
  defensible (which capacities, which solar curves, which hosts), so the choices are fixed below.
  Imports: this folder's scripts import experiment 694's trackers and loaders, experiment 695's grid
  helpers, and experiment 697's engine draw, intervals, and local-time helpers. The solar-farm
  loader of experiment 405 is copied here, because importing it would need `torch`.
hypothesis: >-
  The weather-driven variation of the solar generation behind D8's meter makes the tracker's
  daylight labels of the gas engine worse, beyond the harm that the lower level of the reading and
  the fold of the reading at zero cause.
prediction: >-
  (P1) At a host whose curve of reading against solar output matches D8's, with an engine that has
  no pattern injected, a run that carries the half-hourly output of one real solar farm scores
  worse from 10:00 to 15:00 local time than its smooth twin (the same capacity times the farm's
  own average curve of that month and half-hour, which has the farm's level and daily shape and no
  weather), by at least one harm criterion: F1 lower by 0.05 or more, the false-flag share on
  half-hours with no injected output higher by 0.02 or more, or the bright-minus-dull flagged share
  higher by 0.05 or more. The same capacity shows the harm at D8's reading from before the engine
  and at D3.
decision_rule: >-
  Each contrast is paired (same host, same seed, same injected engine) with a 95% interval from 2,000
  resamples of whole ISO weeks, one draw of weeks for every run of a host. A contrast meets a
  criterion when its point estimate reaches the threshold and its interval excludes zero on the
  harmful side. Tested in this order. Inconclusive (the audit blocks the test): D8's reading from
  before the engine fails the curve gate with the farm shape at both capacities, or its solar-free
  twin does not reproduce a midday F1 of 0.767 within 0.02, or the paired half-width of the F1 of
  the farm weather contrast at D8 exceeds 0.05 (the other contrasts do not enter), or the farm
  series covers under 90% of the daylight half-hours of a host that runs (D8 before the engine, and
  D3 when D3 passes the farm gate). Supported: at one capacity the farm weather contrast meets a
  criterion at D8's reading from before the engine and at D3 (at every capacity that ran at D8
  alone, if D3 fails the gate). Partly supported: any one of the 3 criteria at any capacity that
  ran is met by the farm weather contrast at D8 without the conditions of the supported branch, or
  by the CAMS weather contrast at D8. Refuted: at D8, for every capacity and shape that passed the
  curve gate (at least one farm contrast and one CAMS contrast), every weather contrast has an
  interval wholly inside its threshold. Not supported: no weather contrast meets a criterion and
  the refuted branch does not hold. A run on D8's own off-spells can lower `supported` to `partly
  supported` and cannot raise a verdict. The level contrast is reported in its own sentence and
  never changes the verdict word. No smallest worthwhile gain applies, because no forecast is
  scored and no production code changes.
benchmark: none (no forecast is scored)
row_set_id: none
metric: >-
  F1, precision, recall, false-flag share, and bright-minus-dull flagged share, 10:00 to 15:00 local
code: ml-experiments/00698-gas-engine-state-tracker-part-5-does-pv-harm-the-disaggregation/
---

**This file is a plan in the shape of a planned `result.md`, and nothing in it has been fitted,
tracked, or scored.** The experiment asks whether solar panels behind the meter of D8 (a primary
substation) make the tracker of
[experiment 694](../00694-gas-engine-state-tracker-part-1/result.md) label the gas engine behind the
same meter worse. The tracker infers from D8's half-hourly reading how many units of the engine run.
The engine record we hold is called Gas A here, and the page shows it only inside scores pooled over
a whole run.

**The experiment is a diagnostic: known solar output and a known engine are injected into real
readings, and the scores show which part of the solar output does any harm.** Three solar curves
are injected at the same capacity. The first is the output of one real solar farm, the second is the
irradiance of CAMS (the Copernicus Atmosphere Monitoring Service's satellite-derived product) at
the host's own location turned into output, and each has a smooth twin that carries the same
level and the same daily and seasonal curve but no weather. The farm's smooth twin is built from
the farm's own average output in each month and half-hour, and the CAMS twin from the CAMS average.
The differences between the runs separate the harm from the lower level and the fold of the reading
(the CAMS smooth twin minus a solar-free twin) from the harm from weather variation (a farm run
minus the farm's smooth twin, or a CAMS run minus the CAMS smooth twin). The **fold** is the
reflection of the reading at zero: the meter reads the size of the net power and not its sign, so
once solar and engine output exceed the demand behind the meter, the reading rises again.

**D8's history has two periods that the page calls eras.** The **net era** is the period before
solar generation began on the 11 kV network around D8, and the **solar era** is the period after it.
Host runs use D8's reading from before the engine (so with no solar generation behind it) and D3's
reading in D3's solar era.

## What the stored labels show, seen before the thresholds were set

**Three label-only checks show that D8's labels in the solar era behave differently from the labels
in the net era, and none of the three shows that solar generation is the cause.** All use the smoothed
labels of experiment 694's tracker, with 95% intervals from 1,000 resamples of whole weeks, and
`e00698_audit.py` rebuilds each one. The checks are the flagged share on weekends and bank holidays
from 09:00 to 15:00 with a negative spark spread, the flagged share at night, and the flagged share
in the brightest third of CAMS irradiance minus the dullest third, from 10:00 to 15:00. The
experiment does not use D8's own label statistics as a test, because every threshold below was set
after they were seen.

**The thresholds were set after seeing numbers, so a verdict rests on a rule fixed before the first
real run and not on an independent test.** The two thresholds of 0.05 (F1, and the
bright-minus-dull flagged share) were set after seeing D8's bright-minus-dull contrast and the
stored controls of experiment 697. The threshold of 0.02 (false-flag share) was calibrated on
experiment 697's level-matched control, which scores 0.020. The two capacities of 7.0 MW and 8.9 MW
and the gate of 1 MVA were set after seeing a table of the median reading by fifth of solar output,
read from the stored files. The capacity of 8.9 MW is the 8.91 MW register row that
[experiment 673](../00673-pv-capacity-for-nged-per-primary-with-register-cross-check/result.md)
published.

## Order of work

- **Step 1, audit and gate, before any tracker run.** A host and capacity runs only if its
  solar-free curve of median reading against solar output lies within 1 MVA of D8's solar-era curve
  in each of the first four fifths, using one set of fifth edges for every series. The gate is
  evaluated for each shape, so a host can run with the farm shape and not the CAMS shape. The
  audit writes the gate only when every check passes, and the job script runs only after a passing
  audit that is newer than the audit's inputs.
- **Step 2, solar curves per host and capacity,** as described above: the farm, its smooth twin,
  the CAMS curve, and the CAMS smooth twin.
- **Step 3, host runs.** The hosts are D8's reading from before the engine and D3's reading in
  the solar era, each with 3 seeds of a patternless engine (experiment 697's day draw of Gas A) and
  one run with no engine. The solar-free twin of D8 is experiment 697's control C1.
- **Step 4, runs on D8's own long off-spells.** Gas A's days are injected into whole local days
  inside spells of 48 half-hours or more that the stored labels call off, 5 seeds in each era.
  Whole local days inside those spells give 57 days in the net era and 16 days (768 half-hours) in
  the solar era, and the audit asserts both counts. At that size the half-width of the recall
  contrast almost certainly exceeds the 0.10 that the verdict requires, so step 4 almost surely
  cannot lower a verdict.

## Limits of the rule, stated before the run

- **The farm's brightness is measured elsewhere.** The thirds of the bright-minus-dull criterion
  are thirds of the host's CAMS shape, while the farm's cloud is at another site. The farm weather
  contrast therefore has a weak link to those thirds, and its bright-minus-dull criterion is close
  to zero by construction. The audit prints the correlation of the farm series with the host's CAMS
  shape within each month and half-hour, and the verdict script reports the statement "weather
  explains D8's swing" for the CAMS contrast beside the farm contrast.
- **The partly supported branch fires on any one of up to 12 contrasts** (3 criteria, 2
  capacities, and 2 shapes). Its false-positive rate is higher than that of the supported branch,
  which needs the same capacity to meet a criterion at 2 hosts.
- **The refuted branch reads every capacity and shape that passed the curve gate at D8.** The
  curve gate removes some capacity and shape pairs, so the reading with all four pairs (both
  capacities, both shapes) could never hold. The verdict script prints a line with that reading
  beside the verdict.
- **Branch conditions read the farm shape.** The inconclusive branch asks about the farm shape at
  D8 and the half-width of the farm weather contrast only. A farm weather contrast that meets a
  criterion at both capacities of D8 and not at D3 returns `partly supported`.
- **The coverage clause counts daylight half-hours.** Night half-hours always count as finite
  farm values, so a share over all half-hours overstates coverage. The clause counts only
  half-hours with positive CAMS shape at the host, and the audit also prints the share of midday
  half-hours and the share of the dullest third that the farm's gaps mask. Farm gaps are filled
  with the farm's cell mean and masked out of every score.
- **The solar-era weekend cheap-spark share is 0.188 under the audit's row definition.** The rows
  are the observed half-hours that are not on a clock-change day, with the drops kept, and the
  figure is the one quoted.

## Anonymisation

**No page, figure, or printed file names Gas A or the solar farm, dates either, or shows either's
output.** The farm keeps experiment 662's letter. Dated rasters of an injected series stay in the
private data folder, and no SVG text may hold a four-digit year.

## Files

The Files section is written in the pull request that adds the finished write-up.
