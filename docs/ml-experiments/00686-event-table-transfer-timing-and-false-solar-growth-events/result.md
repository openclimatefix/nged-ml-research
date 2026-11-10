---
id: "00686"
slug: event-table-transfer-timing-and-false-solar-growth-events
title: >-
  Do the donor and the recipient of a load transfer start together, and what kind of false event does
  solar growth cause in the event table?
theme: lsc-detectors
also_relevant_to: []
status: planned
date: 2026-11-10
kind: other
size: >-
  medium. (1) Stored: private fit tables and one table of decoded false events under
  data/ml-experiments/00686-event-table-transfer-timing-and-false-solar-growth-events/ only.
  (2) Serving path: not touched. (3) Degradation rule: not touched. (4) Designs: more than one is
  defensible for the timing model (the shape of the step, the window, the way the fit difference is
  calibrated) and for the classes of false event, so the plan fixes one of each before any score is
  read and reports the sensitivity to the step shape. (5) Callers: the scripts import the folders
  of experiments 653, 657, and 660 by path; nothing imports this folder. CPU only, 2 cores.
hypothesis: >-
  (1) At the two verified transfers the recipient's step begins hours after the donor's, so a
  detector that requires both onsets to coincide would reject real transfers. (2) Most of the false
  events that experiment 653's masked arm decodes under injected solar growth are slow ramps or
  misfits of the slow level, which a better slow level would remove, and fewer than a third are
  steps inside a run of sunny days.
prediction: >-
  Written before any score was read. P1 (timing): at both verified transfers the best separate
  onsets of the donor and the recipient differ by 12 hours (24 half-hours) or more, the shared-onset
  model fits worse by a likelihood ratio of 4 or more, and the day-block bootstrap rejects a shared
  onset at the 5% level. P2 (method check): when a transfer with the fitted sizes is injected
  simultaneously into 60 other event-free windows of the same two Primaries, the fitted onset gap is
  within 4 half-hours in at least 90% of the windows, for both a constant step and a step that
  follows the donor's daily profile, so a gap of this size in the real data is not made by the
  method. P3 (false events): half or more of the false solar-growth events are slow ramps or
  slow-level misfits, and under a third are steps inside a run of sunny days.
decision_rule: >-
  Fixed before any score was read. P1 is supported if the gap is 12 hours or more, the likelihood
  ratio is 4 or more, and the bootstrap p-value is under 0.05 at both transfers, partly supported if
  that holds at one transfer, and not supported otherwise. If the shared onset is within noise at
  both transfers (likelihood ratio under 4), the page reports that the data support simultaneity
  with a tolerance of the joint onset interval. P2 gates P1: if P2 fails, the page says that the
  method can produce a lag that is not real and does not report a gap as a finding. P3 is supported
  if both of its halves hold, and the page then says that the composite capacity curve could remove
  the first group but not the second. If more than half of the false events are sunny-run steps,
  P3 is not supported and the page recommends against building the trajectory for false events. The
  porting threshold is zero, because no production code changes. Data stop before 2026-07-01, and
  the final-test window is never read.
benchmark: >-
  Experiment 660's separate fits (recipient 14.5 and 15 hours after the donor) for P1; experiment
  653's 0.93 false events per injected Primary-year in the masked arm for P3.
row_set_id: >-
  experiment 258's grid of 15 Primaries, half-hourly, 2022-01-01 to 2026-06-30, cleaned power;
  not comparable with the leaderboard
metric: >-
  Onset gap in half-hours with a day-block bootstrap interval; likelihood ratio of separate against
  shared onsets with a bootstrap null; share of simulated simultaneous transfers recovered within 4
  half-hours; counts and Wilson-interval shares of false events per class.
code: ml-experiments/00686-event-table-transfer-timing-and-false-solar-growth-events/
---
Planned. The write-up replaces this body when the experiment finishes.

## Plan

**Check 1 asks whether the two ends of a verified transfer start together.** The residual of both
Primaries (cleaned power minus the lag-free forecast of experiment 327's arm C) is fitted in a
window of 5 days either side of the nominal edge. Each Primary gets a level, a trend, and a step
with a ramp of 0 to 4 half-hours. The step is constant, half the donor's daily profile, or all of it.
One model gives each Primary its own onset, and the other forces one shared onset. The same is done
for the end of the event. A day-block bootstrap under the shared model gives the null distribution
of the fit difference.

**Check 1 also measures what the method can do.** A transfer with the fitted sizes is injected
simultaneously, and with a 29-half-hour lag, into 60 event-free windows of the same two Primaries, and
the identical fit reads the gap back.

**Check 2 re-runs experiment 653's solar-growth worlds with 24 seeds instead of 2.** Worlds 0 and 1
reuse 653's seeds and must reproduce its 5 and 9 false events in the masked arm. Each false event
is assigned to the first matching class: a step inside a sunny run (edge width 1 hour or less, and the
midday ERA5 clearness index over the first 3 days at least 0.7), a slow ramp (edge width a day or
more, or an edge that does not complete within 2 days), a misfit of the slow level (the injected
change over the first 3 days differs from the slow level's change by 0.5 noise units or more), and
otherwise another step. Shares carry Wilson intervals, and the false-event rate per Primary-year
carries a bootstrap interval over worlds. The solar-growth term is injected at 3 of 15 Primaries per
world over 2.5 years.

**Standing practices.** No forecast is trained or scored, so monthly refits, saved models, and
tail scores do not apply. The cleaned-data rule applies through experiment 327's cleaned power,
and D18's rows before 2025-03-01 are dropped as in experiment 653. The clearness index uses ERA5
shortwave, a recorded weather product available only after the fact, which suits a hindsight
detector. The final-test window from 2026-07-01 is never read.
