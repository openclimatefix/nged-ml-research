---
id: "00653"
slug: edge-state-event-table-penalised-segmentation
title: >-
  Can a penalised segmentation of the Primaries' residuals build a table of load-shift events with
  partners, start times, and end times?
theme: lsc-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
kind: other
size: >-
  medium. (1) Stored: private residual panels, event tables, and score tables under
  data/ml-experiments/00653-edge-state-event-table-penalised-segmentation/ only. (2) Serving path:
  not touched. (3) Degradation rule: not touched. (4) Designs: more than one is defensible for the
  slow level, the penalty, and the event generator, so each is fixed below before any score is
  read, and the slow level has two comparison arms. (5) Callers: this folder's scripts, which read
  the stored arrays of experiments 258, 327, and 580 and import the folders of experiments 130,
  254, and 327 for the grid and the matching rules; nothing imports them. The maintainer's brief
  limits the work to CPU, 2 cores, and no GPU.
hypothesis: >-
  A penalised joint segmentation of the 15 Primaries' residuals from a demand model with no lagged
  power can recover load-shift events (an opposite-sign partner at an electrically connected
  Primary, a start, and an end) better than the single-series detectors already built, because it
  weighs four explanations for each step against one another: an edge transfer, a level shift with
  no partner, a routine step, and a ramp. A slow level that is seasonal, fitted with candidate
  events masked, and then frozen keeps long events from being absorbed into the baseline.
prediction: >-
  Written before any score is read. P1: on injected events of at least 4 noise units that last at
  least 1 day, in held-out years, the MW-weighted recall of edge transfers is 0.70 or more at 1 or
  fewer unconfirmed edge events per edge-year on the real held-out record. P2: the start is within
  2 hours for 70% of found events and the end within 6 hours for 50%. P3: inference on a
  degree-preserving rewired graph classes a third or fewer of the real edge events as edge events.
  P4: on the real record, at least 24 of the 35 logged shifts of more than 1 day are found at an
  alarm rate no higher than LR_b20's. P5 (slow level): absorption in the 21-to-60-day band is 0.10
  or less for the masked frozen slow level and 0.30 or more for the 30-day random walk; recall in
  that band is 0.6 or more; 80% or more of events whose end lies inside the record are classed
  correctly as temporary or permanent; growth of unmetered solar panels and a seasonal bias cause
  0.5 or fewer false events per Primary-year. P6 (verified pair): the D5 and D16 transfer starts
  on 29 to 30 December 2025 and ends on 13 to 14 January 2026, each within 12 hours, with D5 as the
  donor and 1.2 to 2.2 MW at each end, and the slow level moves by 2 points of capacity or less
  across it.
decision_rule: >-
  Fixed before any score is read. Residual: observed cleaned power minus the forecast of the
  lag-free convex demand model of experiment 130, trained on every half-hour with a reading and
  refitted monthly (experiment 327's arm C forecasts), 15 Primaries, 2022-01-01 to 2026-06-30
  (the stored grid; the 2019 to 2021 history is not on it, which the page states). The forecast
  is out of sample in time but its 24-month window contains earlier events, so the residual absorbs
  some of them; the page reports this. Tuning years: 2022 and 2023. Scored years: 2024-01-01 to
  2026-06-30, and no penalty, threshold, or generator setting changes after the first scored
  number is read. The final-test window from 2026-07-01 is never read. Every contrast is read at
  equal alarm rate (the unit of comparison is events found per Primary-year of alarms, because this
  experiment has no bands to widen; the equal-rate version and the version at each method's own
  threshold are printed side by side), and a placebo that is neutral by construction is printed
  beside it: the rewired-graph run changes only which Primaries count as neighbours when classing
  an event, so the decoded per-Primary events and every decision row stay the same. The stop rule
  has three parts. Stop the idea if the rewired-graph run classes half or more of the real edge
  events as edge events, or if recall is under 0.5 and the method finds fewer real shifts than
  RO-NC. Report events longer than 21 days as unresolved, capped at 21 days with a flag, if
  absorption in the 21-to-60-day band exceeds 0.25 or more than 30% of long and permanent events
  are confused with each other. Skip the neural stage (E2) and adopt this table as the event method
  if synthetic recall is 0.85 or more and the real-record count matches LR_b20's. The porting
  threshold is zero, because the table is a research output and changes no production code.
benchmark: >-
  RO-NC, LR_b20, the 24-hour residual detector with the neighbour filter, and the label-free
  pipeline of experiment 327, scored on the 35 logged shifts of more than 1 day with experiment
  254's strict and loose matching; the edge-flow fit of experiment 131; and the 30-day random-walk
  slow level (arm S0) and the unmasked Huber slow level (arm S1) inside this experiment.
row_set_id: >-
  experiment 258's grid of 15 Primaries, half-hourly, 2022-01-01 to 2026-06-30, cleaned power;
  not comparable with the leaderboard
metric: >-
  MW-weighted recall of injected events by duration band; unconfirmed edge events per edge-year on
  real held-out data; start and end error in hours; absorption of an event by the slow level;
  logged shifts found at a stated alarm rate.
code: ml-experiments/00653-edge-state-event-table-penalised-segmentation/
---

**This page is a plan; the results are not in yet.** The experiment asks whether the residuals of
15 Primary substations, read together, can produce a table of load-shift events: when each event
starts and ends, which Primaries it moves power between, and how many megawatts move. It builds the
table by penalised segmentation, which fits a piecewise-constant signal and charges a price for
each change. It checks the table on events it injected itself, on the verified transfer of
December 2025, and on the real record. No neural network is trained.

## Method

**Architecture and algorithm.**

- **Input.** The residual is the observed cleaned power minus the forecast of the lag-free convex
  demand model of [experiment 130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md), as
  stored by [experiment 327](../00327-lsc-label-free-detect-mask-train-pipeline/result.md) (arm C,
  trained on every half-hour that has a reading). Weather is ERA5 and CAMS estimates of past
  weather, so the table is a hindsight product and cannot be made live. Half-hours that the
  `substation_zero` rule flags are missing. D15 and D18 lose every row before 2025-03-01.
- **Noise unit.** For each Primary, one noise unit is the robust spread of the difference between
  successive 6-hour block means, computed with a 90-day moving window so that it follows the season.
- **Slow level (default arm).** A seasonal level per Primary, following the
  slow-level addendum of 9 November 2026 to the design answer
  (`data/ml-experiments/briefs/edge_state_slow_level_addendum.md`): 3 day-of-year harmonics shared by all years plus a linear spline with knots
  every 365 days, fitted with a Huber loss. The fit gives zero weight to any window whose 3-day mean
  residual exceeds 1.5 noise units, widened by 2 days on each side. It refits three times and then
  freezes, so the segmentation cannot change it. D15 and D18 share the pooled seasonal shape and fit
  only its scale. A trend in unmetered solar panels multiplied by CAMS irradiance is left out of
  the default arm, and the page counts the false events that solar growth causes instead.
- **Slow level (comparison arms).** Arm S0 is a 30-day random walk with a penalty on its curvature.
  Arm S1 is the same seasonal fit with a Huber loss and no mask.
- **Segmentation.** Each Primary's residual, with the slow level removed, is cut into 6-hour block
  means. An exact dynamic programme (a Viterbi pass over a hidden semi-Markov chain) decodes the
  states below over the blocks, forwards and then backwards, which gives a posterior for each state
  and an interval for each start and end. The shift states have 12 amplitude levels, log-spaced from
  1 to 20 noise units, in both signs. Staying shifted costs a fixed amount per block, switching on
  costs an onset price, and the duration prior is log-uniform from 3 hours to 180 days. The
  penalties are tuned on the 2022 and 2023 injected worlds only. A final pass refines each start
  and end to the half-hour by maximising the step statistic within 12 hours of the block boundary,
  and reads the edge width (the number of half-hours the step takes to go from 10% to 90%).
- **States per Primary.** Normal, a temporary shift, a permanent shift, a routine step, and a ramp.
  A temporary shift returns to the earlier level. A permanent shift has no return within 180 days
  of hindsight. A routine step belongs to a Primary whose trailing 60 days hold many short solo
  steps (D8's gas engine). A ramp spreads the same change over 4 hours or more.
- **Classing an event.** Events at connected Primaries (the neighbour graph of experiment 383,
  `neighbours_nged.json`) are paired when their starts fall within 6 hours and their ends within
  12 hours, with opposite signs. The pairing minimises a loose conservation cost,
  `(Δdonor + Δrecipients)² / (σ² + (0.5·|Δdonor|)²)`, and the recipient's share of the donor's step
  is limited to 0.3 to 1.5 (experiment 459 and the generator ideas file). A paired event is an edge
  transfer. An unpaired temporary event is a shift without a partner. An event whose end lies beyond
  the record is an open shift, reported with a probability of being temporary. An event at a Primary
  with a high routine rate needs a partner larger than that Primary's noise unit to count.
- **Output.** One row per event: start and end with intervals, Primaries, direction, megawatts per
  Primary, class, duration band, and posterior confidence.

**Training regime.** There is nothing to train. The penalties and thresholds are tuned on the 2022
and 2023 injected worlds, and the 2024 to 2026 years are scored once. NGED's labels enter no
penalty, threshold, or mask.

- **Step 0, before the generator is fixed.** On the two verified transfers, D5 and D16 (29 December
  2025 to 13 January 2026) and D3 and D22 (26 October to 22 November 2024), measure the width of
  each step in half-hours and whether the extra power during the event is a constant number of
  megawatts or a share of the donor's own daily shape. The share is the slope of the event-minus-
  baseline difference on the donor's baseline profile, set against a constant, and the better fit
  by residual variance decides, with both reported. The medians fix the edge width and the shape of
  injected events. If the two transfers disagree, both shapes are injected and scored as separate
  arms. The two transfers are only two episodes, and the page says so.
- **Generator.** Injected events follow the design answer's section 4 and the generator ideas file:
  a donor with 1, 2, or 3 recipients, or an unseen partner; a recipient share of 0.5 to 1.1;
  amounts from 1 to 20 noise units, half of the mass at 2 to 8; durations from 3 hours to 112 days
  (log-uniform) with short-heavy and long-heavy mixtures; start hours and weekdays drawn from the
  logged start times; mostly an exact return, some partial and staged returns; a slow drift inside
  an event; overlap at about 5% of each Primary's time. The addendum adds permanent shifts (an
  instant step, or a ramp over 1 to 8 weeks), permanent transfers, solar-panel growth of 0.2 to
  1 MW a year with an irradiance shape, and a seasonal bias that repeats every year. Hard negatives
  are routine steps that follow the daily shape, ramps of 2 to 12 hours, pulses under 3 hours, solo
  shifts, opposite-sign steps at two unconnected Primaries (the placebo pairs), and a smooth common
  mode shared by all Primaries. MW and MVA Primaries are kept apart; MVA Primaries receive
  `|MVA ± transfer|`.
- **Lab shortcut.** Events are added to the stored residual, not to the power before the demand
  model is refitted, as experiment 297 did with a frozen model. The injected events therefore
  escape the demand model's absorption, so recall is a ceiling for events as large as the real
  ones. Real events already in the residual stay in it as unlabelled noise, and the page marks an
  injected window that overlaps a logged or verified event.

**Measurement.**

- **Injected events, held-out years.** MW-weighted recall by duration band (3 hours to 1 day,
  1 to 7 days, 7 to 21 days, 21 to 60 days, permanent), by size in noise units, and by class. Start
  and end error. Absorption (the share of each event's megawatts taken by the slow level, by band).
  A confusion matrix of temporary against permanent for events whose end lies inside the record.
  False events from solar growth and seasonal bias per Primary-year. A shape check: constant
  against proportional events, and a cross-generator check that trains the penalties on one and
  scores the other (recall may drop by at most 0.05).
- **The verified pair.** D5 and D16 of December 2025 to January 2026, and D3 and D22 of 2024
  (which falls in the tuning years and so is a descriptive check only). Dates, direction,
  amplitude, class, and the movement of the slow level across the event. K3 sat about 11 points of
  capacity above D5's demand before the event, so the check is repeated on the residual of an
  ERA5-driven model, in case the offset belongs to K3.
- **The real record.** Edge events per Primary-year; the 35 logged shifts of more than 1 day found
  by experiment 254's strict and loose rules; the count of calls at D8 against the median Primary
  (it must stay within a factor of 2); the unconfirmed edge events per edge-year (an upper bound,
  because NGED's logs are an incomplete one-off dump, and clean rows may hold events nobody logged).
  The 27 transfers that experiment 580 found in the telemetry are a second reference, with the
  caveat that its detector and this one read the same residual.
- **Baselines and intervals.** RO-NC, LR_b20, the 24-hour residual detector with the neighbour
  filter, the label-free pipeline of experiment 327, and the edge-flow fit of experiment 131, each
  from its stored alarms where they exist. Intervals resample whole events and whole calendar
  months. A real-record claim from the D5 and D16 episode alone is one episode, and the page says
  so.
- **Checks that the method has not learnt the generator.** A rewired-graph run (neutral by
  construction, as the decision rule states), a rate check per Primary-year, and the cross-generator
  check above. The page also prints each score without the equal-rate matching.
- **Compute.** CPU only, 2 cores, no GPU, no fitted network. The residual panel already exists.

## Files

- `ml-experiments/00653-edge-state-event-table-penalised-segmentation/`: the scripts. None exists
  yet; each is named `e00653_*.py`.
- `data/ml-experiments/00327-lsc-label-free-detect-mask-train-pipeline/forecast/C/` and `power/`:
  the stored forecast and the cleaned power per series on the 2022 half-hour grid (read only;
  written by [experiment 327](../00327-lsc-label-free-detect-mask-train-pipeline/result.md)).
- `data/ml-experiments/00580-verified-load-shift-event-set-and-label-audit/work/`: the 27 verified
  transfers and the 184 labelled events (read only; written by
  [experiment 580](../00580-verified-load-shift-event-set-and-label-audit/result.md)).
- `data/ml-experiments/00383-load-shift-xgboost-oracle-first-part-1/work/neighbours_nged.json`: the
  electrical neighbour graph (read only).
- `data/ml-experiments/00653-edge-state-event-table-penalised-segmentation/`: the new residual
  panel, the slow-level fits, the injected worlds, the event tables, and the score tables (written
  here, never committed).
