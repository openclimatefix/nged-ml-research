---
id: "00663"
slug: end-detector-for-the-prevailing-forecast-followup-to-654
title: >-
  Does a learned end detector, reading the freshest half-hours at a Primary and its neighbours, make
  the prevailing forecast drop a load shift sooner after it ends, without more false alarms?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Complex. (1) It stores per-row forecasts, detector weights for each base arm, fold, and seed, and
  the release-candidate parquet files; (2) more than one design tests the hypothesis (a detector
  fitted on frozen base outputs, or a joint refit, and which residual and neighbour inputs it
  reads); (3) it spans the K1, R-haz, and J2 code of experiments 612, 613, 647, 650, 652, and 654,
  whose callers cannot be named without searching; (4) it touches no serving path or degradation
  rule; (5) it changes no contract. Full set of reviews: plan, code, results, write-up, prose.
hypothesis: >-
  A detector that reads the freshest half-hours of (observed power minus the NRA forecast) at a
  Primary and at its electrical neighbours, and outputs the probability that a load shift has ended,
  lets the prevailing forecast of E0 (frozen K1 plus the survival-hazard head R-haz) and of J2 (the
  end-to-end network of experiment 654) drop a shift sooner after it ends, and cuts the share of
  event-free rows on which the prevailing forecast wrongly departs from the NRA forecast.
prediction: >-
  Written before any fit. (P1) E0d minus E0 and J2d minus J2 in scaled CRPS over lead days 1 to 9 is
  at most -0.3 points of capacity in the "ended in the last 24 hours" and "near" strata, and within
  0.1 points of zero in the "far", "middle", and event-free strata. (P2) For injected shifts
  shallower than 5 points of capacity the detector arms score within 0.1 points of their bases in
  the "ended in the last 24 hours" stratum. (P3) The rewired-graph placebo recovers at most half of
  the gain of the real-graph detector. (P4) The share of event-free rows on which the prevailing
  output departs from the NRA output by 3 points or more falls by at least a quarter, relative to
  the base arm, for J2d (J2's base share is the larger, because experiment 654 found J2's prevailing
  output worse than its NRA output on gate-off rows, 4.54 against 4.05).
decision_rule: >-
  Contrast 1: E0d minus E0, and J2d minus J2, scaled CRPS averaged over lead days 1 to 9, in the
  strata "ended in the last 24 hours" and "near" pooled, on the injected bank (training family),
  five development months, seeds averaged within a month. It counts as a gain when the month-block
  95% interval (percentile bootstrap over the 5 months) excludes zero in the negative direction and
  the point estimate is at most -0.3 points; elsewhere the contrast is descriptive. The threshold is
  the size that pays for a second head and a neighbour feed in the serving path. Contrast 2: the
  real-graph detector minus the rewired-graph detector (same bar). One veto, on event-free origins:
  the upper bound of the 95% month-block interval of each detector arm minus its base (unscaled
  CRPS, prevailing output) is below 0.03 points of capacity. The NRA output of a detector arm is the
  base arm's NRA output unchanged, so it is non-inferior by construction. The final-test window is
  read once for the frozen candidates (E0, J2 reported once, and the detector arms) and every window
  contrast is descriptive.
benchmark: >-
  E0 and J2 (the bases), B-haz, R0 (168 h), K1 as context, the matched-compute control C0 of
  experiment 654, and on the window the manual heuristic.
row_set_id: dev_months_origins_2025_10_to_2026_06_export_roster
metric: >-
  CRPS (twice the mean pinball loss over the 11 stored quantile levels) in points of capacity, scaled
  and unscaled, by lead day (1 to 9) and by origin stratum; normalised MAE of the median; q05-q95
  coverage; q01, q05, q95, and q99 pinball loss; follow; detection delay (days) and false-alarm rate
  per Primary-month for the detector alone; share of event-free rows with a departure of 3 points or
  more and its CRPS cost; avoidable-curtailment cost.
production_status: not-for-production
adopted_in: none
---
**Status: planned. The sections below are the plan; the finished page replaces them.**

A Primary is a primary substation. A load shift is a period when the electricity network operator
moves part of one Primary's demand onto, or off, a neighbouring Primary. K1 is the neural network
of experiments 612 and 613 that forecasts the normal running arrangement (NRA), the demand with no
load shift. The prevailing forecast is the demand under the arrangement now in force.
[Experiment 654](../00654-two-output-network-end-to-end-with-end-detector/result.md) built two
bases for it: E0 (frozen K1 plus the survival-hazard head R-haz of experiment 652) and J2 (one
network fine-tuned end to end). Experiment 654 did not run its end detector. This experiment does.

## What it adds to experiment 654

- **End detector** on E0 and on J2, development months only, with the opposite-sign neighbour
  boost, the routine-step-rate column, the all-Primaries median removed, and a rewired-graph
  placebo beside every contrast (Method below).
- **E0 on the final-test window**: the final-fold R-haz head, which experiment 654 did not have, so
  that the window scores E0 as well as J2.
- **False-alarm accounting**: the share of event-free rows on which the prevailing output departs
  from the NRA output by 3 points or more, and its CRPS cost, for every arm.
- **Real events in the window scenarios**, with the count of events per scenario stated as a lower
  bound, instead of only the G4 gate proxy.
- **Time series** from 14 days before to at least 7 days after a real event.

## Findings from reading the earlier experiments (what changed the plan)

- **NGED's labels do not reach the window.** The last labelled event ends at half-hour index
  119,331 of the label origin (2019-09-01); the window starts at 119,760. Experiment 653's table of
  detected real events also ends before 2026-07-01. Window scenarios therefore need events detected
  by this experiment (Method, "Real events"), and the count will be small.
- **The stored banks cannot be regenerated cheaply.** A different duration law for the injected
  events needs K1 run again on every injected lag block (experiment 650's `inject` and `rebuild`
  steps, five seeds, two cuts, five months, CPU hours) and the heads and J2 refitted on the new
  banks. The detector reads only residuals, and residuals do not need K1 re-run, so the plan splits
  the two jobs (below).
- **Stored E0 fits keep quantiles and survival only.** Experiment 652 stored no weights, and a
  detector that multiplies the survival needs the head's depth and spread, so E0 is refitted with
  the same code and seeds (a minute per fit on CPU) and checked against the stored quantiles.

## Method

**Rows and folds.** The five development months (2025-10, 2025-12, 2026-01, 2026-03, 2026-06), the
16 Primaries, leads 24 to 240 hours, daily origins, leave one month out, 2 seeds, as in experiments
652 and 654. Monthly refit, cleaned data, D15 and D18 cut before 2025-03-01, as the standing
practices say. Every fitted network is saved under the experiment's data folder.

**Injected world for the base-versus-detector contrasts: experiment 650's bank, unchanged.**
Durations are 3 to 14 days (half the events) and 28 to 112 days (half); depth log-uniform 3 to 50
points of capacity; constant or proportional shape; step or 6 to 48 hour ramp onset. This is the
bank on which E0 and J2 were scored, so the contrast against the base is a paired comparison on
identical rows. The page states that its length distribution has no events of 1 to 3 or 14 to 28
days, and that it is not the log-uniform 1 to 112 day law of experiment 653's injector.

**Injected world for the detector alone: experiment 653's duration law.** A second world draws the
duration log-uniform from 1 to 112 days with a random end, plus a short band from 3 hours to 1 day
(20% of events), keeps experiment 650's depths, shapes, and receivers, and adds the hard negatives
(routine solo steps that follow the daily profile, pulses under 3 hours, ramps of 2 to 12 hours,
opposite-sign steps at two unconnected Primaries, a smooth common-mode perturbation). It is injected
into the residual panel only, because the detector reads residuals and no base arm is run on it.
Un-injected time is unlabelled. It supplies the detector's delay and false-alarm figures and a
check that the detector trained on the first world transfers to a different duration law. It is
described as a deviation from "the 653 injector, unchanged", with the reason above.

**Residual panel.** For each Primary and half-hour, observed cleaned power minus K1's median at the
day-ahead lead (24 to 48 hours, from the previous day's origin), in capacity fractions, divided by
the Primary's noise unit (1.4826 times the median absolute deviation of event-free residuals in the
training months). The injected shift is added to the observed power before the subtraction. The
median across Primaries at each half-hour is removed (common-mode weather error), and a constant
per Primary (the median of the training months' event-free residuals) stands in for the slow
seasonal level. A free 30-day level is not used, because it would absorb a long shift.

**Detector inputs, per Primary and origin.** The freshest 48 hours as 16 three-hour block means, the
eight daily means before that, the same two summaries averaged over connected neighbours (the 29
edges of `neighbours_nged.json`), the opposite-sign neighbour step (the neighbour's change in daily
mean over the last day, signed against the Primary's own change; a score boost, never a
requirement), the routine-step rate (the count of daily-mean changes larger than 3 noise units over
the previous 60 days, per day), and the origin features experiment 647's heads already read. The
detector is a two-layer network with 64 units per layer, whose output is `q_end`, the probability
that the shift has ended. The prevailing quantiles use the survival `p_base * (1 - q_end)` in the
mixture of experiment 647. The rewired-graph placebo replaces the neighbour sets by a
degree-preserving rewiring of the 29 edges (fixed seed, no edge kept), and the no-neighbour variant
drops the neighbour inputs.

**Fit.** The detector is fitted on frozen base outputs (the NRA quantiles, depth, spread, and
base survival) with the loss of experiment 652 (weighted pinball of the mixture quantiles plus 0.05
times the binary cross-entropy of the survival against the still-on label), one detector per base,
month, and seed, the other four months training. Hard negatives enter the residual inputs of 30% of
the training rows. For J2, depth, spread, and base survival come from a forward pass of the saved J2
networks on the same rows (the models are in experiment 654's `models/` folder).

**Arms.**

- E0, E0d (real graph), E0d-rewired, E0d-no-neighbours.
- J2, J2d (real graph), J2d-rewired.
- References, not refitted: B-haz, R0, K1 (context), C0, and the experiment 654 results for J1.

**Strata, scenarios, and metrics** are those of experiments 652 and 654: strata by the state of the
shift at the origin (event-free, event-free and more than 14 days from any event, onset ahead, far,
middle, near, ended in the last 24 hours, just ended within 7 days), lead days 1 to 9, scaled and
unscaled side by side with one scaling factor per arm, month, and seed fitted across strata. Every
metric table reports the normalised MAE of the median, q05-q95 coverage, the tail losses at 0.01,
0.05, 0.95, and 0.99, and the avoidable-curtailment cost. The four load-shift scenarios (all rows;
outside events and 7 days after; inside; the 7 days after an event ends) are reported on the
injected bank and, descriptively, on real events.

**Real events on the development months.** NGED's labelled events and experiment 653's detected
events (counts stated, lower bounds). The unlogged-event caveat applies to every table.

## Final-test window (2026-07-01 to 2026-10-31)

The maintainer approved a second opening for E0 and the detector arms on 2026-11-09. Every
candidate choice uses development months only.

- **Final-fold E0.** R-haz is fitted on all five development months and applied to the window's
  K1 quantiles and origin features (experiment 620's extended inputs). The detector arms are
  fitted the same way.
- **Freeze, message, then score once.** `final/freeze2.json` lists E0, J2 (as frozen in
  experiment 654), the detector arms, and the hashes of every fitted file. The manager is messaged,
  and the window is scored once. J2, J1, C0, K1, and the manual heuristic are not rescored; their
  window forecasts are reused from experiment 654.
- **Real events in the window.** Labels: none (above). Detected events: experiment 653's
  segmenter is run on a K1-based residual panel for the window. If it cannot be calibrated there
  without reading the window to tune it, the window scenarios fall back to the G4 gate proxy and
  the page says so and gives the count of spells.
- **Scenarios and plots** as experiment 654: the four scenarios, intervals blocked by origin month
  (4 months, coarse), counts as lower bounds.

## Figures (at most two panels each)

- Loss versus lead day by stratum, bases against detector arms, with intervals.
- Detector alone: detection delay (in whole days, because origins are daily) by depth, and false
  alarms per Primary-month on event-free origins, on the 650-world and the 653-law world.
- False-alarm share and its CRPS cost per arm.
- Time series from 14 days before to at least 7 days after a real event, with the observed power,
  K1, the NRA output, and the prevailing output of the base and the detector arm: the verified
  D5/D16 pair and the largest event in the window; anonymised wherever a metered generator is
  involved; at least 280 pixels per panel.
- Day-ahead and 14-day example forecasts.

## Compute and process

- GPU: one process at a time, only for the J2 forward passes (minutes). CPU: at most 4 cores for
  the head and detector fits, each a minute or so. Ceiling 4 GPU-hours (expected under one).
  Memory: `MemoryMax=16G` per job, `OOMPolicy=continue`.
- Standing practices followed: monthly refit, cleaned data, short-history cut, fitted networks saved,
  stored predictions reused (K1, the 650 bank, the J2 fits, E0's stored quantiles as a check),
  tails and curtailment cost reported, example forecasts and event plots, draft pull request.
- Reviews: one Sonnet plan review, one Sonnet code review before the first fit, one results review,
  one write-up review, one prose review.
- Release candidate parquet files (E0 and the detector arms, 16 Primaries, five development months,
  and the window) with a plain-words README in
  `data/ml-experiments/00663-*/release_candidate/`, written as each arm finishes.
- Reports to the manager, one line each: plan PR open, fits started, freeze2 written, window
  scored, PR merged.
