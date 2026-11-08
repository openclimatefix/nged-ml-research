---
id: "00650"
slug: extended-shift-departure-head-against-decayed-recent-error
title: >-
  Does a departure head trained on shifts with random ends, varied shapes, and varied depths beat K1
  shifted by its own decayed recent error, once band widening is removed?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
  - calibration-and-uncertainty
status: planned
kind: other
protocol: demand-v2
size: >-
  Medium. (1) It stores one rebuilt K1 forecast table per shifted-bank pass and the head weights (a
  few thousand parameters each), and no new K1 weights; (2) more than one design would defensibly
  test the hypothesis (heads trained on real bases or on generated bases, with and without a
  matched placebo), so it runs the arms side by side; (3) it reuses experiment 647's head, scorer,
  and K1 rebuild, experiment 581's bank interface, and experiment 613's K1 inference, whose callers
  are named in the plan; (4) it touches no serving path or degradation rule; (5) it changes no
  contract.
hypothesis: >-
  A departure head trained on an extended shift generator (shifts that end at random times, constant
  and proportional shapes, step and ramp onsets, depths from 3 to 50 points of capacity, donor and
  receiver conserved across real neighbours) forecasts the arrangement now in force better than K1
  shifted by its own decayed recent error (R0 at tau 168 h), on a shift family that no arm trains
  on, after the band-widening share of the gain is removed.
prediction: >-
  Written before any head is fitted. On the held-out shift family, after equal width scaling, R-new
  scores a CRPS below R0-168 by more than 0.05 points of capacity (the minimum detectable effect),
  and below R-old by more than 0.05. After a shift ends, R-new is no worse than K1 by more than 0.5
  points (R-old was 3.7 worse). On real event-free dev rows R-new is no worse than K1 by more than
  0.03 points. The placebo scores within 0.05 of K1 after equal width scaling.
decision_rule: >-
  R-new counts as beating R0-168 when the month-block 95% interval (percentile bootstrap over the 5
  development months, seeds averaged within a month) of the scaled CRPS difference on the
  out-of-support cells excludes zero in the predicted direction and the point estimate exceeds 0.05
  points. With 5 blocks the interval is coarse, so it is reported with the fold-by-fold table.
  The 0.05 limit on the after-shift loss (0.5) and the 0.03 limit on event-free rows are vetoes:
  if either fails, R-new is reported as not ready for porting whatever the contrast says. R-old is
  context and enters no stop rule. The first month pair
  (2025-10 and 2025-12) is a feasibility gate, not a test, because two month blocks cannot support
  an interval; a claim needs all 5 months. Drop R-new if its point estimate is not below R0-168
  on the first pair. Drop arm G if it is worse than R-new by more than 0.05, or if the realness
  classifier's AUC is above 0.6. The smallest gain worth porting is 0.05 points over R0-168 on the
  out-of-support cells: R0-168 needs no fit, so a second neural network must beat it clearly. The
  value is the brief's fixed minimum detectable effect, chosen before any score, and no power
  calculation backs it; experiment 647 measured a 0.9 point gain on an in-distribution bank.
benchmark: >-
  R0-168 (K1's quantiles shifted by K1's own mean residual over valid days 2 to 8 before the origin,
  times exp(-lead/168 h)), K1 (experiments 612 and 613), and R-old (experiment 647's H1 as merged).
row_set_id: dev_months_origins_2025_10_to_2026_06_export_roster
metric: >-
  CRPS (twice the mean pinball loss over the 11 stored quantile levels) in points of capacity after
  equal width scaling, q05-q95 coverage, and q01, q05, q95, and q99 pinball loss, by phase of the
  shift and lead band.
---

## Plan

Experiment 647 found that a departure head beats K1 inside injected shifts by 10.0 points of
capacity, but 60% to 68% of that is band widening, R0 at tau 168 h is only 0.9 points behind, and the
head is 3.7 points worse than K1 after a synthetic shift ends. The bank of experiment 581 ends every
shift on day 14 with a constant shape, so both its training and its test cases share one grid. This
experiment retrains the head on an extended shift generator and tests it on a shift family that no
arm trains on. Development months only: nothing on or after 2026-07-01 is read, and there is no D14
or D18 scoring. Cadence: no K1 refit (K1 is frozen and stored); head fits are leave-one-month-out as
in experiment 647. Cleaned power is used throughout (the K1 inputs and the observed power come from
the stored clean stack), and the D15 and D18 cut before 2025-03-01 does not bite on these months.

**Arms** (matched rows, epochs, and seeds; first fold pair 2025-10 and 2025-12 first):

- **R0-168:** K1 shifted by its own decayed recent error, tau 168 h. No fit.
- **R-old:** experiment 647's H1, reloaded from its stored weights.
- **R-new:** H1 retrained on real unshifted bases plus the extended generator (below).
- **Placebo:** R-new's training with the shifts shuffled across rows.
- **G (only if time allows):** the extended generator on lag-free forecast bases from experiment
  598's table.

**Extended generator** (new `bank` module, same interface as experiment 581's `contaminate`):
half constant, half proportional to the Primary's own load (the realised mix is reported per
group, because MVA receivers take constant shapes only); step or 6 to 48 h ramp onsets; depth
log-uniform from 3 to 50 points of capacity; duration drawn from 3 to 14 days or 4 to 16 weeks,
with a random end (before, inside, or after the 10-day horizon); the donor's loss reaches one or
two real electrical neighbours at r times the loss, r from 0.5 to 1 (partial conservation);
power is never summed across MW
and MVA Primaries (MVA receivers take constant-shape events only, with a stated power factor of
0.97). Training passes use seeds disjoint from the test passes.

**Held-out family** (never trained on). Three attributes lie outside the training support, and each
gets its own reported cell: ramp onsets of 72 to 120 h (training ramps are 6 to 48 h), depth 0.65 of
capacity (training stops at 0.50), and a duration of 45 days with a shift that starts within the
horizon (training has no 29 to 55 day durations). The other cells (depths 0.10, 0.20, 0.40;
durations 5, 10, 21 days; constant shape; the training neighbour wiring and r) are inside the
training support and are reported as in-support cells. Each pass gives every Primary one cell, the
cells rotate across passes as in experiment 581, and both signs appear in every cell across the 4
passes. The decision contrast is the out-of-support cells pooled; the in-support cells are context.

**Scoring.**

- Held-out family: CRPS by phase (before, inside, after the shift). Every arm is also scored after
  equal width scaling, defined here before any score is read: one factor per arm, month, and phase,
  applied to all 11 quantiles about the median, chosen so that the q05-q95 band covers 0.90 of
  rows (the nominal level, not K1's own coverage). The same rule applies to every arm, including K1
  and R0-168. The factor is fitted on the scored rows, so it is an oracle, and is stated as such.
  Unscaled CRPS and coverage are reported beside it. The scaled contrast is the decision contrast.
- Real event-free dev rows: cost and false alarms against K1, limit 0.03 points.
- The verified D5 and D16 pair: descriptive only. The unlogged-event caveat applies: clean rows may
  hold events no label covers, so the event-free cost is optimistic.
- Tails: q01, q05, q95, q99 pinball loss and coverage next to CRPS, per group and per Primary.
- Avoidable-curtailment cost in three row sets (inside, 7 days after, all rows) on the six MW
  Primaries, using the experiment 601 function, as in experiment 647.
- G only: a gradient-boosted classifier on the head's input vector, trained on half the origins and
  scored by AUC on the other half, separating generated-base inputs from real-base inputs (AUC above
  0.6 means the head can tell them apart).
- Checks: median minus mean forecast per Primary, the mean-forecast error beside each quantile
  contrast, and the seven audit triggers.

**Leakage controls.** K1 is frozen and was fitted on the five development months with the
leave-one-out design of experiments 612 and 613, which score each month with a K1 not trained on
it, except that K1(Dec) is in-sample on the October rows (experiment 647's limit); this holds for
every arm alike. The shifted bank changes the observed power of days 2 to 8 before the origin, so
R0-168's residual is computed on the shifted series, as in experiment 647. Head fits are
leave-one-month-out: the scored month's generated passes are never in training, the placebo shuffle
is within the training fold only, and the epoch count is the fixed 30 with early stopping on a
validation month, never on the scored month or the held-out family. The scorer asserts that the
largest valid time read is before 2026-07-01.

**Compute.** CPU only, 14 cores at most, 16 GB per job, one OMP thread per process, spawn start
method. The cost is running K1's five seed networks (saved weights, 5.7 MB) on each new shifted bank:
12 training passes and 4 held-out passes per month, first for 2025-10 and 2025-12 (2 seeds and 2
cut days per pass as in experiment 613's inject: 160 inference runs for the first pair). The first
code step times one pass on 14 cores and writes the forecast-table size per pass into this page; the
disk to free is that size times 16 passes times 5 months, plus the head data. Jobs run 2 at a time
so as to stay inside 14 cores and 24 GB. Arm G adds 12 passes on the lag-free bases and is counted
only if time allows. Forecasts of
experiments 613 and 647 for unshifted rows and the 647 bank are reused for R-old and for the
event-free rows; nothing is refitted. The reuse check (same series, origins, leads, widening) is the
first step of the code. All outputs need free space on `/mnt/data`, which is full at planning time,
so no fit starts until space is freed.

**Reviews.** A fresh Sonnet agent reviews this plan; another reviews the code before the first run;
the results and write-up are each reviewed by a fresh Sonnet agent. No Opus unless a verdict could
change.

## Results

Not run yet.
