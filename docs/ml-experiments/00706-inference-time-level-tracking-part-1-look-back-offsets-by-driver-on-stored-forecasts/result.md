---
id: "00706"
slug: inference-time-level-tracking-part-1-look-back-offsets-by-driver-on-stored-forecasts
title: >-
  Can the slow level of a Primary substation's net demand be tracked at forecast time, from the
  forecast errors of the last 14 days, split by irradiance and temperature?
theme: lsc-detectors
also_relevant_to:
  - lagged-power-and-error-correction
  - calibration-and-uncertainty
  - switching-labels-in-the-forecast
status: planned
kind: other
protocol: demand-v2
size: >-
  Anything else. (1) Stored: forecasts of four arms, the residual table, and, in stage 2 only, the
  head's state files, all private under the experiment's data folder; nothing under `src/`. (2) More
  than one design defensibly tests the hypothesis (binned or regressed drivers, which stored forecast
  is "expected", how the share is learned). (3) Only this folder's scripts call the new code, and
  they import the folders of experiments 647, 660, 667, and 688 and `studies.*`. No download, no
  spend. Reviews: Fable design, Opus design review, Opus code review before each first run, Opus
  result review, Sonnet numbers review, Sonnet prose review, and an audit if a trigger fires.
hypothesis: >-
  A look-back over the last 14 days of the stored normal-running forecast's errors, computed at each
  forecast origin and split by driver, lowers the forecast error on rows free of load-shift events
  beyond what the Candidate's monthly-knot offset and its lagged-power and recent-gap inputs already
  give. In stage 2, a share learned from the size, timing, and electrical neighbours of a shift lets
  the normal-running forecast take slow drift and leave a switching shift out.
prediction: >-
  Written before any score is read, CRPS in points of capacity, leads 0 to 240 h, 2025-12 to
  2026-06. C1: the lead-band global offset minus the stored normal-running forecast on event-free rows
  is -0.05 to -0.15, with day 1 showing less gain because the Candidate already tracks a 1-to-7-day
  level. C2: the driver split minus the global offset is between -0.02 and +0.02 (expected to fail),
  with one predicted gain of -0.05 on sunny daylight half-hours from April to June at the signed-MW
  six. C3 (stage 2): the learned share cuts the copied share of an injected shift by a third to a half
  against the no-network rule. C4 (stage 2): the real neighbour graph and a rewired graph differ by
  under 0.1 points, for each of N-flag and N-gate. C6 (stage 2): N-gate minus N-flag on the copied share
  and on event-free CRPS is within 0.1 points, and we expect N-gate to be no worse. C5: the faithful prevailing arm minus the stored prevailing rows is 0 to +0.1.
decision_rule: >-
  Contrasts are paired on series, origin, and valid time and read at 99.17% (Bonferroni over six
  contrasts). A real-row contrast counts when its month-block bootstrap interval and the t test on
  the monthly paired differences both exclude zero (critical value 3.71 for 6 degrees of freedom),
  and each interval's half-width and minimum detectable effect is printed. C1 and C2 are read in the
  signed-MW six and the MVA-only eight separately, and both groups must pass. The smallest worthwhile
  gain for C1 is 0.05 points, because the offset adds a serving step that keeps 14 days of issued
  forecasts and telemetry, and 0.05 equals what fresher lags bought in experiment 572. C1 passes when
  the 99% interval lies wholly below -0.05. C2 passes at zero with an interval that excludes zero. A
  point estimate worse than +0.02 in either group blocks `supported`. The full verdict table covers
  every combination of C1, C2, and C3 and is in the design file. A failed known-answer gate caps the
  affected contrast at `inconclusive`.
benchmark: >-
  The stored normal-running and prevailing rows of the one-year backtest files (experiment 668),
  which carry the Candidate's monthly-knot offset D5, 24-hour and 7-day lags, and recent-gap and
  neighbour inputs; recipe R (experiment 115), a 28-day median shift (experiment 642), and a
  month-by-half-hour climatology.
row_set_id: backtest_year_2025_12_to_2026_06_daily_12utc_pre_holdout
metric: >-
  CRPS and normalised mean absolute error of the median forecast, in points of capacity, with q01,
  q05, q95, and q99 coverage and pinball loss, cold winter nights, and sunny April to June half-hours.
---

**Status: planned. No fit and no score has been run.**

**This experiment asks whether the slow drift in a Primary substation's net demand can be corrected
at forecast time, with no new fitted network.** At each forecast origin, we take the errors of the
stored forecast over the last 14 days, compute their mean, and add a share of it to the forecast. We
also split the errors by irradiance and by cold temperature, to see whether solar-farm drift and
cold-night drift need separate corrections. We then score the corrected forecasts against what
happened on 14 Primaries from 2025-12 to 2026-06.

The question follows the advice of [experiment 690](../00690-slow-level-state-on-primaries-13-and-16-july-august/result.md),
whose slow level halved the error in hindsight and whose advice is to fit the level causally. The
look-back mean follows recipe R of [experiment 115](../00115-differentiable-physics-w1-diurnal-shape-state-on-ens/result.md).

## Stages and approval

- **Stage 1 (approved scope):** the residual table, the known-answer tests, the global-offset, driver-split,
  and prevailing arms on the stored forecasts (CPU only, 4 cores), and contrasts C1, C2, and C5.
- **Stage 2 (maintainer approved 2026-11-18, under the stop rule):** a learned gate decides how much
  of each look-back offset the forecast applies. Four earlier experiments failed on the
  neighbour-based switching share (165, 175, 652, 663), and the maintainer approved a fifth attempt
  with a second design that has no binary switching-event flag. Two stage 2 arms run side by side:
  N-flag, the earlier design (a switching probability trained with a binary cross-entropy term and a
  change-point split of the offset), and N-gate, in which the head learns
  how far to trust each look-back offset (global, high-irradiance, cold, and the neighbours' offsets
  at several window lengths), as the Candidate's lag encoder learns how far to trust each lagged
  power input, and it trains on synthetic switching events with the CRPS loss only. Stage 2 also
  runs only if stage 1's global or driver-split arm improves event-free CRPS by a point estimate of
  0.02 or more. The stage 2 design is rewritten before any stage 2 script is written.

## Plan

The full design, with the Opus review and its triage, is the private file
`data/ml-experiments/briefs/level_tracking/design.md`. Its main points:

- **Arms.** G-lb (lead-band-matched look-back mean, factor 0.75, 14 days) is the primary global
  offset. D adds a ridge-regressed irradiance slope and cold-temperature slope. P-7d and P-cp are
  prevailing arms. N-flag (switching probability), N-gate (learned per-feature gate, no flag), and N-rule are the stage 2 arms. The stored normal-running and prevailing rows,
  recipe R, a 28-day median shift, hindsight ceilings, and a climatology row are references.
- **Look-ahead.** The target weather comes from the ensemble run of 00:00 UTC on the origin's day,
  available at 09:00 UTC, and the script asserts this row by row. The look-back window ends 6 h
  before the origin. The printed example is Primary 17 (D5), origin 2026-03-10 12:00 UTC, valid time
  2026-03-15 13:00 UTC.
- **Known-answer tests and placebos.** KA1a (additive step, forecast held fixed), KA1b (re-run bank,
  reported), KA2 (PV-shaped and cold-night injections), PL1 (non-neighbour window), PL2 (shuffled
  drivers), and, in stage 2, KA3, PL3, and PL4.
- **Holdout.** No row with a valid time on or after 2026-07-01 is read.
- **Standing practices.** Cleaned data, stored predictions reused first, tails and cold nights
  reported, the four load-shift scenarios, at least six event plots, and a draft PR.
- **Audit triggers likely to fire.** KA2 or PL2 failing, a C1 gain that contradicts experiments 683
  and 642, a nested arm beating the arm that contains it, and leaving out D5, D15, or D18.
