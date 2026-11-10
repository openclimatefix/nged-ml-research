---
id: "00690"
slug: slow-level-state-on-primaries-13-and-16-july-august
title: >-
  Does the slow-level state track the July to August level change at Primaries 13 and 16?
theme: lsc-detectors
also_relevant_to: []
status: planned
date: 2026-11-10
kind: other
size: >-
  medium. (1) Stored: private tables under
  data/ml-experiments/00690-slow-level-state-on-primaries-13-and-16-july-august/ only.
  (2) Serving path: not touched. (3) Degradation rule: not touched. (4) Designs: one defensible
  design, the masked frozen seasonal slow level of the edge-state addendum, as coded in experiment
  653. (5) Callers: this folder's scripts, which import the folders of experiments 653 and 685;
  nothing imports them. CPU only, 2 cores.
hypothesis: >-
  The slow level of the edge-state addendum (a smooth seasonal level fitted with candidate windows
  masked, then frozen) follows the July to August 2026 change in level that the stored
  normal-running forecasts miss at Primaries 13 and 16.
prediction: >-
  At both ids 13 and 16 the level estimate crosses half of the July to August residual change
  within 14 days of the residual's own half-crossing, and adding it to q50 cuts the median absolute
  error of July to September by at least 25%, while the two control Primaries move by less than 5%.
  Expected to fail, because the design masks long deviations on purpose.
decision_rule: >-
  Supported only if both ids meet both conditions and both controls move by less than 5%. If either
  id misses, the slow-level state does not explain the miss and the miss stays recorded as
  unexplained; no new method is proposed.
benchmark: >-
  The stored normal-running q50 of the one-year backtest (primaries.parquet, NRA rows), with no
  correction.
row_set_id: >-
  one-year backtest normal-running rows of primaries.parquet, one row per half-hour (the row with
  the smallest lead), 2026-05-01 to 2026-09-30; not comparable with the leaderboard
metric: >-
  Weekly median of observed power minus q50; correlation and lag of the weekly level estimate with
  it; days from the start to half of the change; median absolute error and q05 to q95 coverage in
  July to September with and without the level added.
code: ml-experiments/00690-slow-level-state-on-primaries-13-and-16-july-august/
---

**Planned. The plan below is fixed before any residual is read.** The maintainer approved the
experiment on 2026-11-10, including reading July to September 2026 descriptively. Nothing is chosen
from that window.

## Plan

- **Arm 1, the miss.** Observed power (`ground_truth_primaries.parquet`, cleaned `power`) minus the
  stored `q50`, on normal-running rows, by week from 2026-05 to 2026-09 at ids 13 and 16. One row
  per half-hour: the stored row with the smallest lead. The clean-power level sits beside it.
- **Arm 2, the state.** The slow level of experiment 653 (`fit_slow`, arm `masked`: 3 harmonics, a
  yearly spline, Huber fit, three masking passes), run on experiment 685's residual panel (the
  lag-free forecast against power, 2022 to 2026-10-31), because the stored backtest holds only 8
  months before July and the design needs years. Noise units and the mask threshold are frozen on
  the residual before 2026-07-01, as in experiment 685. The weekly level estimate is compared with
  arm 1's weekly median residual: correlation, the lag of the best agreement (−28 to +28 days), and
  the day each crosses half of its change. Half of the change means half of the difference between
  the mean over 2026-05-01 to 2026-06-14 and the mean over 2026-07-15 to 2026-08-31.
- **Arm 3, controls.** The two ids, other than 11, 13, 16, and 18, with the smallest mean absolute
  distance of the stored normal-running q05 to q95 coverage from 0.90 over 2026-07 and 2026-08. The
  rule is fixed here; the ids are chosen by it afterwards.
- **Arm 4, diagnostic.** The level estimate (taken from the arm 2 level at the half-hour, in
  megawatts) is added to q50 over 2026-07 to 2026-09. Reported: the change in median absolute error
  and in q05 to q95 coverage, with day-block bootstrap intervals (7-day blocks, 2,000 draws). This
  is a diagnostic with hindsight, never a forecast.
- **Secondary arms, from the addendum, labelled so:** S1 (Huber, no mask) and S0 (30-day random
  walk) on the same residual, because the main arm masks long deviations by design. They change no
  verdict.
- **Reuse.** No model is fitted. The stored forecasts, the ground truth, experiment 685's panel,
  and experiment 653's slow-level code are reused.
- **Limits.** July to September 2026 lies in the final-test window and is read descriptively
  only. NGED's event labels are an incomplete dump, so "no transfer" is a lower bound. Primaries
  are named and dated; no generator is shown.
