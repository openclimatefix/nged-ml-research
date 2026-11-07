---
id: "00637"
slug: ara-detector-rescore-of-the-bakeoff-final-test
title: >-
  How do the five bake-off systems score on July to October 2026 once days flagged as an abnormal
  running arrangement are removed?
theme: production-reference-and-best-guess
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  Medium. (1) It stores one small rule file and score tables, nothing in the serving path; (2) the
  detector admits more than one defensible design, so its choice is frozen on development days
  first; (3) it reuses experiment 620's scoring and the detector experiments' labels; (4) it
  touches no serving path and no degradation rule; (5) it changes no contract.
hypothesis: >-
  The networks' weak score against the manual heuristic on the window comes mainly from days on
  which a Primary ran in an abnormal arrangement, so removing days that a frozen detector flags
  narrows the gap between K1 and the heuristic.
prediction: >-
  Written before the window was scored with this detector. K1's CRPS minus the heuristic's falls
  from +0.10 points of capacity to below +0.05 on kept rows, and D14 and D18 lose most of their
  flagged days.
decision_rule: >-
  Descriptive second reading of a window that was already read once. Never used to rank or tune.
  Intervals are day-block bootstrap 95% intervals; the Bonferroni count is printed.
benchmark: Experiment 620's all-rows scores and its rule R1 second reading.
row_set_id: final_test_origins_2026_07_01_to_2026_10_31
metric: CRPS and MAE on kept rows, pooled and per Primary, with wins against the heuristic.
---

## Frozen detector (written before the window was touched)

- **Input:** cleaned observed power only. No network forecast is read.
- **Daily statistic:** for each Primary and UTC day, the day's mean of observed power minus the
  heuristic (all 13 members), and minus the climatology fitted to data before 2025-07-01. Each
  residual has the cross-Primary daily median subtracted. The two are averaged.
- **Scale:** each Primary's median and 1.4826 x median absolute deviation over 2025-07-01 to
  2026-06-30.
- **Rule:** flag days with |z| above k, bridge one-day gaps, keep runs of at least n days. Operating
  point k = 3.5, n = 2. Stricter k = 4.0, looser k = 3.0 (same n).
- **Chosen on development days only** (2025-07-01 to 2026-06-30), by the mean of the F1 on injected
  known-answer steps (4 worlds of 30 steps, 0.5 to 8 MW, 2 to 28 days) and the F1 on the verified
  real transfers over 1 day. At the operating point: injected precision 0.46 and recall 0.58; real
  recall 0.31, with 55 flagged days outside any labelled event (a lower bound on false alarms, since
  NGED's labels are incomplete). Injected recall by size: 0.5 MW 0.14, 1 MW 0.43, 2 MW 0.84, 4 MW and
  above 1.00.
- **Rule file** `ara_rule.json` SHA-256: `e5fb5cc08a67327d74dbc93ee51ba4e31dfc25bb25d8d4fab426bb32762b0566`.

## Plan

- Apply the frozen rule once to 2026-07-01 to 2026-10-31. Drop every row of a flagged Primary-day
  for all five systems alike.
- Report the share of rows and Primary-days flagged, the flag rate by month, whether the D14 and
  D18 episodes are recovered, CRPS and MAE per system on kept rows (pooled, per Primary, wins
  against the heuristic) with day-block bootstrap intervals, both sensitivities, and the overlap with
  rule R1.
- State what the detector cannot see: transfers of about 1 MW or less.
- Figures per the `dataviz` skill: at most two panels, one arm per panel, a legend, and no line drawn
  as the normal running arrangement.
