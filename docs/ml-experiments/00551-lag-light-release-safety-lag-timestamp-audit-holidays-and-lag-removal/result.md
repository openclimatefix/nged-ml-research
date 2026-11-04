---
id: "00551"
slug: lag-light-release-safety-lag-timestamp-audit-holidays-and-lag-removal
title: >-
  Is the lag-light demand network safe to release: do its lag timestamps look ahead, does it hold
  up in Christmas week and on bank holidays, and does it survive removed or late power readings?
theme: lagged-power-and-error-correction
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
size: >-
  medium. Stored: new score tables and one audit file under this experiment's private data folder
  only. Serving path: not touched (this repository holds none). Degradation rule: not touched.
  Designs: one defensible design per part, because the briefs fix the audit, the slices, and the
  perturbations. Callers: the scripts import experiments 087, 397, 428, 454, 470, 510, 511, 521,
  and 544 (and read 539's stored weights and predictions). One fresh Sonnet code review before the
  first scoring run, no further review unless it flags many problems.
hypothesis: >-
  The lag-light demand network R-DW (experiment 539) and its dropout variant R-DROP (experiment 544)
  read only power readings that existed at the forecast origin, are no worse than L-1d in Christmas
  week, and degrade gracefully when readings are removed or late.
prediction: >-
  Written before any score is read. P1 (audit): zero rows where a lag's source time is later than
  the origin minus 3.5 h, where the age column differs from the origin-to-source gap, or where the
  missing flag disagrees with a missing value. P2 (calendar): R-DW is no worse than L-1d by more
  than 0.10 CRPS in Christmas week at the eight. P3 (removal): R-DROP with every reading removed is
  within 0.15 CRPS of L-1d at the eight; R-DW without dropout loses more than 0.5 when its readings
  are removed; the part fails if R-DROP loses more than 0.3.
decision_rule: >-
  P1 fails on a single violation, which voids the R-DW and R-DROP headlines and stops the line. P2
  and P3 are read as point values, because a slice of a few days gets no interval claim; any slice
  with fewer than 10 days shows per-year signs only. Intervals, where shown, are month-block
  bootstraps over 5 blocks (near the minimum to maximum range of the folds). The minimum detectable
  effect is L-1d's two-seed spread, and the Bonferroni count is stated as a number in the page.
benchmark: >-
  L-1d (experiment 511), A-pool (experiment 425), R-DW (experiment 539), R-DROP (experiment 544),
  and the lag-free recipe with trailing temperature windows (experiment 548, arm E), all read from
  their stored forecasts or stored weights.
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at the eight MVA-only and the six signed-MW Primaries; D15
  and D18 report-only; July 2026 unused.
---

Plan only. The write-up replaces this page when the scores are in.
