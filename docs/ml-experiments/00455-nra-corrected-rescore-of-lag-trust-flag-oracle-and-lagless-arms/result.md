---
id: "00455"
slug: nra-corrected-rescore-of-lag-trust-flag-oracle-and-lagless-arms
title: >-
  Does an exact record of which power readings lie inside a load-shift event help a lagged and
  lagless mixture when the forecast is scored against the normal running arrangement?
theme: lagged-power-and-error-correction
also_relevant_to:
  - lsc-detectors
status: planned
kind: other
protocol: demand-v2
size: >-
  simple. Stored: one table of per-cell contrasts in this experiment's private folder. No fit and
  no new forecast: it rescores stored forecasts of experiments 404 and 445 against experiment 435's
  NRA-corrected actuals, on one CPU core. One design (experiment 404's audit script, extended to
  more arms). Production serving path: not touched (this repository holds none). Degradation rule:
  not touched. Callers: the script imports experiment 404's modules.
hypothesis: >-
  Scored against NRA-corrected actuals instead of metered actuals, the hard flag-oracle mixture
  (O-hard) scores lower than B-mix (a fixed 50/50 mixture of the lagged and lagless forecasts)
  on rows inside real load-shift events, so detecting events would matter under the NRA target.
prediction: >-
  Written before any score is read. Experiment 445 found O-hard +0.465 worse than B-mix against
  metered actuals at the signed-MW six. P1. Against NRA-corrected actuals the contrast moves
  towards zero by under 0.5 points of capacity but stays above zero at depth factor 1.0, so the
  445 refutation stands. P2. The lagless head scores worse than B-mix at every depth. P3. Dropping
  the largest event of each month changes the sign of at least one O-hard contrast.
decision_rule: >-
  Contrast = arm minus B-mix on the same rows inside padded real events (target inside a padded
  event or in the 96 half-hours after one), mean of month means (seeds within a month averaged
  first), signed-MW six as the headline and the MVA-only eight reported apart. Interval = month-block
  bootstrap (resampling the 5 months, 20,000 draws) at the Bonferroni level for 48 contrasts (8 arms
  against B-mix, at 3 depth factors, in 2 groups; the with-and-without-largest-event variants
  repeat the same contrast and are not counted again). If O-hard minus B-mix is below zero with a
  Bonferroni interval that excludes zero, with and without the largest event of each month, at
  all three depth factors (0.75, 1.0, 1.25), detection matters under the NRA target. Otherwise
  the 445 refutation stands under both targets. A 5-month bootstrap at the Bonferroni level cannot
  resolve a tail beyond the smallest and largest resampled mean, so the rule is conservative: a
  contrast that passes it is large relative to the spread of months. The minimum detectable effect
  (2.8 times the standard error of the monthly differences) is reported beside each contrast.
  The smallest gain worth reporting is zero, because the O-rule is a diagnostic ceiling and is not
  ported. Interim and unaudited.
benchmark: >-
  B-mix, experiment 404's stored fixed 50/50 mixture of A0-inj and the lagless table.
row_set_id: experiment_404_scoring_rows
metric: >-
  Mass-weighted pinball loss (CRPS-like) in points of capacity from 11 quantiles, as in experiment
  404's report; no widening beyond what the stored forecasts carry.
take_home_message: >-
  Planned. Result pending.
advice: []
code: ml-experiments/00455-nra-corrected-rescore-of-lag-trust-flag-oracle-and-lagless-arms/
---

**Planned.** This experiment rescores stored forecasts and fits nothing. It adds the lagless head
and the three flag-oracle mixtures of experiment 445 (O-hard, O-soft, O-floor) to the rescore of
experiment 404 against NRA-corrected actuals.

- **Row 1 of the review is answered by experiment 404's audit** (`e00404_audit.py` and the
  `work/audit/` files `cells.parquet`, `pooled_event_weighted.csv`, and `event_counts.csv`). Nothing
  here repeats it.
