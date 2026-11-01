---
id: "00453"
slug: effective-capacity-upper-bound-data-look
title: >-
  Do the metered wind and solar farms' power series hold implausibly high values, and should
  effective capacity be the maximum, the 99.99th, the 99.9th, or the 99th percentile?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
status: planned
kind: other
protocol: wind-v1
date: 2026-11-01
size: >-
  simple. A CPU-only look at the stored cleaned power series and the stored forecasts of
  experiments 429, 430, and 450; no fit. Serving path: not touched. Sonnet only.
hypothesis: >-
  The farms' historical power holds few implausible high values, so an effective capacity set at
  the p99.9 or the maximum would cut the share of actuals above 1 and the share above q99 without
  harming the CRPS.
prediction: >-
  Written before any script ran. Most half-hours above p99 are plausible (a smooth rise to and from
  the value, with strong wind or clear sky). A few single-half-hour spikes or stuck runs exist at
  one or two series. The p99.9 gives the best trade-off for wind and solar. Farm A's share above
  q99 falls below 1.5% at p99.9.
decision_rule: >-
  Recommend, per technology, the highest percentile whose upper tail is at least 90% plausible
  after the screening rules in the plan, and which does not raise the pooled CRPS (in MW) by more
  than 1% against the p99 rule. If the maximum contains an implausible value, the maximum is
  rejected whatever its score.
benchmark: >-
  The current rule: the 99th percentile of each series' cleaned power over its whole record.
row_set_id: fold_a_tuesdays_v1
metric: >-
  Share of actuals above 1; share of actuals above the stored q99; CRPS of the stored forecasts
  rescaled to each capacity.
verdict: inconclusive
production_status: not-for-production
adopted_in: none
code: ml-experiments/00453-effective-capacity-upper-bound-data-look/
---

**Plan (before any script has run).** Interim and unaudited.

- **Series.** Farm A, Farm B, Farm C (experiment 429's wind farms) and the six solar series of
  experiment 430, labelled Solar A to Solar F, of which Solar A and Solar B are the two the brief
  names. No name or ID appears in any file.
- **Step 1.** Describe each series' upper tail: maximum, p99.99, p99.9, p99, the number and the
  dates of half-hours above each; plots of power against time near the top and of the sorted tail.
- **Step 2.** Classify each half-hour above p99 as implausible if it is a single-half-hour spike
  (above both neighbours by more than 20% of capacity), part of a stuck run (four or more equal
  consecutive readings at the top), or follows a level step (the 48-half-hour rolling median
  before and after differs by more than 15% of capacity). Otherwise it is plausible. For wind,
  relate the high values to ERA5 100 m wind speed; for solar, to the clear-sky irradiance computed
  from the Haurwitz formula.
- **Step 3.** For each test month of experiment 429 and 430 (11 and 12 months), set the capacity at
  the maximum, p99.99, p99.9, or p99 of the series' power before that month, rescale the stored
  forecast quantiles and actuals, clip the quantiles to 0 to 1.05 of the new capacity, and report
  the share of actuals above 1, the share above q99, and the CRPS in MW.
- **Reviews.** A fresh Sonnet reviews the analysis code and the classification before the numbers
  are used; fresh Sonnets review the numbers and the write-up.
