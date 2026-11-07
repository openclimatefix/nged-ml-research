---
id: "00630"
slug: roster-upper-demand-tail-calibration-event-days-and-bias-correction
title: >-
  Is the roster networks' upper demand tail too low on ordinary days, and would a bias correction
  fitted on earlier months remove the excess out of sample?
theme: calibration-and-uncertainty
also_relevant_to: []
status: planned
kind: other
protocol: demand-v2
date: 2026-11-07
size: >-
  simple. Stored: tables and figures in this experiment's private folder only; no forecast is fitted
  or written for reuse. One defensible design (re-score the exported quantiles; the correction is a
  per-Primary, per-lead-band shift or rescale fitted on earlier months). Callers: this folder's
  scripts only, which read the roster export and the ECMWF ensemble temperature and call no
  reviewed module beyond `studies.*`. One Sonnet code review before the first scored run, one Sonnet
  results review, one Sonnet write-up review.
hypothesis: >-
  The roster networks (K1, the class-2 candidate, K3) exceed their q90 and q95 for more actuals than
  nominal (the manager's check on all rows: 14% to 17% above q90, 8% to 10% above q95) on ordinary
  days as well as on load-shift event days, at most Primaries, so the excess is a bias of the
  median that a shift fitted on the first three development months removes on the last two.
prediction: >-
  Written before any score is read. P1. The candidate's event-free q90 exceedance stays above 12% in
  at least 3 of the 5 months (`bias on ordinary days`), and event rows add less than 5 points to it.
  P2. The excess is larger at lead 0 to 24 hours than at 120 hours and over, and larger on cold
  nights than on warm hours. P3. A per-Primary, per-lead-band shift brings the event-free q90
  exceedance of the last two months inside its 95% interval around 10%, and q95 stays outside its
  interval around 5% (a shift does not widen the upper tail). P4. The rescale does the opposite:
  q95 inside, q90 outside. P5. The CRPS change from either correction is within 1%.
decision_rule: >-
  Fixed before any score is read. Rows: every row of the export with an observed power reading,
  lead 0 to 240 hours, for the five development months (2025-10, 2025-12, 2026-01, 2026-03,
  2026-06; the month is the month of the forecast origin). Never the July to October 2026 window.
  Exceedance is the share of rows with `observed_power` strictly above the quantile (strictly below
  for q05 and q10). Event-free rows are the export's `event_free` rows (no labelled or verified
  load-shift within 7 days at that Primary); event rows are all others. NGED's logs may miss
  events, so event-free rows can hold unlogged load-shifts and every event count is a lower bound.
  `event-driven` if the candidate's event-free q90 exceedance is within 2 points of 10% (8% to 12%)
  in the pooled five months and its event-row q90 exceedance is at least 5 points higher.
  `bias on ordinary days` if the candidate's event-free q90 exceedance is above 12% in at least 3
  of the 5 months. If both or neither hold, the answer is `mixed`. `correction usable` if, on the
  last two months, a correction fitted on the first three (2025-10, 2025-12, 2026-01) brings the
  candidate's event-free q90 and q95 exceedance each to a value whose 95% interval contains the
  nominal level, and its CRPS (twice the mean pinball loss over the 11 quantiles) is within 1% of
  the uncorrected candidate on the same rows. Two corrections are fitted per Primary and lead band
  (0 to 24, 24 to 72, 72 to 120, 120 to 240 hours) on event-free rows only: (S) an additive shift
  d, the median of observed power minus q50, added to every quantile above q50 (then sorted); (R) a
  factor c on a grid from 1.0 to 3.0 in steps of 0.05, applied to the distance above q50 of every
  quantile above q50, chosen so the training q90 exceedance is closest to 10%. K1 and K3 are
  reported by the same rule as references. Intervals: 20,000 resamples of days within each month
  (the independent unit is the day; half-hour rows are autocorrelated), 95% percentile interval;
  no month-block resampling because five months is too few. No multiplicity correction because
  the three words and the correction are the only gating questions. No rescue tuning after a
  failed bar.
benchmark: >-
  The uncorrected exported forecasts of the same network on the same rows, and the nominal levels
  (5%, 10%, 90%, 95%).
row_set_id: roster_export_2026-11-07_dev_months
metric: >-
  Exceedance at q05, q10, q90, q95 by month, lead band, hour of day, Primary, and temperature
  tercile (cold nights separately); mean of observed power minus q50 in MW and as a share of the
  Primary's median demand; CRPS and coverage of the corrected forecasts on the last two months.
production_status: not-for-production
adopted_in: none
---

**Plan only. The results and the write-up replace this page when the experiment finishes.**

The manager's quick check (read before this plan was written) found that on all rows of the roster
export the networks exceeded q90 for 14% to 17% of actuals and q95 for 8% to 10%, with the lower
tail near nominal. [Experiment
626](https://github.com/openclimatefix/nged-ml-research/blob/main/docs/ml-experiments/00626-widen-the-tails-of-the-51-scenario-member-values/result.md)
found the same pattern in an earlier demand forecast: widening its tails did not fix its q90, which
pointed to a median that sits too low. This experiment splits the roster's excess by event days,
month, lead, hour, Primary, and temperature, and tests whether a correction fitted on earlier months
removes it.

- Standing practices followed: cleaned data (the export's `observed_power` is cleaned; flagged
  half-hours are null and excluded), the D15 and D18 cut is not needed (the export holds 16
  Primaries only), no fit, no weather input to any network.
- Temperature source: the ECMWF ensemble 2-metre temperature (mean of members) in the production
  Delta table, at the Primary's H3 cell, from the latest run that started at or before the valid
  time. It is a forecast-time quantity, available when the forecast is made. Terciles are taken
  per Primary over all its rows; a cold night is a half-hour from 18:00 to 06:00 UTC in the lowest
  tercile.
- Caveats: the Candidate's band widths were set on the same months they are applied to; the
  export holds one verified real transfer (D5 and D16, December 2025).
