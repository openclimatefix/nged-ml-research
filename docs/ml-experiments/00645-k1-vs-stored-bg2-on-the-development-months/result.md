---
id: "00645"
slug: k1-vs-stored-bg2-on-the-development-months
title: >-
  Did K1 beat BG2 on the five development months, when both were scored on the same rows?
theme: production-reference-and-best-guess
also_relevant_to: []
status: planned
kind: other
protocol: demand-v2
size: >-
  Simple. (1) It stores one scored table per cut and no model; (2) one design, a free re-score of
  stored forecasts on the rows they share; (3) it reuses the scorer of experiment 620 and the
  detector of experiment 637, whose callers are known; (4) it touches no serving path and no
  degradation rule; (5) it changes no contract.
hypothesis: >-
  Experiment 633's BG2 lead over K1 on July to October 2026 is a feature of that window and of
  BG2's fresher power lags, not a lead BG2 already held on the development months. On the five
  development months, K1 scores no worse than the stored BG2 of experiment 87 on event-free rows at
  the 13 Primaries other than D14, D15, and D18.
prediction: >-
  Written before any BG2 row is joined to the K1 rows. On the five development months K1's CRPS
  minus BG2's is below +0.05 points of capacity on event-free rows at those 13 Primaries, pooled over
  the four lead bands. On all rows at 16 Primaries it lies between -0.30 and +0.30, against the
  window's -0.71 (K1 minus BG2 +0.71 in 633's sign).
decision_rule: >-
  Descriptive. The development months were already read by experiments 87, 425, 449, 470 and 613, so
  the result decides nothing and ranks nothing. The reading counts as "K1 held its own on the
  development months" when the pooled CRPS contrast (K1 minus BG2) on event-free rows without D14,
  D15, and D18 has a month-block 95% interval whose upper end is below +0.10; as "BG2 led" when the
  interval lies wholly above +0.10. Anything else is reported as unclear. Threshold 0.10 is half the
  window's lead; no port is at stake.
benchmark: >-
  Experiment 87's stored BG2 (arm `p_s6_bg2`), the manual heuristic, and experiment 633's window
  numbers.
row_set_id: dev_months_origins_2025_10_to_2026_06_export_roster
metric: >-
  CRPS (twice the mean pinball loss over 11 levels) and MAE of the median in points of capacity,
  with 90% coverage and q01, q05, q95, and q99 pinball loss and coverage.
---

**Plan, written before any BG2 row is joined to K1's rows.** The page is filled in after the run.

## Question and design

- **Question:** did K1 beat experiment 87's stored BG2 on the five development months (origins in
  2025-10, 2025-12, 2026-01, 2026-03, and 2026-06)? Experiment 633 found BG2 ahead of K1 by 0.71
  points on July to October 2026, and [the leakage audit](https://openclimatefix.github.io/nged-ml-research/ml-experiments/00633-bakeoff-final-test-bg2-and-lag-enabled-bsp-gsp-network/result/)
  found no stored development-month comparison of the two.
- **Systems:** K1, the Candidate, K3, the manual heuristic, the climatology, and BG2.
- **Reused, not refitted:** the roster export `backtest_export_roster_2026-11-07/` (K1, Candidate,
  K3, five seeds averaged, from experiments 612 and 613) and `p_s6_bg2` of experiment 87 (the full
  five-member BG2 with weekly refits, spread factor 1.15, 292 daily origins at 09:00 UTC, real lags
  with the newest reading 3.5 hours old). The plan follows from reading 87's code
  (`origin_lags`, `NEWEST_AGE`); the page confirms it from the stored boosters' inputs.
- **Rows:** the join of the two on (`time_series_id`, `power_fcst_init_time`, `valid_time`), leads
  under 240 hours (BG2 stores to 351), valid times before 2026-07-01, and a finite cleaned
  `observed_power`. The page prints the common row count per month, the rows dropped from each side
  and why, and the Primary-days with no BG2 forecast (experiment 87 stored 292 of the 365 daily runs).
- **Scorer:** a copy of the 620 scorer's formulas (`e00620_score.py`, imported, not edited), with BG2
  added as a system. Climatology is fitted before the first day of each month. No window day is read.
- **Cuts:** pooled at 16 Primaries; without D15 and D18; without D14, D15, and D18; each on all rows
  and on event-free rows. The event-free flag is experiment 637's frozen detector (rule file hash
  checked against the committed `ara_rule.json`, applied to development days, which it was chosen on)
  and, as a second reading, the export's own `event_free` column. By Primary, by lead band (day 1,
  day 2, days 3 to 5, days 6 to 10), and by month.
- **Intervals:** month-block bootstrap (origin months resampled, 5 blocks, so the intervals are
  coarse and the page says so) and, beside it, the day-block bootstrap of experiment 620.
- **Comparison with the window:** one table of K1 minus BG2 per lead band, on the development months
  and on experiment 633's window (numbers quoted from its page, not recomputed).
- **Standing practices:** cleaned data (flagged half-hours are the export's nulls, counted per month);
  D15 and D18 cut before 2025-03-01 (inherited from both sources, checked); no refit (monthly
  cadence of K1 and weekly cadence of BG2 are inherited and stated); tails reported beside CRPS;
  generators not involved. Past-weather caveat: both sources use the 00 UTC ensemble run, as the
  audit confirmed; no new weather is read.
- **Cost:** CPU only, about an hour, no download.

## Reviews

Code review (fresh Sonnet) before the first scored number; write-up review (numbers and prose) after.
