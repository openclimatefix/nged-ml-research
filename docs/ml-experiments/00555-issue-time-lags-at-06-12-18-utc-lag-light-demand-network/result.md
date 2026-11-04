---
id: "00555"
slug: issue-time-lags-at-06-12-18-utc-lag-light-demand-network
title: >-
  Do the power readings available at the 06, 12, and 18 UTC issue times improve the demand forecast
  over the 00 UTC weather run aged to that issue time?
theme: lagged-power-and-error-correction
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: other
size: >-
  Medium. What gets stored: new per-row score tables and forecast tables under this experiment's
  private data folder only, no fitted model. Designs: one defensible design, because the stored
  networks of experiments 538, 539, and 544 are scored unchanged on rows rebuilt for each issue
  time. Callers: the scripts import experiments 087, 397, 428, 454, 470, 510, 511, 521, and 551.
  One fresh Sonnet code review before the first scoring run. No serving path or degradation rule is
  touched.
hypothesis: >-
  NGED wants an updated forecast every 6 hours. The ECMWF ensemble runs once a day, so a forecast
  issued at 06, 12, or 18 UTC can only refresh its power readings. A lag-light demand network that
  reads the day-ago and week-ago readings, scored with the readings available at the issue time
  (readings up to the issue time minus 3.5 h data delay) and the 00 UTC weather run aged to the
  issue time, forecasts better than the lag-free network that reads no readings.
prediction: >-
  Written 2026-11-04 before any score is read. P1. At the 12 UTC issue time, R-DW (experiment 539)
  lowers CRPS against the lag-free L-1d (experiment 511) at lead times of 0 to 12 h from the issue
  time, pooled over the eight MVA-only Primaries, by at least 0.2 points of capacity; it fails if
  the lowering is under 0.1. P2. The same holds at the 06 and 18 UTC issue times (R-DW minus L-1d
  below -0.1 at 0-12 h at the eight). Experiment 539 found -0.217 at 0-6 h and -0.351 at 6-12 h from
  the 09:00 UTC origin, so P1 is expected to hold, and the question is how much of that gain
  survives when the weather is 3 to 21 hours older than the issue time.
decision_rule: >-
  Supported if P1 holds with a month-block bootstrap interval that excludes zero. Not supported if
  the P1 difference is above -0.1. Between those two values the verdict is small gain.
  Intervals are 95% month-block bootstraps with 5 blocks (near the minimum to maximum of the five
  folds), with per-fold signs and the minimum detectable effect (L-1d's two-seed spread) beside every
  contrast. The counted tests are R-DW minus L-1d at 4 issue times x 4 lead bands (0-6, 6-12,
  12-24, 24-48 h) x 2 row sets (the eight and the six), so the Bonferroni count is 32. Every other
  contrast (R-DROP, R9, D15 and D18) is report-only. The smallest gain worth porting is 0.2 points
  because the change adds three pipeline runs a day.
benchmark: >-
  L-1d (experiment 511), the lag-free network with the 00 UTC weather run, scored on the same rows
  and lead bands counted from the issue time. Its forecast does not change with the issue time, only
  the rows left (valid times after the issue time).
row_set_id: experiment_470_intersection_v1
metric: >-
  CRPS in points of effective capacity at the eight MVA-only and the six signed-MW Primaries (the
  six: rows with no labelled event); D15 and D18 are report-only; July 2026 is unused. Five folds of
  511 (2025-10, 2025-12, 2026-01, 2026-03, 2026-06).
---
**Plan only; no score has been read.** Experiment 555 asks whether power readings that are fresher
than the 09:00 UTC origin of the daily ECMWF run improve the forecast issued at 00, 06, 12, and 18
UTC.

## Design

- **Forecasts.** The stored R-DW (539), R-DROP (544), and R9 (538, the network that also reads the
  newest reading) weights are loaded unchanged, one daily refit per origin day and seed, and scored
  on test rows rebuilt for each issue time. No network is refitted.
- **Issue times.** Each origin day's ECMWF 00 UTC run (usable from 09:00 UTC) is aged to the issue
  time on the same day (12 and 18 UTC) and the next day (00 and 06 UTC). A row is kept if its valid
  time is after the issue time.
- **What changes at an issue time.** The weather columns are those of the 00 UTC run at the valid
  time. The lead and its square root are counted from the issue time, and the day-ago and week-ago
  readings (value, missing flag, age; for R9 also the newest reading) are retaken from the latest
  same-clock half-hour at or before the issue time minus 3.5 h.
- **Benchmark.** L-1d's stored forecast of the same run, which does not depend on the issue time.
- **Origin-time asserts, row by row at every issue time.** The weather run's publication time
  (init + 9 h) is at or before the issue time; every lag source is at or before the issue time minus
  3.5 h; each lag age equals the target minus its source; the rebuilt table at the 09:00 UTC origin
  reproduces the stored table. One example row is printed for each issue time.
- **Cost.** About 0.3 GPU-hours of inference, one process under 8 GB.

## Caveats written before the run

- The refit used for an origin day was fitted for the 09:00 UTC origin, so it is up to one day older
  than a refit a live service would use at 00 and 06 UTC on the next day.
- The networks were trained only on 09:00 UTC origins. A network that sees the lead counted from
  another origin is outside its training rows, so a loss here may reflect that, and a short
  fine-tune on issue-time rows is the follow-up.
- NGED's event logs are incomplete, so clean rows may hold unlabelled load-shift events.
