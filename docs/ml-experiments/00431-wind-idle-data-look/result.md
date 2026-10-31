---
id: "00431"
slug: wind-idle-data-look
title: >-
  Do idle half-hours at the three metered wind farms look different from genuine low output, and do
  they follow a weekly maintenance cycle?
theme: generator-forecasting
also_relevant_to: []
status: planned
kind: other
size: >-
  simple. Stored: summary tables and a private gallery under
  data/ml-experiments/00431-wind-idle-data-look/ only. Serving path: not touched. Degradation rule:
  not touched. Designs: one defensible design (a descriptive look at three series, no fit).
  Callers: this folder's scripts only, which read experiment 210's stored `joined.parquet`.
hypothesis: >-
  H1: the idle floor of each metered wind farm has a texture (exact zeros, repeats, a quantised step,
  small first differences) that genuine low output at the same power level does not have. H2: idle
  half-hours at high wind (a whole-farm outage or maintenance) follow a weekly cycle, because
  maintenance is done on weekday daytimes and weather-driven idling has no weekly cycle.
prediction: >-
  Written before any number is read. H1 fails for a farm if the best single texture statistic gives
  an area under the receiver operating characteristic curve (AUC) below 0.65 for sure-idle rows
  against sure-generating low-output rows. H2 fails if the share of idle half-hours at high wind on
  weekday daytimes is under 1.5 times the share at weekends and at night.
decision_rule: >-
  Sure-idle rows: ERA5 100 m speed below 2 m/s and power inside experiment 210's idle band (5% to
  95% of power at ERA5 wind below 2 m/s, per farm). Sure-generating low-output rows: power between
  3% and 8% of capacity at ERA5 wind of 6 m/s or more. Texture statistics, computed per half-hour
  from the farm's own series only: exact-zero flag, repeat-of-previous-value flag, rolling standard
  deviation of first differences, and lag-1 and lag-2 autocorrelation of first differences, each
  over 3-hour and 24-hour trailing and centred windows, amplitude relative to the farm's idle-floor
  p95, and amplitude relative to capacity. The quantisation step is the smallest non-zero absolute
  first difference (reported, not scored). Each statistic's AUC is the larger of AUC and 1 minus
  AUC, with a bootstrap interval over calendar-year blocks (2,000 resamples). Stop the cheap texture
  detector D1 if the best statistic's AUC is below 0.65 at all three farms; otherwise keep it. For
  H2, "high wind" is ERA5 100 m speed above 1.5 times the farm's cut-in from experiment 210, "idle"
  is inside the idle band, "weekday daytime" is Monday to Friday 08:00 to 17:59 UTC, and "weekend or
  night" is all other hours. Keep the whole-farm outage class I2 and the weekday covariate if the
  ratio of the two idle shares is at least 1.5 at any farm (interval reported); otherwise merge I2
  into "unexplained outage".
  Amendment before any number was read (code review): the sure-generating set has power of at
  least 3% of capacity and the sure-idle set sits inside the idle band, so the level of power alone
  separates them at some farms. The gate therefore applies only to level-free statistics (the
  rolling standard deviation of first differences, the lag-1 and lag-2 autocorrelations, the
  repeat-of-previous-value flag and its rolling share), and to each of them as a level-matched
  AUC: the AUC within bins of 0.5% of capacity, averaged with weight equal to the product of the
  two class counts. The exact-zero flag, its rolling share, and the two amplitude statistics are
  level statistics: they are reported raw and never gate D1. A farm whose two sets share no
  power bin cannot test D1; that is reported as untestable and does not count as an AUC below
  0.65. The sign of each statistic is fixed once on the full sample, and the bootstrap resamples
  the signed AUC. A constant window's autocorrelation counts as 0. The high-wind speed for H2 also
  has a sensitivity run with the empirical cut-in of experiment 210. The two sure sets are defined by the same wind speed as the 210 mask,
  so the gate tests texture, not the mask. No threshold on forecast gain is set, because no forecast
  is fitted.
benchmark: >-
  None. This is a descriptive precursor to the wind idle-noise detector design (Stage 0).
row_set_id: >-
  all half-hours of the three wind farms from 2019-09-17 to 2026-09-20 with a power reading, as in
  experiment 210; not comparable with the leaderboard
metric: >-
  AUC of each texture statistic with a year-block bootstrap interval; ratio of idle shares at high
  wind by weekday and hour.
code: ml-experiments/00431-wind-idle-data-look/
---

Plan. Cadence: no fit, so no refit. Cleaned data: the three series are customer generators, which
the production cleaning rule never flags (experiment 210); the half-hours with a missing power
reading are absent. The short-history cut applies to demand Primaries only. The data are anonymised
as Farm A, Farm B, and Farm C.
