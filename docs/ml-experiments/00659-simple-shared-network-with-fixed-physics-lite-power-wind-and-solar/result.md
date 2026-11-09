---
id: "00659"
slug: simple-shared-network-with-fixed-physics-lite-power-wind-and-solar
title: >-
  Does one simple pooled network, with or without a fixed physics-lite power input, forecast the 3
  metered wind farms and the 6 metered solar farms as well as the current defaults?
theme: generator-forecasting
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
protocol: wind-v1
date: 2026-11-09
size: >-
  Medium. (1) Stores Parquet forecasts and fitted-network state files in the private data folder
  only. (2) More than one design is defensible (which raw inputs, which physics-lite power), so
  the brief fixes them before any score is read. (3) Spans code from experiments 210, 347, 430,
  499, and 615, which this folder imports or copies. (4) Touches no serving path and no
  degradation rule. (5) Changes no contract. The final test is never read.
hypothesis: >-
  A pooled network with two hidden layers of 64 units and a three-component Gaussian mixture, given
  only the ensemble mean and spread of raw weather, scores within the minimum detectable effect of
  the current default of its technology (wind default 2, solar `a2_member`). Adding the mean and
  spread of a fixed physics-lite power across the 51 members lowers the score further.
prediction: >-
  Written before any new forecast was scored. P1: the wind S arm scores within 0.15 points of
  capacity of wind default 2 over 24 to 240 h. P2: wind and solar PL+ score at least 0.05 points
  better than S. P3: the PL arms (physics-lite power only) score within 0.1 points of PL+. P4:
  replacing idle readings by zero lowers every wind score by at least 0.1 points. P5: recomputing
  capacity as the 99.9th percentile of the previous 730 days changes the scores of the two
  defaults by under 0.1 points.
decision_rule: >-
  Descriptive, for the maintainer to choose. An arm counts as no worse than its default when the
  99% month-block interval of "arm minus default" lies below 0.15 points (wind) or 0.10 points
  (solar). An arm counts as better than another when the interval excludes zero. The Bonferroni
  count and the minimum detectable effect are printed beside every contrast. Cost of porting:
  the S arm removes the physics plant, so any no-worse result is worth porting; PL+ adds a fixed
  function with no fitted parameters, so a gain of 0.05 points is the threshold.
benchmark: >-
  Wind default 2 and solar `a2_member` (quarterly refit) from experiment 615's stored outputs,
  plus `climatology`.
row_set_id: development_months_daily_origins_v1
metric: >-
  Normalised mean absolute error of the median, and quantile score over 11 levels in points of
  capacity, CRPS, q05-q95 coverage, tail shares.
---
Plan page. The write-up replaces this page when the fits have finished.

**Question.** One technology at a time, can a network that sees only the ensemble mean and spread
of weather, or of a fixed physics-lite power, forecast as well as the more elaborate defaults?
**Design.** See `briefs/wind_solar_shared_architecture_experiment_brief.md` in the private data
folder; the brief is the specification and is not repeated here.
