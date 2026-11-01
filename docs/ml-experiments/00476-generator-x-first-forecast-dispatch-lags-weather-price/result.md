---
id: '00476'
slug: generator-x-first-forecast-dispatch-lags-weather-price
title: >-
  Can a mixture density network forecast Generator X better than same-hour climatology and
  last-output persistence, and do weather or price inputs help?
theme: generator-forecasting
also_relevant_to: []
status: planned
kind: other
size: >-
  medium. Stored: private fitted networks and private prediction tables under
  data/ml-experiments/00476-generator-x-first-forecast-dispatch-lags-weather-price/ only. Serving
  path: not touched. Degradation rule: not touched. Designs: more than one defensible design (which
  inputs a dispatch-driven series needs), so the four arms below are the design. Callers: this
  folder's scripts, which import reviewed experiment code (87, 397, 428) and `studies.*`.
hypothesis: >-
  H1: Generator X's output has autocorrelation (short spells, a two-peak daily cycle) that a mixture
  density network fed the origin lags and calendar inputs can use, so it scores better than
  same-hour climatology and than last-output persistence over days 1 to 10. H2: ensemble weather
  at the site and lagged market price add nothing, because experiment 473 found no usable
  relationship with weather proxies and only a weak one with price.
prediction: >-
  Written before any fit. Arm A0 (calendar, lags) beats climatology by 3% to 10% at 12 to 36 hours
  and by under 2% beyond 3 days; it beats persistence beyond 6 hours and loses to persistence under
  3 hours. The weather arm (A1) and the price arm (A2) each stay within one minimum detectable
  effect of A0. The weather-and-calendar arm without lags (A3) is within one minimum detectable
  effect of climatology. The idle level is a small import, so exact-zero probability is not a mass
  to model. The sign diagnostic beats the cell frequency of negative half-hours by under 5%.
decision_rule: >-
  Fixed here, before any score is read. Scored on four screen folds (2026-06, 2026-01, 2025-10,
  2026-03), 24-month training windows, monthly refit, 3 seeds, equal rows, leads up to 240 hours.
  Metric: quantile score (twice the mean pinball loss over 11 levels, 0.01 to 0.99) in points of
  the series' maximum absolute output. Intervals: bootstrap over blocks of one fold and one origin
  day, 99% level, labelled interim (four folds). Minimum detectable effect: 1.96 x sqrt(2) x
  (seed sd of A0 per fold) / sqrt(3), from A0's three seeds, stated before any contrast. Contrasts
  (Bonferroni count 5, so the 99% level): C1 A0 minus climatology, C2 A0 minus persistence, C3 A1
  minus A0, C4 A2 minus A0, C5 A3 minus climatology. Stop rule: if no arm beats both climatology
  and persistence at days 1 to 10 by more than the minimum detectable effect (interval below zero
  on both), report that Generator X is not forecastable from the available inputs, and stop.
  Smallest gain worth porting: any, for the lag and calendar arms (no new feed); for A1 and A2 one
  minimum detectable effect, because each needs a live feed that does not exist today.
benchmark: >-
  Same-hour climatology (local month, half-hour, weekend flag, quantiles of the training window)
  and last-output persistence (the newest usable reading as a point forecast), on the same rows.
row_set_id: none
metric: >-
  quantile score in points of maximum absolute output at days 1 to 10 and in the 12 to 36 hour
  band; q01, q05, q95, q99 pinball loss and coverage; sign diagnostic; scores on idle and active
  half-hours separately
---
**Plan only; no fit has run.** This experiment is the first forecast of Generator X, the one series
that experiment 473 kept for stage 2. Generator X's output is two-sided: 58.9% of its half-hours
are negative, and its idle level is a small import. The question is whether any input we hold
(recent output, calendar, weather, price) makes a forecast better than the two references.

## Design

- **Network.** The pooled-family mixture density network of experiments 397 and 428, fitted to
  Generator X alone: three Gaussians, hidden width 64, two layers, dropout 0.1 (setting g09),
  AdamW, early stopping on held-out weeks, quantiles at 0.01 to 0.99. No clip at zero and no
  folded likelihood. It is the 428 loop's S1 network without the injected-event and cross-fit parts
  (those are demand-only).
- **Arms.** A0 calendar and the three origin lags with a missing flag. A1 is A0 plus the ensemble
  mean and standard deviation of seven weather variables, one pass. A2 is A0 plus market index
  price at the three lag times. A3 is calendar and weather, no lags.
- **Weather.** The stored ensemble summary of a Primary within 1 km of the site stands in for the
  site's own cell (private mapping). It runs from 2024-04-01, so the 2025-10 fold trains on about
  18 months and the other folds on up to 24. All arms train on the same rows.
- **Calendar.** Sine and cosine of the half-hour of day and of the weekday, the lead time, and no
  day-of-year column, so the network cannot memorise the calendar position.
- **Cleaning.** Cleaned power, as in experiment 473: no half-hour of Generator X carries a cleaning
  flag, so no flagged half-hour is dropped. Exact zeros are kept as valid readings.
- **Cadence.** Monthly refit, 24-month window. Folds and seeds as in the decision rule.
- **Reuse.** No stored forecast of Generator X exists (experiment 473 fitted nothing), so every arm
  is fitted. Code reused from experiments 87 (`origin_lags`, the dense half-hourly grid), 397 (the
  network and mixture loss), and 428 (the training loop pattern and scoring).
- **Cost.** 4 arms x 4 folds x 3 seeds = 48 fits. One fit is timed first; the total goes to the
  manager before the rest. One GPU process under 8 GB, one feeder core.

## Scoring

- **Absolute scores** beside every contrast, in points of maximum absolute output, at days 1 to 10
  and in the 12 to 36 hour band, with a 1.0, 1.15, and 1.3 widening sweep.
- **Tails.** q01, q05, q95, and q99 coverage and pinball loss.
- **Idle and active.** Scores on idle (absolute output under 5% of the maximum) and active
  half-hours separately, and the probability the network gives to a negative output.
- **Sign.** Import versus export as its own diagnostic: Brier score and accuracy of the predicted
  probability of a negative reading, against the cell frequency of negative half-hours.
- **Not applicable.** The avoidable-curtailment cost needs an agreed economic story and none has
  been agreed. A production forecast would need a live price feed, which does not exist today.

## Anonymisation

Generator X is the only label in any file outside the private data folder: no name, identifier,
plant class, capacity in MW, or connection point appears in this folder or in any message.
