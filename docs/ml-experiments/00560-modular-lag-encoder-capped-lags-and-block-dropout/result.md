---
id: "00560"
slug: modular-lag-encoder-capped-lags-and-block-dropout
title: >-
  Does a lag encoder beside the weather and datetime encoders let the modular demand network use
  24-hour-capped power lags as well as the lag-light network does?
theme: lagged-power-and-error-correction
also_relevant_to:
  - retraining-and-training-history
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: new network weights, forecasts, and scores in a private folder. More than one
  defensible design (what the lag encoder reads, how "expected" is built, shared or unshared
  weights per lag). Spans code whose callers need a search (experiments 397, 418, 425, 510, 511,
  545, and 558). Serving path and degradation rule: not touched (this repository holds none).
  Cost: about 3 GPU-h.
hypothesis: >-
  A lag encoder trained in ERA5/CAMS pretraining, next to the weather and datetime encoders of
  experiment 545's M0, lets the modular network use power lags that are at least 24 hours old.
  Feeding the encoder what the weather and datetime encoders say about each lag's own time
  ("expected") lets it read the lag as a deviation from normal, which a lag alone does not give.
prediction: >-
  Written before any fit. P1: variant 3 (lags plus expected) beats variant 1 (lags alone) by at
  least 0.05 CRPS at 24-120 h at the eight. P2: variant 4 (expected only, no actual lagged power)
  is within 0.03 of M0, so the encoder adds nothing without power. P3: the best arm is within 0.04
  of A-pool-24 (experiment 558) and beats R-DW-24 (experiment 558). P4: the injected-shift leak of
  the best arm is at most 6.0 points of capacity (L-1d 4.49; M0 7.3 to 7.7). Any of these may fail.
decision_rule: >-
  Contrasts are paired by row. Intervals are 5-block month bootstraps (near the minimum to maximum
  of the folds), with per-fold signs and the minimum detectable effect beside each. Counted
  contrasts for the Bonferroni correction: P1 (1 band group), P2 (1), P3 (2), so 4 tests; the
  lead-band breakdown is descriptive. The threshold for porting is 0.03 CRPS, because a lag encoder
  adds a network part and a pretraining input. Stop rule: if variant 3 does not beat variant 1 by
  0.03, stage B (b) is skipped. Wording is "interval excludes zero" or "not shown".
---

# Plan: a modular lag encoder with capped lags (interim, unaudited)

**Take-home message (plan only).** No number exists yet. The experiment tests whether the modular
network of experiment 545 can use power readings at least 24 hours old through a third encoder,
and which way of feeding that encoder works best.

## Lag inputs

Lags are those of experiment 558 (L1): the day lag and the week lag taken at least 24 h before the
origin, no newest-value lag, each as value, missing flag, and age. Block dropout is 558's: in 30%
of training days all lags are masked, and in 15% of those only the day lag is. This experiment
re-uses 558's lag builder and does not rewrite it. The lag encoder is pretrained on ERA5/CAMS rows
with the same capped lags and dropout, then fine-tuned as in experiment 545 (fresh mixer that also
reads ensemble spread and lead).

## Arms

- **Stage A** (one joint encoder over both lags):
    1. lags alone;
    2. lags plus the datetime features;
    3. lags plus "expected", the embeddings that copies of the datetime encoder and the weather
       encoder (shared weights) give for the weather and datetime of each lag's own time;
    4. expected only, with no lagged power (the control).
- **Stage B**, for the best of variants 1 to 3: (a) one lag encoder per lag, weights shared across
  lags, seeing one lag with its weather and datetime embeddings at a time; (b) the same with
  unshared weights.
- References, stored: M0 (545 and 554), L-1d, and, from 558 once it is scored, R-DW-24 and
  A-pool-24.

## Checks and reporting

- Every lag time is at least 24 h before the origin and every weather input comes from the run
  issued at or before the origin, asserted row by row, with one example row printed per lag per
  lead band. The weather at a lag's own time comes from the freshest run issued at or before the
  origin, never from ERA5/CAMS at fine-tuning or scoring; if the run summaries cannot supply it,
  variant 3 and Stage B are reported as blocked rather than approximated.
- CRPS by the seven lead bands plus 0-6 h and 6-12 h at the eight and the six; D15 and D18
  report-only; rows with no labelled event; leave-D8-out; injected-shift leak beside every score.
- For every arm, scores with lags removed and with lags aged by 1, 2, and 3 extra days (inference
  only). Metered generators appear only as Farm A to C and Solar A to F. Numbers are interim and
  unaudited.
- 2 seeds, 5 folds of 511, daily refit with the 90-day half-life; July 2026 unused. No XGBoost.
  One fresh Sonnet code review before the first number is used.

## Dependency and cost

GPU fits wait until 558's lag builder is on a pushed branch. Cost is about 3 GPU-h across 6 arms
(pretraining is the largest part); the first pretraining is timed before the rest start, and the
manager is asked before going over.
