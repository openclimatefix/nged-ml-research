---
id: "00516"
slug: lagless-demand-network-learned-weather-window-cnn
title: >-
  Does a small convolutional encoder over a 24-hour weather window improve a lag-free demand network
  that cannot see recent power?
theme: demand-inputs
also_relevant_to:
  - weather-products-and-blending
status: planned
kind: other
protocol: demand-v2
size: >-
  complex. Stored: pretrained and fine-tuned lag-free networks with a window encoder, and their raw
  quantile forecasts, under this experiment's private data folder only; every earlier experiment
  is read and never written. Serving path: not touched (this repository holds none). Degradation
  rule: not touched. Designs: more than one defensible (centred or trailing window, which seven
  variables, how ERA5 windows are made to match ensemble windows, what the encoder returns), so
  the plan fixes each before any score is read. Callers: this folder's scripts import experiments
  397, 428, 448, 510, 513, and 515, and the `studies` evaluation code. One Sonnet plan review, one
  Sonnet code review before the first fit that produces quoted numbers, one Sonnet results review,
  and two Sonnet write-up reviews.
hypothesis: >-
  A lag-free demand network sees weather at the valid time only, yet buildings warm up and cool
  down slowly and sunshine through windows offsets heating on a cold day. A small convolutional
  encoder over the 24 hours of ensemble-mean weather around the valid time can learn which part of
  that history matters, so it improves the lag-free network, and it improves it by more than the
  hand-built trailing windows of experiment 515 (the second PR of the maintainer's 2026-11-02
  weather-window brief; the hand-built arms are experiment 515's).
prediction: >-
  Written before any fit. P1: the centred encoder (C-CNN) changes CRPS against the 37-column base by
  -0.08 to +0.02 points at both groups on all rows; P1 fails if the gain is larger than 0.08 or the
  loss is larger than 0.02. P2: C-CNN beats the trailing-only encoder (T-CNN) by under 0.03 points,
  because the forward half of the window is the same ensemble mean the base already sees at the
  valid time; P2 fails if the gap is 0.03 or more. P3: the best CNN variant beats the best
  hand-built arm of experiment 515 by under 0.03 points; P3 fails if it wins by 0.03 or more.
  P4: the control network that reads shuffled-day windows (C-CNN-null) scores within the minimum
  detectable effect of the base; P4 fails if it gains more than that, which would mean extra
  parameters and inputs, not weather history, drive any CNN gain. Each prediction can fail.
decision_rule: >-
  See "Pre-registered rules". A variant advances at a group only if its month-block bootstrap
  interval (Bonferroni over 8 primary tests) excludes zero and the point estimate is at most -0.03
  points of CRPS. A variant counts as better than the hand-built windows only if it also beats the
  best experiment 515 arm by at least 0.03 with an interval that excludes zero. A stop reads "not
  resolved", not "no effect". The preference for one simple network means a hand-built window that
  passes is chosen over a CNN of equal skill.
benchmark: >-
  The 37-column lag-free network (experiments 448, 510, and 513's P0 weights) refitted on the same
  folds as experiment 515, with the same seeds; the experiment 515 arms on the same rows;
  K2-S1, climatology, and `manual_heuristic` for scale.
row_set_id: experiment_515_folds_v1
metric: >-
  CRPS in points of effective capacity at the signed-MW six and the MVA-only eight, plus q95 and
  q99 coverage and pinball loss on cold nights (below 5 C, 18:00 to 08:00), and the cold-sunny
  against cold-overcast slices.
---

**This experiment asks whether a learned encoder over the recent weather beats both the 37-column
lag-free network and the hand-built trailing windows of experiment 515.** It is planned, with no fit
run. Experiment 515 owns the folds, the base weights, the minimum detectable effect, and the
later-runs-removed leak check; this experiment reuses them and builds only the half-hourly window
builder and the encoder.

## Question

The lag-free network (37 columns: 14 ensemble mean and spread weather columns at the valid time, 3
solar geometry, 12 calendar, 6 holiday, 2 lead, plus the Primary embedding) sees weather at the
valid time only. Does a one-dimensional convolutional encoder over the 24 hours of ensemble-mean
weather around the valid time improve it? Every earlier neural test of trailing weather (experiments
419, 437, 441, 442) fed the network three origin lags of power, which may have carried the same
information. Experiment 6 (XGBoost) is the only earlier lag-free window test.

## Arms

| Arm | What it is |
|---|---|
| C-CNN | Encoder over a 24 h window centred on the valid time (12 h before to 11.5 h after), 48 half-hourly positions of 7 variables. |
| T-CNN | The same encoder over the 24 h ending at the valid time. |
| C-CNN-null | C-CNN reading shuffled-day windows (experiment 513's placebo rule), so a gain from extra parameters alone is visible. |
| Base | The 37-column network of experiment 515, not refitted here. |

The seven variables are the ensemble means of shortwave, longwave, temperature, dew point, surface
pressure, precipitation, and 10 m wind speed. The encoder is two strided convolutions (16 then 32
channels, kernel 5) flattened (not pooled, so the position of a change inside the window survives)
into an 8-number vector that is concatenated to the 37 columns before the mixture network. The new
first layer is wider, so every arm is pretrained afresh and no weights are zero-initialised.

## The window rule

- **Every window is built on the half-hourly series, at every lead** (maintainer, 2026-11-02). At
  scoring and in fine-tuning the series is the ensemble-mean summary, which is a linear interpolation
  of the ensemble's steps (3-hourly to 144 h, 6-hourly to 360 h; confirmed in the stored NWP of
  the 2025-05-09 run: 85 steps, shortwave empty at step 0).
- **ERA5 windows used in pretraining are subsampled to that step grid and then interpolated**
  (`e00516_window.step_interpolate`), so the training and scoring series are equally smooth. Beyond
  144 h the series carries no information finer than 6 h, and nothing is finer than 3 h on day 1.
  The write-up says so.
- **ERA5 values are the `:00` rows** of `era5_features.parquet` (the `:30` rows are interpolation),
  with July 2026 cut. Shortwave comes from the hourly CAMS file with its hour-end stamps.
- **Windows reaching before the origin read the history** (each earlier run's first day), as in
  experiment 437. Nothing after the row's origin is read from a later run. A history position that
  is missing is held at the nearest finite value, and the number of missed runs is stored beside
  each row as a degradation flag.
- **Scoring uses ensemble windows built exactly as at serving**, never ERA5 windows.

## Folds, seeds, and cost

- **Folds and seeds are experiment 515's** (ten winter months, 3 seeds, no noise arm), and so are the
  base weights and the minimum detectable effect, which this experiment quotes from 515 first.
  Fits wait until 515's window features and later-runs-removed check are done, and reuse them.
- **Fit count, before timing:** 3 arms x 10 folds x 3 seeds = 90 pretrains and 90 fine-tunes.
- **Cost, before timing:** an encoder pretrain is assumed to take 3 to 5 times the 18 to 24 s of a
  37-column pretrain, so 90 pretrains take 1.4 to 3.0 GPU-hours. The first arm is timed and the fit
  count and cost reported to the manager before the remaining fits.
- **Memory:** pretraining windows are gathered on the GPU from one dense array per Primary (7
  variables on the half-hourly grid). Fine-tuning and scoring windows are gathered from one combined
  array per (Primary, run) of the history and the run (about 0.4 GB in total). One GPU process, under
  8 GB.

## Pre-registered rules

- **Bonferroni count 8:** 2 variants x 2 groups x (all rows, cold nights). Reported, not corrected: q95
  and q99 cold-night tests and the cold-sunny against cold-overcast slice.
- **Horizon bands** day 1, days 2 to 3, days 4 to 10; month-block bootstrap intervals with the fold
  count stated and per-fold signs.
- **Noise arm only for a variant that passes**, defined as in experiment 515 (temporally coherent,
  per window column and lead band, offset fitted before the refit).
- **Reporting rules:** "rows with no labelled event", "load-shift", Farm A to C and Solar A to F,
  the OCF palette with no mark in `#fffbf5`, interim numbers labelled interim and unaudited.
- **Merging:** merge commit, never squash; diff only in `ml-experiments/` and `docs/ml-experiments/`.
