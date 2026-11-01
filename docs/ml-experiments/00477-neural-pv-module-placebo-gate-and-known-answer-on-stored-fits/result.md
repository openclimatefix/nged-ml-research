---
id: "00477"
slug: neural-pv-module-placebo-gate-and-known-answer-on-stored-fits
title: >-
  Does the explicit PV module inside the net demand neural network recover a plausible, placebo-gated
  capacity per Primary, and what does the explicit split cost and gain on sunny half-hours?
theme: differentiable-physics-embedded-generation
also_relevant_to:
  - solar-disaggregation
status: planned
kind: other
size: >-
  medium. Stored: fitted networks, per-row forecasts, and injected copies of the data, all in this
  experiment's private folder. Production serving path: not touched. Degradation rule: not touched.
  Designs: more than one gate is defensible, so the gate is a number written before any PV-split
  placebo score is read. Callers: the code imports experiments 87, 397, 405, 409, and 416 and edits
  none of them. Asset graph: not touched.
hypothesis: >-
  The PV module of experiment 416's `C3` (a learned capacity per Primary times a physics curve of
  ensemble-mean irradiance and temperature, subtracted from a demand module that has no direct
  irradiance) recovers an injected farm's capacity at the signed-MW six, while a placebo (the farm's
  daylight profile plus its day-to-day anomaly shuffled within the calendar month) stays under a
  limit set by the same placebo on `A0-doy` (the network with free irradiance inputs and no PV
  module). A network without the power lags recovers more of the injected capacity than `C3`,
  because the lags let the demand module absorb the PV effect.
prediction: >-
  Written before any fit. (1) `C3` recovers 0.4 to 0.9 of a 2 MW and a 4 MW injected farm (ratio of
  the learned capacity increment to the injected MW), at 4 of the 6 Primaries. (2) The placebo
  recovers at most the gate limit at 5 of the 6 Primaries, and the real farm exceeds the placebo by
  more than 2 seed standard deviations at 4 of the 6. (3) The lag-masked PV-split recovers more than
  `C3` at 4 of the 6. (4) Re-scored on the 12 months, `C3` minus `A0-doy` on sunny middays repeats
  experiment 416 (+0.227, interval -0.135 to +0.560).
decision_rule: >-
  See "Pre-registered rules". The explicit PV module is kept if it loses no more than 0.02 CRPS
  points overall or 5% on q99 pinball loss against `A0-doy`, and either gains more than the minimum
  detectable effect on the sunny middays or gives the per-Primary capacity explanation of the plots.
  The 0.02 loss limit cannot be tested at the overall minimum detectable effect of 0.057, so
  adoption rests on the explanation clause and on the capacity plots (stated before any score is
  read). If the placebo gate fails, the capacities are plotted and marked unvalidated.
benchmark: >-
  `A0-doy` (free irradiance, no PV module), `A0-noirr-doy`, `H1`, and `C3p-weak`, all stored by
  experiment 416; the GAM stack, BG2, the manual heuristic, and climatology from their stored
  forecasts. The GAM-flavoured capacity estimate of experiment 474 is the side-by-side reference
  for the capacity plots.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Continuous ranked probability score from the 11 quantiles, in points of effective capacity;
  leads 0.5 to 336 hours; the headline band is days 1 to 10.
code: ml-experiments/00477-neural-pv-module-placebo-gate-and-known-answer-on-stored-fits/
---

**Status: planned.** This page is the plan, written before any fit.

## What is asked

**The question is whether the PV module of the net demand neural network reads the sky well enough
to give a capacity per Primary worth plotting, and whether the explicit split helps or hurts the
net demand forecast.** The brief (`unmetered_pv_neural_module_brief.md`, with its edits of
2026-11-01) asked for 75 refits. The queue-order check found four of the five score arms already
stored by experiment 416, so this experiment reuses them and fits only what is new.

## Review of the plan against the latest evidence

- **Reused, not refitted:** `A0-doy` (A-free), `A0-noirr-doy` (No-irr), `C3` (PV-split),
  `C3p-weak` (weak prior), and `H1` (explicit term plus free irradiance), from experiment 416's
  `work/seed_preds/`, 12 months, 5 seeds, with the `*.capacity.json` files.
- **Base recipe:** experiment 397's plain loop (`e00397_prep`, `e00405_{prep,models,pv,fit}`,
  `e00409_*`, `e00416_*`), day-of-year inputs, no pretraining. Experiment 425's arm A (pretrained)
  is a reference row only.
- **Physics as in experiment 405:** irradiance is the ensemble mean and spread of shortwave in
  training and forecasting alike; CAMS defines the sunny flag at scoring only. Tilt 35 degrees, DC to
  AC ratio 1.15, 0.4% per kelvin. The brief's 1.25 ratio and its 1.1 and 1.5 sensitivity are
  dropped, because they break the reuse of every stored `C3` fit.
- **Scope:** the six signed-MW Primaries (D5, D9, D10, D11, D14, D16), with D15 and D18 report-only.
  D5 (export limit), D12 (reverse flow, no PV module), D15 and D18 are marked in every table. The
  MVA-only Primaries are covered by experiment 474's GAM-side table.
- **Standing rules followed:** cleaned data, the D15 and D18 cut at 2025-03-01, monthly refits,
  saved networks, the tails, the seven-step process, draft pull request.

## New work (about 50 fits, one GPU process under 8 GB, one feeder core)

1. **Placebo and injection (runs first, gates the capacity plots).** On the 2026-06 fold, 3 seeds,
   inject a metered solar farm's output (Solar A and Solar C of the six; named only by letter) into
   all six signed-MW Primaries, with the lags injected as in experiment 405. Conditions: control
   (no injection); held farm at 2 MW and 4 MW for both farms; placebo (the farm's monthly daylight
   profile plus its day-to-day anomaly shuffled within the calendar month, 4 MW); profile only (the
   farm's monthly daylight profile, 4 MW). `C3` takes all of them (7 conditions, 21 fits).
   `A0-doy` takes control, placebo, and profile only (9 fits) to calibrate the gate.
2. **One new arm: `C3-nolag`.** `C3` with the three power-lag values and their missing flags held
   constant in training and at inference. It is fitted on the five fold months of experiment 449
   (3 seeds, 15 fits) and on control and the 4 MW held farm in the 2026-06 fold (6 fits).
3. **Re-scoring on CPU:** the stored arms plus `C3-nolag` on the 12 months and on the five fold
   months of experiment 449, from stored forecasts.
4. **Plots:** capacity per Primary with the prior and the 10th to 90th percentile over seeds, beside
   experiment 474's GAM estimates; implied PV on a sunny and a cloudy day; PV share of net demand by
   hour; recovery against injected MW.

## Pre-registered rules

1. **Recovery** is the seed-mean learned capacity increment (injected minus control, last
   season-year cell) divided by the injected MW, per Primary. The least-squares read-out of
   experiment 409 (`dC_LS`) is reported beside it.
2. **The gate limit** is the equivalent capacity that `A0-doy` shows on the profile-only injection:
   the least-squares regression of the change in its median forecast between control and the
   injection on the ensemble unit shape, over daylight rows of the 2026-06 test month, divided by
   the injected MW. The limit is the larger of the profile-only equivalent and the placebo
   equivalent, seed-averaged, per Primary. **The number is committed to this page, in a separate
   commit, before any PV-split placebo score is read.**
3. **The gate passes at a Primary** when `C3`'s placebo recovery is at most that Primary's limit
   plus 0.20, and the held-farm recovery exceeds the placebo recovery by more than 2 seed standard
   deviations. A failed gate marks the Primary's capacity "unvalidated" in the plots. Scores are
   reported regardless.
4. **Score contrasts** (12 months and the five fold months, signed-MW six, CRPS days 1 to 10 and
   sunny middays): `C3` minus `A0-doy`, `C3` minus `A0-noirr-doy`, `C3-nolag` minus `C3` (fold
   months only). Month-block bootstrap at 95% with 20,000 resamples, Bonferroni across the 3, one
   equal-row intersection (row set `fold_a_daily_gap_v1`), minimum detectable effect first. Sunny
   middays are CAMS clear-sky index at least 0.7, 10:00 to 15:00.
5. **Other scores:** q01, q05, q95, and q99 coverage and pinball loss; the 12 to 36 hour band;
   cold winter nights; Christmas week (report-only); the 1.15 widening; avoidable-curtailment cost
   at f = 0.9, 0.75, and 0.5 for the signed-MW six (`metrics.avoidable_curtailment_cost`); absolute
   scores beside every contrast.
6. **Expected result:** close to experiment 416, which is not new evidence by itself. The 0.02 loss
   limit is untestable at the sunny-midday minimum detectable effect of 0.28 to 0.37 (overall 0.057).

## Production cost of the PV term

The PV term needs a capacity table per Primary, refreshed at each monthly refit, and a physics chain
(irradiance and temperature from the ensemble to plane-of-array power) at forecast time. The write-up
reports this beside the score.
