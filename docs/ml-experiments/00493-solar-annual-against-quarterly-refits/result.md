---
id: '00493'
slug: solar-annual-against-quarterly-refits
title: >-
  Can the pooled solar network for the six metered farms be refitted once a year instead of every
  quarter without losing more than 1% of its quantile score?
theme: metered-solar-farms
also_relevant_to:
- generator-forecasting
- production-reference-and-best-guess
status: planned
date: 2026-11-01
kind: other
size: >-
  simple to medium. Stored: fitted networks and forecasts in this experiment's private folder only.
  Designs: one, the refit schedule is the only difference between arms and experiment 430's code
  fits both. Callers: this folder's two scripts import experiment 430's and experiment 310's
  modules. Serving path and degradation rule: not touched (this repository holds none). Asset
  graph: not touched. Step 4 of the neural simplification series, keeping only the new part.
hypothesis: >-
  Refitting the pooled solar network once a year, on data to the end of August 2025, scores within
  1% of refitting it every quarter, for the one-pass network and for the 51-member encoder, so a
  production solar forecast needs one refit a year instead of four.
prediction: >-
  Written before any fit and before any score is read. Scores are quantile scores in points of
  capacity (lower is better), farm-equal means over experiment 310's scored year, percentage change
  of the annual schedule against the quarterly schedule. C1. `g0_onepass` annual against `g0_onepass`
  quarterly (six seeds each): +0.3% (annual worse), upper end of the interval below +1%. C2.
  `a2_member` annual against `a2_member` quarterly (six seeds): +0.3%, upper end below +1%. C3.
  `g0_onepass` annual against the baseline `a2_member` quarterly: +0.5%, upper end below +1.5%
  (experiment 430 stored +0.22% for the one-pass network against the encoder). Staleness grows
  through the year, so the loss is zero in the first quarter by construction (control), and largest
  in the last quarter; the last quarter alone may exceed 1%. Each prediction can fail.
decision_rule: >-
  Annual refits are called equivalent to quarterly refits for a contrast when the upper end of its
  98.33% month-block interval (12 scored months, 10,000 draws, Bonferroni count 3 for C1 to C3, family
  level 5%) lies below the limit, and no one farm's point estimate exceeds +1%. The limit is 1%, or
  the minimum detectable effect from the seed spread if that is larger (printed first). An equivalence
  claim at the minimum detectable effect cannot show a loss smaller than the minimum detectable
  effect. A point estimate inside the limit with an interval that crosses it is inconclusive, not
  equivalent. If a contrast fails, the quarterly schedule stays for that network. Control: on the
  init months of the first quarter (2025-09 to 2025-11) annual and quarterly forecasts come from the
  same fit and seed, so their scores must agree to 0.01% (GPU nondeterminism only); if not, the run
  stops and reports. No arm is adopted by this experiment.
benchmark: >-
  Experiment 430's stored quarterly forecasts of `g0_onepass` (seeds 0 to 5) and `a2_member`
  (seeds 0 to 5), read from its private folder and never written. Baseline of the neural
  simplification series for solar: `a2_member` with four quarterly refits.
row_set_id: >-
  Experiment 310's rows: farms A to F, scored year 2025-09 to 2026-08, day bands d1, d2, d3, d4_5,
  d6_7, d8_10, rows that every compared arm holds with finite quantiles. Same as experiment 430.
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), pinball loss
  and coverage at 0.01, 0.05, 0.95, and 0.99, the 12 to 36 hour band, per farm, minimum detectable
  effect from the seed spread
code: ml-experiments/00493-solar-annual-against-quarterly-refits/
---

**This plan asks whether the six metered solar farms need four refits of the neural network a year,
or one.** National Grid Electricity Distribution (NGED) meters the farms, labelled A to F here. A
51-member ensemble weather forecast drives each network. Experiment 430 refitted every network every
quarter, and found that a one-pass network on the ensemble mean and spread scored within 0.22% of an
encoder that reads all 51 members. We now refit each of those two networks once, on data to 31
August 2025, and let that one network forecast all 12 scored months. Step 4 of the neural
simplification series (`briefs/neural_simplification_series_brief.md`) keeps only this half:
experiment 430 already stored the one-pass against encoder answer (6.842 against 6.810 over all
bands, three seeds), so we quote it and do not rerun it.

## Design

- **Arms.** `annual_g0_onepass` and `annual_a2_member`: experiment 430's `g0_onepass` and `a2_member`
  configurations with one refit at 2025-09 that predicts the runs of all 12 months. Same recipe,
  validation window, ceiling rule, and seeds 0 to 5 (six fits per arm, 12 fits in all).
- **References.** Experiment 430's quarterly `g0_onepass` and `a2_member`, seeds 0 to 5, read from
  its private folder. Quarterly refits, own history, no table P, as stored.
- **Contrasts.** C1: annual against quarterly for `g0_onepass`. C2: the same for `a2_member`. C3:
  `annual_g0_onepass` against quarterly `a2_member` (the simplest annual stack against the baseline).
  Bonferroni count 3.
- **Control.** In the first quarter the annual fit and the first quarterly fit have the same cutoff,
  data, and seed, so their forecasts must agree.
- **Read-outs.** Each contrast over all bands and in the 12 to 36 hour band; per farm A to F; by
  quarter of the scored year (months since the refit, 0 to 3, 3 to 6, 6 to 9, 9 to 12); the tails
  (0.01, 0.05, 0.95, and 0.99 coverage and pinball loss); the minimum detectable effect printed first.
- **Weather at forecast time.** Every input is ensemble data of the 00 UTC run, as in experiment 430.
  No lagged power. ERA5 and CAMS are not used.
- **Standing practices.** Monthly-refit and demand-specific rules do not apply (the question is the
  refit schedule, at the quarterly cadence of the stored references). Rows and cleaning are those
  of experiment 430. No D15 or D18 cut applies. Fitted networks are saved in the private folder.

## Serving table (filled when finished)

| Stack | Networks per pass | Inputs | Refits per year | Score against baseline |
|---|---|---|---|---|
| `a2_member`, quarterly (baseline) | 1 network, 51 members through its encoder | member-level and summary columns | 4 | 0 |
| `g0_onepass`, quarterly | 1 | ensemble mean and spread | 4 | experiment 430 |
| `g0_onepass`, annual | 1 | ensemble mean and spread | 1 | planned: C3 |

## Reviews

- A fresh Sonnet review of the two scripts before the first fit, a fresh Sonnet re-derivation of the
  headline numbers after, a fresh Sonnet review of the write-up, then the prose review.

## Compute

About 12 fits, under 0.5 GPU-hours, one GPU process under 8 GB with one feeder core. The scoring
reads stored forecasts.
