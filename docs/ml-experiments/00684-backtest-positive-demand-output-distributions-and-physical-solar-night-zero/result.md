---
id: "00684"
slug: backtest-positive-demand-output-distributions-and-physical-solar-night-zero
title: >-
  Can output distributions that cannot go below zero, and a solar output that is zero at night by
  construction, replace the post hoc clamps in the one-year backtest files?
theme: production-reference-and-best-guess
also_relevant_to:
  - calibration-and-uncertainty
  - generator-forecasting
status: planned
kind: other
size: >-
  Complex. (1) It stores refitted demand networks and quantile forecasts for two test months and one
  solar refit; (2) two output families for demand and one construction for solar are defensible,
  and the plan fixes them; (3) it patches the loss and the quantile read-out of a pipeline that
  spans experiments 397, 510, 560, 574, 591, 613, 620, and 667, whose callers cannot be named
  without searching; (4) it touches no serving path and no degradation rule; (5) it changes no
  contract.
hypothesis: >-
  A positive output family at the four Primaries that cannot go below zero, and a solar output that
  is the physics-lite power at the half-hour's own zenith times the network's ratio, remove the
  negative lower quantiles and the non-zero night quantiles of the backtest files without costing
  accuracy.
prediction: >-
  Written before any fit. P1: after the change no q01 is negative at ids 14, 15, 26, and 17. P2:
  CRPS at those four series changes by within 0.01 points of capacity against the control (the
  stored output of experiment 667), per family. P3: q05 to q95 coverage is not worse than the
  control by more than 0.01. P4: at Solar B and Solar E no night half-hour carries a non-zero
  quantile, and the daylight CRPS changes by within 0.01 points of capacity against experiment
  666's output.
decision_rule: >-
  Stop rule for demand: if a family worsens the CRPS of any of the four series by 0.05 points or
  more, keep the clamp and drop that family. A family passes if P1, P2, and P3 hold for all four
  series. Solar passes if P4 holds at both farms. The model-level fix is compared with the post
  hoc clamp of the stored files as a fourth arm. No ranking beyond that, and no claim of benefit
  from two test months.
benchmark: >-
  The stored forecasts of experiment 667 (demand, normal-running output) and experiment 666 (solar),
  with and without the post hoc clamp of the stored backtest files.
row_set_id: backtest_year_2025_11_to_2026_10_daily_12utc
metric: >-
  CRPS and normalised median error in points of capacity, q05 to q95 coverage, and q01, q05, q95,
  and q99 coverage and pinball loss, by lead band; the share of rows with a lower quantile at or
  below zero.
production_status: not-for-production
adopted_in: none
---

**Plan only: nothing has run.** The one-year backtest files hold two defects that the files'
checks found: negative lower quantiles at the Primaries that meter apparent power (MVA), and small
non-zero solar quantiles at night. A post hoc clamp removes both. This experiment asks whether the
forecasting models can instead produce the right answer themselves, by refitting them with an output
that cannot go below zero (demand) or that is multiplied by the physics-lite power at each
half-hour's own sun position (solar), and whether the refit costs accuracy.

## Plan

**Scope.** Demand: Primaries with ids 14, 15, and 26 (MVA, median within one band-width of zero on
65% to 97% of rows, upper half-width 1.2 to 1.5 times the lower) and id 17 (MW, observed power
never negative, kept as a control that must not get worse). Original ids appear only in this plan.
Solar: Solar B and Solar E (original ids 22 and 30). Only the normal-running (NRA) output of
experiment 667 is refitted. The prevailing output mixes the NRA quantiles with a shifted copy, so a
fix there waits for the NRA result.

**Demand arms.** Each arm is a refit of experiment 667's Candidate on two folds, with test months
2026-02 and 2026-05 (the fold recipe is experiment 667's, monthly refit, 2 seeds, D15 and D18 cut
before 2025-03-01, cleaned power). Test month 2026-08 lies inside the final-test window and is not
read. Only the output family changes, and only at ids 14, 15, 26, and 17. Every other Primary keeps
the Gaussian mixture, so a Gaussian mixture also remains for ids 10 and 12, which do go negative.

- **Control:** the stored forecasts of experiment 667 for the same folds. The stored clamped file
  (`primaries.parquet`) is the fourth comparison.
- **Truncated Gaussian mixture:** the same three-component Gaussian mixture, normalised over
  power at or above zero. The negative log likelihood adds the log of the mixture's mass above zero,
  and the quantiles are the mixture's quantiles conditioned on power at or above zero, read by
  bisection on the conditional cumulative distribution.
- **Gamma mixture:** three gamma components on power over its Primary's scale, with each
  component's mean and coefficient of variation read from the head's outputs. The negative log
  likelihood uses the gamma log density (observed power is clipped to 1e-4 of capacity), and the
  quantiles are read by bisection on the mixture's cumulative distribution (the regularised
  incomplete gamma function). A log-normal mixture is not run: it needs a positive offset that
  re-creates negative quantiles.

**What is refitted.** The lag, weather, and date-time encoders keep the weights pre-trained on ERA5
and CAMS, and the K1 chain's step-2 weights are reused. The fresh mixer and the daily fine-tune are
refitted with the new loss, because the Candidate re-initialises its mixer at that stage
(`fresh_mixer`). The limit is therefore that the encoders were pre-trained under the Gaussian loss,
and the write-up says so. The check that comes first in implementation is that a refit with the
Gaussian loss reproduces experiment 667's stored quantiles on one fold to within seed-level
differences. If it does not, the plan is revised and the manager told.

**Solar arm.** One quarterly refit, 2026-02-01 (it scores origins from February to April 2026, all
before 2026-07-01), 3 seeds, at Solar B and Solar E, and the other four farms to keep the pooled
network the same as in experiment 666. The physics-lite power already uses the half-hour's own
zenith cosine, so it is exactly zero at night. The network now outputs a mixture on the ratio of
observed power to the physics-lite power (centred at 1), the loss carries the Jacobian of that
change of scale, and each quantile is the physics-lite power times the ratio's quantile, clamped at
zero and at capacity. Night half-hours (zenith at the middle of the half-hour at or above 90
degrees) have zero physics-lite power, so their quantiles are zero by construction, with no gate.
Training rows are the daylight half-hours with positive physics-lite power. The control is the
stored output of experiment 666.

**Scoring.** The scoring code is experiment 667's (`e00667_score.py`) and experiment 666's
(`e00666_score.py`): CRPS (twice the mean pinball loss over 11 levels) and median error in points of
capacity, by the four lead bands, on the same rows as the control. The page reports, per series and
lead band: CRPS; q01, q05, q95, and q99 coverage and pinball loss; q05 to q95 coverage; and the
share of rows with a lower quantile at or below zero. Solar also reports the number of night
half-hours with a non-zero quantile. Intervals: the paired difference to the control, by a block
bootstrap over origin days, and no multiplicity claim because the experiment is descriptive.

**Figures.** Take-home first; forecast fans with 50% and 90% bands for the new heads against the
control on a few days for ids 14 and 26; the fitted mixtures on two example half-hours; solar night
half-hours before and after; a 14-day-horizon fan for one origin and a day-ahead fan. Lines 1 px
wide, at least 280 px per panel, distinct hues. Generators are anonymised (Solar B and Solar E, no
coordinates, no calendar dates on a generator's series).

**Standing practices.** Monthly refits; cleaned data with the flagged half-hours counted per fold;
the D15 and D18 cut; every fitted network saved under the experiment's data folder; stored
predictions reused as controls; tails reported; no load-shift event is scored. The ERA5 caveat does
not arise because the inputs are the ENS forecast and the lags are at least 24 hours old, as in
experiment 667.

**Compute.** About 1 GPU-hour for the demand refits (4 fits; the Candidate's `fit` step is a small
share of experiment 667's 0.5 GPU-hours per fold) and about 20 minutes for solar, inside the
ceiling of 4 GPU-hours, with one GPU process at a time and at most 4 CPU cores, sharing the GPU with
experiment 677.

**Reviews.** One Sonnet plan review, one Sonnet code review before the first fit, a Sonnet check of
the printed numbers, and one Sonnet prose review. The audit triggers are run on the results.

## Stop rules

- Any series worsens by 0.05 points or more under a family: keep the clamp for that family.
- The Gaussian-loss reproduction check fails: stop and report before any new fit.
- A fit raises or exceeds the GPU budget: stop and report to the manager.
