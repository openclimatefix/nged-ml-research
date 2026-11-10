---
id: "00684"
slug: backtest-positive-demand-output-distributions-and-physical-solar-night-zero
title: >-
  Can output distributions that cannot go below zero, and a solar night gate set at the half-hour's
  own sun position, replace the post hoc clamps in the one-year backtest files?
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
  A positive output family at the four Primaries that cannot go below zero, and a hard zero at
  every solar half-hour whose sun is at or below the horizon at the middle of the half-hour, remove the
  negative lower quantiles and the non-zero night quantiles of the backtest files without reducing
  accuracy.
prediction: >-
  Written before any fit. P1: after the change no q01 is negative at ids 14, 15, 26, and 17. P2:
  CRPS at those four series changes by within 0.01 points of capacity against the Gaussian refit
  (the second control below), per family. P1 holds by construction for the two new families, so P2
  and P3 are the tests that can fail. P3: q05 to q95 coverage is not lower than the
  control's by more than 0.01. P4 (arm a, the gate): at Solar B and Solar E no night half-hour
  (sun elevation at the middle of the half-hour at or below zero) carries a non-zero quantile, and
  daylight CRPS and q95 and q99 coverage change by within 0.01 against the stored files of
  experiment 666. P5 (arm b, only if arm a leaves a score gap at dawn and dusk): feeding the
  physics-lite power at each half-hour's own zenith to the network improves dawn and dusk CRPS (sun
  elevation under 5 degrees) by at least 0.05 points of capacity, else arm b is dropped.
decision_rule: >-
  Stop rule for demand: if a family worsens the CRPS of any of the four series by 0.05 points or
  more, keep the clamp and drop that family. A family passes if P1, P2, and P3 hold for all four
  series. Solar arm a passes if P4 holds at both farms. A worsening of 0.01 to 0.05 points fails P2 without
  triggering the stop rule, and is reported as a cost. A change is read against the Gaussian refit,
  whose distance from the stored files is the seed noise floor. The model-level fix is compared
  with the post hoc clamp of the stored files as a fourth arm. No ranking beyond that, and no claim
  of benefit from two test months.
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
forecasting models can instead produce the right answer themselves: by refitting them with an
output that cannot go below zero (demand), and by a hard zero at night set at each half-hour's own
sun position (solar). It also asks whether the change reduces accuracy.

## Plan

**Scope.** Demand: Primaries with ids 14, 15, and 26 (MVA, median within one band-width of zero on
65% to 97% of rows, upper half-width 1.2 to 1.5 times the lower) and id 17 (MW, observed power never
negative, kept as a control that must not get worse). Solar: Solar B and Solar E. Original ids stay
out of this page. Only the normal-running (NRA) output of experiment 667 is refitted. The prevailing
output mixes the NRA quantiles with a shifted copy, so a fix there waits for the NRA result.

**Demand arms.** Each arm is a refit of experiment 667's Candidate on two folds, with test months
2026-02 and 2026-05 (the fold recipe is experiment 667's, monthly refit, 2 seeds, D15 and D18 cut
before 2025-03-01, cleaned power). Test month 2026-08 lies inside the final-test window and is not
read. Only the output family changes, and only at ids 14, 15, 26, and 17. Every other Primary keeps
the Gaussian mixture, so a Gaussian mixture also remains for ids 10 and 12, which do go negative.

- **Controls:** the stored forecasts of experiment 667 for the same folds, and a Gaussian refit
  with the same chain as the new arms (the noise floor). The stored clamped file
  (`primaries.parquet`) is the fourth comparison: its clamp (negative lower quantiles set to zero at
  the three MVA Primaries) is applied to the control rows. The other 12 Primaries are refitted in
  the same shared network and are scored too, as a check that the change does not harm them.
- **Truncated Gaussian mixture:** the same three-component Gaussian mixture, normalised over
  power at or above zero. The negative log likelihood adds the log of the mixture's mass above zero,
  and the quantiles are the mixture's quantiles conditioned on power at or above zero, read by
  bisection on the conditional cumulative distribution.
- **Gamma mixture:** three gamma components on power over its Primary's scale, with each
  component's mean and coefficient of variation read from the head's outputs. The negative log
  likelihood uses the gamma log density (observed power is clipped to 1e-4 of capacity), and the
  quantiles are read by bisection on the mixture's cumulative distribution (the regularised
  incomplete gamma function). A log-normal mixture is not run: it would need a positive offset that
  re-creates negative quantiles.

**What is refitted.** The Candidate's `fit` loads the K1 chain's step-2 weights and continues them,
so the mixer is not re-initialised there. A new output family therefore re-runs the chain from the
step-2 stage: `step2`, `table`, `fit`, `ptable`, and `pfit` of `chain.sh`, per fold and seed, under
a new `E00667_ROOT`. The encoder pre-training (`pretrain`) is reused unchanged, so the encoders were
pre-trained under the Gaussian loss, and the write-up says so. The new loss must be patched into
`step2` and `fit` (`e.row_nll`), and the cross-fit tables keep the Gaussian read-out because they
only supply the Candidate's inputs; the code review checks both. The check that comes first in
implementation is that a Gaussian-loss run of this chain reproduces experiment 667's stored
quantiles on one fold to within seed-level differences. If the check fails, the plan is revised and
the manager told.

**Injecting the family.** `mixture_nll` and `mixture_quantiles` see only the standardised target, so
the implementation passes the Primary index and the per-Primary target mean and standard deviation
into `row_nll`, `serving_quantiles`, and the window read-out, and a per-row mask on the Primary
selects the family (one shared head, read as Gaussian at 12 Primaries and as the new family at the
four). Truncated Gaussian: the log of the mass above zero is computed with `logsumexp` of
`log_ndtr` terms so that it does not underflow when the median is near zero, and the bisection's
lower bound is the standardised zero. Gamma: each component's mean is a softplus and its
coefficient of variation a clamped exponential of the head outputs, and the head bias starts the
components at equal weights with means spread around the Primary's mean.

**Solar arms.** One quarterly fold, refit date 2026-02-01 (it scores origins from February to
April 2026, all before 2026-07-01), at Solar B and Solar E. Arm a (the gate) keeps the stored
networks of experiment 666 unchanged and re-predicts, with no refit: every quantile is set to zero
where the sun's elevation at the middle of the half-hour is at or below zero (the zenith rule the
scores use), and nothing else changes. A multiplier by the physics-lite power is not used, because
it forces dawn and dusk quantiles too low and scales down the upper quantiles of sunny days. Arm b
runs only if arm a leaves a score gap at dawn and dusk: the physics-lite power at each half-hour's
own zenith (`pl_mean` of `build_day`) becomes a network input, and Solar B and Solar E are refitted
for the same fold, 3 seeds (about 0.5 GPU-hours). Neither a zero-inflated head nor a clear-sky-index
target (experiment 430) is used. The report scores night, dawn and dusk (sun elevation under 5
degrees), and the rest of daylight separately. The control is the stored output of experiment 666.

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

**Compute.** About 2 GPU-hours for the demand refits (three chains of two folds, each about 0.3 of
experiment 667's 0.5 GPU-hours per fold without the encoder pre-training) and about 0.5 GPU-hours
for solar arm b if it runs, inside the ceiling of 4 GPU-hours, with one GPU process at a time and
at most 4 CPU cores, sharing the GPU with experiment 677.

**Reviews.** One Sonnet plan review, one Sonnet code review before the first fit, a Sonnet check of
the printed numbers, and one Sonnet prose review. The audit triggers are run on the results.

## Stop rules

- Any series worsens by 0.05 points or more under a family: keep the clamp for that family.
- The Gaussian-loss reproduction check fails: stop and report before any new fit.
- A fit raises or exceeds the GPU budget: stop and report to the manager.
