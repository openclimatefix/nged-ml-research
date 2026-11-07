---
id: "00634"
slug: pv-probe-capacity-step-world-and-energy-recovery
theme: solar-disaggregation
also_relevant_to:
  - production-reference-and-best-guess
status: planned
kind: other
size: >-
  medium. Stored: pretrained network weights, probe frames, and probe results in this experiment's
  private folder. Serving path: not touched. Degradation rule: not touched. Designs: more than one
  way to inject a capacity step and to define the truth a calendar-blind network can know is
  defensible, so the construction and every bar are fixed below before any step-world recovery is
  read. Callers: this folder's scripts import experiments 397, 405, 409, 428, 448, 510, 554, 621, and
  623 and edit none of them. Asset graph: not touched. One GPU process, about 1 GPU-hour (84 fits of
  about 30 s each, plus four data builds), against a ceiling of 3.
hypothesis: >-
  (1) Measured in injected energy instead of injected AC megawatts, the donor-day probe of a lag-free
  neural network recovers a real injected farm at a factor closer to 1 and tighter across Primaries
  and farms than the 0.65 to 0.71 it reads in megawatts. (2) After a real-farm capacity step of +2
  MW or +4 MW at a chosen month, the calibrated probe shows at least 70% of the step by the first
  summer cutoff after the step, and under 0.3 MW before the step.
prediction: >-
  Fixed before any energy re-score or step-world recovery was read. (1) The median energy-based
  recovery of the 4 MW real-farm injection lies in 0.80 to 1.00, and its range across the six
  Primaries is narrower than the megawatt-based range of 0.65 to 0.71 relative to its median (the
  coefficient of variation falls). (2) The lag-free networks have no input that tells a row's year,
  so a network averages the injected capacity over the training rows it can reach. We therefore
  expect the calibrated step at the first summer cutoff to equal the post-step share of the donor
  rows (about 0.5 for the spring step, about 0.25 for the summer step), and to miss the 70% bar. (3)
  The shuffled-shape step reads under 0.3 MW at every cutoff. (4) The seed noise floor, the
  probed capacity of one uninjected network minus another, has a 95th percentile under 0.3 MW. (5)
  The widened interval covers the constant-injection truth in at least 60% of leave-one-farm-out
  real-farm injections.
decision_rule: >-
  Fixed before any step-world recovery is read. Bars as written in the prediction and in "Bars"
  below. The step bar is judged on the calibrated step (raw divided by the leave-one-farm-out factor
  of experiment 621's constant-injection networks) at the first cutoff after the step that holds a
  summer of post-step rows. Stop rule: if the lag, the number of months from the step to the first
  cutoff at which the calibrated recovery reaches 0.7, exceeds 9 months, or never occurs at the four
  cutoffs, report season-year values only and say that the probe cannot track a step inside a
  year. No rescue tuning. The threshold for using the probe in production is not set here.
benchmark: >-
  Experiment 621's plain-network constant-injection recoveries (same rows, shapes, probe, and seeds),
  re-scored in energy, and the uninjected control of each cutoff.
row_set_id: >-
  Not a forecast score. Pretraining rows are experiment 554's ERA5-and-CAMS rows at the cutoffs
  2025-07, 2025-10, 2026-03, and 2026-06 (development data only; nothing dated 2026-07-01 or later).
  Probe dates are 12 Wednesdays from 2025-04-09 to 2025-09-10, half-hour ends from 10:00 to 15:00
  UTC.
metric: >-
  Recovery per injected AC megawatt and per injected megawatt-hour over the probed cells; effective
  AC capacity in MW per Primary from the donor-day regression; calibrated step; day-block bootstrap
  80% interval widened by the leave-one-farm-out spread of the factor.
---

# Capacity-step world and energy-based recovery of the donor-day PV probe

**We ask whether a probe of a lag-free neural network can follow a step in solar capacity, and
whether its 0.68 recovery of a real farm is mostly a units effect.** The probe asks how much a
network's forecast of demand falls when the sky clears. Experiments
[621](../00621-pv-probe-injected-known-answer-and-placebo-test/result.md),
[623](../00623-pv-probe-monotone-arm-known-answer/result.md), and
[624](https://github.com/openclimatefix/nged-ml-research/pull/291) found that the probe reads about
0.68 of the megawatts of a real farm injected into a network's training target, but 0.97 of a
physics curve. Experiment 624 showed that a real farm scaled to a given 99th-percentile output
carries 0.72 to 0.84 of the energy of the physics curve scaled the same way. This experiment (1)
re-scores the stored probe outputs in energy and (2) injects a capacity step into the training
target and measures how much of it the probe shows at later training cutoffs.

## Plan

**Standing practices followed.** Cleaned data, development months only, the six MW Primaries, and
anonymised farms (Solar A to F, by letter). Not followed: no forecast is scored, so there is no
tails, curtailment, or example-forecast reporting. Earlier advice followed: report a calibrated
figure beside the raw one, widen intervals by the leave-one-farm-out spread of the factor, and drop
the cloud template from the headline (experiments 623 and 624).

**Part 1: energy-based recovery (CPU, no fitting; reported before any fit).** For each stored
recovery in experiments 621, 623, and 624, `recovery_energy = recovery_MW x sum(u) / sum(farm)`.
Here `u` is the physics unit curve of the Primary's own CAMS and ERA5 weather and `farm` is the
farm's normalised series, each summed over the probe cells (12 dates, half-hour ends from 10:00 to
15:00 UTC, cells where the farm has no gap). The injected energy is the injected megawatts times
`sum(farm)`. The energy the probe implies is the probed capacity times `sum(u)`. Report per Primary,
per farm, and per cutoff, beside the megawatt figure, for the plain network (621, 623, and 624) and
the monotone network (623). The same ratio over all of the six Primaries' pretraining rows is
reported as a cross-check of experiment 624's audit (0.72 to 0.84). The physics-curve world is the
control: its energy ratio is 1 by construction.

**Part 2: capacity-step world.** Four injected conditions, each added to the target of the six MW
Primaries from the first day of a step month, using the real-farm shape of experiment 621's
rotation 0 (farm A, B, C, F, A, B): `step_spring_mw2` and `step_spring_mw4` from 2025-04-01, and
`step_summer_mw2` and `step_summer_mw4` from 2025-07-01. Two placebo conditions use the same
calendar and energy with the sky link removed (the shuffled shape of experiment 621):
`shufstep_spring_mw4` and `shufstep_summer_mw4`. An `control` condition has no injection. Every
condition shares one row mask, as in experiment 621. The network, rows, loss, early stopping, and
seeds (0, 1, 2) are experiment 554's `pretrain` through experiment 621's `fit`; the probe is
experiment 621's `probe`. Cutoffs: 2025-07, 2025-10 (the first summer cutoff after both steps),
2026-03, and 2026-06. That is 7 conditions x 3 seeds x 4 cutoffs = 84 fits.

**What a network that cannot tell the year can know.** The pretraining inputs carry the day of the
year and the time of day and no year, so a pretrained network cannot tell a row before the step
from a row after it. A calibrated probe would then read the average of the injected capacity over
the rows the probe reaches. The probe's donor days are within 21 days of the probe date's day of the
year and at the same half-hour, so the truth a calendar-blind network can know is
`mw x post-step share of the donor rows`. We compute that share per cutoff and Primary from the
probe's donor rows and report the recovery against both the full step and this share.

**Bars (fixed now).**

- **Step bar.** The calibrated step at 2025-10 is at least 70% of the injected step, at 4 of the 6
  Primaries, for both step months and both sizes.
- **Before-the-step bars.** (a) The shuffled-shape step reads under 0.3 MW at every cutoff. (b) The
  seed noise floor (the probed capacity of one uninjected network minus another, per Primary and
  cutoff) has a 95th percentile under 0.3 MW. (c) At 2025-07 the summer-step networks train on rows
  identical to the control's, so their step reads zero by construction; the check confirms the
  probe returns the same number.
- **Coverage bar.** The nominal 80% interval, widened by the leave-one-farm-out spread of the
  factor, covers the constant-injection truth of the 2 MW and 4 MW real-farm networks of experiment
  621 in at least 60% of injections (calibrated). For the step worlds it is reported against both the
  full step and the calendar-blind share.
- **Lag.** Months from the step to the first cutoff at which the calibrated recovery of the full
  step reaches 0.7 at the median Primary; stop rule above.

## Part 1 results

To be filled in.

## What the next experiment should answer

To be filled in after the results are read.
