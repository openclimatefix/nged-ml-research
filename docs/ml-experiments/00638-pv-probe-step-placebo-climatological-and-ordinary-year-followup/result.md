---
id: "00638"
slug: pv-probe-step-placebo-climatological-and-ordinary-year-followup
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  medium. Stored: pretrained network weights and probe results in this experiment's private folder.
  Serving path: not touched. Degradation rule: not touched. Designs: more than one way to build a
  placebo with no sky link and to choose steps in ordinary years is defensible, so the construction,
  the start dates, and every bar are fixed below before any result is read. Callers: this folder's
  scripts import experiment 634's modules (and, through them, experiments 397 to 623) and edit
  none of them. Asset graph: not touched. One GPU process, about 0.6 GPU-hours (69 fits of about 30
  s each, plus four data builds), against a ceiling of 2.
hypothesis: >-
  Experiment 634's failure to follow a capacity step (a calibrated read of 0.32 to 0.38 of a 4 MW
  spring-2025 step, and 1.2 to 2.1 MW read from a shuffled-shape placebo) was caused by the unusually
  sunny spring of 2025 (clearness 0.76 against 0.61 to 0.69 in other years), and not by a limit of
  the probe that holds in any year. The shuffled-shape placebo kept a month-level sky link, so a
  placebo whose shape is the farm's multi-year day-of-year mean, with no link to the sky of any one
  date, reads close to zero.
prediction: >-
  Fixed before any result was read. (1) For the spring-2025 step, the climatological placebo's
  calibrated read is at most half the real step's read at each of the cutoffs 2025-10, 2026-03, and
  2026-06, and at most 0.15 in absolute terms. (2) The ordinary-spring step (from 2023-04-01, with
  about 0.55 to 0.62 of the donor rows after the step and a clearness difference of at most 0.02) is read
  within 0.2 of its post-step donor share at every cutoff, and its climatological placebo reads at
  most 0.15. (3) The summer-2024 step's read is reported against its donor share and clearness difference, with no bar. (4) Moving the probe dates 364 days earlier changes the median calibrated read, over the cutoffs of one step, by at most 0.05; single reads may move by 0.1, as experiment 634's did. This is a diagnostic and does not enter the decision rule.
decision_rule: >-
  Fixed before any result was read. Stop rule from the brief: if the climatological placebo of the
  spring-2025 step reads at least half of the real step's calibrated read (a ratio of medians over
  Primaries and seeds), and the real read is above 0.2, at two or more of the three cutoffs
  2025-10, 2026-03, and 2026-06, stop, report, and do not tune. The sunny-year
  explanation is supported if prediction (1) holds and the ordinary-spring step is read within 0.2
  of its donor share (prediction 2). A limit of the probe that holds in any year is supported if
  the stop rule fires, or if the ordinary-spring step is read at under 0.3 of its size while its
  donor share is above 0.5. Any other outcome, including both explanations satisfied at once, is inconclusive. No threshold for using the probe in
  production is set here.
benchmark: >-
  Experiment 634's networks and probe reads: the uninjected control, the constant injections
  (`farm_mw4_r0`, `farm_mw4_r1`), the real spring-2025 and summer-2025 steps, and the
  shuffled-shape placebos, re-read unchanged.
row_set_id: >-
  Not a forecast score. Pretraining rows, frames, cutoffs (2025-07, 2025-10, 2026-03, 2026-06),
  probe dates (12 Wednesdays from 2025-04-09 to 2025-09-10), and half-hour ends are experiment
  634's (development data only; nothing dated 2026-07-01 or later).
metric: >-
  Calibrated step read (probed AC capacity of the injected network minus its control, divided by
  the injected AC megawatts and by the leave-one-farm-out factor), the ratio of the placebo's read
  to the real step's read, and the coverage of the widened 80% interval, at season-year level.
---

# Why experiment 634's capacity step was missed: the sunny spring or a limit of the probe?

**We asked whether a probe of a neural network failed to follow a rise in solar capacity because the
spring in which the rise was injected was unusually sunny, or because the probe cannot follow a
rise inside its training years in any year.** Experiment
[634](../00634-pv-probe-capacity-step-world-and-energy-recovery/result.md) added a real solar
farm's output to the demand of six MW Primary substations from 1 April 2025, and the probe found
0.32 to 0.38 of it. A placebo with the same calendar and a shuffled shape was found at about half
that. The audit of experiment 634 found that spring 2025 was the sunniest spring in the data and that
the placebo kept the actual monthly mean of the farm's output. We now repeat the test with steps that
start in other years and with a placebo that has no link to the sky of any date. This page is the
plan, written before any result exists.

## Arms

**Four steps, each injected as 4 MW of a real farm's output (the real step) and as 4 MW of the
same farm's day-of-year mean profile (the climatological placebo).**

| Step | Starts | What the brief asked for | Post-step share of donor rows | Clearness after minus before the step |
|---|---|---|---|---|
| `spring23` | 2023-04-01 | An ordinary-spring step | 0.55 to 0.62 | 0.004 to 0.021 |
| `summer24` | 2024-07-01 | A sunny-summer step (see below) | 0.20 to 0.32 | 0.023 to 0.090 |
| `spring25` | 2025-04-01 | Experiment 634's step | 0.12 to 0.26 | 0.049 to 0.100 |
| `summer25` | 2025-07-01 | Experiment 634's step | 0.00 to 0.16 | -0.052 to 0.012 |

- **The shares and clearness differences are approximate, computed from the frames of experiment 634
  over all rows from 20 March to 1 October before any network is trained.** The analysis script
  recomputes them on the probe's own donor rows. Clearness is the ratio of the measured to the
  clear-sky shortwave irradiance at 10:00 to 15:00 UTC. The ordinary-spring step starts in a year
  whose April-to-June clearness (0.68) sits with 2020 to 2023 (0.64 to 0.69), and more than half of
  the donor rows follow it.
- **No summer in the data was sunny.** July-to-September clearness is 0.59 to 0.63 in every year
  from 2020 to 2025. The sunny-summer step of the brief cannot be built from a sunny summer, so the
  `summer24` step stands in for it: its post-step period includes the sunny springs of 2025 and 2026,
  which gives a clearness difference between those of the ordinary-spring and spring-2025 steps.
  The four steps then give four clearness contrasts, and the page plots each step's read against its
  contrast.
- **The climatological placebo** is the farm's mean normalised output per day of the year and
  half-hour over every year before 2026-07-01, smoothed over 7 days either side. It has the farm's
  seasonal and daily shape and carries no weather of any single date. Its energy over the post-step
  window is 0.93 to 1.10 of the real farm's, per farm.
- **Previous-year base.** Every probe is run twice: with the 12 probe dates of 2025, and with each
  date moved 364 days earlier (a base in 2024, before the `spring25` and `summer25` steps). Experiment
  634 ran the second only for its own steps.

**Reused without retraining:** experiment 634's `control`, `farm_mw4_r0`, and `farm_mw4_r1` networks
(for the paired difference and the leave-one-farm-out factor) and its real `spring25` and `summer25`
steps. **Retrained:** 6 conditions (`step_spring23_mw4`, `step_summer24_mw4`, and the four
`climstep_*_mw4`) x 3 seeds x 4 cutoffs, less the `climstep_summer25` cases at 2025-07 where no row
follows the step (the network would equal the control). That is 69 fits, about 0.6 GPU-hours.

## Bars (fixed before any result was read)

- **Stop rule** and **support rules**: as in the front matter.
- **Placebo.** A climatological placebo's calibrated read is under 0.15 in absolute value, and
  under half of the real step's read, for every step with a real read above 0.2. A read under
  0.244 MW (experiment 634's noise floor, 0.06 before calibration) is not distinguishable from zero.
- **Coverage.** The widened 80% interval of experiment 634 is compared with the post-step donor
  share of each real step (the reference a date-blind network can know; reads barely move with the
  probe date) and with zero for each placebo. Reported by step and cutoff, not pooled. Coverage of
  "the full step" is not a bar, because about half the probe dates of `summer25` precede its start.
- Season-year capacities only: no read is reported as a capacity that changed inside a window.

## Standing practices and checks

- **Followed:** cleaned data, development months only, the six MW Primaries, anonymised farms (A, B,
  C, F), the experiment 634 code reused and unedited, and reads reported against the control.
- **Not applicable:** no forecast is scored, so there is no tail calibration, curtailment cost, or
  example forecast, and no median-minus-mean check.
- **Earlier advice followed** (experiment 634): report probed capacity per training window, build the
  placebo from a multi-year climatological profile, and check the clearness of the months a step covers.
- **Audit triggers:** a failed placebo (the stop rule) and a result that contradicts its prediction
  would each trigger a fresh-agent audit before the prose review.
