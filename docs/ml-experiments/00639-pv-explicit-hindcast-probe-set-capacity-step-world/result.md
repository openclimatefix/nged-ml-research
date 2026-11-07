---
id: "00639"
slug: pv-explicit-hindcast-probe-set-capacity-step-world
theme: solar-disaggregation
also_relevant_to: []
status: planned
kind: other
size: >-
  complex. Stored: pretrained network weights, test-window quantiles, and a capacity table in this
  experiment's private folder. Serving path: not touched. Degradation rule: not touched. Designs:
  more than one way to turn a probed capacity into a timeline of capacities is defensible, so the
  construction, the contrasts, and every bar are fixed below before any result is read. Callers: the
  scripts import the modules of experiments 634, 638, and 474 (and, through them, earlier ones) and
  edit none of them. Asset graph: not touched. One GPU process, about 1.8 GPU-hours (about 180 fits
  of 30 to 40 s), against a ceiling of 3.
hypothesis: >-
  A demand network that is given a solar capacity from outside, and subtracts a physics solar curve
  of that capacity from its target, forecasts sunny middays better than a network that learns the
  solar response from free irradiance inputs, after a capacity step that the probe has measured.
  The gain, if any, comes from tracking capacity and not from a better mean score.
prediction: >-
  Fixed before any result was read. (C1) In the ordinary-spring step world (4 MW added to the
  demand of the six MW Primaries from 2023-04-01), the explicit-PV network set by the calibrated
  probe has a lower sunny-midday CRPS than the free-irradiance network, on the 60 days after each
  training cutoff, by at least the minimum detectable effect. My expectation is a gain of 0.05 to
  0.15 CRPS points, because the free network already reads 0.82 to 0.88 of the step in experiment
  638. (C2) On the real development months the explicit network loses at most 0.02 CRPS points
  over all half-hours. (C3) The explicit network beats both placebos (the capacities permuted
  across Primaries, the physics curve built from a shuffled-day sky) by the minimum detectable
  effect on sunny middays. The earlier mean-score nulls (experiments 349, 352, 416, 477, 501)
  predict no gain on all half-hours.
decision_rule: >-
  Fixed before any result was read. CRPS is in points: percent of effective capacity, pooled over
  the six MW Primaries, the three cutoffs, and the sunny half-hours of the 60 days after each
  cutoff (10:00 to 15:00 UTC, clear-sky index at least 0.7). A contrast is the free network's CRPS
  minus the explicit network's, averaged over seeds, so a positive value favours the explicit
  network. A contrast counts as a gain when its day-block bootstrap interval over the seeds'
  averages (95%, Bonferroni-corrected over the four contrasts C1, C3a, C3b, and C2) excludes zero
  and the contrast is at least the minimum detectable effect (MDE). The MDE is the half-width of
  the 95% interval of a null contrast: the free network's seeds 0, 2, and 4 against its seeds 1
  and 3, on the same rows. C1 is supported if it is a gain. C2 is supported if the upper end of
  the interval of the explicit network's loss on all half-hours is at most 0.02. C3 is supported
  if both placebo contrasts are gains. Stop rule from the brief: if C1 fails, report and do not
  build the version that uses ensemble weather forecasts. Complexity tier: a change that adds a
  pipeline step (a capacity input) would need a threshold above zero, so no threshold for
  production is set here.
benchmark: >-
  The free-irradiance network (experiment 554's M0 pretraining, the recipe of experiment 565),
  with seeds 0 to 2 taken from experiments 634 (real world) and 638 (step world), and seeds 3 and
  4 trained here; the fixed capacity table of experiment 474 (rung R4) as an explicit network.
row_set_id: >-
  Not a published row set. Training rows are experiment 634's pretraining rows at the cutoffs
  2025-07, 2026-03, and 2026-06 (development data only). Test rows are the same ERA5-and-CAMS
  rows with a valid time from the cutoff to 60 days later (30 days at 2026-06), one row per
  Primary and valid time, cleaned power with farm-gap rows dropped as in experiment 634. Nothing
  dated 2026-07-01 or later is used.
metric: >-
  CRPS in points of effective capacity from the 11 quantiles of each mixture, on sunny middays and
  on all half-hours; capacity-timeline error in MW against the injected 4 MW; probed step per
  injected megawatt and per injected megawatt-hour.
---

# Explicit solar term with a probed capacity: does it follow a capacity step better than free irradiance?

**We ask whether a demand network that is told how much solar capacity sits behind a substation
forecasts sunny middays better than a network that has to learn the solar response from its
irradiance input.** The capacity comes from the probe of experiments
[621](../00621-pv-probe-injected-known-answer-and-placebo-test/result.md),
[623](../00623-pv-probe-monotone-arm-known-answer/result.md), and
[638](../00638-pv-probe-step-placebo-climatological-and-ordinary-year-followup/result.md). The
weather is the recorded weather (ERA5 and CAMS), so a gain here shows that the explicit term uses
capacity well. It does not show a gain for a forecast that uses weather forecasts. The result is
pending.

## Plan

### Standing practices followed and departed from

- **Followed:** cleaned data, the D15 and D18 cut, saved weights, reuse of stored networks, one
  paired contrast per question, and day-ahead and 14-day forecast plots (here one sunny week and one
  cloudy week instead, because the rows are hindcasts without leads).
- **Departed from:** monthly refits (one pretraining per cutoff, as in experiment 634); the
  load-shift slices (no event is injected); the screen on four test months (three cutoffs only,
  chosen below).
- **Earlier advice followed:** experiment 638 advises a step that starts in an ordinary spring
  with more than half the donor rows after it, so the step world is its `spring23` step. The
  placebo is the shuffled-day sky and the permutation of capacities, as the design review asks.

### Worlds, networks, and arms

- **Real world:** experiment 634's `control` rows (no injection). **Step world:** the `spring23`
  step of experiment 638 (a real farm's output scaled to 4 MW added to the six MW Primaries' demand
  from 2023-04-01; farms A, B, C, F, A, B in order, anonymised by letter).
- **Network:** experiment 554's `ModularMdn` at stage `pre`, trained with experiment 634's
  `train`. The **free arm** is that network on the net target. Every **explicit arm** trains the same
  network on the gross target `y + C(t) * inc * u(t)` (`u` is experiment 634's physics unit curve
  of the row's own CAMS-and-ERA5 weather, `inc` the inverse effective capacity of the Primary), and
  subtracts `C(t) * inc * u(t)` from each quantile at prediction. The shortwave input stays.
- **Capacity timeline.** `C(t) = (k + e_k) * R4(season-year of t) + (D + e_D) * 1[t >= step] + delta`.
  `R4` is experiment 474's table (`ac_p50`, per Primary and season-year). `k` rescales the table to
  the probe's level: the probed capacity of the free real-world network (median over seeds) divided
  by `R4` at season-year 2025, clipped to 0.5 to 1.5. `D` is the probed capacity of the free
  step-world network minus that of the free real-world network, floored at zero, and zero in the
  real world. `e_k` and `e_D` are drawn once per epoch from a normal distribution whose relative
  spread is the quadrature sum of the seed spread, the bootstrap spread, the factor spread, and 0.04
  for the spread of the factor between farms. `delta` is a trainable offset per Primary with a
  normal prior of 0.25 MW.
- **Arms.**
    - `free`: the free arm.
    - `cal`: `k` and `D` divided by the leave-one-farm-out factor of experiment 638 (0.67 to 0.69).
    - `raw`: `k` and `D` as probed.
    - `r4`: the fixed table, `k = 1`, `D = 0`.
    - `oracle`: the table with `D` = 4 MW exactly (step world only; a ceiling, not a contrast).
    - `perm`: the `cal` timelines of the six Primaries cyclically shifted by two.
    - `shuf`: the `cal` timeline with `u` built from the weather of another date. The date is drawn
      within the same Primary, the same half-hour of the day, and the same 21-day block of the year.

### Scoring and contrasts

- **Test rows:** the 60 days after each cutoff (the pretraining rows end 3.5 hours before it), so
  every test row postdates training. Targets include the step from 2023-04-01. **Sunny half-hours:**
  10:00 to 15:00 UTC with a CAMS clear-sky index of at least 0.7.
- **C1:** `free` minus `cal`, step world, sunny middays. **C2:** `free` minus `cal`, real world, all
  half-hours; support needs `cal` to lose at most 0.02. **C3a, C3b:** `perm` minus `cal` and `shuf`
  minus `cal`, step world, sunny middays. Also reported without a bar: `raw` and `r4` against `cal`,
  `oracle` against `cal`, the contrasts in the real world, and the same contrasts on all half-hours.
- **Seeds and intervals:** seeds 0 to 4, day-block bootstrap over test days with 2000 resamples.
  The MDE is defined in the decision rule.
- **Probe reads in two units:** the probed step per injected megawatt and per injected megawatt-hour
  over the probe dates, using experiment 634's farm-to-curve energy ratios.
- **Stop:** if C1 fails, the write-up reports that and no version with ensemble weather forecasts
  is planned.

### Review order

A fresh Sonnet agent reviews the code before the first fit. A fresh Sonnet agent reviews the
write-up against the printed report, and a second runs the prose review. An audit follows if a
trigger in the `ml-experiment` skill fires (for example, a nested arm beating the arm that
contains it, or a placebo that does not lose).
