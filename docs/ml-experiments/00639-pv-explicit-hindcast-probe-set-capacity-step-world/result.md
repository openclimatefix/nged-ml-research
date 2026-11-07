---
id: "00639"
slug: pv-explicit-hindcast-probe-set-capacity-step-world
theme: solar-disaggregation
also_relevant_to: []
status: finished
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
  averages (level 98.75%: 95% Bonferroni-corrected over the four contrasts C1, C3a, C3b, and C2)
  excludes zero and the contrast is at least the minimum detectable effect (MDE). The MDE is the
  half-width of the interval of a null contrast at the same level: the free network's seeds split
  into three and two, averaged over all ten splits, on the same rows. C1 is supported if it is a
  gain. C2 is supported if the upper end of
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
verdict: partly supported
best_arm: >-
  In the step world, the explicit solar term set by the calibrated probe: continuous ranked
  probability score (CRPS) on sunny middays 6.09 percentage points of capacity against 8.99 for the
  free network. In the real months the fixed capacity table scores best (3.20 against 3.30 for the
  calibrated probe and 4.05 for the free network).
headline: >-
  A neural network given a physics solar curve and a probed capacity forecasts sunny middays with
  a lower CRPS than a neural network that learns solar from irradiance (0.75 points lower on real
  months, 2.90 lower in a world with a 4 MW capacity step). Swapping the capacities between
  Primaries lowers the step-world score by only 0.14 points, below the 0.18 the test can
  resolve, so that test does not show that each Primary's own capacity matters.
lessons:
  - >-
    The free network's median forecast is 2.7 to 5.0 points of capacity too high on sunny middays
    in the real months (4.0 to 10.3 in the step world), against -0.4 to 2.3 (real) and -2.6 to 1.6
    (step) for the calibrated explicit network. In the real months, shifting the free forecast by
    its own bias at each cutoff leaves the explicit network 0.07 ahead, so the gain there is mostly
    a bias. In the step world the free network's forecast distribution is also about twice as wide.
  - >-
    One Primary carries most of the gain, and two of the six Primaries score slightly better with
    the free network, so the gain is not uniform.
  - >-
    The exact 4 MW step scores 0.21 points worse than the probe's calibrated 3.2 to 3.8 MW. The
    injected farm's energy is about 0.78 of the physics curve's at the same capacity, which would
    explain it, but the random capacity draws that regularise the calibrated arm could too.
  - >-
    The shuffled-day-sky placebo scores 3.3 (real) and 8.8 (step) points worse, so the explicit
    term only helps when its curve is built from the weather of the day being forecast.
take_home_message: >-
  Giving a demand neural network a physics solar curve, with a capacity set by the probe, forecasts
  sunny middays clearly better than letting the network learn solar from irradiance, when the weather
  inputs are recorded values and not forecasts. Much of the gain is that the free network's median
  forecast runs too high. Swapping capacities between Primaries lowered the step-world score by too
  little to detect, so this test does not show that the probed capacity tracks a capacity change.
  The next checks are a version with weather forecasts, and a free network whose forecast is shifted
  by its bias.
advice:
  - >-
    Score a free-irradiance network beside a copy shifted by its own bias whenever an explicit solar
    term is tested, because the free network's sunny-midday median ran 2.7 to 5.0 points of capacity
    too high in the real months (see Checks, "The gain is mostly a bias").
  - >-
    Include a placebo that gives each Primary another Primary's capacity, with enough rows that the
    minimum detectable effect is below the loss you expect, because the swap cost only 0.14 points in
    the step world against a minimum detectable effect of 0.18 (see the contrasts figure).
  - >-
    Try the probe's calibrated step before the injected nameplate step when testing an explicit solar
    term, because the exact 4 MW step scored 0.21 points worse (see the contrasts figure).
  - >-
    Repeat the explicit-term test with ensemble weather forecasts before reading it as a forecast
    gain, because this experiment used recorded weather (see Limits).
production_status: not-for-production
adopted_in: none
---

# A physics solar term with a probed capacity forecasts sunny middays better, but mostly by fixing a bias

**A demand neural network given a physics solar curve and a probed solar capacity forecasts sunny
middays better than a neural network that learns the solar response from irradiance, but much of
the gain is a bias in the free network, and the test does not show that each Primary's own capacity
matters.** The error measure is the continuous ranked probability score (CRPS), in percentage points
of each Primary's effective capacity (smaller is better). Sunny middays are the half-hours from
10:00 to 15:00 UTC with a clear-sky index of at least 0.7.

We asked whether a demand neural network that is told how much solar capacity sits behind a Primary
substation forecasts sunny middays better than one that learns the solar response from its
irradiance inputs. The capacity comes from the probe of experiments
[621](../00621-pv-probe-injected-known-answer-and-placebo-test/result.md),
[623](../00623-pv-probe-monotone-arm-known-answer/result.md), and
[638](../00638-pv-probe-step-placebo-climatological-and-ordinary-year-followup/result.md), which
reads the capacity a trained network has learned by feeding it weather from other days. It is
combined with the capacity table of experiment
[474](../00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals/result.md). We trained the
networks on recorded weather (ERA5, from the European Centre for Medium-Range Weather Forecasts,
and CAMS, from the Copernicus Atmosphere Monitoring Service) up to three cutoffs (July 2025, March
2026, and June 2026), and scored the 60 days after each, at the six Primaries metered in megawatts.
We scored two worlds. In the real world the demand is the recorded demand. In the step world, 4 MW
of a real solar farm's output has been added to the demand since April 2023.

On sunny middays the explicit network's CRPS was 3.30 against 4.05 for the free network in the real
world, and 6.09 against 8.99 in the step world. A placebo that swapped the capacities between
Primaries scored only 0.14 points worse than the explicit network in the step world. That is below
the 0.18 points that the test can resolve. Two of the three pre-registered predictions hold: the
gain in the step world, and no loss on the real months. The third holds for one of its two
placebos: the shuffled-sky placebo loses clearly, and the capacity-swap placebo does not. The
hypothesis is partly supported.

![CRPS on sunny middays by arm](crps_by_arm.svg)

## Definitions

- **Free network**: experiment 554's pretraining network (a mixture density neural network) on the
  net demand, which is demand with the solar generation netted off. It is the recipe of experiment
  [565](../00565-drift-embedding-variants-six-designs/result.md), and it learns the solar response
  from its irradiance input. Seeds 0 to 2 are experiment 634's (real world) and 638's (step world);
  seeds 3 and 4 are trained here.
- **Explicit network**: the same neural network trained on gross demand, which is net demand plus a
  physics solar curve (the curve for 1 MW) times a capacity timeline in megawatts. The forecast
  subtracts the same term. The timeline is `k` (a scale factor on experiment 474's table) times the
  table, plus `D` (the capacity step in megawatts, zero in the real world) from the step date, plus
  a trainable offset. Arms: `cal` (`k` and `D` read by the probe and divided by the
  leave-one-farm-out recovery factor of 0.67 to 0.69, the share of an injected farm's capacity the
  probe recovers), `raw` (as probed), `r4` (the table unchanged), `oracle` (the table plus exactly
  4 MW from the step), and the placebos `perm` (the capacity timelines shifted by two Primaries) and
  `shuf` (the curve built from the weather of another date).
- **CRPS** is pooled over test rows, with each row in percentage points of its Primary's effective
  capacity.
- **Minimum detectable effect (MDE)**: the half-width of the interval of a null contrast, which is
  the free network's seeds {0, 2, 4} against {1, 3} (averaged over all ten 3-and-2 splits), at the
  same 98.75% level as the contrasts (95%, Bonferroni-corrected over four contrasts). It is 0.067
  in the real world and 0.181 in the step world.

## Results

**The explicit network beats the free network by far more than the MDE.** Each entry is the arm's
CRPS minus the `cal` arm's, so a positive value favours `cal`. Intervals are day-block bootstraps at
the 98.75% level.

| Arm minus `cal` | Real, sunny | Step, sunny |
|---|---|---|
| `free` | +0.75 [0.51, 1.01] | +2.90 [2.15, 3.65] |
| `raw` | -0.05 [-0.15, 0.05] | +0.20 [-0.10, 0.50] |
| `r4` (fixed table) | -0.10 [-0.14, -0.05] | +1.78 [1.15, 2.40] |
| `oracle` | not run | +0.21 [0.04, 0.38] |
| `perm` placebo | +0.31 [0.22, 0.39] | +0.14 [0.06, 0.22] |
| `shuf` placebo | +3.30 [3.02, 3.58] | +8.81 [8.09, 9.61] |

![Each arm minus the calibrated explicit arm](contrasts_vs_calibrated.svg)

- **C1 holds.** In the step world `free` minus `cal` is +2.90, above the MDE of 0.181, and 16 times
  the MDE. The MDE exceeds the 0.10 limit set in the plan, so the plan's predicted gain of 0.05 to
  0.15 could not have been resolved. The prediction was also too low: the observed gain is 19 to 58
  times the predicted one.
- **C2 holds.** On all half-hours of the real months `cal` is 0.13 points better than `free`
  [0.08, 0.19], so it loses nothing. The windows cover March to August (the June 2026 window is 30
  days).
- **C3 fails.** The shuffled-sky placebo loses by 8.81. The capacity-swap placebo loses by 0.14 in
  the step world, below the MDE of 0.18, so the test does not show that each Primary's own capacity
  matters there. The injected 4 MW is the same at every Primary, so swapping the timelines changes
  only the table part. In the real world, which is outside the pre-registered C3, the swap loses
  0.31, above the real-world MDE of 0.067, so the Primary's own capacity does help there.
- **The best real-world arm is the fixed table.** `r4` beats `cal` by 0.10 in the real world. The
  best arm in `best_arm` is chosen in the step world, where the table has no step. `raw` and `cal`
  differ by 0.2 points or less, and `raw` does not beat `cal` by the MDE.
- **The probe's step.** The raw probed step is 2.1 to 2.6 MW of the 4 MW injected (0.53 to 0.65 per
  injected megawatt), and the calibrated step is 3.2 to 3.8 MW (0.80 to 0.95). Per injected
  megawatt-hour, divide by the farm's energy share of 0.78 of the physics curve's (experiment 634's
  rows basis): the calibrated step is about 1.0 to 1.2 and the raw step about 0.7 to 0.8.

## Checks

- **The gain is mostly a bias in the real world.** The free network's median forecast is too high on
  sunny middays by 5.0, 4.2, and 2.7 points of capacity at the three cutoffs in the real world
  (calibrated explicit network: 2.3, -0.4, 0.8), and by 7.7, 10.3, and 4.0 in the step world
  (calibrated explicit network: 1.6, -2.6, 0.2). Shifting the free forecast by the free network's
  own bias at each cutoff leaves `cal` ahead by about 0.07 in the real world (the audit's figure).
  Shifting it by that bias at each cutoff and Primary, a correction available only in hindsight,
  makes the free network 0.52 better than `cal` in the real world and leaves `cal` 0.90 ahead in
  the step world. The free network is also overconfident (its 50% interval covers 38% to 49% of
  sunny rows), and in the step world its 5% to 95% width is about twice that of `cal`.
- **The gain is uneven.** By cutoff, `free` minus `cal` is +0.85, +0.89, and +0.27 (real), and
  +3.12, +3.73, and +0.48 (step). One of the six Primaries carries +3.05 (real) and +7.15 (step);
  two Primaries score slightly better with `free` in the real world (-0.39 and -0.29). Without that
  Primary the real-world gain is still +0.27 on average.
- **Error of the median forecast beside CRPS.** The mean absolute error of the median on sunny
  middays is 5.81 (`free`) against 4.44 (`cal`) in the real world, and 11.74 against 8.37 in the
  step world, in the same direction as the CRPS. The median minus the approximate mean (from the 11
  quantiles) is -0.17 for `free` and 0.02 for `cal` in the real world, and -4.30 against -0.38 in
  the step world.
- **Triggers.** An audit ran, with a private brief and report. The trigger "a gain after five
  earlier nulls on the mean score" was explained as a bias, partly. The trigger "a nested arm beats
  the arm that contains it" (`oracle` loses to `cal`) has the energy-ratio explanation above, but
  regularisation by the random draws of `k` and `D` each epoch could also explain it, and these
  files cannot separate the two. The trigger "a later experiment depends on the sign" holds, and the
  result is unreplicated. The audit reproduced every headline CRPS with its own code, and found no
  lookahead and no unequal training between arms (free networks stopped after 30 to 54 epochs,
  explicit networks after 39 to 52).
- **Finding not applied.** The audit asked for experiment 474's table for the one dominant Primary
  without later data. That was not run, because the table's own fit used the whole history, and
  Limits states this.

![One Primary, one week of the real world](example_week.svg)

## Limits

- **Recorded weather, not forecasts.** The weather is ERA5 and CAMS at the valid time, so this is a
  hindcast. A gain here does not show a gain for a forecast that uses ensemble weather forecasts.
- **Short lead range.** The rows have no forecast lead, so the 1 to 10 day range that the network
  operator (NGED) cares about is untested.
- **Capacity table.** Experiment 474's table was fitted on data that includes the test windows. The
  networks use only its season-years that ended before the cutoff, and hold the last value.
- **What C1 tests.** The free network has trained on two to three years of post-step rows, and `k`
  and `D` come from probing that same network, so the test shows the value of the explicit
  structure, not tracking of a capacity that appeared shortly before the cutoff.
- **The test windows.** Three cutoffs cover 150 days from March to August, with 4,397 sunny rows
  pooled over six Primaries (the bootstrap resamples whole days). The shuffled-sky donor falls in
  the same 21-day block of the year, and a test row's input-spread columns use one lead band.
- **Not reported.** The plan listed one sunny and one cloudy example week, and a plot of all
  cutoffs. Only one example week is shown.

## What the next experiment should answer

**Does the gain survive ensemble weather forecasts and a free network shifted by its bias?** Repeat
the contrast with ensemble weather forecasts, include a free network shifted to the latest level
after one refit month, and test a step that starts within 60 days of the cutoff.

## Plan (pre-registered)

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

### Changes after the plan review

- **Table lookahead.** Experiment 474's R4 table was fitted on the whole history. Each explicit
  network uses only season-years that ended before its cutoff (2024 and earlier at all three
  cutoffs, because a season-year ends in June) and holds the last value for later dates. The
  table's own fit still saw later months, so the write-up states this as a caveat.
- **What C1 tests.** The free network has trained on two to three years of post-step rows, and `k`
  and `D` come from probing that same network. C1 therefore tests whether an explicit term built
  from the probed level forecasts at least as well as the network's own implicit level. It does not
  test tracking of a capacity that appeared shortly before the cutoff, and the write-up says so. A
  late-step world is not part of this plan.
- **Calibrated or raw.** Dividing by the factor may overstate the PV energy, because the real farm
  carries 0.72 to 0.84 of the unit curve's energy. All of `raw`, `cal`, and `oracle` are reported on
  sunny middays, with each arm's implied PV energy against the injected energy. If `raw` beats
  `cal`, the write-up names `raw` as the better explicit arm, and C1 and C3 are repeated for it.
- **Minimum detectable effect.** The null contrast and its split are computed and printed before
  any explicit network is fitted, at the same confidence level as the contrasts. If it exceeds 0.10
  points, C1 is reported as unable to resolve the predicted gain.
- **Training detail.** Explicit networks standardise the gross target with the net target's mean
  and standard deviation. The offset `delta` enters the gross target and trains by gradient.
  Validation for early stopping uses the mean capacity (no draw) with the current `delta`. The loop
  is experiment 634's `train` with that change only. Seeds 3 and 4 of the free networks use
  experiment 634's `train` unchanged.
- **Test rows.** A new builder takes experiment 448's pretraining rows with the valid-time filter
  moved to the test window. It keeps one row per Primary and valid time (the first origin), excludes
  no purged week (the rows postdate training), and applies experiment 634's farm-gap mask. The free
  and explicit arms score the same rows.
- **Definitions.** A sunny half-hour has `x[:, 0]` (the shortwave column) divided by the Haurwitz
  clear-sky value at least 0.7. Pooling weights every test row equally, in points of that Primary's
  effective capacity. C2 covers March to August only, so it says nothing about winter. The `perm`
  placebo shifts the table and the step increment together, and the `shuf` placebo is a sanity check.

### Review order

A fresh Sonnet agent reviews the code before the first fit. A fresh Sonnet agent reviews the
write-up against the printed report, and a second runs the prose review. An audit follows if a
trigger in the `ml-experiment` skill fires (for example, a nested arm beating the arm that
contains it, or a placebo that does not lose).
