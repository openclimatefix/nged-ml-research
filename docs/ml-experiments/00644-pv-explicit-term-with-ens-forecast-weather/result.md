---
id: "00644"
slug: pv-explicit-term-with-ens-forecast-weather
theme: solar-disaggregation
also_relevant_to: []
status: finished
kind: other
size: >-
  complex. Stored: fine-tuned neural-network weights and test-window quantiles in this experiment's
  private folder. Serving path: not touched. Degradation rule: not touched. Designs: more than one
  way to feed the explicit solar term from ensemble weather is defensible (summary or members, and
  how the 14-day tracker is dated), so the choices are fixed below before any result is read.
  Callers: the scripts import the modules of experiments 510, 554, 634, 638, 639, and 643 and edit
  none of them. Asset graph: not touched. One GPU process; ceiling 4 GPU-hours.
hypothesis: >-
  The sunny-midday gain of experiment 639's explicit solar term over the free network survives when
  the weather inputs are ECMWF ensemble forecasts at lead 1 to 10 days instead of CAMS and ERA5,
  and it is more than the gain of a bias correction that uses recent errors.
prediction: >-
  Fixed before any result was read. The explicit network (arm b) beats the free network (arm a) on
  sunny middays at every lead band (day 1-2, 3-5, 6-10) by at least the bar, with a smaller gain at
  day 6-10 than at day 1-2 because forecast cloud error grows with lead and swamps the capacity
  term. My expectation is 0.3 to 0.6 CRPS points at day 1-2 and 0.1 to 0.3 at day 6-10. I expect
  the 14-day tracker (d) to stay competitive in the real months (experiment 643 found it 0.48
  points ahead), so (b) against the best of (d) and (e) is the contrast most likely to fail. The
  explicit network loses no more than 0.02 points over all half-hours. In the step world the
  explicit network beats the free network and the tracker after the step.
decision_rule: >-
  Fixed before any result was read. CRPS is in points: percent of effective capacity, pooled over
  the six MW Primaries, the three cutoffs (2025-07, 2026-03, 2026-06), and the sunny half-hours
  (10:00 to 15:00 UTC, clear-sky index at least 0.7) of the 60 days after each cutoff, averaged
  over seeds. A contrast is the control's CRPS minus the explicit network's, so a positive value
  favours the explicit network. Intervals are day-block bootstraps at 99% (Bonferroni-corrected
  over the planned contrasts), over seeds as the study skill requires. The bars are experiment
  639's minimum detectable effects, 0.067 (real months) and 0.181 (step world), because the added
  step is a capacity table that someone must update, so it needs a threshold above zero, and no
  larger production threshold is set here. Contrast (i): arm b against arm a, and against the best
  of d and e (the lower sunny-midday CRPS of the two, picked per lead band on the same rows), by
  lead band (day 1-2, 3-5, 6-10); a gain is an interval that excludes zero with a point estimate of
  at least the bar. Contrast (ii): over all half-hours, b minus a is no worse than +0.02. Contrast
  (iii): in the step world, b beats a and d after the step by at least the step-world bar.
  Placebos: capacity permuted across the six Primaries (scaled by 0.5 and 1.5 as experiment 643
  advised, beside the plain swap) and a shuffled-day PV term must each lose to b by at least the
  bar. Recommendation: if (i) holds at every lead band and (ii) holds, recommend b as a release
  candidate and name its cost (the capacity table and its update cadence). If (i) holds only at day
  1-2, say so, because NGED care about 1 to 10 days. If neither b nor d beats a by the bar, stop the
  explicit-PV line. If the capacity-swap placebo fails again, report that the gain does not depend
  on the right capacity.
benchmark: >-
  Arm a (free network, ensemble inputs) and the floors, climatology and the manual heuristic, scored
  on the same rows. Experiment 639's and 643's hindcast results are context only, since they use
  different weather inputs.
row_set_id: >-
  Experiment 639's test windows (60 days after 2025-07, 2026-03, and 2026-06; 30 days at 2026-06),
  with every ensemble run in those windows at lead 0 to 240 hours rather than the first run per
  valid time, so a row is one Primary, valid time, and run. Cleaned power, farm-gap rows dropped.
  Nothing dated 2026-07-01 or later is used.
metric: >-
  CRPS in points of effective capacity from the 11 quantiles of each mixture, on sunny middays and
  on all half-hours, by lead day; MAE of the median forecast; q05-q95 coverage; the
  avoidable-curtailment-cost proxy of earlier studies.
verdict: refuted
best_arm: >-
  The free network corrected by its own mean error over the previous 14 days (the 14-day tracker):
  sunny-midday CRPS 3.41, 3.82, and 4.48 points of capacity at lead days 1-2, 3-5, and 6-10 in the
  real months, against 3.75, 4.28, and 5.05 for the free network and 4.11, 4.53, and 5.33 for the
  explicit network.
headline: >-
  With ensemble forecast weather the explicit solar network is worse than the plain free network on
  sunny middays at every lead band, by 0.37, 0.25, and 0.27 CRPS points (real months, intervals
  exclude zero), so the explicit-PV line stops. The 14-day tracker beats the free network by 0.34,
  0.46, and 0.57 points.
lessons:
  - >-
    The sunny-midday gain of the explicit solar term in experiment 639 came with known weather and
    does not survive forecast weather: the explicit network loses to the free network on 5 of 6
    Primaries, at all 10 lead days, and in each of the 5 seeds at lead days 1-2.
  - >-
    The loss is not confined to days the ensemble mean calls cloudy. The audit found the explicit
    network worse at every forecast-sunniness level, even where the ensemble-mean clear-sky index
    is at least 0.85.
  - >-
    The shuffled-day placebo is far worse than every other arm, so the physics curve carries real
    information; the capacity-permuted placebo is only 0.08 points worse than the right capacity.
  - >-
    A correction from the last 14 days of a free network's own errors is the best arm at every lead
    band, so the production candidate for the sunny-midday bias is a tracker, not a capacity table.
take_home_message: >-
  Telling the demand network each Primary substation's solar capacity does not help once the weather
  is an ensemble forecast: it scores worse than the plain network at every lead from 1 to 10 days. A
  simple correction from the previous 14 days' errors beat both by 0.3 to 0.6 points of capacity on
  sunny middays, so the explicit-capacity line stops and the tracker is the candidate to test next,
  with its own placebo and a version that needs only forecast sunniness.
advice:
  - >-
    Do not build an explicit solar-capacity term into the ensemble-weather demand network, because it
    lost to the free network at all lead bands in experiment 644.
  - >-
    Test a trailing 14-day error correction of the free network as the sunny-midday bias remedy,
    because it beat the free network by 0.34 to 0.57 points in experiment 644 (not independently
    audited; it is applied on realised-sunny half-hours).
  - >-
    Do not carry a gain found with known weather (CAMS and ERA5) over to forecast weather without a
    forecast-weather rerun, because experiment 639's 0.75-point gain became a 0.25 to 0.37-point loss.
production_status: not-for-production
adopted_in: none
---

# The explicit solar term does not survive forecast weather, but a 14-day error correction does

**Telling the demand neural network each Primary substation's solar capacity makes its sunny-midday
forecast worse once the weather inputs are ensemble forecasts. A correction from the previous 14
days' errors makes the same forecast better.** The explicit-PV line therefore stops here, and the
correction is the candidate to test next.

We asked whether the gain of [experiment
639](../00639-pv-explicit-hindcast-probe-set-capacity-step-world/result.md), which gave the network
the solar capacity behind each Primary substation and had it subtract a physics solar curve, holds
when the weather is a forecast (ECMWF ensemble, leads 0 to 240 hours) instead of the recorded
weather. The **free network** learns the solar response from weather alone. The **explicit network**
is told the capacity. The **14-day tracker** is the free network plus its own mean error over the
previous 14 days, as in [experiment
643](../00643-pv-explicit-term-against-bias-shifted-free-network-and-more-placebo-permutations/result.md).
We scored the six MW Primaries on sunny middays (10:00 to 15:00 UTC, observed clear-sky index at
least 0.7) in the 60 days after each of three cutoffs, five seeds. The error is the continuous
ranked probability score (CRPS) in points of each Primary's capacity, and smaller is better.

## Results

![CRPS by lead day, real months](crps_by_lead_real.svg)

| Real months, sunny middays | Day 1-2 | Day 3-5 | Day 6-10 |
|---|---|---|---|
| Free network (a) | 3.75 | 4.28 | 5.05 |
| Explicit, calibrated capacity (b) | 4.11 | 4.53 | 5.33 |
| Explicit, raw capacity (c) | 3.92 | 4.41 | 5.24 |
| Explicit, table capacity (f) | 3.99 | 4.44 | 5.23 |
| 14-day tracker (d) | 3.41 | 3.82 | 4.48 |
| Free + training-month constant (e) | 4.30 | 4.94 | 5.81 |
| Capacity-permuted placebo | 4.19 | 4.63 | 5.39 |
| Shuffled-day placebo | 5.40 | 5.60 | 6.09 |

- **Contrast (i) fails.** Free network minus explicit network is -0.37 [-0.54, -0.20], -0.25
  [-0.45, -0.08], and -0.27 [-0.40, -0.14] at the three bands (negative favours the free network;
  intervals are 99.7% day-block bootstraps, Bonferroni over 19 contrasts). The best of d and e (d at
  every band) beats the explicit network by 0.71, 0.71, and 0.84.
- **Contrast (ii) passes.** Over all half-hours the explicit network is 0.007 points worse than the
  free network [-0.04, +0.03], inside the 0.02 limit.
- **Contrast (iii) fails.** In the step world the explicit network ties the free network (-0.45
  [-0.99, +0.05], -0.20 [-0.63, +0.19], +0.04 [-0.35, +0.38]) and loses to the tracker by 1.2 to
  1.3 points. Here the 4 MW step lies wholly before the ensemble rows begin, so the contrast tests
  carrying capacity from pretraining to fine-tuning, not tracking a step.
- **Placebos.** The explicit network beats the capacity-permuted placebo by only 0.08, 0.10, and
  0.07 points, below the bars (0.084 to 0.123), so the gain, such as it is, does not depend on the
  right capacity. It beats the shuffled-day placebo by 0.8 to 1.3.
- **Absolute scores and tails.** Sunny-midday q05-q95 coverage is 88.0% (a), 89.2% (b), 91.9% (d),
  and 82.3% (e). The mean absolute error of the median over all sunny-midday leads, in points of
  capacity, is 6.33 (a), 6.67 (b), 5.58 (d), and 7.37 (e).

![CRPS by lead day, step world](crps_by_lead_step.svg)

![One day at one Primary](example_day.svg)

## Checks

- **Triggers fired:** a placebo that should lose to the explicit network does not (trigger 1), and
  the result contradicts experiment 639 (trigger 2). A fresh Sonnet audit re-derived the CRPS
  contrasts from the stored quantiles and could not break them: b is worse at all 10 lead days, in
  all 5 seeds at lead days 1-2, with every leave-one-Primary-out contrast keeping its sign. The
  per-cutoff signs are not uniform: the 2026-06 cutoff at day 3-5 favours b by 0.06.
- **Tested explanation:** the audit tested whether the smooth ensemble-mean curve hurts on days the
  ensemble calls cloudy. The loss appears at every forecast-sunniness level, so that explanation is
  not supported. The ensemble spread was not available to the audit, so a spread explanation is
  untested.
- **Median minus mean forecast per site:** not computed, because only the 11 quantiles are stored.
- **Not audited:** the 14-day tracker (d) and the constant shift (e), which are post-processing in
  `e00644_analyse.py`. Both are applied on realised-sunny half-hours, a selection a production
  tracker would not have.

## What the next experiment should answer

Does the 14-day tracker still beat the free network when it is applied by forecast sunniness
instead of observed sunniness, and does a shuffled-error placebo fail to match it? The explicit
capacity term is not worth further work in this form.

## Departures and caveats

- The cached ensemble tables hold the mean and standard deviation per run, not the 51 members, so
  the physics curve uses the ensemble-mean shortwave. A member-level version is untested.
- The 0.5 and 1.5 capacity-scaling placebos and the curtailment-cost proxy were not run (see the
  plan above).
- The second month of each test window uses the standardiser of its own fold, not the cutoff
  month's, so its inputs are slightly rescaled relative to the pretraining, equally for all arms.
- Code: this experiment imports experiments 510, 554, 634, 638, 639, and 643 and edits none of
  them.
