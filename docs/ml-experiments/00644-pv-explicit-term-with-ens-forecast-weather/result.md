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
  favours the explicit network. Intervals are day-block bootstraps (resampling whole days) at 99.7%,
  a Bonferroni correction over 19 contrasts, pooled over the five seeds. The bars start from
  experiment 639's minimum detectable effects, 0.067 (real months) and 0.181 (step world), and the
  analysis raises each to the seed-split null of its lead band when that is larger. The added step
  is a capacity table that someone must update. The explicit network therefore needs a gain above
  zero, and no larger production threshold is set here. Contrast (i): arm b against arm a, and
  against the best
  of d and e (the lower sunny-midday CRPS of the two, picked per lead band on the same rows), by
  lead band (day 1-2, 3-5, 6-10); a gain is an interval that excludes zero with a point estimate of
  at least the bar. Contrast (ii): over all half-hours, b minus a is no worse than +0.02. Contrast
  (iii): in the step world, b beats a and d after the step by at least the step-world bar.
  Placebos: capacity permuted across the six Primaries (experiment 639's shift 2; the 0.5 and 1.5
  scalings were not run) and a shuffled-day PV term must each lose to b by at least the
  bar. Recommendation: if (i) holds at every lead band and (ii) holds, recommend b as a release
  candidate and name its maintenance burden (the capacity table and its update cadence). If (i)
  holds only at day
  1-2, say so, because NGED care about 1 to 10 days. If neither b nor d beats a by the bar, stop the
  explicit-PV line. If the capacity-swap placebo fails to lose to the explicit network, report that
  the gain does not depend
  on the right capacity.
benchmark: >-
  Arm a (free network, ensemble inputs) and the floors, climatology and the manual heuristic, scored
  on the same rows. Experiment 639's and 643's hindcast results are context only, since they use
  different weather inputs.
row_set_id: >-
  Experiment 639's test windows (the 60 days after 2025-07, 2026-03, and 2026-06, cut at
  2026-07-01),
  with every ensemble run in those windows at lead 0 to 240 hours rather than the first run per
  valid time, so a row is one Primary, valid time, and run. Cleaned power, farm-gap rows dropped.
  Nothing dated 2026-07-01 or later is used.
metric: >-
  Continuous ranked probability score (CRPS) in percent of effective capacity (points) from the 11
  quantiles of each forecast distribution, on sunny middays and on all half-hours, by lead day; mean
  absolute error of the median forecast; coverage of the 5% to 95% quantile interval. The
  curtailment-cost proxy was not computed.
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
    does not carry over to ensemble-forecast weather: the explicit network loses to the free network
    on 5 of 6 Primaries, at all 10 lead days, and in each of five seeds at lead days 1-2.
  - >-
    The loss is not confined to days the ensemble mean calls cloudy. A fresh reviewer re-deriving
    the scores found the explicit
    network worse at every forecast-sunniness level, even where the ensemble-mean clear-sky index
    is at least 0.85.
  - >-
    The shuffled-day placebo is far worse than every other arm, so the physics curve carries real
    information; the capacity-permuted placebo is only 0.07 to 0.10 points worse than the right
    capacity.
  - >-
    A correction from the last 14 days of a free network's own errors is the best arm at every lead
    band, so the method to test as a remedy for the sunny-midday bias is a tracker, not a capacity
    table.
take_home_message: >-
  Telling the demand network each Primary substation's solar capacity does not help once the weather
  is an ensemble forecast: it scores worse than the plain network at every lead from 1 to 10 days. A
  simple correction from the previous 14 days' errors beat the plain network by 0.3 to 0.6 points
  of capacity on sunny middays and the explicit network by 0.7 to 0.8, so the explicit-capacity line
  stops and the tracker is the candidate to test next,
  with its own placebo and a version that needs only forecast sunniness.
advice:
  - >-
    Do not build an explicit solar-capacity term into the ensemble-weather demand network, because
    it
    lost to the free network at all lead bands in experiment 644.
  - >-
    Test a trailing 14-day error correction of the free network as the sunny-midday bias remedy,
    because it beat the free network by 0.34 to 0.57 points in experiment 644 (not independently
    audited; it is applied on realised-sunny half-hours).
  - >-
    Do not carry a gain found with known weather (CAMS and ERA5) over to forecast weather without a
    forecast-weather rerun, because experiment 639's gain of 0.75 CRPS points became a loss of 0.25
    to 0.37 in experiment 644.
production_status: not-for-production
adopted_in: none
---

# The solar capacity does not help the forecast, but a 14-day error correction does

**Telling the neural network that forecasts a Primary substation's demand the solar capacity behind
that substation makes its forecast for 10:00 to 15:00 UTC on clear days worse than the same network
without the capacity, once the weather inputs are ensemble forecasts. A correction that adds the
network's own mean error over the previous 14 days makes the same forecast better, by 0.3 to 0.6
continuous ranked probability score (CRPS) points of capacity.** We therefore stop testing whether
to give the network the solar capacity, and the 14-day correction is the method to test next.

[Experiment 639](../00639-pv-explicit-hindcast-probe-set-capacity-step-world/result.md) gave a
neural network the solar capacity behind each Primary substation and had it subtract a physics solar
curve. That experiment used estimates of past weather (CAMS and ERA5). We asked whether its gain
holds with forecast weather: the European Centre for Medium-Range Weather Forecasts (ECMWF)
ensemble, at leads 0 to 240 hours. The **free network** learns the solar response from weather
alone. The **explicit network** is told the capacity (arm b uses a capacity calibrated against
injected solar farms, arm c the raw probed capacity, and arm f the table of experiment 474). The
**14-day tracker** (arm d) is the free network plus its own mean error over the previous 14 days, as
in [experiment
643](../00643-pv-explicit-term-against-bias-shifted-free-network-and-more-placebo-permutations/result.md).
Arm e shifts the free network by a constant fitted on held-out fine-tuning rows. The **step world**
adds 4 MW of a real solar farm's output to every Primary substation's demand from April 2023.
We scored the six MW Primaries on sunny middays (10:00 to 15:00 UTC, observed clear-sky index at
least 0.7, the ratio of observed to clear-sky irradiance) in the 60 days after each of three cutoff
dates (the last day the network is pretrained on), with five seeds. CRPS is in percent of each
Primary's effective capacity ("points"), and smaller is better.

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

- **The explicit network does not beat the free network at any lead band.** Free network minus
  explicit network is -0.37 [-0.54, -0.20], -0.25
  [-0.45, -0.08], and -0.27 [-0.40, -0.14] at the three bands (negative favours the free network;
  intervals are 99.7% day-block bootstraps, Bonferroni over 19 contrasts). The best of d and e (d at
  every band) beats the explicit network by 0.71, 0.71, and 0.84.
- **Over all half-hours the two networks tie.** The explicit network is 0.007 points worse [-0.04,
  +0.03]. The point estimate is inside the planned 0.02 limit and the interval's upper end is not.
- **In the step world the explicit network does not beat the free network or the tracker.** It ties
  the free network (-0.45
  [-0.99, +0.05], -0.20 [-0.63, +0.19], +0.04 [-0.35, +0.38]) and loses to the tracker by 1.2 to
  1.3 points. In this test the 4 MW step lies wholly before the ensemble rows begin. The contrast
  therefore tests carrying capacity from pretraining to fine-tuning, not tracking a step.
- **The explicit network barely beats a wrong capacity.** The explicit network beats the
  capacity-permuted placebo by 0.08, 0.10, and 0.07
  points. That is below the bar at day 1-2 (0.084) and day 6-10 (0.123) and marginally above it at
  day 3-5 (0.097 against 0.093), so the explicit network's small edge over a wrong capacity is not
  established. It beats the shuffled-day placebo by 0.8 to 1.3. The bars are recomputed per band
  as the larger of experiment 639's value (0.067 real, 0.181 step) and the seed-split null on these
  rows: 0.084, 0.093, and 0.123 in the real months; 0.205, 0.246, and 0.266 in the step world.
- **The 14-day tracker's intervals are the best calibrated.** Coverage of the 5% to 95% quantile
  interval on sunny middays (nominal 90%) is 88.0% (a), 89.2% (b), 91.9% (d),
  and 82.3% (e). The mean absolute error of the median over all sunny-midday leads, in points of
  capacity, is 6.33 (a), 6.67 (b), 5.58 (d), and 7.37 (e).

![CRPS by lead day, step world](crps_by_lead_step.svg)

![One day at one Primary](example_day.svg)

## Checks

- **Two review conditions fired:** a placebo that should lose to the explicit network does not, and
  the result contradicts experiment 639. A fresh Sonnet sub-agent re-derived the CRPS contrasts from
  the stored quantiles and could not break them. The explicit network is worse at all 10 lead days
  and in all five seeds at lead days 1-2. Every contrast keeps its sign when one Primary is left out
  in turn. The
  per-cutoff signs are not uniform: the 2026-06 cutoff at day 3-5 favours b by 0.06.
- **Tested explanation:** the sub-agent tested whether the smooth ensemble-mean curve hurts on days
  the
  ensemble calls cloudy. The loss appears at every forecast-sunniness level, so that explanation is
  not supported. The ensemble spread was not available to the sub-agent, so a spread explanation is
  untested.
- **Median minus mean forecast per site:** not computed, because only the 11 quantiles are stored,
  so a skew in the forecasts is unchecked.
- **Not audited:** the 14-day tracker (d) and the constant shift (e), which are post-processing in
  `e00644_analyse.py`. Both are applied on realised-sunny half-hours, a selection a production
  tracker would not have.

## What the next experiment should answer

**** Does the 14-day tracker still beat the free network when it is applied by forecast sunniness
instead of observed sunniness, and does a shuffled-error placebo fail to match it? The explicit
capacity term is not worth further work in this form.

## Departures and caveats

- The cached ensemble tables hold the mean and standard deviation per run, not the 51 members, so
  the physics curve uses the ensemble-mean shortwave. A member-level version is untested.
- The 0.5 and 1.5 capacity-scaling placebos and the curtailment-cost proxy were not run (see the
  plan in the first commit of this pull request).
- The second month of each test window scales its inputs with the mean and standard deviation of its
  own fold, not of the cutoff month, so its inputs are slightly rescaled relative to the
  pretraining, equally for all arms.
- Code: this experiment imports experiments 510, 554, 634, 638, 639, and 643 and edits none of
  them.
