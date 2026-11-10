# Unmetered PV capacity: what we know

**Probing a trained neural network reads the whole-summer solar capacity behind each Primary
substation, at about 0.68 of an injected farm before calibration, and every attempt to follow a
change in capacity, or to use the capacity to improve the forecast, has failed.** The probe's
estimates overlap the regression table of experiment 474 at all six MW Primaries and lie above the
registered capacity at 8 of the 12 Primaries that have a probe interval. They cannot say how
capacity changed from one year to the next. An explicit solar term built from the probed capacity
gains about 0.75 points of capacity on sunny middays when the weather is recorded, and loses 0.25 to
0.37 points when the weather is an [ensemble forecast](../../glossary.md#ens-era5-and-cams). The
work is paused after 10 November 2026. This page lets a reader restart it in January from the merged
result pages alone.

## Take-home message

**What works.** A [donor-day probe](#the-probe-and-its-calibration) of a lag-free neural network,
divided by the recovery of injected real solar farms, estimates 2.4 to 3.9 MW of unmetered solar
behind each of the six Primaries metered in MW. Its intervals overlap the regression table of
[experiment 474](../../00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals/result.md)
at all six, with medians 0.94 to 1.14 times that table ([experiment
649](../../00649-pv-capacity-table-season-year-from-stored-probes/result.md)). At the five Primaries
metered in MVA that never fold, a linear probe divided by 0.79 reads 1.25 to 6.62 MW ([experiment
673](../../00673-pv-capacity-for-nged-per-primary-with-register-cross-check/result.md)).

**What does not work.** The probe cannot follow a capacity step inside its training window, reads a
cloud-following load as solar, and has season-year resolution at best. The explicit solar term does
not improve a forecast made with ensemble weather. The loss is largest at day 1 to 2, at 0.37
points, and does not grow with lead. Explanations that have been tested and ruled out are the time
step, the interpolation and time stamps, ensemble members and cell temperature, a bias adapter, and
missing spread. Forecast cloud error remains the inferred cause, with no direct test.

**What to try next.** The ranked options are in [the January options](#options-for-january). The
cheapest are a per-Primary yearly capacity trajectory from register curves anchored to the probe,
and a direct test of the cloud-error explanation on stored quantiles.

## What we know and what does not work

### The probe and its calibration

**The probe reads about 0.68 of a real solar farm's megawatts, and calibration undoes most of
that.** The probe takes a lag-free neural network, fixes its calendar inputs, and reads the change
in forecast demand when donor-day weather is supplied. Experiment
[621](../../00621-pv-probe-injected-known-answer-and-placebo-test/result.md) injected the output of
real metered solar farms into the target and read 0.68 of the injected megawatts (0.66 to 0.70 at
each of six Primaries). Placebos read about 0.1 (95th percentile 0.08 to 0.16, bar 0.20). The
verdict was "not supported", because the intervals missed the truth and a cloud-following load could
not be told from solar.

**Most of the shortfall is the shape of a real farm, and the sky is not the cause.** A real farm of
a given peak produces about 20% less energy than the idealised curve of the same peak, and a probe
that recovered energy exactly would read about 0.77 ([experiment
624](../../00624-pv-probe-contrastive-injection-arms/result.md)). In energy the probe recovers 0.78
to 0.88 ([experiment 634](../../00634-pv-probe-capacity-step-world-and-energy-recovery/result.md)).
A metered neighbouring farm 5 to 25 km away as a second sky input lifted recovery from 0.68 to only
0.69 ([experiment 635](../../00635-pv-probe-metered-neighbour-pv-as-sky-input/result.md)). A
monotone network and a contrastive training loss gave 0.71 and broke the network, respectively
([experiment 623](../../00623-pv-probe-monotone-arm-known-answer/result.md), experiment 624).

**A constant injected capacity calibrates to 0.95 to 1.04, so the probe is usable as a whole-summer
estimate.** The leave-one-farm-out recovery factor is 0.68 (0.67 to 0.69). Experiment 649 divides
stored probe outputs by this factor and gets an 80% interval that covers the injected truth in 0.81
of 96 leave-one-farm-out injections (reviewer variants 0.78 to 0.91).

### Capacity steps and the sunny-spring confound

**The probe cannot follow a capacity step inside its training window, and part of what it reads is
the weather of the step's months.** Experiment 634 injected a step of 2 or 4 MW starting in April or
July 2025. The calibrated read of the spring step was 0.32 to 0.51, and that of the summer step was
0.0 to 0.1, against a bar of 0.70. Spring 2025 was unusually sunny: the clearness index was 0.76 in
April to June 2025, against 0.61 in 2024 and 0.65 to 0.69 in 2020 to 2023. A placebo that shuffled
the sunshine read about a third of the spring step.

**A step that starts in an ordinary spring is followed.** [Experiment
638](../../00638-pv-probe-step-placebo-climatological-and-ordinary-year-followup/result.md) found
0.82 to 0.88 of a 4 MW step that started in the spring of 2023, against 0.32 to 0.49 for the same
step in 2025. Its placebo read 0.07 to 0.12 in 2023 and 0.08 to 0.28 in 2025. Experiment 638's
verdict was "partly supported", and the summer-step reads (0.45 to 0.55 for a step in summer 2024,
0.00 to 0.11 in summer 2025) show that no sunny summer exists in the record to test against.

**Rule for any capacity-step claim: a step in a sunnier-than-usual season cannot be read until a
climatological placebo built from the multi-year day-of-year mean has been scored alongside it.**
Experiment 638 recommends reporting per training window until the excess below is explained.

### The explicit solar term and forecast weather

**With recorded weather the explicit solar term gains; with ensemble weather it loses.** [Experiment
639](../../00639-pv-explicit-hindcast-probe-set-capacity-step-world/result.md) gave a demand network
a physics solar curve with the probed capacity. On sunny middays at the six MW Primaries with
recorded weather, the free network scored 0.75 points worse (98.75% interval 0.51 to 1.01).
[Experiment 644](../../00644-pv-explicit-term-with-ens-forecast-weather/result.md) repeated the test
with ECMWF ensemble weather. The free network then scored better by 0.37, 0.25, and 0.27 points at
day 1 to 2, day 3 to 5, and day 6 to 10 (99.7% intervals -0.54 to -0.20, -0.45 to -0.08, and -0.40
to -0.14). The explicit term lost at five of the six Primaries, at all 10 lead days, and at every
level of forecast sunniness. Its verdict was "refuted".

**Much of the recorded-weather gain was the free network's bias.** In [experiment
643](../../00643-pv-explicit-term-against-bias-shifted-free-network-and-more-placebo-permutations/result.md)
the free network's median ran 4.1 points too high on sunny middays, and a 14-day error correction
added after the results beat the explicit term in the real months (-0.48, interval -0.66 to -0.30).
One Primary carried most of the gain in experiments 639 and 643, and two Primaries scored better
with the free network.

**Six explanations for the ensemble-weather loss have been tested, and five are ruled out.**

- The ensemble's 3-hour and 6-hour time steps: averaging recorded weather into 3-hour and 6-hour
  windows still leaves a gain of 0.84 and 0.77 points ([experiment
  661](../../00661-pv-ens-time-resolution-shortwave-interpolation-audit-and-bias-adapter/result.md)).
- The shortwave interpolation and time stamps: the audit found zero time-stamp shift and a pipeline
  identical to production to 0.0 W/m2 (experiment 661).
- Ensemble members and cell temperature: a curve per member with a Faiman cell temperature still
  lost by 0.35, 0.23, and 0.21 points ([experiment
  655](../../00655-pv-member-level-cell-temperature-term-and-forecast-sunniness-tracker/result.md)).
- A learned bias adapter: it stayed 0.13 to 0.29 points behind the free network (experiment 661).
- Missing spread: building the distribution as gross demand minus a random member's solar curve lost
  by 0.29 to 0.38 points ([experiment
  664](../../00664-pv-sum-of-distributions-spread-and-recalibration/result.md)).
- **Forecast cloud error is the inferred cause and has no direct test.** The 14-day error correction
  keeps almost none of its gain when it uses only forecast sunniness (experiment 655), which is
  consistent with that explanation and also with selecting scoring rows on observed sunniness.

### Cost of converting to apparent power

**Under NGED's stated power factor 1 with a small reactive floor, the linear probe loses about 0.5
points at half the midday reading.** That is the level of the five non-folding Primaries metered in
MVA. The 80% interval is -1.0 to 7.7 points, so a loss of up to 7.7 points is not excluded
([experiment 679](../../00679-pv-mva-conversion-loss-reactive-floor-at-half-level/result.md)). The
power-factor 0.97 assumption of [experiment
656](../../00656-pv-probe-through-the-mva-fold-at-eight-primaries/result.md), [experiment
662](../../00662-pv-probe-through-the-mva-fold-matched-level-redesign/result.md), and [experiment
671](../../00671-pv-mva-fold-probe-fair-denominator-and-d3-d4-disagreement/result.md) gave 10.7
points (7.8 to 17.1). The reactive floor in experiment 679 is an assumption of 0.10 times the median
sunny-midday reading, and the real meters' reactive power is unknown.

**At the three folding Primaries (D8, D12, and D22) and at D5 and D10, the capacity is
report-only.** The fold model's extra capacity at D2, D3, D4, and D19 comes almost entirely from two
free reactive-power terms, so experiment 671 treats the linear probe as the better-supported reading
and the fold model as an upper bound.

## Earlier attempts

**Fourteen earlier experiments tried to estimate capacity; one method passed its gates, on a weaker
placebo.** The table is from the Opus design review of 7 November 2026. It judged the probe "a
neural version" of experiments 235 and 479, with a likely recovery of 0.75 to 0.90.

| Experiment | Method | What it found |
|---|---|---|
| [129](../../00129-differentiable-physics-pv-capacity-known-answer-test/result.md) | Convex generalised additive model, one capacity per season-year | Passed all four gates (recovery at 5 of 5 Primaries, placebo 0.12); the range 1.2 to 1.6 MW is a scenario spread |
| [235](../../00235-pv-capacity-matched-day-pairs/result.md) | Matched sunny-minus-cloudy day pairs | Recovery 0.758; a cloud-following load added 0.295; December-to-January placebo 0.33 |
| [241](../../00241-pv-capacity-calibrated-interval/result.md) | Divide by the median recovery of other farms | Within 10% at 5 of 5; the 80% interval covers 59 to 79%, and 0 to 2% with a cloud load |
| [269](../../00269-pv-capacity-gain-decomposition/result.md) | Pooled empirical plant table driven by CAMS | 0.892 (99th-percentile ratio) and 0.966 (capacity ratio); satellite sunshine is not the limit |
| [349](../../00349-primary-demand-default-3-exp-269-capacity-in-the-stack/result.md), [352](../../00352-primary-demand-default-3-capacity-response-surface/result.md) | Better capacity inside the generalised additive model stack | Sunny score flat in capacity (0.047 worse; flat surface) |
| [405](../../00405-mdn-explicit-pv-term-learned-capacity-known-answer/result.md) | Mixture-density network with learned-capacity solar term | A day-shuffled placebo kept 0.60 to 0.91 of real recovery (limit 0.20) |
| [409](../../00409-mdn-solar-term-day-of-year-and-anomaly-known-answer/result.md) | Same, with a day-of-year term and anomaly-only injection | Anomaly read-out 0.03 to 0.05 MW against seed noise of 0.18 to 0.59 MW; the test could not decide |
| [416](../../00416-mdn-explicit-solar-term-better-powered-follow-up/result.md) | Explicit solar term, 5 seeds, 12 months | Sunny-midday +0.227 points, interval -0.135 to +0.560; capacities 0.4 to 0.7 of the reference, 1.5 at D5 |
| [474](../../00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals/result.md) | Placebo-gated ladder on the generalised additive model | No rung passed: profile placebo 0.25 to 0.40, and 0.49 to 0.51 with cloud terms |
| [477](../../00477-neural-pv-module-placebo-gate-and-known-answer-on-stored-fits/result.md) | Neural PV module on stored fits, lag-masked | Placebo read 0.90 to 0.95 of the injected MW against 0.86 to 0.92 for the real farm |
| [479](../../00479-pv-capacity-anomaly-regression-rung/result.md) | Within-month anomaly regression | Placebos pass, mostly by construction; recovery 0.62 to 0.83; a cloud load fakes 0.2 to 0.3 of a 2 MW farm |
| [501](../../00501-explicit-pv-column-from-the-474-capacity-table/result.md) | Explicit PV column from the experiment 474 table | Sunny-midday -0.014 points against 0.225 detectable (the fifth null) |
| [504](../../00504-neural-demand-leak-free-recent-cams-irradiance-inputs/result.md) | Recent CAMS columns in the neural network | Leak-free columns null; target-time CAMS -0.318 at the eight MVA Primaries and -0.467 at the six MW Primaries |

**Two figures from the same review bound what a method can detect.** Experiment 129 found capacity
growing 0.4 to 2.8 MW over about 4 years, which is about 0.1 to 0.7 MW per year. The within-month
minimum detectable capacity of experiment 479 is 0.05 to 0.35 MW.

## Facts missing from the result pages

- **Polygon size.** Only the 16 Primaries have an `area_wkt` service-area polygon. The areas run
  from 14.7 to 215.7 km2, with a median of 102 km2, against about 25 km2 for one CAMS pixel. A
  median polygon therefore spans about 4 pixels. Source: the multi-pixel CAMS brief and the first
  probe design brief, which are not merged pages.
- **Power factor.** NGED states that its meters read power factor 1 with a small reactive floor,
  relayed by the maintainer. [Experiment
  673](../../00673-pv-capacity-for-nged-per-primary-with-register-cross-check/result.md) and
  experiment 679 use that statement, and the floor size is an assumption.
- **Probe above a date-blind regression.** In experiment 638 the probe reads 0.1 to 0.2 raw above a
  regression of the injected capacity on the unit curve and temperature for a real step, and 0.05 to
  0.1 above it for a placebo. The cause was not found, and weekday, holiday, and pandemic columns
  are an untested candidate.
- **One-sided bias of the free network on sunny middays.** In experiment 664, realised power falls
  below the free network's 5% quantile in 9.3% of half-hours at day 1 to 2, and above its 95%
  quantile in 1.9%. The lower band is too narrow and the upper band too wide, so a recalibration of
  the width cannot fix it.
- **Probe against the register floor.** In experiment 673 the probe interval lies below the
  registered floor at D2, D3, D5, and D21. The D2 flag depends on the calibration factor (with 0.68
  the interval overlaps the floor). At D5 one site of 2.70 MW accounts for most of the 4.55 MW
  floor. The sources disagree on D10 and D11: experiment 673's main table shows both above the
  floor, and adding the Feed-in Tariff band of 50 kW to 1 MW moves D10 below the floor and D11 to
  "overlaps". [Experiment 267](../../00267-pv-capacity-early-years-and-register-floor/result.md)
  found a 2021 shortfall at D10 and D11 that remains unexplained.
- **Register curves for recent domestic PV.** The Feed-in Tariff report covers 2010-04-01 to
  2025-03-31 by postcode district only, the Embedded Capacity Register lists sites of 50 kW and
  above, and the Renewable Energy Planning Database lists mostly sites of 150 kW and above. The
  result pages do not say that a register-based curve is unavailable for recent domestic PV, but no
  register covers it.
- **Multi-pixel CAMS.** A multi-pixel CAMS download was written and never run. Displacement of the
  sky by 5 to 17 km cost 0.02 to 0.04 in experiment 624, and a metered neighbour's output added
  0.008 in experiment 635.
- **Wind capacity estimation is parked.** [Experiment
  617](../../00617-wind-turbine-step-capacity-known-answer-and-fit/result.md) failed its
  known-answer test.

## Open questions

- **What causes the loss with ensemble weather?** Forecast cloud error is inferred and untested.
- **How did capacity change from year to year?** The probe has season-year resolution at best, and
  the slow drift of 0.1 to 0.7 MW per year is below the step size that a detector can read.
- **Why does the probe sit 0.1 to 0.2 MW above a date-blind regression?** The cause is unknown.
- **Do the MVA figures hold?** They depend on the unknown reactive power of the meters, and the five
  Primaries that sit below their register floors (D2, D3, D5, D21) need NGED's own list of connected
  solar.

## Options for January

**These are options in ranked order, and none of them is agreed work.**

1. **A per-Primary yearly capacity trajectory, 2019 to 2026.** Build it from register curves
   anchored to the probe. It asks whether a composite curve, made of the season-year capacities of
   experiments 129, 474, and 479 from 2021, the probe of experiment 649, a cumulative register curve
   by polygon, and an unregistered-share term, tracks the drift. It costs about a day on CPU. It
   depends on nothing.
2. **A direct test of the cloud-error explanation.** Regress the explicit-minus-free gain on
   forecast cloud error, using the stored quantiles of experiments 644 and 655. It costs about half
   a day and depends on nothing.
3. **An asymmetric sunny-midday quantile bias correction for the free network.** It asks whether the
   one-sided bias in experiment 664 can be removed. It costs about 1 to 2 GPU-hours and has a
   serving-path implication, so the maintainer decides first.
4. **Native floor-arm worlds at the five non-folding MVA Primaries.** It asks whether the 0.5-point
   loss of experiment 679 holds at D2, D3, D4, D19, and D21. It costs about 0.5 GPU-hours and needs
   the code of experiment 662 extended.
5. **Explicit PV as a graph-detector arm.** It depends on option 1.
6. **NGED connection records.** They would settle D2, D3, D5, and D21, and are an option only if the
   maintainer later allows it.
