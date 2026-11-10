# Unmetered PV capacity: what we know

**A probe of a trained neural network reads the whole-summer capacity of photovoltaic (PV) solar
panels behind each Primary substation, with a calibration measured on injected real solar farms.** The probe followed a step in capacity only when the step began in
an ordinary spring. An explicit solar term built from the probed capacity improved the forecast with
estimates of past weather and worsened it with ensemble forecast weather. The maintainer paused the
work after 10 November 2026. This page is meant to let a reader restart it in January.

**Sources.** Every figure comes from a merged result page of the theme, linked at its first use.
Items marked "(brief)" come from briefs in `data/ml-experiments/briefs/` of the production
repository (`pv_probe_design_review_answer.md`, `pv_probe_brief_changes_after_621_623.md`,
`pv_next_c_multi_pixel_cams_brief.md`, `pv_session_reply_solar_drift.md`, and
`pv_session_reply_for_graph_labeller.md`), which are not on the docs site.

## Take-home message

**What works.** The probe, divided by the recovery of injected real solar farms, estimates 2.4 to
3.9 MW of unmetered solar behind each of the six Primaries metered in MW. Its intervals overlap the
regression table of [experiment
474](../../00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals/result.md) at all six,
with medians 0.94 to 1.14 times that table ([experiment
649](../../00649-pv-capacity-table-season-year-from-stored-probes/result.md)). At the five
Primaries metered in MVA that never fold, a linear probe divided by a recovery factor of 0.79 reads
1.25 to 6.62 MW ([experiment
673](../../00673-pv-capacity-for-nged-per-primary-with-register-cross-check/result.md)).

**What does not work.** The probe followed a step in capacity only when the step began in an
ordinary spring, and it reads a cloud-following load as solar. Its values have season-year
resolution at best. An explicit solar term does not improve a forecast made with ensemble weather.
The loss is largest at day 1 to 2, at 0.37 points, and does not grow with lead.

**Tested explanations for that loss are ruled out one at a time.** They are the ensemble time step,
the interpolation and time stamps, ensemble members and cell temperature, a bias adapter, and
missing spread. Forecast cloud error is the inferred cause, with no direct test.

**What to try next.** The ranked options are in [Options for January](#options-for-january). The
cheapest are a per-Primary yearly capacity trajectory from register curves anchored to the probe,
and a direct test of the cloud-error explanation on stored quantiles.

## Terms used on this page

- **Primaries.** The 6 MW Primaries are D5, D9, D10, D11, D14, and D16, whose meters read signed
  real power in megawatts (MW). The five non-folding MVA Primaries are D2, D3, D4, D19, and D21,
  whose meters read apparent power in megavolt-amperes (MVA). The three folding MVA Primaries are
  D8, D12, and D22. A "fold" occurs when solar output exceeds demand: the real flow turns negative,
  and apparent power, which is never negative, folds it back to a positive reading.
- **Probe.** The probe takes a neural network with no inputs from recent power readings (a
  "lag-free" network), fixes its calendar inputs, supplies the weather of other ("donor") days, and
  reads the change in forecast demand as solar megawatts.
- **Recovery.** The probe's reading divided by the megawatts injected into the target. Recovery is
  measured by adding the output of real metered solar farms to a substation's demand and probing it
  (a known-answer test).
- **Placebo.** A test that injects nothing, or a shape unrelated to the sky, so that a reading above
  zero is false signal.
- **Free network.** The demand neural network without a solar term. The **explicit solar term** is
  a physics solar-output curve scaled by the probed capacity and given to the demand neural network.
- **Verdicts.** "Supported", "partly supported", "not supported", and "refuted" refer to the
  pre-registered decision rule of the experiment cited.
- **Sunny midday.** A half-hour between 10:00 and 15:00 UTC with an observed clear-sky index of at
  least 0.7 ([experiment
  639](../../00639-pv-explicit-hindcast-probe-set-capacity-step-world/result.md)).
- **Points.** Quantile-score or CRPS points of capacity; lower is better, and a positive gain means
  the explicit term scored better.
- **Past weather.** Estimates of past weather (ERA5 reanalysis and CAMS satellite irradiance, see
  the [glossary](../../glossary.md#ens-era5-and-cams)), as against ECMWF ensemble forecasts.

## What we know and what does not work

### Reading capacity from a trained neural network

**The probe reads about 0.68 of a real solar farm's megawatts, and calibration removes part of that
shortfall.** [Experiment 621](../../00621-pv-probe-injected-known-answer-and-placebo-test/result.md)
injected the output of real metered solar farms and read 0.68 of the injected megawatts (0.66 to
0.70 at each of six Primaries). Placebos read about 0.1 (95th percentile 0.08 to 0.16, bar 0.20).
The verdict was "not supported", because recovery missed the 0.85 to 1.15 bar, the raw intervals
covered the truth in 0 of 144 injections, and a cloud-following load could not be told from solar.

**Part of the shortfall is the shape of a real farm; the sky is not the cause.** A real farm of a
given peak produces about 20% less energy than the idealised curve of the same peak, and a probe
that recovered energy exactly would read about 0.77 ([experiment
624](../../00624-pv-probe-contrastive-injection-arms/result.md)). In energy the probe recovers
0.78 to 0.88 ([experiment
634](../../00634-pv-probe-capacity-step-world-and-energy-recovery/result.md)). Clipping and sky
displacement overlap with this explanation. A metered neighbouring farm 5 to 25 km away as a second
sky input lifted recovery from 0.68 to 0.69 ([experiment
635](../../00635-pv-probe-metered-neighbour-pv-as-sky-input/result.md)). A monotone network lifted
it to 0.71
([experiment 623](../../00623-pv-probe-monotone-arm-known-answer/result.md)). A contrastive
training loss gave a recovery of 0.00 in every world (experiment 624).

**A whole-summer capacity can be read, with caveats.** A constant injected capacity calibrates to
0.95 to 1.04, and the leave-one-farm-out recovery factor is 0.68 (0.67 to 0.69). Experiment 649
divides stored probe outputs by that factor, and its 80% interval covers the injected truth in 0.81
of 96 leave-one-farm-out injections (experiment 634 read 0.66 with a different widening, so the two
are not comparable). The calibration rests on four injected farms that resemble each other, so it
may not transfer to a fleet of rooftop panels.

### Capacity steps and the sunny spring of 2025

**A step that began in the very sunny spring of 2025 was read at 0.32 to 0.51, and part of what the
probe reads is the weather of the step's months.** Experiment 634 injected a step of 2 or 4 MW from
April or July 2025. The calibrated read of the spring step was 0.32 to 0.51 and that of the summer
step was 0.0 to 0.1, against a bar of 0.70. The clearness index was 0.76 in April to June 2025,
against 0.61 in 2024 and 0.65 to 0.69 in 2020 to 2023.

**A step that begins in an ordinary spring is followed.** [Experiment
638](../../00638-pv-probe-step-placebo-climatological-and-ordinary-year-followup/result.md) read
0.82 to 0.88 of a 4 MW step that began in spring 2023, with a placebo at 0.07 to 0.12. It read 0.32
to 0.49 for the same step in spring 2025, where a placebo built from the multi-year day-of-year
mean read 0.08 to 0.28. The verdict was "partly supported". No July-to-September in the record has
a clearness index above 0.63, so a step in summer 2024 stands in for a sunny summer, and it read
0.45 to 0.55.

**A step in a sunnier-than-usual season should be scored beside a climatological placebo before it
is read as capacity.** This is a rule drawn from experiments 634 and 638, and experiment 638
recommends reporting per training window until the excess in the facts below is explained.

### Adding a physics solar term to the demand neural network

**With estimates of past weather the explicit solar term gains, and with ensemble weather it
loses.** Experiment 639 gave a demand network a physics solar curve with the probed capacity. On
sunny middays at the six MW Primaries with past weather, the free network scored 0.75 points worse
(98.75% day-block interval 0.51 to 1.01). [Experiment
644](../../00644-pv-explicit-term-with-ens-forecast-weather/result.md) repeated the test with
ECMWF ensemble weather. The free network then scored better by 0.37, 0.25, and 0.27 points at day 1
to 2, day 3 to 5, and day 6 to 10 (99.7% intervals, raised for 19 comparisons: -0.54 to -0.20,
-0.45 to -0.08, and -0.40 to -0.14). The explicit term lost at five of the six Primaries, at all 10
lead days, and at every level of forecast sunniness. The verdict of experiment 644 was "refuted".

**Much of the past-weather gain was the free network's bias.** In [experiment
643](../../00643-pv-explicit-term-against-bias-shifted-free-network-and-more-placebo-permutations/result.md)
the free network's median ran 4.1 points too high on sunny middays. A 14-day error correction,
added after the results, scored 0.48 points better than the explicit term in the real months
(interval -0.66 to -0.30). One Primary carried most of the gain in experiment 639, and two
Primaries scored slightly better with the free network.

**Five explanations for the ensemble-weather loss were not supported by one test each, and a
sixth is inferred.**

- The ensemble's 3-hour and 6-hour time steps: averaging past weather into 3-hour and 6-hour
  windows still leaves a gain of 0.84 and 0.77 points ([experiment
661](../../00661-pv-ens-time-resolution-shortwave-interpolation-audit-and-bias-adapter/result.md)).
  Experiment 661 calls its separate test across lead bands weak.
- The shortwave interpolation and time stamps: the audit found zero time-stamp shift and a pipeline
  identical to production to 0.0 W/m2 (experiment 661).
- Ensemble members and cell temperature: a curve per member with a Faiman cell temperature still
  lost by 0.35, 0.23, and 0.21 points ([experiment
  655](../../00655-pv-member-level-cell-temperature-term-and-forecast-sunniness-tracker/result.md)).
- A learned bias adapter stayed 0.13 to 0.29 points behind the free network (experiment 661).
- Missing spread: building the distribution as gross demand minus a random member's solar curve
  lost by 0.29 to 0.38 points ([experiment
  664](../../00664-pv-sum-of-distributions-spread-and-recalibration/result.md)).
- **Forecast cloud error is the inferred cause and has no direct test.** The loss at every level of
  forecast sunniness (experiment 644) means it is not confined to days the ensemble calls cloudy.
  The 14-day error correction keeps almost none of its gain when it uses only forecast sunniness
  (experiment 655). That is what forecast cloud error would produce. It is also what would happen
  if scoring rows selected on observed sunniness favoured the arm that knew it.

### Converting real power to apparent power at the MVA Primaries

**Under NGED's stated power factor 1 with a small reactive floor, the linear probe loses about 0.5
points at half the midday reading.** Half the midday reading is the level of the five non-folding
MVA Primaries. The 80% interval is -1.0 to 7.7 points, so a loss of 7.7 points is not excluded
([experiment 679](../../00679-pv-mva-conversion-loss-reactive-floor-at-half-level/result.md)).
Experiment 679 measured 10.7 points (7.8 to 17.1) under the power factor 0.97 world that [experiment
656](../../00656-pv-probe-through-the-mva-fold-at-eight-primaries/result.md) and [experiment
662](../../00662-pv-probe-through-the-mva-fold-matched-level-redesign/result.md) assumed, and
[experiment 671](../../00671-pv-mva-fold-probe-fair-denominator-and-d3-d4-disagreement/result.md)
measured 5 to 12 points there at four MW Primaries. The reactive floor of experiment 679 is an
assumption of 0.10 times the median sunny-midday reading, and the real meters' reactive power is
unknown.

**The three recovery factors for the MVA linear probe give figures that differ by about 16%.**
Experiment 673 divides by 0.79, the signed-world recovery of experiment 662's first rotation, and
lists 0.68 as a sensitivity. Experiment 679 measures a signed recovery of 0.746 and recommends the
signed-world factor alone.

**The three folding Primaries (D8, D12, and D22) are report-only.** D5 and D10 are report-only for
the MVA conversion only, because their midday readings of 0.88 and 1.2 MW put a 2 MW test farm past
the fold. Their 6 MW capacities stand. At D2, D3, D4, and D19 the extra capacity of the fold model
comes almost entirely from two free reactive-power terms, so experiment 671 treats the linear probe
as the better-supported reading and the fold model as an upper bound.

## Earlier attempts

**Of 14 earlier experiments, experiment 129 passed all its gates, against a weaker placebo.** The
table is from a design review written on 7 November 2026 (brief). The review judged the probe a
neural version of experiments 235 and 479, with a likely recovery of 0.75 to 0.90.

| Experiment | Method | What it found |
|---|---|---|
| [129](../../00129-differentiable-physics-pv-capacity-known-answer-test/result.md) | Convex generalised additive model, one capacity per season-year | Passed all four gates (recovery at 5 of 5 Primaries, placebo 0.12); the range 1.2 to 1.6 MW is a scenario spread |
| [235](../../00235-pv-capacity-matched-day-pairs/result.md) | Matched sunny-minus-cloudy day pairs | Recovery 0.758; a cloud-following load added 0.295; December-to-January placebo 0.33 |
| [241](../../00241-pv-capacity-calibrated-interval/result.md) | Divide by the median recovery of other farms | Within 10% at 5 of 5; the 80% interval covers 59 to 79%, and 0 to 2% with a cloud load |
| [269](../../00269-pv-capacity-gain-decomposition/result.md) | Pooled empirical plant table driven by CAMS | 0.892 (99th-percentile ratio) and 0.966 (capacity ratio); satellite sunshine is not the limit |
| [349](../../00349-primary-demand-default-3-exp-269-capacity-in-the-stack/result.md), [352](../../00352-primary-demand-default-3-capacity-response-surface/result.md) | Better capacity inside the generalised additive model stack | Sunny score flat in capacity (0.047 worse; flat surface), per the review |
| [405](../../00405-mdn-explicit-pv-term-learned-capacity-known-answer/result.md) | Mixture-density network with learned-capacity solar term | A day-shuffled placebo kept 0.60 to 0.91 of real recovery (limit 0.20) |
| [409](../../00409-mdn-solar-term-day-of-year-and-anomaly-known-answer/result.md) | Same, with a day-of-year term and anomaly-only injection | Anomaly read-out 0.03 to 0.05 MW against seed noise of 0.18 to 0.59 MW; the test could not decide |
| [416](../../00416-mdn-explicit-solar-term-better-powered-follow-up/result.md) | Explicit solar term, 5 seeds, 12 months | Sunny-midday +0.227 points, interval -0.135 to +0.560; capacities 0.4 to 0.7 of the reference, 1.5 at D5 |
| [474](../../00474-pv-capacity-placebo-gated-ladder-and-per-primary-intervals/result.md) | Placebo-gated ladder on the generalised additive model | No rung passed: profile placebo 0.25 to 0.40, and 0.49 to 0.51 with cloud terms |
| [477](../../00477-neural-pv-module-placebo-gate-and-known-answer-on-stored-fits/result.md) | Neural PV module on stored fits, lag-masked | Placebo read 0.90 to 0.95 of the injected MW against 0.86 to 0.92 for the real farm |
| [479](../../00479-pv-capacity-anomaly-regression-rung/result.md) | Within-month anomaly regression | Placebos pass, mostly by construction; recovery 0.62 to 0.83; a cloud load fakes 0.2 to 0.3 of a 2 MW farm |
| [501](../../00501-explicit-pv-column-from-the-474-capacity-table/result.md) | Explicit PV column from the experiment 474 table | Sunny-midday -0.014 points against 0.225 detectable (the fifth null) |
| [504](../../00504-neural-demand-leak-free-recent-cams-irradiance-inputs/result.md) | Recent CAMS columns in the neural network | Leak-free columns null; target-time CAMS -0.318 at the eight MVA Primaries and -0.467 at the six MW Primaries |

**Two figures from the same review bound what a method can detect (brief).** Experiment 129 found
capacity growing 0.4 to 2.8 MW over about 4 years, which is about 0.1 to 0.7 MW per year. The
within-month minimum detectable capacity of experiment 479 is 0.05 to 0.35 MW.

## Findings that are not on one merged page

**The polygons are small relative to the weather grid (brief).** Only the 16 Primaries have an
`area_wkt` service-area polygon. The areas run from 14.7 to 215.7 km2, with a median of 102 km2,
against about 25 km2 for one CAMS pixel, so a median polygon spans about 4 pixels. A multi-pixel
CAMS download was written and never run (brief). Displacement of the sky by 5 to 17 km lowered
recovery by under 0.05 in experiment 624 (the brief says 0.02 to 0.04), and a metered neighbour's
output moved recovery by 0.008 in the unpaired medians of experiment 635 (paired difference -0.004,
90% interval -0.010 to +0.026).

**NGED states that its meters read power factor 1 with a small reactive floor.** The maintainer
relayed the statement, and experiments 673 and 679 use it. The size of the floor is an assumption.

**The probe reads above a regression that cannot see the date.** In experiment 638 a regression of
the injected capacity on the unit curve and temperature reads 0.39 to 0.42 raw for a real step in
spring 2023, against 0.57 to 0.59 for the probe. The remaining excess is 0.1 to 0.2 raw (a fraction
of the injected megawatts) for a real step and 0.05 to 0.1 for a placebo. The cause was not found.
Weekday, holiday, and pandemic columns are an untested candidate.

**The free network is biased one way on sunny middays.** In experiment 664, realised power falls
below the free network's 5% quantile in 9.3% of half-hours at day 1 to 2, and above its 95%
quantile in 1.9%. The lower band is too narrow and the upper band too wide, so a recalibration of
the width cannot fix the bias.

**The probe interval lies below the registered floor at D2, D3, D5, and D21.** The "floor" is the
sum of registered solar capacity inside the service area, with metered sites removed
([experiment
673](../../00673-pv-capacity-for-nged-per-primary-with-register-cross-check/result.md)).
The D2 flag depends on the recovery factor: with 0.68 the interval overlaps the floor. At D5 one
site of 2.70 MW accounts for most of the 4.55 MW floor. A sensitivity in experiment 673 that adds
the Feed-in Tariff band of 50 kW to 1 MW moves D10 below the floor and D11 and D16 to "overlaps".
[Experiment 267](../../00267-pv-capacity-early-years-and-register-floor/result.md) found a 2021
shortfall at D10 and D11 that remains unexplained.

**No register gives a dated capacity curve for domestic PV installed after March 2025 (brief).**
The Feed-in Tariff report covers 2010-04-01 to 2025-03-31 by postcode district, the Embedded
Capacity Register lists sites of 50 kW and above, and the Renewable Energy Planning Database lists
mostly sites of 150 kW and above. A cumulative registered curve could be built from commissioning
dates for earlier years.

**Wind capacity estimation is parked.** [Experiment
617](../../00617-wind-turbine-step-capacity-known-answer-and-fit/result.md) failed its
known-answer test.

## Open questions

- **What causes the loss with ensemble weather?** Forecast cloud error is inferred and untested.
- **How did capacity change from year to year?** The probe has season-year resolution at best, and
  the estimated drift of 0.1 to 0.7 MW per year overlaps the within-month minimum detectable
  capacity of 0.05 to 0.35 MW.
- **Why does the probe read above a regression that cannot see the date?** The cause is unknown.
- **Do the MVA figures hold?** They depend on the unknown reactive power of the meters. D2, D3, D5,
  and D21 sit below their register floors and need NGED's own list of connected solar.

## Options for January

**These options are listed in order of the information each gives per hour of work, and none is
agreed work.**

1. **A per-Primary yearly capacity trajectory, 2019 to 2026.** It asks whether a composite curve
   tracks the capacity drift. The curve combines the season-year capacities of experiments 129,
   474, and 479 from 2021, the probe of experiment 649, a cumulative register curve by service-area
   polygon, and a term for the unregistered share. It takes about a day on CPU and depends on
   nothing.
2. **A direct test of the cloud-error explanation.** It regresses the explicit-minus-free gain on
   forecast cloud error, using the stored quantiles of experiments 644 and 655. It takes about half
   a day and depends on nothing.
3. **An asymmetric sunny-midday quantile bias correction for the free network.** It asks whether the
   one-sided bias of experiment 664 can be removed. It takes about 1 to 2 GPU-hours and has a
   serving-path implication, so the maintainer decides first.
4. **Injected-farm tests at the five non-folding MVA Primaries with the reactive floor built in.**
   They ask whether the 0.5-point loss of experiment 679 holds at D2, D3, D4, D19, and D21. They
   take about 0.5 GPU-hours and need the code of experiment 662 extended.
5. **Explicit PV as a graph-detector arm.** It depends on option 1.
6. **NGED connection records.** They would settle D2, D3, D5, and D21, and are an option only if
   the maintainer later allows requesting them.
