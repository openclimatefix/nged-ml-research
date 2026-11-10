# Generator forecasting: state and next steps

**A small neural network that reads only the ensemble weather forecast is the final model for both
the 3 metered wind farms and the 6 metered solar farms.** The wind neural network (called S) reads
10 weather columns, 3 lead-time columns, and 3 farm columns (which farm it is). The inputs are
summaries of the ECMWF ENS ensemble forecast (see [ENS, ERA5, and
CAMS](../../glossary.md#ens-era5-and-cams)). The solar neural network (called PLP) reads weather
means and spreads, the sun's elevation and azimuth, the lead time, and a fixed hand-written power
formula. [Final models](../../final-models.md) describes both. Neither is adopted for production.

**Scores are only comparable within one experiment.** Experiments 659 and 666 report mean pinball
loss. Experiment 668 and the final models report CRPS (continuous ranked probability score), which
is twice the mean pinball loss. Both are in points of capacity (see the
[glossary](../../glossary.md#capacity-and-points-of-capacity)), and smaller is better, so a negative
contrast favours the first arm named. A q99 is the 99% quantile of a forecast, and about 1% of
observations should exceed it. Experiments 666 and 668 give the only final-model scores, and both
come from one backtest year that overlaps the development months. Experiment 659 uses 5 development
months (October, December, January, March, and June origins). Experiments 429, 461, and 466 use 53
Tuesday runs, and Experiments 54 and 57 use 52.

## What works

**A simple pooled neural network is no worse than the elaborate defaults it replaces.** In
[Experiment
659](../../00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md), wind
S scored 3.684 against 3.766 for wind default 2 (the earlier wind recipe) on rows with idle readings
zeroed. The difference is -0.082 (99% interval -0.151 to -0.008), and the interval is sensitive to
the resampling seed.

**The wind lead over wind default 2 comes from idle readings, so the two are tied.** On all rows the
difference is -0.036 (-0.087 to +0.020), and on kept rows it is +0.019 (-0.018 to +0.066).
[Experiment 429](../../00429-unified-forecaster-wind-ablations/result.md) found the same tie between
a plain pooled network and wind default 2 (10.533 against 10.518, where 10.533 is the arm that
passes members through an encoder and a one-pass version scores 10.541). Wind S is justified by
being simpler.

**Solar PLP scored 3.279 against 3.308 for the earlier solar default.** The difference is -0.028
(-0.089 to +0.012), a tie. The fixed power formula improved solar by 0.021 and changed wind by
+0.001, so wind S reads no formula.

**Calendar columns hurt wind.** Adding hour-of-day and day-of-year columns made wind S 0.094 worse
(0.012 to 0.180) in Experiment 659.

**Capacity at the 99.9th percentile of earlier power is safe for wind.** In [Experiment
461](../../00461-wind-and-solar-retrain-with-p99-9-effective-capacity/result.md), retraining the
wind network with this capacity kept the score (-0.17%, inside fit-to-fit spread) and left all 3
farms under the 1.5% limit on observations above q99.

**A year-long backtest exists.** [Experiment
666](../../00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md) produced quantile
forecasts to 14 days for 365 daily origins, and wind S matched Experiment 659 on the rows both cover
(4.174 against 4.179). [Experiment
668](../../00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
resamples origin days (a 95% day-block bootstrap), where earlier experiments resampled months.

## What does not work

**Weighting recent training rows more does not help.** No half-life from 4.5 months to 3 years
improved the wind or solar score by the margin, and wind scored 0.22% to 0.25% worse with short
half-lives ([Experiment 466](../../00466-wind-and-solar-recency-weighting/result.md)). Recency
weighting stays closed until the ECMWF ensemble history is longer than 2 years.

**The 14-day error correction made the wind interval include zero and solar worse.** The correction
changed the CRPS by +0.437 (+0.215 to +0.664) for solar and +0.413 (-0.003 to +0.924) for wind
([Experiment 23](../../00023-lagged-power-and-error-correction-solar-wind/result.md)). A
recent-capacity feature changed it within noise ([Experiment
26](../../00026-capacity-features-solar-wind/result.md)). The final wind and solar networks read no
power lag.

**A hidden Markov model could not track wind-farm capacity steps.** It found 31%, 32%, and 14% of
outages of 2 or more turbines lasting 14 days or more, and raised 3.5 false steps a year at two
farms, because only 11% to 21% of days have wind strong enough to show full output ([Experiment
617](../../00617-wind-turbine-step-capacity-known-answer-and-fit/result.md)).

**Texture-based idle detectors did not beat the existing idle mask.** They missed 3-hour idle blocks
(recall 0.08 to 0.33), and no detector passed the gate at any wind farm ([Experiment
444](../../00444-wind-idle-detectors-label-free-bake-off/result.md)).

**A second weather product added little for solar and a small amount for wind.** The UKV forecast
(the Met Office's 2 km weather forecast) lowered the wind quantile score by 0.198 at 24 to 54 hours
ahead, measured on a research archive that differs from the live feed ([Experiment
54](../../00054-product-trained-xgboost-wind-farms/result.md)). For solar the gain was -0.035, which
the data cannot resolve ([Experiment
57](../../00057-product-trained-xgboost-solar-farms/result.md)).

**One metered generator that is neither wind nor solar is not forecastable from the inputs held.**
In [Experiment 476](../../00476-generator-x-first-forecast-dispatch-lags-weather-price/result.md),
no network version beat same-hour climatology: the best gain was 0.015 against a minimum detectable
effect of 0.019.  Lagged market price was the one input worth a dispatch-data experiment, and no
live price feed exists. The next step is a different kind of input, such as the operator's dispatch
schedule.

**A rolling capacity does not give one rule for wind and solar.** In [Experiment
505](../../00505-rolling-p99-9-effective-capacity-for-wind-and-solar/result.md), a 730-day window
(the window the final models use) gave a pooled share above q99 of 0.849% for wind and 1.449% for
solar. It fails the 1.5% limit at Solar B (1.506%). The advice is the all-history p99.9 capacity for
wind and the old capacity for Solar A and Solar E.

**The ensemble's 100 m wind is about 1.1 m/s slower than the farms' power implies.** A speed factor
closed 70% to 79% of the gap to the ERA5 oracle ([Experiment
307](../../00307-wind-farm-power-as-wind-sensor/result.md)). Wind S does not use that design.

## Open weaknesses

**The upper tail on very sunny days is too low in every solar arm.** On 27 sunny farm-days, 10.6% to
12.7% of observations exceeded q99 (Experiment 659). Experiment 666's own sunny-day definition gives
2.8%, so the two figures are not comparable and the problem is unresolved.

**The wind upper tail is too low in plain neural networks.** Before recalibration, 2.17% to 2.78% of
observations exceeded q99 in every arm of Experiment 429 (wind default 2: 2.15%), against a limit of
1.5%. A table of quantile rescalings clipped at the farm's largest earlier output brought every farm
under the limit ([Experiment
450](../../00450-wind-upper-tail-check-with-table-p-and-sweep/result.md), [Experiment
451](../../00451-farm-a-upper-tail-exceedances-and-capacity-state/result.md)).

**The final wind file has no such table, and its share above q99 is over the limit.** The share is
0.016 at days 0 to 1 and 0.023 at days 10 to 14 (Experiment 666), against 0.015.

**Wind coverage is below nominal on high-wind days and in some months.** On kept rows in the 5
development months, q05-to-q95 coverage in January and March was 0.88 to 0.89 for S and 0.855 and
0.867 for wind default 2 (Experiment 659).  On the 44 high-wind days, coverage is 0.82 for S and
0.83 for wind default 2, with 4.4% to 5.2% above q99.

**Wind S is no better than climatology beyond day 10.** It scores 9.86 at days 10 to 14, against
9.68 for climatology (Experiment 666).

**Solar PLP is worse than climatology from November to March at days 10 to 14.** Experiment 666
shows the gap at 0.03 to 0.42 points, although the mean absolute error gain is 5.7%. PLP is better
than climatology at days 10 to 14 from April to September.

**The solar median is high in spring and low in autumn.** One year with quarterly refits cannot
separate a seasonal effect from refit age.

**The dawn and dusk bands are badly calibrated.** A night gate at each half-hour's own sun position
gave coverage of 0.69 and 0.75 at dawn and dusk at two farms. Experiment 684 covered only 3 months
of origins ([Experiment
684](../../00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/result.md)).

## Options

**Options 1 and 2 come from the open questions of the [final models](../../final-models.md) page.**
Options 3, 4, and 5 come from Experiments 505, 54, and 57.

1. An option is to test a calendar input or more frequent refits for solar, to see whether they
   remove the seasonal bias in the median.
2. An option is to test the open question "Whether wind needs a power lag or a different long-lead
   rule", given wind's score against climatology at days 10 to 14.
3. An option is to test a 730-day capacity window with a share limit judged by an interval rather
   than a point estimate, plus a weekly refit at Solar A and Solar E (Experiment 505).
4. An option is to build a matched history of the live UKV feed and test it for wind at lead days 1
   and 2. This needs a source that holds live UKV at those leads, which Open-Meteo's history does
   not (it reaches lead day 0 only; Experiment 54).
5. An option is to test a satellite-based nowcast at lead day 0 for solar farms, because UKV and a
   second global ensemble each added little. Experiment 57 lists it as an untested candidate.

## Parked

**The wind refit-schedule test is parked.** In the development months, a single fit, a quarterly
refit, and a twice-monthly refit scored within 0.013 of each other for wind S (Experiment 659).

**Load-shift event detection and unmetered PV capacity** are paused until January and belong to the
demand themes.
