# Metered solar farms: state and next steps

**This page summarises what the experiments on the six metered solar farms (Solar A to F) have
found, and which experiments an option could run next.** Every figure comes from a merged result
page, linked at its first use. The scores are the quantile score and the continuous ranked
probability score (CRPS), both defined in the [glossary](../../glossary.md), which also defines ENS
(the ensemble weather forecast), ERA5, and CAMS (the Copernicus Atmosphere Monitoring Service).
"Points of capacity" are quantile-score units as a fraction of the farm's capacity, and in a signed
contrast a negative value is better. The Solar A to F labels follow the earlier experiment pages,
and the labels in [experiment
668](../../00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)
differ. [Final models](../../final-models.md) describes the final solar network.

## Take-home message

**A small neural network that reads only the ensemble weather forecast forecasts the six farms about
as well as any more elaborate design we tried.** In the one-year backtest of the final solar
network, the CRPS on daylight half-hours is 37.1% better than climatology at days 0 to 1 (95%
day-block interval 34.6% to 39.6%) and 7.6% better at days 10 to 14 (6.0% to 9.0%). Climatology is
the farm's own power for the same month and half-hour of day in earlier history. The backtest is one
year that overlaps the development months ([experiment
668](../../00668-documentation-of-the-one-year-backtest-files-models-training-and-performance/result.md)).

**The weather input and the upper tail may be what limits the forecast.** The network settings, the
plant model, pretraining, and extra weather sources each moved the score by 2% or less. No
experiment has yet isolated the weather input as the cause of the shortfall on very sunny days.

Two options bear on this. One is to test another irradiance product or a blend of products, and the
other is to add XGBoost plant-response columns (options 1 and 2 below).

## What works

**The final network beats climatology on average at days 10 to 14, but not in every season.** The
network PLP (a shared mixture-density neural network over the six farms, with 3 Gaussian components)
reads the ensemble mean and spread of the weather, the sun's elevation and azimuth, the lead, and a
fixed physics-lite power, with no calendar or farm column ([experiment
659](../../00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md),
[experiment 666](../../00666-one-year-backtest-of-simple-wind-and-solar-networks/result.md)). At
days 10 to 14 it is worse than climatology from November to March (by 0.03 to 0.42 points of
capacity) and better from April to September.

**The assembled forecast keeps its skill from 1 to 10 days ahead.** The blend of [experiment
310](../../00310-metered-farm-long-lead-scoring/result.md) has a quantile score 38.7% better than
climatology at day 1 (interval -44.2% to -33.3%, 98.75%) and 10.1% better at days 8 to 10. The
mixture-density network alone is 37.9% better at day 1. These results use the physical plant, which
the final network lacks. A plant-fed XGBoost model has no skill over climatology at days 8 to 10.

**One pass over the ensemble mean and spread is enough.** [Experiment
430](../../00430-unified-forecaster-solar-and-pretraining-ablations/result.md) tested four richer
designs: passing the 51 ensemble members through an encoder, a plant model inside the network,
pretraining on ERA5 and CAMS, and point masses at zero and at capacity. None beat the one-pass
network by the 1% margin.

**A pooled network matches separate networks.** One network over the six farms scores within 0.8% of
a separate network for each farm ([experiment
345](../../00345-pooled-solar-wind-mixture-density-network/result.md)). The interval is about 4
points wide. Adding wind farms makes solar 0.7% to 2.3% worse, so the two fuels are trained
separately.

**A physical plant model adds little once the neural network sees the weather.** A weather-only
forecast scores within 0% to 3.4% of the plant-based forecast up to 10 days ahead, and the intervals
include zero except for XGBoost, which is 2.8% better ([experiment
313](../../00313-metered-farm-pure-xgboost-and-mdn-vs-plant/result.md)).

**The neural network can be refitted once a year.** Moving from quarterly to annual refits changes
the quantile score by less than 0.6% for both one-pass networks tested ([experiment
493](../../00493-solar-annual-against-quarterly-refits/result.md)). Experiment 493 did not test the
network PLP.

**Hard zeros at night are safe.** A zero set at each half-hour's own sun position removes every
non-zero night quantile and changes no daylight score, tested at two farms ([experiment
684](../../00684-backtest-positive-demand-output-distributions-and-physical-solar-night-zero/result.md)).
Solar B has small negative readings on 8% of night half-hours ([experiment
433](../../00433-solar-idle-noise-data-look/result.md)), so the zero is a forecast rule, not a
property of the data.

## What does not work

**The upper tail on very sunny days is too low in every design.** On very sunny days 10.6% to 12.7%
of observations exceed the 0.99 quantile in every simple arm ([experiment
659](../../00659-simple-shared-network-with-fixed-physics-lite-power-wind-and-solar/result.md)). A
band-widening rule fitted on earlier months did not improve the score. [Experiment
318](../../00318-metered-farm-mdn-head-more-features/result.md) found that clock and cloud-spread
inputs lower the 0.99 pinball loss by about 5% at days 3 to 7, which leaves the overall score
unchanged and does not cure the sunny-day exceedance share.

**Dawn and dusk are badly calibrated.** At the two farms tested, the coverage of the 5% to 95% band
is 0.69 and 0.75, and 0.30 and 0.24 of observations fall at or below the 0.01 quantile (experiment
684). The 0.01-quantile shares on this page are not comparable, because each is measured on
different rows.

**The median is biased by season.** It is high in spring and low in autumn, and the network has no
calendar input (experiment 666). One year with quarterly refits cannot separate a seasonal effect
from refit age.

**Network settings, more history, and other weather sources add little.**

- None of 25 one-at-a-time changes to the neural network improves the score by 0.5% ([experiment
  443](../../00443-solar-network-hyperparameter-and-architecture-sweep/result.md)).
- Adding the Met Office UKV forecast to the ensemble lowers the day-ahead error by 1.8% and 1.3%
  ([experiment 229](../../00229-metered-farm-ens-ukv-blend/result.md)), and adding the German
  ICON-D2 forecast as well gains about 1.0% ([experiment
  246](../../00246-metered-farm-ens-ukv-icond2-blend/result.md)).
- The ensemble's irradiance on the cells around a farm improves the quantile score by 0.36%
  ([experiment 292](../../00292-metered-farm-ens-neighbourhood/result.md)).
- Nearly 6 years of past-weather rows added to the training lower the quantile score by 1.7%
  ([experiment 275](../../00275-metered-farm-long-history-neural-head/result.md)).

**Per-farm plant parameters do not help.** Giving each farm its own panel tilt, azimuth, heat loss,
and inverter limit raises the fraction of output explained by 0.003 (0.953 against 0.950); the same
plants fit only 0.91 with the ensemble weather ([experiment
218](../../00218-metered-solar-farm-plant-fit/result.md)).

## Open questions

**Exact zeros may not explain the lower-tail shortfall.** [Experiment
321](../../00321-metered-farm-lower-tail-recalibration-and-tail-head/result.md) shows that exact
zeros (3.0% of daylight half-hours) account for the earlier 3.3% to 3.7% at or below the 0.01
quantile. Strictly below it, the share is 1.2% to 1.4% with the farm's own history, 1.4% to 2.1%
with none, and 4.6% at Solar E. [Experiment
430](../../00430-unified-forecaster-solar-and-pretraining-ablations/result.md) reports 5.1% to 5.3%
at or below the quantile on its own rows, which zeros alone may not explain. The final network has
0.028 to 0.032 below the 0.01 quantile, about 3 times too frequent (experiment 666).

**The cause of the sunny-day shortfall is not isolated.** [Experiment
293](../../00293-metered-farm-plant-tilt-and-clip/result.md) found that a higher or softer cap on
the plant's power does nothing, so the shape of the plant is not the cause. Experiment 218 points at
the irradiance forecast, and no experiment has tested that directly.

**Two farms (Solar D and Solar E) have idle runs of 17 to 24 days on clear days.** Outages are too
rare in the scored year to test an availability state: five farms have at most 34 outage half-hours
and the sixth has 165 ([experiment 301](../../00301-metered-farm-outage-state-band/result.md)).

**The value of the forecast for curtailment is unresolved.** Forecasting each farm's output cut the
curtailment cost by 40% at one substation and raised it by 24% at another, which is smaller than the
uncertainty ([experiment 255](../../00255-metered-farm-node-curtailment-value/result.md)).

## Options for the next experiments

**Each option comes from the research plan or from a merged page's "What to try next".**

1. **An option is to test another irradiance product or a blend of products** (research plan, from
   experiment 218). The question is whether it raises the fraction of output explained by the plant
   model by at least 0.02 over CAMS alone. Experiments 443 and 292 name the input as the next place
   to look, and [experiment
   618](../../00618-solar-effective-capacity-sunny-rows-and-known-answer-test/result.md) names ERA5
   surface radiation for Solar D.
2. **An option is to give an XGBoost model the plant's mean response and the spread of the 51 plant
   responses as two extra columns, refitted weekly** (research plan, from experiment 224). The
   plan's failure rule is a quantile-score gain under 2%, or a 0.99 pinball loss above the plant's.
3. **An option is to remove or measure the seasonal bias in the median** with a calendar input or
   monthly refits (experiment 666 and Final models).
4. **An option is to ask NGED (National Grid Electricity Distribution) what happened at Solar B**,
   where the output in an earlier period was 11% lower, not higher (experiment 618).
5. **An option is to score the final network with the node metric once each farm is paired with its
   substation** ([experiment 300](../../00300-metered-farm-best-guess-assembly/result.md)), which
   needs more substations than experiment 255 had.

## What is parked, and why

**Capacity tracking from the farms' output is paused until January.** It belongs to the
unmetered-PV-capacity work that the maintainer paused. Experiment 618 finds a sudden change of 8% or
more at Solar A, B, C, and F, with at most 0.7 false changes a year. [Experiment
636](../../00636-metered-solar-monthly-slope-capacity-known-answer-test/result.md) stopped its line
under its stop rule.
