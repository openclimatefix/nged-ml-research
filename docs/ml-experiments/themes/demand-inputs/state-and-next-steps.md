# Demand inputs: state and next steps

**Extra weather, calendar, price, and satellite inputs have given the demand forecast little beyond
the inputs already in the final model.** The [overview page](index.md) holds the XGBoost results.
This page covers the neural-network results, which came later, and ranks the options for further
tests. [Final models](../../final-models.md) describes what the production port builds. In this page,
"the Candidate" is the demand neural network that the final models use for the 16
[Primary substations](../../glossary.md#primary-substation).

**Terms.** CRPS is the continuous ranked probability score, an error measure for probabilistic
forecasts, in points of capacity. A negative contrast means the first arm is better. The six MW
Primaries are the six Primary substations whose meters report signed megawatts (MW). The eight MVA
Primaries report apparent power only, in megavolt-amperes (MVA). A cold night is a half-hour below 5
degrees C between 18:00 and 08:00 local time. A placebo is the same columns with the values taken
from a shuffled other day, so the level is right and the timing is wrong. The minimum detectable
effect is the smallest difference that seed spread alone would resolve. ENS is the ECMWF ensemble
weather forecast, ERA5 is the fifth-generation ECMWF reanalysis, and CAMS is the Copernicus
Atmosphere Monitoring Service satellite estimate of sunlight ([glossary](../../glossary.md)). M0 is
the plain modular neural network that the add-on experiments use as their control.

## What we know

**No input group tested on a neural network beat its control by the registered bar over a whole
year.** Eight bundles of thermal-lag, solar-gain, look-ahead, calendar, tariff, and national-signal
columns did not lower the CRPS of the pooled neural network by 0.03 points in 3 of 4 folds, and
seven of the eight were worse at the six MW Primaries ([Experiment
437](../../00437-neural-demand-thermal-lag-solar-gain-and-tariff-feature-bundles/result.md)).

**When the same columns entered pre-training through a matching encoder, no group beat its placebo
by the minimum detectable effect.** The closest was the thermal-lag and solar-gain group (B1) at the
six MW Primaries: -0.135 (interval -0.31 to +0.04), against a minimum detectable effect of 0.30
([Experiment
569](../../00569-modular-network-inputs-through-matching-encoders-in-pretraining-and-fine-tuning-run-1/result.md)).

**Adding the 14 thermal-lag and solar-gain columns to the modular neural network changed error
little and lowered the leak of injected shifts.** Against the control without them, the columns moved
the CRPS by +0.010 at the eight MVA Primaries and -0.046 at the six MW Primaries in the 0 to 24 hour
band. The leak of an injected shift into the forecast fell from 7.7 to 5.4 points against the
placebo ([Experiment
574](../../00574-part-d-primaries-only-modular-recipe-lag-encoder-offset-and-add-one-arms/result.md)).
Experiment 677, which builds up the Candidate one component at a time, is pending, and
[Final models](../../final-models.md) will state its result.

**The one candidate input group is the thermal-lag and trailing-temperature group, and only on cold
nights.** The gain is small and shifts between experiments:

- Experiment 437 found a gain of 0.022 at the eight MVA Primaries in 2 of 4 folds.
- [Experiment
  442](../../00442-thermal-lag-cold-night-test-on-six-winter-folds/result.md) found the cold-night
  contrast at the eight fell from the 0.05 or more of [Experiment
  441](../../00441-difficult-row-rescore-of-the-neural-demand-input-bundles/result.md) to -0.014
  (-0.102 to +0.138), and at the six it was -0.086 (-0.241 to +0.141).
- [Experiment
  515](../../00515-lagless-demand-network-trailing-weather-window-features/result.md) found
  trailing temperature windows lowered cold-night CRPS at the eight by 0.150 (10 of 10 folds) while a
  placebo lowered it by 0.071, so the experiment called the effect not resolved.
- [Experiment
  562](../../00562-modular-lag-free-network-null-input-retry-with-placebos/result.md) found a
  thermal-lag and solar-gain group 0.231 better than its control on cold nights at the six, from 3
  monthly blocks, at a minimum detectable effect of 0.227.

**Experiment 526 supersedes Experiment 515 because it used 11 seeds.** The two experiments share
folds. [Experiment
526](../../00526-lagless-demand-network-trailing-weather-window-placebo-repeat/result.md) found the
real windows beat the placebo by 0.093 (-0.133 to -0.053, 9 of 10 folds) at the eight, only just
above the minimum detectable effect of 0.083. Nothing resolved at the six or in summer. The seam
check failed: that check compares the windows built from lagged weather with windows that fill the
hours before the forecast origin with the run's first value.

**Satellite sunshine helps only when it is read at the target half-hour.** Two columns of realised
CAMS irradiance lowered the XGBoost CRPS at all 16 Primaries in the point estimates, with the
interval wholly below zero at 15 ([Experiment
480](../../00480-xgboost-cams-columns-at-mva-only-primaries-lab-conditions/result.md): -0.355 at the
eight, -0.476 at the six). A live service lacks that input for the first day ahead, because the
satellite data arrives about a day late. Satellite values a live service could hold, from whole days
at least 36 hours old, raised the neural network's CRPS by 0.006 at the eight and 0.010 at the six
([Experiment
504](../../00504-neural-demand-leak-free-recent-cams-irradiance-inputs/result.md)).

**The ensemble mean of temperature and shortwave carries the XGBoost gain from the ensemble.** Two
mean columns lowered the CRPS by 0.052 at the six and 0.029 at the eight, and the spread columns
added almost nothing ([Experiment
394](../../00394-xgboost-ensemble-mean-and-std-columns-separately/result.md)).

## What does not work

**Learned or larger components did not beat the hand-made inputs.**

- A convolutional encoder over a 24-hour weather window scored 0.07 to 0.13 worse than the
  37-column base on all rows, with every interval including zero ([Experiment
  516](../../00516-lagless-demand-network-learned-weather-window-cnn/result.md)).
- A shared calendar encoder scored 0.27 worse than the hand-made calendar columns at the six
  ([Experiment 423](../../00423-pooled-mdn-shared-date-time-encoder/result.md)).
- Larger encoders did not help beyond seed noise at the eight ([Experiment
  579](../../00579-encoder-capacity-and-architecture-arms-on-the-modular-network/result.md)).

**Price did not help, even as a training-time-only input.** A hindsight price shape for the whole day
gave -0.048 at the six and +0.037 at the eight, against a minimum detectable effect of about 0.17.
The arms that would have trained a teacher neural network on the price (price dropout and a teacher
network) were therefore not run ([Experiment
513](../../00513-lagless-demand-network-whole-day-price-shape-at-training-time/result.md)). A price
encoder trained beside the weather encoders made the fine-tuned forecast 0.29 worse at the six
([Experiment
570](../../00570-price-encoder-part-1-known-answer-test-and-pretrained-stack/result.md)). The
XGBoost price tests are on the [overview page](index.md).

**The lockdown flag is neither shown to be needed nor shown to be removable.** In [Experiment
517](../../00517-lagless-demand-network-lockdown-flag-removal/result.md), arm C0 keeps the flag, C1
removes the flag and keeps the 2020 to 2021 lockdown rows, and C2 removes the flag and drops those
rows. C1 scored 0.029 worse than C0 at the six. C2 scored 0.044 better than C1 at the eight, and
0.033 better than C0 at the eight. The experiment judges that it cannot rank the arms, because
leaving out one Primary at a time weakens both results and every difference is below the minimum
detectable effect of 0.12 to 0.13.

**The 95% and 99% upper quantiles of the six MW Primaries cover only 76% and 83% of cold
winter-night half-hours** ([Experiment
441](../../00441-difficult-row-rescore-of-the-neural-demand-input-bundles/result.md)). That shortfall
is larger than any input's gain, so calibration is a likelier lever on cold nights than another
input.

## Open questions

- **Does the trailing-temperature gain at the eight survive on a held-out winter?** Experiments 515
  and 526 share folds, so a new winter is the first independent test.
- **How much of Experiment 480's satellite gain survives when the columns come from the ensemble
  forecast?** The experiment named this as its next step.

## Next-experiment options

**The options are ranked by how much a production port depends on the answer.**

1. **An option is to wait for Experiment 677 and then test removing the 14 thermal-lag and
   solar-gain columns from the Candidate.** Experiments 574 and 569 found the columns neutral on
   error, so the question is whether the port can drop them and the pipeline step that builds them.
   Experiment 677's build-up is pending (see [Final models](../../final-models.md)), and it may
   answer the question first. The plan page lists no separate experiment for this option.
2. **An option is to retest trailing temperature on more winter folds with the placebo and seam
   check of Experiment 526.** The question is whether the 0.093 gain at the eight is real. It depends
   on option 1, because a column the final model drops needs no retest. Experiment 562 asked for a
   larger cold-night test on the best lagged neural network.
3. **An option is to re-run Experiments 569 and 562 with a second seed and a re-pretrained plain
   control.** Experiment 569 named this step, because its pre-trained control scored worse than the
   stored weights of plain M0. It matters mainly if option 1 shows the columns help.
4. **An option is to score Experiment 480's satellite columns with ensemble-forecast values.** The
   question is what remains of the 0.355 to 0.476 gain in a live service. Its result applies to the
   XGBoost path more than to the final neural network.
5. **An option is to add more seeds for the larger datetime encoder at the six.** Experiment 579
   suggested it, and its own conclusion is that no further architecture arms are needed, so the
   production value is small.

## What is parked

**The convolutional window encoder is parked.** Experiment 516 recommended spending no more fits on
learned windows until Experiment 515's placebo question was resolved, and Experiment 526 resolved it
only partly.

**Price inputs are parked.** The oracle bound of Experiment 513 is not resolvable from zero, and
Experiment 570 failed its known-answer test, so the next step named there is a controlled test with a
larger injected generator before any real-data price work. Modelling batteries and price-following
generators as a separate target is a roadmap question, not an experiment of this theme.

**The lockdown flag stays as the default.** Experiment 517 cannot rank the arms, and the flag is
zero in every scored month, so the production port can follow the default of the final model.

**Load-shift event detection is paused until January.** It is the subject of the
[lsc-detectors theme](../lsc-detectors/state-at-january-pause.md), and the options above do not
depend on it.
